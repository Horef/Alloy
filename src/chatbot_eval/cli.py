from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

from .adapters import HttpChatbotAdapter
from .config import load_settings
from .documents import load_chunks
from .evaluator import Evaluator
from .generator import GenerationOptions, SilverSetGenerator
from .io import read_questions, write_evaluations, write_questions
from .llm import GeminiStructuredLLM
from .logging_utils import configure_logging
from .models import ChatbotResult, SilverQuestion
from .report import write_report
from .results_io import ResultColumns, read_premade_results
from .topics import infer_topics

logger = logging.getLogger(__name__)


def _ensure_config(path: Path) -> None:
    if path.exists():
        return
    example = Path("config.example.toml")
    if example.exists() and path.name == "config.toml":
        shutil.copyfile(example, path)
        raise SystemExit(f"Created {path}. Review it, create .env from .env.example, then rerun.")
    raise SystemExit(f"Config file not found: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chatbot-eval", description="Generate and evaluate chatbot question sets")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars for production/non-interactive runs")
    parser.add_argument("--log-file", type=Path, help="Write operational logs to this file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate a reviewable silver question set")
    generate.add_argument("--documents", type=Path, required=True)
    generate.add_argument("--output", type=Path, default=Path("outputs/questions"))
    generate.add_argument("--max-questions", type=int)
    generate.add_argument("--topic", help="Restrict generation to this requested topic")
    generate.add_argument("--topic-count", type=int, help="Maximum questions for --topic")
    generate.add_argument("--unanswerable-ratio", type=float)

    evaluate = commands.add_parser("evaluate", help="Call a chatbot and judge its responses")
    evaluate.add_argument("--questions", type=Path, required=True)
    evaluate.add_argument("--chatbot-url", required=True)
    evaluate.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    evaluate.add_argument("--question-field", default="question")
    evaluate.add_argument("--answer-field", default="answer", help="Dotted response path, e.g. data.answer")
    evaluate.add_argument("--context-field", default="retrieved_context", help="Dotted response path")
    evaluate.add_argument("--header", action="append", default=[], help="HTTP header as Name=Value; repeatable")
    evaluate.add_argument("--approved-only", action="store_true")

    evaluate_file = commands.add_parser("evaluate-file", help="Judge premade chatbot questions and answers")
    evaluate_file.add_argument("--results", type=Path, required=True, help="Input .xlsx, .csv, or .jsonl file")
    evaluate_file.add_argument("--sheet", help="Excel sheet name; defaults to the active sheet")
    evaluate_file.add_argument("--output", type=Path, default=Path("outputs/file-evaluation"))
    evaluate_file.add_argument("--question-column")
    evaluate_file.add_argument("--expected-answer-column")
    evaluate_file.add_argument("--answer-column")
    evaluate_file.add_argument("--source-column")
    evaluate_file.add_argument("--id-column")
    evaluate_file.add_argument("--topic-column")
    evaluate_file.add_argument("--answerable-column")
    evaluate_file.add_argument("--error-column")
    evaluate_file.add_argument("--infer-topics", action="store_true", help="Infer consistent Hebrew topics with Gemini when no topic column exists")
    return parser


def _headers(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --header {value!r}; expected Name=Value")
        key, content = value.split("=", 1)
        result[key] = os.path.expandvars(content)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _ensure_config(args.config)
    settings = load_settings(args.config)
    log_path = args.log_file or (Path(settings.log_file) if settings.log_file else None)
    configure_logging(log_path, args.log_level or settings.log_level)
    progress_enabled = settings.progress_enabled and not args.no_progress
    llm = GeminiStructuredLLM(settings.api_key, settings.max_retries)
    logger.info("command_started command=%s progress_enabled=%s", args.command, progress_enabled)
    if args.command == "generate":
        maximum = args.max_questions or settings.max_questions
        if maximum < 1:
            raise ValueError("--max-questions must be positive")
        ratio = settings.unanswerable_ratio if args.unanswerable_ratio is None else args.unanswerable_ratio
        if not 0 <= ratio < 1:
            raise ValueError("--unanswerable-ratio must be in [0, 1)")
        chunks = load_chunks(args.documents, settings.chunk_chars, settings.chunk_overlap_chars, progress_enabled)
        options = GenerationOptions(
            max_questions=maximum, batch_chunks=settings.batch_chunks,
            min_topic_questions=settings.min_topic_questions, max_topic_share=settings.max_topic_share,
            unanswerable_ratio=ratio, requested_topic=args.topic, requested_topic_count=args.topic_count,
        )
        questions, topics = SilverSetGenerator(llm, settings.generation_model, progress_enabled).generate(chunks, options)
        csv_path, jsonl_path = write_questions(questions, args.output)
        print(f"נוצרו {len(questions)} שאלות ב-{len(topics)} נושאים שזוהו.")
        print(f"קובץ CSV לבדיקה: {csv_path}\nקובץ JSONL עם מקורות: {jsonl_path}")
        if len(questions) < maximum:
            print(f"הערה: נוצרו פחות שאלות מהמבוקש ({len(questions)}/{maximum}), משום ששאלות ללא ביסוס או שאלות כפולות הוסרו.")
        logger.info("command_completed command=generate question_count=%d output=%s", len(questions), args.output)
        return 0

    if args.command == "evaluate-file":
        columns = ResultColumns(
            question=args.question_column, expected_answer=args.expected_answer_column,
            answer=args.answer_column, source=args.source_column, id=args.id_column,
            topic=args.topic_column, answerable=args.answerable_column, error=args.error_column,
        )
        pairs = read_premade_results(
            args.results, sheet_name=args.sheet, columns=columns, progress_enabled=progress_enabled,
        )
        if args.infer_topics:
            infer_topics(pairs, llm, settings.judge_model, progress_enabled=progress_enabled)

        class NeverCalledAdapter:
            def ask(self, question: SilverQuestion) -> ChatbotResult:
                raise RuntimeError("Premade evaluation must not invoke a chatbot")

        records = Evaluator(NeverCalledAdapter(), llm, settings.judge_model, progress_enabled).judge_results(pairs)
        details_csv, details_jsonl = write_evaluations(records, args.output)
        summary_json, report_html = write_report(records, args.output)
        print(f"הוערכו {len(records)} תוצאות קיימות.\nפירוט: {details_csv}\nנתונים גולמיים: {details_jsonl}\nסיכום: {summary_json}\nדוח: {report_html}")
        logger.info("command_completed command=evaluate-file record_count=%d output=%s", len(records), args.output)
        return 0

    questions = read_questions(args.questions, approved_only=args.approved_only)
    if not questions:
        raise ValueError("No questions selected for evaluation")
    adapter = HttpChatbotAdapter(
        args.chatbot_url, args.question_field, args.answer_field, args.context_field,
        settings.request_timeout_seconds, _headers(args.header),
    )
    records = Evaluator(adapter, llm, settings.judge_model, progress_enabled).evaluate(questions)
    details_csv, details_jsonl = write_evaluations(records, args.output)
    summary_json, report_html = write_report(records, args.output)
    print(f"הוערכו {len(records)} שאלות.\nפירוט: {details_csv}\nנתונים גולמיים: {details_jsonl}\nסיכום: {summary_json}\nדוח: {report_html}")
    logger.info("command_completed command=evaluate record_count=%d output=%s", len(records), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
