from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
from pathlib import Path

from .adapters import HttpChatbotAdapter
from .artifacts import EvaluationCheckpoint, RunManifest, evaluation_fingerprint, input_inventory
from .config import load_settings
from .documents import load_chunks
from .evaluator import Evaluator
from .generator import GenerationOptions, SilverSetGenerator
from .insights import generate_insights, write_insights
from .io import read_questions, write_evaluations, write_questions
from .llm import GeminiStructuredLLM
from .logging_utils import configure_logging
from .models import ChatbotResult, SilverQuestion
from .prompt_generator import SystemPromptGenerator, write_prompt_package
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
    generate.add_argument("--user-variation-ratio", type=float, help="Share of the total set reserved for realistic user phrasings")
    generate.add_argument("--ambiguous-variation-share", type=float, help="Share of user variations that should require clarification")
    generate.add_argument(
        "--exclude-questions", type=Path,
        help="Existing silver CSV/JSONL whose questions must not be generated again",
    )

    generate_prompt = commands.add_parser("generate-prompt", help="Generate a reviewable Hebrew system prompt from a document base")
    generate_prompt.add_argument("--documents", type=Path, required=True)
    generate_prompt.add_argument("--output", type=Path, default=Path("outputs/prompt"))
    generate_prompt.add_argument("--assistant-name", default="העוזר הדיגיטלי")
    generate_prompt.add_argument("--audience", default="משתמשי הארגון")

    evaluate = commands.add_parser("evaluate", help="Call a chatbot and judge its responses")
    evaluate.add_argument("--questions", type=Path, required=True)
    evaluate.add_argument("--chatbot-url", required=True)
    evaluate.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    evaluate.add_argument("--question-field", default="question")
    evaluate.add_argument("--answer-field", default="answer", help="Dotted response path, e.g. data.answer")
    evaluate.add_argument("--context-field", default="retrieved_context", help="Dotted response path")
    evaluate.add_argument("--header", action="append", default=[], help="HTTP header as Name=Value; repeatable")
    evaluate.add_argument("--approved-only", action="store_true")
    evaluate.add_argument("--generate-insights", action="store_true", help="Generate an optional Hebrew cross-result insights block")
    evaluate.add_argument("--hide-correct-answer-metrics", action="store_true", help="Hide correct-answer metrics from the report summary and topic table")
    evaluate.add_argument("--resume", action="store_true", help="Resume completed rows from a compatible checkpoint")
    evaluate.add_argument("--checkpoint", type=Path, help="Checkpoint JSONL path; defaults inside the output directory")

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
    evaluate_file.add_argument("--generate-insights", action="store_true", help="Generate an optional Hebrew cross-result insights block")
    evaluate_file.add_argument("--hide-correct-answer-metrics", action="store_true", help="Hide correct-answer metrics from the report summary and topic table")
    evaluate_file.add_argument("--resume", action="store_true", help="Resume completed rows from a compatible checkpoint")
    evaluate_file.add_argument("--checkpoint", type=Path, help="Checkpoint JSONL path; defaults inside the output directory")
    return parser


def _headers(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --header {value!r}; expected Name=Value")
        key, content = value.split("=", 1)
        result[key] = os.path.expandvars(content)
    return result


def _optional_insights(args, records, llm, model):
    if not args.generate_insights:
        return None, None
    try:
        insights = generate_insights(records, llm, model)
        return insights, write_insights(insights, args.output)
    except Exception:
        logger.exception("insight_generation_failed; continuing without insights")
        return None, None


def _checkpointed_evaluation(args, items, evaluator: Evaluator, *, premade: bool):
    fingerprints = {
        question.id: evaluation_fingerprint(question, result if premade else None)
        for question, result in items
    }
    checkpoint = EvaluationCheckpoint(
        args.checkpoint or args.output / "evaluation_checkpoint.jsonl",
        fingerprints,
        resume=args.resume,
    )
    completed = checkpoint.load() if args.resume else {}
    pending = [(question, result) for question, result in items if question.id not in completed]
    logger.info(
        "evaluation_resume_state completed=%d pending=%d checkpoint=%s",
        len(completed), len(pending), checkpoint.path,
    )
    if premade:
        new_records = evaluator.judge_results(pending, on_record=checkpoint.append) if pending else []
    else:
        new_records = evaluator.evaluate([question for question, _ in pending], on_record=checkpoint.append) if pending else []
    for question, _ in items:
        if question.id in completed:
            completed[question.id].question.topic = question.topic
    by_id = {**completed, **{record.question.id: record for record in new_records}}
    return [by_id[question.id] for question, _ in items], len(completed), checkpoint.path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _ensure_config(args.config)
    settings = load_settings(args.config)
    log_path = args.log_file or (Path(settings.log_file) if settings.log_file else None)
    configure_logging(log_path, args.log_level or settings.log_level)
    progress_enabled = settings.progress_enabled and not args.no_progress
    llm = GeminiStructuredLLM(settings.api_key, settings.max_retries)
    logger.info("command_started command=%s progress_enabled=%s", args.command, progress_enabled)
    if args.command == "generate-prompt":
        with RunManifest(
            args.output, command=args.command, settings=settings, inputs=[args.documents],
            parameters={"assistant_name": args.assistant_name, "audience": args.audience},
        ) as manifest:
            chunks = load_chunks(args.documents, settings.chunk_chars, settings.chunk_overlap_chars, progress_enabled)
            topic_generator = SilverSetGenerator(llm, settings.generation_model, progress_enabled)
            topics = topic_generator.discover_topics(chunks, settings.batch_chunks)
            package = SystemPromptGenerator(llm, settings.generation_model).generate(
                chunks, topics, assistant_name=args.assistant_name, audience=args.audience,
            )
            prompt_path, package_path = write_prompt_package(package, args.output)
            manifest.complete(
                topic_count=len(topics), chunk_count=len(chunks),
                outputs=input_inventory([prompt_path, package_path]),
            )
        print(f"Generated a reviewable system prompt across {len(topics)} discovered topics.")
        print(f"System prompt: {prompt_path}\nReview package: {package_path}")
        logger.info("command_completed command=generate-prompt topic_count=%d output=%s", len(topics), args.output)
        return 0

    if args.command == "generate":
        maximum = args.max_questions or settings.max_questions
        if maximum < 1:
            raise ValueError("--max-questions must be positive")
        ratio = settings.unanswerable_ratio if args.unanswerable_ratio is None else args.unanswerable_ratio
        if not 0 <= ratio < 1:
            raise ValueError("--unanswerable-ratio must be in [0, 1)")
        variation_ratio = settings.user_variation_ratio if args.user_variation_ratio is None else args.user_variation_ratio
        ambiguous_share = settings.ambiguous_variation_share if args.ambiguous_variation_share is None else args.ambiguous_variation_share
        if not 0 <= variation_ratio < 1:
            raise ValueError("--user-variation-ratio must be in [0, 1)")
        if not 0 <= ambiguous_share <= 1:
            raise ValueError("--ambiguous-variation-share must be in [0, 1]")
        if ratio + variation_ratio >= 1:
            raise ValueError("unanswerable_ratio + user_variation_ratio must be below 1")
        if args.topic_count is not None and args.topic_count < 1:
            raise ValueError("--topic-count must be positive")
        manifest_inputs = [args.documents] + ([args.exclude_questions] if args.exclude_questions else [])
        with RunManifest(
            args.output, command=args.command, settings=settings, inputs=manifest_inputs,
            parameters={
                "max_questions": maximum, "topic": args.topic, "topic_count": args.topic_count,
                "unanswerable_ratio": ratio, "user_variation_ratio": variation_ratio,
                "ambiguous_variation_share": ambiguous_share,
            },
        ) as manifest:
            chunks = load_chunks(args.documents, settings.chunk_chars, settings.chunk_overlap_chars, progress_enabled)
            excluded_questions = ()
            if args.exclude_questions:
                excluded_questions = tuple(q.question for q in read_questions(args.exclude_questions))
                logger.info("generation_exclusions_loaded count=%d", len(excluded_questions))
            options = GenerationOptions(
                max_questions=maximum, batch_chunks=settings.batch_chunks,
                min_topic_questions=settings.min_topic_questions, max_topic_share=settings.max_topic_share,
                unanswerable_ratio=ratio, user_variation_ratio=variation_ratio,
                ambiguous_variation_share=ambiguous_share,
                max_candidate_rounds=settings.max_candidate_rounds,
                requested_topic=args.topic, requested_topic_count=args.topic_count,
                excluded_questions=excluded_questions,
            )
            questions, topics = SilverSetGenerator(llm, settings.generation_model, progress_enabled).generate(chunks, options)
            csv_path, jsonl_path = write_questions(questions, args.output)
            manifest.complete(
                question_count=len(questions), topic_count=len(topics), chunk_count=len(chunks),
                outputs=input_inventory([csv_path, jsonl_path]),
            )
        print(f"Generated {len(questions)} questions across {len(topics)} discovered topics.")
        print(f"Review CSV: {csv_path}\nProvenance JSONL: {jsonl_path}")
        if len(questions) < maximum:
            print(f"Note: generated fewer than requested ({len(questions)}/{maximum}) because unsupported or duplicate candidates were discarded.")
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

        with RunManifest(
            args.output, command=args.command, settings=settings, inputs=[args.results],
            parameters={
                "sheet": args.sheet, "infer_topics": args.infer_topics,
                "generate_insights": args.generate_insights, "resume": args.resume,
            },
        ) as manifest:
            records, resumed_count, checkpoint_path = _checkpointed_evaluation(
                args, pairs, Evaluator(NeverCalledAdapter(), llm, settings.judge_model, progress_enabled), premade=True,
            )
            insights, insights_path = _optional_insights(args, records, llm, settings.judge_model)
            details_csv, details_jsonl = write_evaluations(records, args.output)
            summary_json, report_html = write_report(
                records, args.output, insights,
                show_correct_answer_metrics=not args.hide_correct_answer_metrics,
            )
            output_paths = [details_csv, details_jsonl, summary_json, report_html, checkpoint_path]
            if insights_path:
                output_paths.append(insights_path)
            manifest.complete(
                record_count=len(records), resumed_count=resumed_count,
                outputs=input_inventory(output_paths),
            )
        print(f"Evaluated {len(records)} premade results.\nDetails: {details_csv}\nRaw details: {details_jsonl}\nSummary: {summary_json}\nReport: {report_html}")
        if insights_path:
            print(f"Insights: {insights_path}")
        logger.info("command_completed command=evaluate-file record_count=%d output=%s", len(records), args.output)
        return 0

    questions = read_questions(args.questions, approved_only=args.approved_only)
    if not questions:
        raise ValueError("No questions selected for evaluation")
    adapter = HttpChatbotAdapter(
        args.chatbot_url, args.question_field, args.answer_field, args.context_field,
        settings.request_timeout_seconds, _headers(args.header),
    )
    with RunManifest(
        args.output, command=args.command, settings=settings, inputs=[args.questions],
        parameters={
            "chatbot_url_sha256": hashlib.sha256(args.chatbot_url.encode()).hexdigest(),
            "question_field": args.question_field, "answer_field": args.answer_field,
            "context_field": args.context_field, "approved_only": args.approved_only,
            "generate_insights": args.generate_insights, "resume": args.resume,
        },
    ) as manifest:
        items = [(question, ChatbotResult(question_id=question.id, answer="")) for question in questions]
        records, resumed_count, checkpoint_path = _checkpointed_evaluation(
            args, items, Evaluator(adapter, llm, settings.judge_model, progress_enabled), premade=False,
        )
        insights, insights_path = _optional_insights(args, records, llm, settings.judge_model)
        details_csv, details_jsonl = write_evaluations(records, args.output)
        summary_json, report_html = write_report(
            records, args.output, insights,
            show_correct_answer_metrics=not args.hide_correct_answer_metrics,
        )
        output_paths = [details_csv, details_jsonl, summary_json, report_html, checkpoint_path]
        if insights_path:
            output_paths.append(insights_path)
        manifest.complete(
            record_count=len(records), resumed_count=resumed_count,
            outputs=input_inventory(output_paths),
        )
    print(f"Evaluated {len(records)} questions.\nDetails: {details_csv}\nRaw details: {details_jsonl}\nSummary: {summary_json}\nReport: {report_html}")
    if insights_path:
        print(f"Insights: {insights_path}")
    logger.info("command_completed command=evaluate record_count=%d output=%s", len(records), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
