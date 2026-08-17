from __future__ import annotations

import argparse
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
from .report import write_report


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
    llm = GeminiStructuredLLM(settings.api_key, settings.max_retries)
    if args.command == "generate":
        maximum = args.max_questions or settings.max_questions
        if maximum < 1:
            raise ValueError("--max-questions must be positive")
        ratio = settings.unanswerable_ratio if args.unanswerable_ratio is None else args.unanswerable_ratio
        if not 0 <= ratio < 1:
            raise ValueError("--unanswerable-ratio must be in [0, 1)")
        chunks = load_chunks(args.documents, settings.chunk_chars, settings.chunk_overlap_chars)
        options = GenerationOptions(
            max_questions=maximum, batch_chunks=settings.batch_chunks,
            min_topic_questions=settings.min_topic_questions, max_topic_share=settings.max_topic_share,
            unanswerable_ratio=ratio, requested_topic=args.topic, requested_topic_count=args.topic_count,
        )
        questions, topics = SilverSetGenerator(llm, settings.generation_model).generate(chunks, options)
        csv_path, jsonl_path = write_questions(questions, args.output)
        print(f"Generated {len(questions)} questions across {len(topics)} discovered topics.")
        print(f"Review CSV: {csv_path}\nProvenance JSONL: {jsonl_path}")
        if len(questions) < maximum:
            print(f"Note: generated fewer than requested ({len(questions)}/{maximum}) because unsupported or duplicate candidates were discarded.")
        return 0


    questions = read_questions(args.questions, approved_only=args.approved_only)
    if not questions:
        raise ValueError("No questions selected for evaluation")
    adapter = HttpChatbotAdapter(
        args.chatbot_url, args.question_field, args.answer_field, args.context_field,
        settings.request_timeout_seconds, _headers(args.header),
    )
    records = Evaluator(adapter, llm, settings.judge_model).evaluate(questions)
    details_csv, details_jsonl = write_evaluations(records, args.output)
    summary_json, report_html = write_report(records, args.output)
    print(f"Evaluated {len(records)} questions.\nDetails: {details_csv}\nRaw details: {details_jsonl}\nSummary: {summary_json}\nReport: {report_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
