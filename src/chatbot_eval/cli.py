from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

from .adapters import HttpChatbotAdapter
from .artifacts import (
    EvaluationCheckpoint, RunManifest, StructuredCallCheckpoint, evaluation_fingerprint, atomic_write_text, file_sha256, input_inventory,
)
from .cache import CorpusAnalysisCache
from .config import load_settings
from .evaluator import Evaluator, judge_contract_fingerprint
from .generator import GenerationOptions, SilverSetGenerator, reground_fingerprint, topic_discovery_fingerprint
from .history import (
    discover_previous_run, read_current_prompt, read_evaluation_insights,
    read_evaluation_contract, read_evaluation_records,
)
from .insights import generate_insights, insights_fingerprint, write_insights
from .io import read_questions, write_evaluations, write_questions
from .llm import GeminiStructuredLLM
from .logging_utils import configure_logging
from .models import ChatbotResult, Outcome, SilverQuestion
from .prompt_generator import SystemPromptGenerator, prompt_generation_fingerprint, write_prompt_package
from .prompt_policy import POLICY_VERSION
from .report import write_report
from .review import MergeDiagnostics, merge_review_file, write_review_file
from .results_io import ImportDiagnostics, ResultColumns, read_premade_results
from .contracts import model_identity, records_hash
from .topics import infer_topics, topic_inference_fingerprint

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
    generate.add_argument("--sequential-ids", action="store_true", help="Use legacy Q0001-style run-local IDs")
    generate.add_argument(
        "--exclude-questions", type=Path,
        help="Existing silver CSV/JSONL whose questions must not be generated again",
    )
    generate.add_argument("--resume", action="store_true", help="Resume successful structured generation calls")
    generate.add_argument("--checkpoint", type=Path, help="Generation-call checkpoint; defaults inside output")
    _add_cache_arguments(generate)

    generate_prompt = commands.add_parser("generate-prompt", help="Generate a reviewable Hebrew system prompt from a document base")
    generate_prompt.add_argument("--documents", type=Path, required=True)
    generate_prompt.add_argument("--output", type=Path, default=Path("outputs/prompt"))
    generate_prompt.add_argument("--assistant-name", default="העוזר הדיגיטלי")
    generate_prompt.add_argument("--audience", default="משתמשי הארגון")
    generate_prompt.add_argument("--previous-run", type=Path, help="Previous Alloy output directory or evaluation_details.jsonl used as improvement evidence")
    generate_prompt.add_argument("--insights", type=Path, help="Optional evaluation_insights.json; overrides insights discovered in --previous-run")
    generate_prompt.add_argument("--current-prompt", type=Path, help="Current prompt file, prompt_package.json, or prompt output directory")
    generate_prompt.add_argument("--instruction-profile", choices=["compact", "guided"])
    generate_prompt.add_argument("--answer-policy", choices=["balanced", "conservative"])
    _add_cache_arguments(generate_prompt)

    evaluate = commands.add_parser("evaluate", help="Call a chatbot and judge its responses")
    evaluate.add_argument("--questions", type=Path, required=True)
    evaluate.add_argument("--chatbot-url", required=True)
    evaluate.add_argument("--deployment-id", default="", help="Nonsecret identity for chatbot prompt/model/tenant deployment; change when behavior or routing changes")
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
    evaluate.add_argument("--compare-with", type=Path, help="Previous Alloy output directory or evaluation_details.jsonl to compare in the report")
    evaluate.add_argument("--soft-compare", action="store_true", help="Show approximate deltas on matched questions even when the benchmark or evaluation contract changed; warnings remain but are informational")
    evaluate.add_argument("--retry-errors", action="store_true", help="Retry checkpointed chatbot/judge errors while preserving successes")
    evaluate.add_argument("--max-concurrency", type=int, help="Concurrent chatbot/judge workers; default is config value 1")
    _add_cache_arguments(evaluate)

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
    evaluate_file.add_argument("--expected-behavior-column")
    evaluate_file.add_argument("--question-form-column")
    evaluate_file.add_argument("--parent-question-id-column")
    evaluate_file.add_argument("--question-type-column")
    evaluate_file.add_argument("--difficulty-column")
    evaluate_file.add_argument("--reference-claims-column")
    evaluate_file.add_argument("--supporting-quotes-column")
    evaluate_file.add_argument("--reference-sources-column")
    evaluate_file.add_argument("--infer-topics", action="store_true", help="Infer consistent Hebrew topics with Gemini when no topic column exists")
    evaluate_file.add_argument("--generate-insights", action="store_true", help="Generate an optional Hebrew cross-result insights block")
    evaluate_file.add_argument("--hide-correct-answer-metrics", action="store_true", help="Hide correct-answer metrics from the report summary and topic table")
    evaluate_file.add_argument("--strict", action="store_true", help="Fail if any imported row is invalid or skipped")
    evaluate_file.add_argument("--resume", action="store_true", help="Resume completed rows from a compatible checkpoint")
    evaluate_file.add_argument("--checkpoint", type=Path, help="Checkpoint JSONL path; defaults inside the output directory")
    evaluate_file.add_argument("--compare-with", type=Path, help="Previous Alloy output directory or evaluation_details.jsonl to compare in the report")
    evaluate_file.add_argument("--soft-compare", action="store_true", help="Show approximate deltas on matched questions even when the benchmark or evaluation contract changed; warnings remain but are informational")
    evaluate_file.add_argument("--retry-errors", action="store_true", help="Retry checkpointed judge errors while preserving other rows")
    evaluate_file.add_argument("--max-concurrency", type=int, help="Concurrent judge workers; default is config value 1")
    _add_cache_arguments(evaluate_file)

    review_export = commands.add_parser(
        "review-export", help="Export a human-readable review file from a generated silver question set",
    )
    review_export.add_argument("--questions", type=Path, required=True, help="Silver CSV or JSONL to export for review")
    review_export.add_argument("--output", type=Path, default=Path("outputs/review"), help="Destination for questions_for_review.csv/.md")
    review_export.add_argument("--hebrew-columns", action="store_true", help="Write Hebrew column headers for reviewers (the 'id' column stays English); review-merge reads either language")

    review_merge = commands.add_parser(
        "review-merge",
        help="Merge an edited human review file back onto the canonical silver set, preserving technical columns",
    )
    review_merge.add_argument("--canonical", type=Path, required=True, help="Original silver JSONL/CSV that holds all technical columns")
    review_merge.add_argument("--review", type=Path, required=True, help="Edited questions_for_review.csv")
    review_merge.add_argument("--output", type=Path, default=Path("outputs/questions-reviewed"), help="Destination for the merged silver CSV/JSONL")

    reground = commands.add_parser(
        "reground",
        help="Regenerate only the grounding (sources, quotes, reference claims) for human-edited questions against the documents",
    )
    reground.add_argument("--questions", type=Path, required=True, help="Silver CSV/JSONL containing questions to re-ground")
    reground.add_argument("--documents", type=Path, required=True, help="Document root the questions were generated from")
    reground.add_argument("--output", type=Path, default=Path("outputs/questions-regrounded"), help="Destination for the regrounded silver CSV/JSONL")
    reground.add_argument("--all", action="store_true", help="Re-ground every answerable question, not only those marked needs_reground")
    reground.add_argument("--evidence-limit", type=int, default=12, help="Maximum candidate chunks offered to the model per question")
    _add_cache_arguments(reground)

    report = commands.add_parser("report", help="Regenerate a report from completed Alloy evaluation JSONL")
    report.add_argument("--results", type=Path, required=True, help="Current output directory or evaluation_details.jsonl")
    report.add_argument("--output", type=Path, default=Path("outputs/report"))
    report.add_argument("--compare-with", type=Path, help="Previous output directory or evaluation_details.jsonl")
    report.add_argument("--insights", type=Path, help="Optional evaluation_insights.json or its output directory")
    report.add_argument("--hide-correct-answer-metrics", action="store_true")
    report.add_argument("--soft-compare", action="store_true", help="Show approximate deltas on matched questions even when the benchmark or evaluation contract changed; warnings remain but are informational")
    return parser


def _add_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, help="Override the shared local workflow cache directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--refresh-cache", action="store_true", help="Recompute and replace matching workflow cache entries")
    group.add_argument("--no-cache", action="store_true", help="Do not read or write the workflow cache")


def _workflow_cache(args, settings, corpus_root: Path | None = None) -> CorpusAnalysisCache:
    if args.cache_dir is not None:
        directory = args.cache_dir
    else:
        configured = Path(settings.cache_directory)
        directory = configured if configured.is_absolute() else args.config.resolve().parent / configured
    enabled = settings.cache_enabled and not args.no_cache
    if enabled and corpus_root is not None and directory.resolve().is_relative_to(corpus_root.resolve()):
        raise ValueError("The cache directory must be outside --documents so cache files cannot become corpus inputs")
    return CorpusAnalysisCache(
        directory,
        enabled=enabled,
        refresh=args.refresh_cache, model_identity=model_identity(settings),
    )


def _analysis_cache(args, settings) -> CorpusAnalysisCache:
    if getattr(args, "output", None) and args.output.resolve().is_relative_to(args.documents.resolve()):
        raise ValueError("The output directory must be outside --documents")
    return _workflow_cache(args, settings, args.documents)


def _headers(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --header {value!r}; expected Name=Value")
        key, content = value.split("=", 1)
        result[key] = os.path.expandvars(content)
    return result


def _optional_insights(args, records, llm, model, *, cache, settings):
    status = {"status": "disabled", "records_sha256": records_hash(records)}
    status_path = args.output / "evaluation_insights_status.json"
    def save():
        args.insights_status = status.copy()
        atomic_write_text(status_path, json.dumps(status, ensure_ascii=False, indent=2))
    if not args.generate_insights:
        save()
        return None, None
    try:
        insights = cache.load_insights(
            records, model=model, transport=settings.gemini_transport,
            max_prompt_chars=settings.insights_max_prompt_chars,
            implementation_sha256=insights_fingerprint(),
            discover=lambda: generate_insights(records, llm, model, max_prompt_chars=settings.insights_max_prompt_chars),
        )
        path = write_insights(insights, args.output)
        status.update(status="generated", insights_sha256=file_sha256(path))
        save()
        return insights, path
    except Exception as exc:
        logger.error("insight_generation_failed error_type=%s; continuing without insights", type(exc).__name__)
        status.update(status="failed", error_type=type(exc).__name__)
        save()
        return None, None


def _write_generation_diagnostics(diagnostics: dict, output_dir: Path) -> Path:
    """Persist final generation accounting without changing the generator API."""
    path = output_dir / "generation_diagnostics.json"
    atomic_write_text(path, json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n")
    return path


def _checkpointed_evaluation(
    args, items, evaluator: Evaluator, *, premade: bool, contract: dict,
):
    run_signature = hashlib.sha256(
        json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    fingerprints = {
        question.id: evaluation_fingerprint(question, result if premade else None, contract=contract)
        for question, result in items
    }
    checkpoint = EvaluationCheckpoint(
        args.checkpoint or args.output / "evaluation_checkpoint.jsonl",
        fingerprints,
        resume=args.resume,
        run_signature=run_signature,
    )
    completed = checkpoint.load() if args.resume else {}
    judge_retries = []
    if args.retry_errors:
        retry_ids = {
            identifier for identifier, record in completed.items()
            if record.outcome == Outcome.JUDGE_ERROR
            or (not premade and record.outcome == Outcome.CHATBOT_ERROR)
        }
        for identifier in sorted(retry_ids):
            previous = completed.pop(identifier)
            if not premade and previous.outcome == Outcome.JUDGE_ERROR:
                judge_retries.append((previous.question, previous.result))
        logger.info("evaluation_retry_errors selected_count=%d", len(retry_ids))
    retry_judge_ids = {q.id for q, _ in judge_retries}
    pending = [(question, result) for question, result in items
               if question.id not in completed and question.id not in retry_judge_ids]
    logger.info(
        "evaluation_resume_state completed=%d pending=%d checkpoint=%s",
        len(completed), len(pending), checkpoint.path,
    )
    if premade:
        new_records = evaluator.judge_results(pending, on_record=checkpoint.append) if pending else []
    else:
        new_records = evaluator.evaluate([question for question, _ in pending], on_record=checkpoint.append) if pending else []
    if judge_retries:
        new_records.extend(evaluator.judge_results(judge_retries, on_record=checkpoint.append))
    for question, _ in items:
        if question.id in completed:
            completed[question.id].question.topic = question.topic
    by_id = {**completed, **{record.question.id: record for record in new_records}}
    return [by_id[question.id] for question, _ in items], len(completed), checkpoint.path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "retry_errors", False) and not args.resume:
        raise ValueError("--retry-errors requires --resume")
    if args.command == "generate" and args.resume and args.refresh_cache:
        raise ValueError("generate --resume cannot be combined with --refresh-cache; start a fresh run instead")
    _ensure_config(args.config)
    offline_commands = {"report", "review-export", "review-merge"}
    settings = load_settings(args.config, require_api_key=args.command not in offline_commands)
    log_path = args.log_file or (Path(settings.log_file) if settings.log_file else None)
    configure_logging(log_path, args.log_level or settings.log_level)
    progress_enabled = settings.progress_enabled and not args.no_progress
    logger.info("command_started command=%s progress_enabled=%s", args.command, progress_enabled)
    if args.command == "report":
        records = read_evaluation_records(args.results)
        previous_records = read_evaluation_records(args.compare_with) if args.compare_with else None
        current_contract = read_evaluation_contract(args.results)
        previous_contract = read_evaluation_contract(args.compare_with) if args.compare_with else None
        insights = read_evaluation_insights(args.insights) if args.insights else None
        inputs = [args.results] + ([args.compare_with] if args.compare_with else []) + ([args.insights] if args.insights else [])
        with RunManifest(
            args.output, command=args.command, settings=settings, inputs=inputs,
            parameters={
                "compare_with": bool(args.compare_with),
                "show_correct_answer_metrics": not args.hide_correct_answer_metrics,
                "soft_compare": args.soft_compare,
            },
        ) as manifest:
            summary_json, report_html = write_report(
                records, args.output, insights,
                show_correct_answer_metrics=not args.hide_correct_answer_metrics,
                previous_records=previous_records,
                current_contract=current_contract, previous_contract=previous_contract,
                soft_compare=args.soft_compare,
            )
            manifest.complete(record_count=len(records), outputs=input_inventory([summary_json, report_html]))
        print(f"Generated report from {len(records)} completed records.\nSummary: {summary_json}\nReport: {report_html}")
        if previous_records:
            print(f"Compared with {len(previous_records)} previous records.")
        return 0

    if args.command == "review-export":
        questions = read_questions(args.questions)
        with RunManifest(
            args.output, command=args.command, settings=settings, inputs=[args.questions],
            parameters={"question_count": len(questions), "hebrew_columns": args.hebrew_columns},
        ) as manifest:
            review_csv, review_md = write_review_file(questions, args.output, hebrew_columns=args.hebrew_columns)
            manifest.complete(
                question_count=len(questions),
                outputs=input_inventory([review_csv, review_md]),
            )
        print(
            f"Exported {len(questions)} questions for human review.\n"
            f"Edit this file: {review_csv}\nRead-only view: {review_md}\n"
            "After review, run 'review-merge' with the original silver file to rebuild the full technical set."
        )
        logger.info("command_completed command=review-export question_count=%d output=%s", len(questions), args.output)
        return 0

    if args.command == "review-merge":
        canonical = read_questions(args.canonical)
        merge_diagnostics = MergeDiagnostics()
        merged = merge_review_file(canonical, args.review, diagnostics=merge_diagnostics)
        with RunManifest(
            args.output, command=args.command, settings=settings, inputs=[args.canonical, args.review],
            parameters={"merge_diagnostics": merge_diagnostics.as_dict()},
        ) as manifest:
            csv_path, jsonl_path = write_questions(merged, args.output)
            manifest.complete(
                question_count=len(merged),
                merge_diagnostics=merge_diagnostics.as_dict(),
                outputs=input_inventory([csv_path, jsonl_path]),
            )
        print(
            f"Merged {len(merged)} reviewed questions from {merge_diagnostics.canonical_total} canonical.\n"
            f"Reviewer removed {merge_diagnostics.deleted_by_reviewer}; edited {merge_diagnostics.content_edited}; "
            f"{merge_diagnostics.factual_edits_flagged} factual edits flagged for re-grounding.\n"
            f"Reviewed silver CSV: {csv_path}\nProvenance JSONL: {jsonl_path}"
        )
        if merge_diagnostics.factual_edits_flagged:
            print(
                "Note: questions whose question/expected_answer text was edited had their approval "
                "downgraded to 'needs_reground' because stored sources and claims may no longer match."
            )
        logger.info("command_completed command=review-merge merged=%d output=%s", len(merged), args.output)
        return 0

    llm = GeminiStructuredLLM(
        settings.api_key,
        settings.max_retries,
        transport=settings.gemini_transport,
        apigee_api_key=settings.apigee_api_key,
        apigee_base_url=settings.apigee_base_url,
        request_timeout_seconds=settings.gemini_request_timeout_seconds,
    )
    if args.command == "generate-prompt":
        profile = args.instruction_profile or settings.prompt_instruction_profile
        answer_policy = args.answer_policy or settings.prompt_answer_policy
        previous_records = None
        previous_insights = None
        if args.previous_run:
            previous_records, previous_insights = discover_previous_run(args.previous_run)
        if args.insights:
            previous_insights = read_evaluation_insights(args.insights)
        current_prompt = read_current_prompt(args.current_prompt) if args.current_prompt else ""
        cache = _analysis_cache(args, settings)
        prompt_inputs = [args.documents]
        prompt_inputs.extend(path for path in (args.previous_run, args.insights, args.current_prompt) if path)
        with RunManifest(
            args.output, command=args.command, settings=settings, inputs=prompt_inputs,
            parameters={
                "assistant_name": args.assistant_name, "audience": args.audience,
                "uses_previous_results": bool(previous_records),
                "uses_previous_insights": bool(previous_insights),
                "uses_current_prompt": bool(current_prompt), "instruction_profile": profile, "answer_policy": answer_policy,
                "prompt_generation_fingerprint": prompt_generation_fingerprint(), "policy_version": POLICY_VERSION,
                "cache_enabled": cache.enabled, "refresh_cache": cache.refresh,
            },
        ) as manifest:
            chunks, chunk_key = cache.load_chunks(
                args.documents, settings.chunk_chars, settings.chunk_overlap_chars, progress_enabled,
            )
            topic_generator = SilverSetGenerator(llm, settings.generation_model, progress_enabled)
            topics = cache.load_topics(
                chunks, chunk_key, model=settings.generation_model, transport=settings.gemini_transport,
                batch_chunks=settings.batch_chunks, implementation_sha256=topic_discovery_fingerprint(),
                discover=lambda: topic_generator.discover_topics(chunks, settings.batch_chunks),
            )
            prompt_generator = SystemPromptGenerator(
                llm, settings.generation_model,
                document_context_chars=settings.prompt_max_document_chars,
                evaluation_context_chars=settings.prompt_max_evaluation_chars,
                auxiliary_context_chars=settings.prompt_max_auxiliary_chars,
                instruction_profile=profile, answer_policy=answer_policy,
            )
            package = cache.load_prompt_package(
                chunk_key=chunk_key, topics=topics,
                assistant_name=args.assistant_name, audience=args.audience,
                previous_records=previous_records or [], previous_insights=previous_insights,
                current_prompt=current_prompt, model=settings.generation_model,
                transport=settings.gemini_transport,
                document_context_chars=settings.prompt_max_document_chars,
                evaluation_context_chars=settings.prompt_max_evaluation_chars,
                auxiliary_context_chars=settings.prompt_max_auxiliary_chars,
                instruction_profile=profile, answer_policy=answer_policy,
                implementation_sha256=prompt_generation_fingerprint(),
                generate=lambda: prompt_generator.generate(
                    chunks, topics, assistant_name=args.assistant_name, audience=args.audience,
                    previous_records=previous_records, previous_insights=previous_insights,
                    current_prompt=current_prompt,
                ),
            )
            prompt_path, package_path = write_prompt_package(package, args.output)
            manifest.complete(
                topic_count=len(topics), chunk_count=len(chunks),
                prompt_generation_fingerprint=prompt_generation_fingerprint(), policy_version=package.policy_version,
                cache=cache.summary(),
                outputs=input_inventory([prompt_path, package_path]),
            )
        print(f"Generated a reviewable system prompt across {len(topics)} discovered topics.")
        print(f"System prompt: {prompt_path}\nReview package: {package_path}")
        logger.info("command_completed command=generate-prompt topic_count=%d output=%s", len(topics), args.output)
        return 0

    if args.command == "reground":
        all_questions = read_questions(args.questions)
        if args.all:
            targets = [q for q in all_questions if q.answerable and q.expected_behavior.value == "answer"]
        else:
            targets = [q for q in all_questions if q.review_status.lower() == "needs_reground"]
        if not targets:
            selector = "answerable answer-task" if args.all else "needs_reground"
            raise SystemExit(f"No {selector} questions found in {args.questions}; nothing to re-ground.")
        cache = _analysis_cache(args, settings)
        with RunManifest(
            args.output, command=args.command, settings=settings,
            inputs=[args.questions, args.documents],
            parameters={
                "target_count": len(targets), "total_questions": len(all_questions),
                "reground_all": args.all, "evidence_limit": args.evidence_limit,
                "reground_fingerprint": reground_fingerprint(),
                "cache_enabled": cache.enabled, "refresh_cache": cache.refresh,
            },
        ) as manifest:
            chunks, _ = cache.load_chunks(
                args.documents, settings.chunk_chars, settings.chunk_overlap_chars, progress_enabled,
            )
            generator = SilverSetGenerator(llm, settings.generation_model, progress_enabled)
            target_ids = {q.id for q in targets}
            regrounded_count = 0
            failures: list[dict] = []
            merged: list[SilverQuestion] = []
            for question in all_questions:
                if question.id not in target_ids:
                    merged.append(question)
                    continue
                updated, reason = generator.reground_one(question, chunks, evidence_limit=args.evidence_limit)
                if updated is not None:
                    merged.append(updated)
                    regrounded_count += 1
                else:
                    # Keep the original row untouched; record why it could not be regrounded.
                    merged.append(question)
                    failures.append({"id": question.id, "reason": reason})
                    logger.warning("reground_failed id=%s reason=%s", question.id, reason)
            csv_path, jsonl_path = write_questions(merged, args.output)
            manifest.complete(
                question_count=len(merged), regrounded=regrounded_count,
                failed=len(failures), failures=failures[:50],
                reground_fingerprint=reground_fingerprint(),
                cache=cache.summary(),
                outputs=input_inventory([csv_path, jsonl_path]),
            )
        print(
            f"Re-grounded {regrounded_count}/{len(targets)} targeted questions.\n"
            f"Silver CSV: {csv_path}\nProvenance JSONL: {jsonl_path}"
        )
        if failures:
            print(
                f"{len(failures)} question(s) could not be re-grounded and were left unchanged "
                "(still marked needs_reground). See run_manifest.json for reasons."
            )
        print(
            "Re-grounded questions were reset to review_status=pending; review and approve them "
            "before evaluating with --approved-only."
        )
        logger.info("command_completed command=reground regrounded=%d output=%s", regrounded_count, args.output)
        return 0

    if args.command == "generate":
        maximum = settings.max_questions if args.max_questions is None else args.max_questions
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
        cache = _analysis_cache(args, settings)
        with RunManifest(
            args.output, command=args.command, settings=settings, inputs=manifest_inputs,
            parameters={
                "max_questions": maximum, "topic": args.topic, "topic_count": args.topic_count,
                "unanswerable_ratio": ratio, "user_variation_ratio": variation_ratio,
                "ambiguous_variation_share": ambiguous_share,
                "stable_question_ids": settings.stable_question_ids and not args.sequential_ids,
                "question_type_targets": settings.question_type_targets,
                "cache_enabled": cache.enabled, "refresh_cache": cache.refresh,
                "resume": args.resume,
            },
        ) as manifest:
            chunks, chunk_key = cache.load_chunks(
                args.documents, settings.chunk_chars, settings.chunk_overlap_chars, progress_enabled,
            )
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
                stable_question_ids=settings.stable_question_ids and not args.sequential_ids,
                question_type_targets=tuple(settings.question_type_targets.items()),
                requested_topic=args.topic, requested_topic_count=args.topic_count,
                excluded_questions=excluded_questions,
            )
            generation_signature = hashlib.sha256(json.dumps({
                "chunk_key": chunk_key,
                "options": options.__dict__,
                "model": settings.generation_model,
                "transport": settings.gemini_transport, "model_identity": model_identity(settings),
                "implementation": topic_discovery_fingerprint(),
            }, ensure_ascii=False, sort_keys=True, default=list).encode("utf-8")).hexdigest()
            generation_checkpoint = args.checkpoint or args.output / "generation_checkpoint.jsonl"
            generation_llm = StructuredCallCheckpoint(
                generation_checkpoint, llm, signature=generation_signature, resume=args.resume,
            )
            generator = SilverSetGenerator(generation_llm, settings.generation_model, progress_enabled)
            topics = cache.load_topics(
                chunks, chunk_key, model=settings.generation_model, transport=settings.gemini_transport,
                batch_chunks=settings.batch_chunks, implementation_sha256=topic_discovery_fingerprint(),
                discover=lambda: generator.discover_topics(chunks, settings.batch_chunks),
            )
            questions, topics = generator.generate(chunks, options, topics=topics)
            csv_path, jsonl_path = write_questions(questions, args.output)
            generation_diagnostics = dict(generator.last_generation_diagnostics)
            diagnostics_path = _write_generation_diagnostics(generation_diagnostics, args.output)
            manifest.complete(
                question_count=len(questions), topic_count=len(topics), chunk_count=len(chunks),
                generation_diagnostics=generation_diagnostics,
                cache=cache.summary(),
                resumed_generation=args.resume,
                outputs=input_inventory([csv_path, jsonl_path, diagnostics_path, generation_checkpoint]),
            )
        print(f"Generated {len(questions)} questions across {len(topics)} discovered topics.")
        print(f"Review CSV: {csv_path}\nProvenance JSONL: {jsonl_path}\nGeneration diagnostics: {diagnostics_path}")
        if len(questions) < maximum:
            print(f"Note: generated fewer than requested ({len(questions)}/{maximum}) because unsupported or duplicate candidates were discarded.")
        logger.info("command_completed command=generate question_count=%d output=%s", len(questions), args.output)
        return 0

    if args.command == "evaluate-file":
        previous_records = read_evaluation_records(args.compare_with) if args.compare_with else None
        previous_contract = read_evaluation_contract(args.compare_with) if args.compare_with else None
        columns = ResultColumns(
            question=args.question_column, expected_answer=args.expected_answer_column,
            answer=args.answer_column, source=args.source_column, id=args.id_column,
            topic=args.topic_column, answerable=args.answerable_column, error=args.error_column,
            expected_behavior=args.expected_behavior_column, question_form=args.question_form_column,
            parent_question_id=args.parent_question_id_column, question_type=args.question_type_column,
            difficulty=args.difficulty_column, reference_claims=args.reference_claims_column,
            supporting_quotes=args.supporting_quotes_column, reference_sources=args.reference_sources_column,
        )
        import_diagnostics = ImportDiagnostics()
        pairs = read_premade_results(
            args.results, sheet_name=args.sheet, columns=columns, progress_enabled=progress_enabled,
            strict=args.strict, diagnostics=import_diagnostics,
        )
        concurrency = settings.max_concurrency if args.max_concurrency is None else args.max_concurrency
        if concurrency < 1:
            raise ValueError("--max-concurrency must be positive")
        cache = _workflow_cache(args, settings)

        class NeverCalledAdapter:
            def ask(self, question: SilverQuestion) -> ChatbotResult:
                raise RuntimeError("Premade evaluation must not invoke a chatbot")

        with RunManifest(
            args.output, command=args.command, settings=settings,
            inputs=[args.results] + ([args.compare_with] if args.compare_with else []),
            parameters={
                "sheet": args.sheet, "infer_topics": args.infer_topics, "import_diagnostics": import_diagnostics.as_dict(),
                "generate_insights": args.generate_insights, "resume": args.resume, "strict": args.strict,
                "compare_with": bool(args.compare_with), "soft_compare": args.soft_compare, "retry_errors": args.retry_errors,
                "max_concurrency": concurrency,
            },
        ) as manifest:
            if args.infer_topics:
                candidates = [q for q, _ in pairs if q.topic in {"", "premade", "לא סווג"}]
                assignments = cache.load_question_topics(
                    candidates, model=settings.judge_model, transport=settings.gemini_transport,
                    batch_size=250, implementation_sha256=topic_inference_fingerprint(),
                    discover=lambda: infer_topics(
                        pairs, llm, settings.judge_model, progress_enabled=progress_enabled,
                    ),
                )
                for question in candidates:
                    question.topic = assignments.get(question.id, "לא סווג")
            contract = {
                "kind": "premade", "judge_model": settings.judge_model,
                "transport": settings.gemini_transport,
                "judge_implementation": judge_contract_fingerprint(), "model_identity": model_identity(settings),
                "max_answer_chars": settings.judge_max_answer_chars,
                "max_context_chars": settings.judge_max_context_chars,
            }
            manifest.data["evaluation_contract"] = contract
            records, resumed_count, checkpoint_path = _checkpointed_evaluation(
                args, pairs, Evaluator(
                    NeverCalledAdapter(), llm, settings.judge_model, progress_enabled,
                    max_answer_chars=settings.judge_max_answer_chars,
                    max_context_chars=settings.judge_max_context_chars,
                    max_concurrency=concurrency,
                ), premade=True, contract=contract,
            )
            insights, insights_path = _optional_insights(
                args, records, llm, settings.judge_model, cache=cache, settings=settings,
            )
            details_csv, details_jsonl = write_evaluations(records, args.output)
            summary_json, report_html = write_report(
                records, args.output, insights,
                show_correct_answer_metrics=not args.hide_correct_answer_metrics,
                previous_records=previous_records,
                current_contract=contract, previous_contract=previous_contract,
                soft_compare=args.soft_compare,
            )
            output_paths = [details_csv, details_jsonl, summary_json, report_html, checkpoint_path, args.output / "evaluation_insights_status.json"]
            if insights_path:
                output_paths.append(insights_path)
            manifest.complete(
                record_count=len(records), resumed_count=resumed_count, insights_status=args.insights_status,
                cache=cache.summary(),
                outputs=input_inventory(output_paths),
            )
        print(f"Evaluated {len(records)} premade results.\nDetails: {details_csv}\nRaw details: {details_jsonl}\nSummary: {summary_json}\nReport: {report_html}")
        if insights_path:
            print(f"Insights: {insights_path}")
        logger.info("command_completed command=evaluate-file record_count=%d output=%s", len(records), args.output)
        return 0

    previous_records = read_evaluation_records(args.compare_with) if args.compare_with else None
    previous_contract = read_evaluation_contract(args.compare_with) if args.compare_with else None
    questions = read_questions(args.questions, approved_only=args.approved_only)
    if not questions:
        raise ValueError("No questions selected for evaluation")
    concurrency = settings.max_concurrency if args.max_concurrency is None else args.max_concurrency
    if concurrency < 1:
        raise ValueError("--max-concurrency must be positive")
    request_headers = _headers(args.header)
    cache = _workflow_cache(args, settings)
    adapter = HttpChatbotAdapter(
        args.chatbot_url, args.question_field, args.answer_field, args.context_field,
        settings.request_timeout_seconds, request_headers,
        max_retries=settings.chatbot_max_retries,
        retry_base_seconds=settings.chatbot_retry_base_seconds,
        pacing_seconds=settings.chatbot_pacing_seconds,
        max_response_bytes=settings.chatbot_max_response_bytes,
        require_json_content_type=settings.chatbot_require_json_content_type,
    )
    with RunManifest(
        args.output, command=args.command, settings=settings,
        inputs=[args.questions] + ([args.compare_with] if args.compare_with else []),
        parameters={
            "chatbot_url_sha256": hashlib.sha256(args.chatbot_url.encode()).hexdigest(),
            "question_field": args.question_field, "answer_field": args.answer_field,
            "context_field": args.context_field, "approved_only": args.approved_only,
            "generate_insights": args.generate_insights, "resume": args.resume,
            "compare_with": bool(args.compare_with), "soft_compare": args.soft_compare, "retry_errors": args.retry_errors,
            "max_concurrency": concurrency,
        },
    ) as manifest:
        items = [(question, ChatbotResult(question_id=question.id, answer="")) for question in questions]
        contract = {
            "kind": "live", "judge_model": settings.judge_model,
            "transport": settings.gemini_transport,
            "judge_implementation": judge_contract_fingerprint(), "model_identity": model_identity(settings),
            "max_answer_chars": settings.judge_max_answer_chars,
            "max_context_chars": settings.judge_max_context_chars,
            "chatbot_url_sha256": hashlib.sha256(args.chatbot_url.encode()).hexdigest(),
            "question_field": args.question_field, "answer_field": args.answer_field,
            "context_field": args.context_field,
            "header_names": sorted(request_headers), "deployment_id": args.deployment_id,
            "request_timeout_seconds": settings.request_timeout_seconds,
            "chatbot_max_retries": settings.chatbot_max_retries,
            "chatbot_pacing_seconds": settings.chatbot_pacing_seconds,
            "chatbot_retry_base_seconds": settings.chatbot_retry_base_seconds,
            "chatbot_max_response_bytes": settings.chatbot_max_response_bytes,
            "chatbot_require_json_content_type": settings.chatbot_require_json_content_type,
            "max_concurrency": concurrency,
        }
        manifest.data["evaluation_contract"] = contract
        records, resumed_count, checkpoint_path = _checkpointed_evaluation(
            args, items, Evaluator(
                adapter, llm, settings.judge_model, progress_enabled,
                max_answer_chars=settings.judge_max_answer_chars,
                max_context_chars=settings.judge_max_context_chars,
                max_concurrency=concurrency,
            ), premade=False, contract=contract,
        )
        insights, insights_path = _optional_insights(
            args, records, llm, settings.judge_model, cache=cache, settings=settings,
        )
        details_csv, details_jsonl = write_evaluations(records, args.output)
        summary_json, report_html = write_report(
            records, args.output, insights,
            show_correct_answer_metrics=not args.hide_correct_answer_metrics,
            previous_records=previous_records,
            current_contract=contract, previous_contract=previous_contract,
            soft_compare=args.soft_compare,
        )
        output_paths = [details_csv, details_jsonl, summary_json, report_html, checkpoint_path, args.output / "evaluation_insights_status.json"]
        if insights_path:
            output_paths.append(insights_path)
        manifest.complete(
            record_count=len(records), resumed_count=resumed_count, insights_status=args.insights_status,
            cache=cache.summary(),
            outputs=input_inventory(output_paths),
        )
    print(f"Evaluated {len(records)} questions.\nDetails: {details_csv}\nRaw details: {details_jsonl}\nSummary: {summary_json}\nReport: {report_html}")
    if insights_path:
        print(f"Insights: {insights_path}")
    logger.info("command_completed command=evaluate record_count=%d output=%s", len(records), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
