from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .models import EvaluationRecord, SilverQuestion


QUESTION_COLUMNS = [
    "id", "topic", "question", "expected_answer", "answerable", "difficulty",
    "rationale", "source_files", "source_locations", "source_excerpts",
    "review_status", "reviewer_notes",
]


def write_questions(questions: Iterable[SilverQuestion], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = list(questions)
    csv_path, jsonl_path = output_dir / "silver_questions.csv", output_dir / "silver_questions.jsonl"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUESTION_COLUMNS)
        writer.writeheader()
        for q in items:
            writer.writerow({
                "id": q.id, "topic": q.topic, "question": q.question,
                "expected_answer": q.expected_answer, "answerable": str(q.answerable).lower(),
                "difficulty": q.difficulty, "rationale": q.rationale,
                "source_files": " | ".join(s.file for s in q.sources),
                "source_locations": " | ".join(s.location for s in q.sources),
                "source_excerpts": " | ".join(s.excerpt for s in q.sources),
                "review_status": q.review_status, "reviewer_notes": q.reviewer_notes,
            })
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for q in items:
            handle.write(q.model_dump_json() + "\n")
    return csv_path, jsonl_path


def read_questions(path: Path, approved_only: bool = False) -> list[SilverQuestion]:
    if path.suffix.lower() == ".jsonl":
        items = [SilverQuestion.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            items = []
            for row in csv.DictReader(handle):
                sources = []
                files = row.get("source_files", "").split(" | ") if row.get("source_files") else []
                locations = row.get("source_locations", "").split(" | ") if row.get("source_locations") else []
                excerpts = row.get("source_excerpts", "").split(" | ") if row.get("source_excerpts") else []
                from .models import SourceRef
                for index, file in enumerate(files):
                    sources.append(SourceRef(file=file, location=locations[index] if index < len(locations) else "", excerpt=excerpts[index] if index < len(excerpts) else ""))
                items.append(SilverQuestion(
                    id=row["id"], topic=row["topic"], question=row["question"],
                    expected_answer=row.get("expected_answer", ""),
                    answerable=row.get("answerable", "true").lower() in {"true", "1", "yes"},
                    difficulty=row.get("difficulty", "medium"), rationale=row.get("rationale", ""),
                    sources=sources, review_status=row.get("review_status", "pending"),
                    reviewer_notes=row.get("reviewer_notes", ""),
                ))
    return [q for q in items if not approved_only or q.review_status.lower() == "approved"]


def write_evaluations(records: list[EvaluationRecord], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, jsonl_path = output_dir / "evaluation_details.csv", output_dir / "evaluation_details.jsonl"
    fields = ["question_id", "topic", "question", "expected_answer", "answerable", "chatbot_answer", "outcome", "correctness", "completeness", "relevance", "groundedness", "explanation", "missing_or_wrong", "latency_ms", "error"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            scores = record.scores
            writer.writerow({
                "question_id": record.question.id, "topic": record.question.topic,
                "question": record.question.question, "expected_answer": record.question.expected_answer,
                "answerable": record.question.answerable, "chatbot_answer": record.result.answer,
                "outcome": record.outcome.value, "correctness": scores.correctness if scores else "",
                "completeness": scores.completeness if scores else "", "relevance": scores.relevance if scores else "",
                "groundedness": scores.groundedness if scores else "", "explanation": scores.explanation if scores else "",
                "missing_or_wrong": scores.missing_or_wrong if scores else "", "latency_ms": record.result.latency_ms or "",
                "error": record.result.error or record.judge_error,
            })
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    return csv_path, jsonl_path
