from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .labels import ANSWER_SCOPE_HEBREW, INCORRECT_TYPE_HEBREW, OUTCOME_HEBREW
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
    fields = [
        "מזהה שאלה", "נושא", "שאלה", "תשובה צפויה", "ניתנת למענה", "תשובת הצ׳אטבוט",
        "מקטעים שאוחזרו", "סיווג", "קוד סיווג", "פרטים נדרשים",
        "פרטים שנענו", "פרטים שנענו נכון", "טענות שגויות", "טענות לא מבוססות",
        "טענות עודפות", "היקף התשובה", "סוג שגיאה", "פרטים שנמצאו באחזור",
        "מספר מקטעים", "מקטעים רלוונטיים", "מקטעים סותרים", "הסבר הבדיקה", "מידע חסר או שגוי",
        "הסבר האחזור", "זמן תגובה במילישניות", "שגיאת מערכת",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            scores = record.scores
            writer.writerow({
                "מזהה שאלה": record.question.id, "נושא": record.question.topic,
                "שאלה": record.question.question, "תשובה צפויה": record.question.expected_answer,
                "ניתנת למענה": "כן" if record.question.answerable else "לא",
                "תשובת הצ׳אטבוט": record.result.answer, "מקטעים שאוחזרו": record.result.retrieved_context,
                "סיווג": OUTCOME_HEBREW[record.outcome], "קוד סיווג": record.outcome.value,
                "פרטים נדרשים": scores.required_points_total if scores else "",
                "פרטים שנענו": scores.answer_points_addressed if scores else "",
                "פרטים שנענו נכון": scores.answer_points_correct if scores else "",
                "טענות שגויות": scores.answer_false_claims if scores else "",
                "טענות לא מבוססות": scores.answer_unsupported_claims if scores else "",
                "טענות עודפות": scores.answer_extraneous_claims if scores else "",
                "היקף התשובה": ANSWER_SCOPE_HEBREW[scores.answer_scope] if scores else "",
                "סוג שגיאה": INCORRECT_TYPE_HEBREW[scores.incorrect_type] if scores else "",
                "פרטים שנמצאו באחזור": scores.retrieval_points_found if scores else "",
                "מספר מקטעים": scores.retrieved_chunks_total if scores else "",
                "מקטעים רלוונטיים": scores.retrieved_chunks_relevant if scores else "",
                "מקטעים סותרים": scores.retrieved_chunks_contradictory if scores else "",
                "הסבר הבדיקה": scores.explanation if scores else "",
                "מידע חסר או שגוי": scores.missing_or_wrong if scores else "",
                "הסבר האחזור": scores.retrieval_explanation if scores else "",
                "זמן תגובה במילישניות": record.result.latency_ms or "",
                "שגיאת מערכת": record.result.error or record.judge_error,
            })
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    return csv_path, jsonl_path
