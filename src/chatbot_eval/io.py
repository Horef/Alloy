from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .labels import ANSWER_SCOPE_HEBREW, EXPECTED_BEHAVIOR_HEBREW, INCORRECT_TYPE_HEBREW, OUTCOME_HEBREW, QUESTION_FORM_HEBREW
from .models import EvaluationRecord, EvidenceQuote, ExpectedBehavior, QuestionForm, QuestionType, SilverQuestion
from .validation import validate_question_set


QUESTION_COLUMNS = [
    "id", "topic", "question", "expected_answer", "answerable", "difficulty", "question_type",
    "question_form", "expected_behavior", "parent_question_id",
    "rationale", "reference_claims", "source_ids", "source_files", "source_locations", "source_excerpts", "supporting_quotes",
    "review_status", "reviewer_notes",
]


def write_questions(questions: Iterable[SilverQuestion], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = validate_question_set(list(questions))
    csv_path, jsonl_path = output_dir / "silver_questions.csv", output_dir / "silver_questions.jsonl"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUESTION_COLUMNS)
        writer.writeheader()
        for q in items:
            writer.writerow({
                "id": q.id, "topic": q.topic, "question": q.question,
                "expected_answer": q.expected_answer, "answerable": str(q.answerable).lower(),
                "difficulty": q.difficulty, "question_type": q.question_type.value, "rationale": q.rationale,
                "question_form": q.question_form.value, "expected_behavior": q.expected_behavior.value,
                "parent_question_id": q.parent_question_id,
                "reference_claims": json.dumps(q.reference_claims, ensure_ascii=False),
                "source_ids": " | ".join(s.source_id for s in q.sources),
                "source_files": " | ".join(s.file for s in q.sources),
                "source_locations": " | ".join(s.location for s in q.sources),
                "source_excerpts": " | ".join(s.excerpt for s in q.sources),
                "supporting_quotes": json.dumps(
                    [quote.model_dump() for quote in q.supporting_quotes], ensure_ascii=False,
                ),
                "review_status": q.review_status, "reviewer_notes": q.reviewer_notes,
            })
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for q in items:
            handle.write(q.model_dump_json() + "\n")
    return csv_path, jsonl_path


def read_questions(path: Path, approved_only: bool = False) -> list[SilverQuestion]:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".jsonl"}:
        raise ValueError("Silver question files must be .csv or .jsonl")
    if suffix == ".jsonl":
        items = [SilverQuestion.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            items = []
            for row in csv.DictReader(handle):
                sources = []
                source_ids = row.get("source_ids", "").split(" | ") if row.get("source_ids") else []
                files = row.get("source_files", "").split(" | ") if row.get("source_files") else []
                locations = row.get("source_locations", "").split(" | ") if row.get("source_locations") else []
                excerpts = row.get("source_excerpts", "").split(" | ") if row.get("source_excerpts") else []
                from .models import SourceRef
                for index, file in enumerate(files):
                    sources.append(SourceRef(
                        source_id=source_ids[index] if index < len(source_ids) else "",
                        file=file, location=locations[index] if index < len(locations) else "",
                        excerpt=excerpts[index] if index < len(excerpts) else "",
                    ))
                raw_claims = row.get("reference_claims", "").strip()
                reference_claims = json.loads(raw_claims) if raw_claims else []
                if not isinstance(reference_claims, list) or not all(isinstance(claim, str) for claim in reference_claims):
                    raise ValueError(f"Question {row.get('id', '')!r} has invalid reference_claims JSON")
                raw_quotes = row.get("supporting_quotes", "").strip()
                supporting_quotes = [EvidenceQuote.model_validate(item) for item in json.loads(raw_quotes)] if raw_quotes else []
                items.append(SilverQuestion(
                    id=row["id"], topic=row["topic"], question=row["question"],
                    expected_answer=row.get("expected_answer", ""),
                    answerable=row.get("answerable", "true").lower() in {"true", "1", "yes"},
                    difficulty=row.get("difficulty", "medium"),
                    question_type=QuestionType(row.get("question_type") or QuestionType.BASIC_KNOWLEDGE.value),
                    question_form=QuestionForm(row.get("question_form") or QuestionForm.CANONICAL.value),
                    expected_behavior=ExpectedBehavior(
                        row.get("expected_behavior")
                        or (ExpectedBehavior.ANSWER.value if row.get("answerable", "true").lower() in {"true", "1", "yes"} else ExpectedBehavior.ABSTAIN.value)
                    ),
                    parent_question_id=row.get("parent_question_id", ""),
                    reference_claims=reference_claims,
                    rationale=row.get("rationale", ""), supporting_quotes=supporting_quotes,
                    sources=sources, review_status=row.get("review_status", "pending"),
                    reviewer_notes=row.get("reviewer_notes", ""),
                ))
    selected = [q for q in items if not approved_only or q.review_status.lower() == "approved"]
    return validate_question_set(selected)


def write_evaluations(records: list[EvaluationRecord], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, jsonl_path = output_dir / "evaluation_details.csv", output_dir / "evaluation_details.jsonl"
    fields = [
        "מזהה שאלה", "מזהה שאלת מקור", "נושא", "צורת שאלה", "התנהגות מצופה", "שאלה", "תשובה צפויה", "ניתנת למענה", "תשובת הצ׳אטבוט",
        "מקטעים שאוחזרו", "סיווג", "קוד סיווג", "פרטים נדרשים",
        "פרטים שנענו", "פרטים שנענו נכון", "טענות שגויות", "טענות לא מבוססות",
        "טענות עודפות", "היקף התשובה", "סוג שגיאה", "פרטים שנמצאו באחזור",
        "מספר מקטעים", "מקטעים רלוונטיים", "מקטעים סותרים", "הסבר הבדיקה", "מידע חסר או שגוי",
        "הסבר האחזור", "זמן תגובה במילישניות", "שגיאת מערכת", "פירוט טענות הייחוס",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            scores = record.scores
            writer.writerow({
                "מזהה שאלה": record.question.id, "מזהה שאלת מקור": record.question.parent_question_id,
                "נושא": record.question.topic, "צורת שאלה": QUESTION_FORM_HEBREW[record.question.question_form],
                "התנהגות מצופה": EXPECTED_BEHAVIOR_HEBREW[record.question.expected_behavior],
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
                "פירוט טענות הייחוס": json.dumps(
                    [assessment.model_dump() for assessment in scores.claim_assessments] if scores else [],
                    ensure_ascii=False,
                ),
            })
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    return csv_path, jsonl_path
