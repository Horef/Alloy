from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, TextIO

from .labels import ANSWER_SCOPE_HEBREW, EXPECTED_BEHAVIOR_HEBREW, INCORRECT_TYPE_HEBREW, OUTCOME_HEBREW, QUESTION_FORM_HEBREW
from .models import EvaluationRecord, EvidenceQuote, ExpectedBehavior, QuestionForm, QuestionType, SilverQuestion, SourceRef
from .validation import normalize_question_payload, validate_question_set


QUESTION_COLUMNS = [
    "id", "topic", "question", "expected_answer", "answerable", "difficulty", "question_type",
    "question_form", "expected_behavior", "parent_question_id",
    "rationale", "reference_claims", "source_ids", "source_files", "source_locations", "source_excerpts", "supporting_quotes",
    "review_status", "reviewer_notes", "sources_json", "csv_escape_version",
    "clarification_acceptable", "acceptable_clarification", "boundary_kind", "closed_book_answerable",
    "anchor", "min_key_points", "broad_level",
]


@contextmanager
def _staged_text_file(
    path: Path, *, encoding: str, newline: str | None = None,
) -> Iterator[tuple[TextIO, Path]]:
    """Write and fsync a same-directory temporary file without publishing it."""
    temporary_path: Path | None = None
    handle: TextIO | None = None
    try:
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding=encoding,
            newline=newline,
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        )
        temporary_path = Path(handle.name)
        yield handle, temporary_path
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
    finally:
        if handle is not None:
            handle.close()
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _safe_csv_value(value):
    """Neutralize spreadsheet formulas while leaving numeric values numeric."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _safe_csv_row(row: dict) -> dict:
    return {key: ("\'" + value if row.get("csv_escape_version") == "2" and isinstance(value, str) and value.startswith("\'")
                  else _safe_csv_value(value)) for key, value in row.items()}


def _restore_csv_value(value: str | None) -> str:
    if value and value.startswith("'") and value[1:].lstrip().startswith(("=", "+", "-", "@")):
        return value[1:]
    return value or ""


def write_questions(questions: Iterable[SilverQuestion], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = validate_question_set(list(questions))
    csv_path, jsonl_path = output_dir / "silver_questions.csv", output_dir / "silver_questions.jsonl"
    with (
        _staged_text_file(csv_path, encoding="utf-8-sig", newline="") as (csv_handle, csv_temporary),
        _staged_text_file(jsonl_path, encoding="utf-8") as (jsonl_handle, jsonl_temporary),
    ):
        writer = csv.DictWriter(csv_handle, fieldnames=QUESTION_COLUMNS)
        writer.writeheader()
        for q in items:
            writer.writerow(_safe_csv_row({
                "sources_json": json.dumps([source.model_dump() for source in q.sources], ensure_ascii=False),
                "csv_escape_version": "2",
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
                "clarification_acceptable": str(q.clarification_acceptable).lower(),
                "acceptable_clarification": q.acceptable_clarification,
                "boundary_kind": q.boundary_kind,
                "closed_book_answerable": str(q.closed_book_answerable).lower(),
                "anchor": str(q.anchor).lower(),
                "min_key_points": q.min_key_points,
                "broad_level": q.broad_level,
            }))
        for q in items:
            jsonl_handle.write(q.model_dump_json() + "\n")
        csv_handle.flush()
        os.fsync(csv_handle.fileno())
        jsonl_handle.flush()
        os.fsync(jsonl_handle.fileno())
        os.replace(csv_temporary, csv_path)
        os.replace(jsonl_temporary, jsonl_path)
    return csv_path, jsonl_path


def csv_question_row(row: dict) -> dict:
    if None in row or any(value is None for value in row.values()):
        raise ValueError("row has a different number of values than headers")
    version = row.get("csv_escape_version")
    return {key: (value[1:] if version == "2" and value.startswith("'") else value)
            if version == "2" else _restore_csv_value(value) for key, value in row.items()}


def question_sources(row: dict) -> list[SourceRef]:
    fields = {"source_ids": "source_id", "source_files": "file", "source_locations": "location", "source_excerpts": "excerpt"}
    if row.get("sources_json") not in (None, ""):
        raw = row["sources_json"]
        try:
            raw = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(raw, list):
                raise ValueError("sources_json must be a list")
            sources = [SourceRef.model_validate(item) for item in raw]
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid sources_json") from exc
        for column, attribute in fields.items():
            if column in row and row[column] != " | ".join(getattr(source, attribute) for source in sources):
                raise ValueError(f"{column} conflicts with canonical sources_json; edit both representations")
        return sources
    columns = {key: row.get(key, "").split(" | ") if row.get(key) else [] for key in fields}
    lengths = {len(values) for values in columns.values()}
    if len(lengths) > 1:
        logging.getLogger(__name__).warning("Legacy source column lengths differ; use original JSONL for lossless provenance recovery")
    count = max(lengths, default=0)
    return [SourceRef(**{attribute: columns[column][i] if i < len(columns[column]) else ""
                         for column, attribute in fields.items()}) for i in range(count)]


def read_questions(path: Path, approved_only: bool = False) -> list[SilverQuestion]:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".jsonl"}:
        raise ValueError("Silver question files must be .csv or .jsonl")
    if suffix == ".jsonl":
        rows = [(number, json.loads(line)) for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if line.strip()]
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            if not headers or any(not header.strip() for header in headers) or len(set(headers)) != len(headers):
                raise ValueError("Blank or duplicate CSV headers are not allowed")
            rows = list(enumerate(reader, 2))
    items = []
    for number, raw in rows:
        try:
            if not isinstance(raw, dict):
                raise ValueError("question must be an object")
            row = csv_question_row(raw) if suffix == ".csv" else raw
            if suffix == ".csv":
                row["sources"] = question_sources(row)
            question = SilverQuestion.model_validate(normalize_question_payload(row))
            validate_question_set([question])
            items.append(question)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"row {number}: {exc}") from exc
    selected = [q for q in items if not approved_only or q.review_status.lower() == "approved"]
    if not selected:
        raise ValueError("No usable questions remain after import and filtering")
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
    with (
        _staged_text_file(csv_path, encoding="utf-8-sig", newline="") as (csv_handle, csv_temporary),
        _staged_text_file(jsonl_path, encoding="utf-8") as (jsonl_handle, jsonl_temporary),
    ):
        writer = csv.DictWriter(csv_handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            scores = record.scores
            writer.writerow(_safe_csv_row({
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
            }))
        for record in records:
            jsonl_handle.write(record.model_dump_json() + "\n")
        csv_handle.flush()
        os.fsync(csv_handle.fileno())
        jsonl_handle.flush()
        os.fsync(jsonl_handle.fileno())
        os.replace(csv_temporary, csv_path)
        os.replace(jsonl_temporary, jsonl_path)
    return csv_path, jsonl_path
