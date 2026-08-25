from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from .models import ChatbotResult, ExpectedBehavior, SilverQuestion, SourceRef
from .progress import track
from .validation import validate_question_set

logger = logging.getLogger(__name__)


ALIASES = {
    "question": {"question", "prompt", "query", "שאלה"},
    "expected_answer": {"expected_answer", "expected answer", "reference_answer", "reference answer", "expected", "תשובה צפויה"},
    "answer": {"answer", "response", "chatbot_answer", "chatbot answer", "model_answer", "model answer", "תשובה מהמודל"},
    "source": {"source", "sources", "evidence", "context", "retrieved_context", "retrieved context", "מקור"},
    "id": {"id", "question_id", "question id", "messageid", "message_id"},
    "topic": {"topic", "category", "נושא"},
    "answerable": {"answerable", "is_answerable", "is answerable"},
    "error": {"error", "exception", "failure"},
}


@dataclass(frozen=True)
class ResultColumns:
    question: str | None = None
    expected_answer: str | None = None
    answer: str | None = None
    source: str | None = None
    id: str | None = None
    topic: str | None = None
    answerable: str | None = None
    error: str | None = None


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().replace("_", " ").split())


def _resolve(headers: list[str], overrides: ResultColumns) -> dict[str, str | None]:
    blank_headers = [index + 1 for index, header in enumerate(headers) if not str(header).strip()]
    if blank_headers:
        raise ValueError(f"Blank column headers are not allowed (positions {blank_headers})")
    normalized_values = [_normalize(header) for header in headers]
    duplicate_headers = sorted(value for value, count in Counter(normalized_values).items() if count > 1)
    if duplicate_headers:
        raise ValueError(f"Duplicate column headers are not allowed: {duplicate_headers}")
    normalized = {_normalize(header): header for header in headers}
    resolved: dict[str, str | None] = {}
    for field, aliases in ALIASES.items():
        override = getattr(overrides, field)
        if override:
            if override not in headers:
                raise ValueError(f"Column {override!r} not found. Available columns: {headers}")
            resolved[field] = override
        else:
            matches = sorted({normalized[_normalize(alias)] for alias in aliases if _normalize(alias) in normalized})
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple columns match {field!r}: {matches}. Use the explicit --{field.replace('_', '-')}-column option."
                )
            resolved[field] = matches[0] if matches else None
    missing = [field for field in ("question", "expected_answer", "answer") if not resolved[field]]
    if missing:
        raise ValueError(f"Could not identify required columns {missing}. Available columns: {headers}. Use explicit --*-column options.")
    return resolved


def _xlsx_rows(path: Path, sheet_name: str | None) -> tuple[str, list[dict[str, Any]]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    if sheet_name and sheet_name not in workbook.sheetnames:
        raise ValueError(f"Sheet {sheet_name!r} not found. Available sheets: {workbook.sheetnames}")
    sheet = workbook[sheet_name] if sheet_name else workbook.active
    values = sheet.iter_rows(values_only=True)
    try:
        headers = [str(value).strip() if value is not None else "" for value in next(values)]
    except StopIteration as exc:
        raise ValueError(f"Workbook sheet {sheet.title!r} is empty") from exc
    rows = [dict(zip(headers, row)) for row in values if any(value not in (None, "") for value in row)]
    workbook.close()
    return sheet.title, rows


def _tabular_rows(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("Every JSONL line must contain an object")
        return "jsonl", rows
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return "csv", list(csv.DictReader(handle))


def _safe_error(value: str, category: str = "stored_error") -> str:
    compact = " ".join(value.split())
    if not compact:
        return ""
    return f"{category}: {compact[:240]}" + ("…" if len(compact) > 240 else "")


def _looks_like_error(answer: str, explicit_error: str) -> str:
    if explicit_error.strip():
        return _safe_error(explicit_error)
    if not answer.strip():
        return "Premade result contains an empty chatbot answer"
    normalized = answer.casefold()
    if "spike arrest violation" in normalized:
        return "throttled: detected Spike Arrest violation in stored response"
    if normalized.lstrip().startswith(("{", "[")) and any(
        token in normalized for token in ("'fault'", '"fault"', "errorcode", "faultstring")
    ):
        return "api_fault: detected structured fault payload in stored response"
    return ""


def _as_bool(value: Any, default: bool = True) -> bool:
    if value in (None, ""):
        return default
    normalized = _normalize(value)
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid answerable value: {value!r}")


def read_premade_results(
    path: Path,
    *,
    sheet_name: str | None = None,
    columns: ResultColumns | None = None,
    progress_enabled: bool = False,
    strict: bool = False,
) -> list[tuple[SilverQuestion, ChatbotResult]]:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        source_name, rows = _xlsx_rows(path, sheet_name)
    elif suffix in {".csv", ".jsonl"}:
        source_name, rows = _tabular_rows(path)
    else:
        raise ValueError("Premade results must be .xlsx, .csv, or .jsonl")
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    headers = list(rows[0].keys())
    mapping = _resolve(headers, columns or ResultColumns())
    logger.info("premade_results_loading path=%s source=%s row_count=%d columns=%r", path, source_name, len(rows), mapping)
    results = []
    skipped: list[str] = []
    for index, row in enumerate(track(rows, enabled=progress_enabled, description="Reading premade results", total=len(rows)), 2):
        try:
            if None in row and any(value not in (None, "") for value in (row.get(None) or [])):
                raise ValueError("row has more values than headers")
            question_text = str(row.get(mapping["question"], "") or "").strip()
            expected = str(row.get(mapping["expected_answer"], "") or "").strip()
            answer = str(row.get(mapping["answer"], "") or "").strip()
            if not question_text or not expected:
                raise ValueError("missing question or reference answer")
            external_id = str(row.get(mapping["id"], "") or "").strip() if mapping["id"] else ""
            source = str(row.get(mapping["source"], "") or "").strip() if mapping["source"] else ""
            explicit_error = str(row.get(mapping["error"], "") or "").strip() if mapping["error"] else ""
            topic = str(row.get(mapping["topic"], "") or "לא סווג").strip() if mapping["topic"] else "לא סווג"
            answerable = _as_bool(row.get(mapping["answerable"])) if mapping["answerable"] else True
        except ValueError as exc:
            reason = f"row {index}: {exc}"
            skipped.append(reason)
            logger.warning("premade_row_skipped row=%d reason=%s", index, exc)
            continue
        question_id = f"ROW{index:05d}"
        refs = [SourceRef(file=path.name, location=f"{source_name}!row {index}", excerpt=source)] if source else []
        question = SilverQuestion(
            id=question_id, topic=topic or "לא סווג", question=question_text,
            expected_answer=expected, answerable=answerable,
            expected_behavior=ExpectedBehavior.ANSWER if answerable else ExpectedBehavior.ABSTAIN,
            reference_claims=[expected] if answerable else [], sources=refs,
            review_status="premade", reviewer_notes=f"Original ID: {external_id}" if external_id else "",
        )
        result = ChatbotResult(
            question_id=question_id, answer=answer, retrieved_context=source,
            metadata={"source_file": str(path), "source_sheet": source_name, "source_row": index, "original_id": external_id},
            error=_looks_like_error(answer, explicit_error),
        )
        results.append((question, result))
    if strict and skipped:
        preview = "; ".join(skipped[:10])
        raise ValueError(
            f"Strict import rejected {len(skipped)} invalid row(s): {preview}"
            + (f"; and {len(skipped) - 10} more" if len(skipped) > 10 else "")
        )
    logger.info("premade_results_loaded usable_rows=%d skipped_rows=%d", len(results), len(rows) - len(results))
    validate_question_set([question for question, _ in results])
    return results
