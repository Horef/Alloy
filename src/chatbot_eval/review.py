"""Human-readable review sidecar and lossless round-trip merge.

Generated silver questions carry many technical columns (provenance JSON, verbatim
supporting quotes, atomic reference claims, CSV escape metadata) that a human reviewer
does not need to see and should not edit by hand. This module exports a small, readable
review file that contains only the fields a reviewer legitimately changes, and merges an
edited review file back onto the canonical technical set so that as much technical
information as possible is preserved for evaluation.

Design
------
* The canonical ``silver_questions.jsonl`` remains the single source of truth for every
  technical field. The review file never has to reconstruct provenance or claims.
* Rows are matched by the stable ``id``. The reviewer must not edit that column.
* On merge the canonical record is copied and only the reviewer-editable fields are
  overlaid. A row missing from the review file is treated as an intentional human
  deletion. A row whose ``id`` is not in the canonical set cannot be grounded and is
  reported rather than silently invented.
* When a reviewer edits a factual field (``question`` or ``expected_answer``) the stored
  provenance, supporting quotes, and reference claims may no longer match. We keep the
  technical fields (so no information is lost) but flag the row so evaluation does not
  silently trust stale grounding: an ``approved`` status is downgraded and a note is
  appended describing what changed.
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .io import _staged_text_file, _safe_csv_value, csv_question_row
from .labels import EXPECTED_BEHAVIOR_HEBREW, QUESTION_FORM_HEBREW
from .models import ExpectedBehavior, QuestionForm, SilverQuestion
from .validation import parse_bool, validate_question_set

logger = logging.getLogger(__name__)

# Columns exposed to the human reviewer. ``id`` is the immutable join key. The three
# read-only reference columns give the reviewer enough context to judge each question
# without exposing raw JSON. ``review_status`` and ``reviewer_notes`` are the decision
# columns; the remaining content columns may be edited in place.
REVIEW_COLUMNS = [
    "id",
    "topic",
    "question",
    "expected_answer",
    "answerable",
    "question_form",
    "expected_behavior",
    "difficulty",
    "sources_readable",
    "supporting_quotes_readable",
    "review_status",
    "reviewer_notes",
]

# Fields the reviewer is allowed to change directly. Everything else is preserved from the
# canonical record on merge.
EDITABLE_CONTENT_FIELDS = (
    "topic", "question", "expected_answer", "answerable",
    "question_form", "expected_behavior", "difficulty",
)
DECISION_FIELDS = ("review_status", "reviewer_notes")

# Editing one of these changes the evaluation task itself, so stored grounding may become
# stale. Such edits are flagged during merge.
FACTUAL_FIELDS = ("question", "expected_answer")

READONLY_HELP = "(עמודת עזר לקריאה בלבד — אין לערוך)"

# Optional Hebrew header labels for reviewers who work in Hebrew. Export can write these
# instead of the canonical English keys; merge accepts either language by normalizing every
# header back to its canonical key. The ``id`` column keeps its English name because it is
# the immutable machine join key and must not be localized or edited.
HEBREW_COLUMN_LABELS = {
    "id": "id",
    "topic": "נושא",
    "question": "שאלה",
    "expected_answer": "תשובה צפויה",
    "answerable": "ניתנת למענה",
    "question_form": "צורת שאלה",
    "expected_behavior": "התנהגות מצופה",
    "difficulty": "רמת קושי",
    "sources_readable": "מקורות (לקריאה בלבד)",
    "supporting_quotes_readable": "ציטוטים תומכים (לקריאה בלבד)",
    "review_status": "סטטוס סקירה",
    "reviewer_notes": "הערות סוקר",
}
# Reverse lookup used by merge to accept Hebrew-headed files. Canonical keys map to
# themselves so a mixed or English file still resolves.
_HEBREW_TO_CANONICAL = {label: key for key, label in HEBREW_COLUMN_LABELS.items()}


def _flatten_sources(question: SilverQuestion) -> str:
    """Render provenance as plain, non-editable reference text."""
    lines = []
    for index, source in enumerate(question.sources, 1):
        header = " · ".join(part for part in (source.file, source.location) if part)
        excerpt = " ".join((source.excerpt or "").split())
        if len(excerpt) > 500:
            excerpt = excerpt[:500] + "…"
        lines.append(f"[{index}] {header}\n{excerpt}".strip())
    return "\n\n".join(lines)


def _flatten_quotes(question: SilverQuestion) -> str:
    lines = []
    for quote in question.supporting_quotes:
        text = " ".join((quote.quote or "").split())
        lines.append(f"({quote.source_id}) {text}" if quote.source_id else text)
    return "\n".join(lines)


def _review_row(question: SilverQuestion) -> dict:
    return {
        "id": question.id,
        "topic": question.topic,
        "question": question.question,
        "expected_answer": question.expected_answer,
        "answerable": "true" if question.answerable else "false",
        "question_form": question.question_form.value,
        "expected_behavior": question.expected_behavior.value,
        "difficulty": question.difficulty,
        "sources_readable": _flatten_sources(question),
        "supporting_quotes_readable": _flatten_quotes(question),
        "review_status": question.review_status,
        "reviewer_notes": question.reviewer_notes,
    }


def write_review_file(
    questions: Iterable[SilverQuestion],
    output_dir: Path,
    *,
    hebrew_columns: bool = False,
) -> tuple[Path, Path]:
    """Export a reviewer-friendly CSV and a read-only Markdown view.

    When ``hebrew_columns`` is set, the CSV header uses Hebrew labels (except the machine
    ``id`` column). ``review-merge`` reads either language, so the choice is purely
    cosmetic for the reviewer. Returns the CSV path (the file the reviewer edits) and the
    Markdown path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    items = validate_question_set(list(questions))
    csv_path = output_dir / "questions_for_review.csv"
    markdown_path = output_dir / "questions_for_review.md"
    fieldnames = [HEBREW_COLUMN_LABELS[key] for key in REVIEW_COLUMNS] if hebrew_columns else list(REVIEW_COLUMNS)
    with (
        _staged_text_file(csv_path, encoding="utf-8-sig", newline="") as (csv_handle, csv_temporary),
        _staged_text_file(markdown_path, encoding="utf-8") as (md_handle, md_temporary),
    ):
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        for question in items:
            # Neutralize spreadsheet formula injection per cell. The review file has no
            # csv_escape_version column, so read-back uses the legacy restore path in
            # csv_question_row, which strips a single leading apostrophe before a formula.
            row = {key: _safe_csv_value(value) for key, value in _review_row(question).items()}
            if hebrew_columns:
                row = {HEBREW_COLUMN_LABELS[key]: value for key, value in row.items()}
            writer.writerow(row)
        md_handle.write(_render_markdown(items))
        for handle, temporary, destination in (
            (csv_handle, csv_temporary, csv_path),
            (md_handle, md_temporary, markdown_path),
        ):
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, destination)
    return csv_path, markdown_path


def _render_markdown(items: list[SilverQuestion]) -> str:
    lines = [
        "# שאלות לסקירה אנושית",
        "",
        "קובץ קריאה בלבד. לעריכה השתמשו ב-`questions_for_review.csv`.",
        f"סך הכול {len(items)} שאלות.",
        "",
    ]
    for question in items:
        lines.append(f"## {question.id} — {question.topic}")
        lines.append("")
        lines.append(f"**שאלה:** {question.question}")
        lines.append("")
        lines.append(f"**תשובה צפויה:** {question.expected_answer}")
        lines.append("")
        lines.append(
            "| ניתנת למענה | צורת שאלה | התנהגות מצופה | רמת קושי | סטטוס סקירה |"
        )
        lines.append("|---|---|---|---|---|")
        lines.append(
            f"| {'כן' if question.answerable else 'לא'} "
            f"| {QUESTION_FORM_HEBREW[question.question_form]} "
            f"| {EXPECTED_BEHAVIOR_HEBREW[question.expected_behavior]} "
            f"| {question.difficulty} | {question.review_status} |"
        )
        lines.append("")
        sources = _flatten_sources(question)
        if sources:
            lines.append("**מקורות:**")
            lines.append("")
            lines.append("```")
            lines.append(sources)
            lines.append("```")
            lines.append("")
        if question.reviewer_notes:
            lines.append(f"**הערות סוקר:** {question.reviewer_notes}")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


@dataclass
class MergeDiagnostics:
    canonical_total: int = 0
    review_total: int = 0
    merged: int = 0
    deleted_by_reviewer: int = 0
    content_edited: int = 0
    factual_edits_flagged: int = 0
    unknown_ids: list[str] = field(default_factory=list)
    duplicate_review_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "canonical_total": self.canonical_total,
            "review_total": self.review_total,
            "merged": self.merged,
            "deleted_by_reviewer": self.deleted_by_reviewer,
            "content_edited": self.content_edited,
            "factual_edits_flagged": self.factual_edits_flagged,
            "unknown_ids": self.unknown_ids,
            "duplicate_review_ids": self.duplicate_review_ids,
        }


def _canonical_header(header: str) -> str:
    """Map an English or Hebrew review header to its canonical field key.

    Unknown headers pass through unchanged so extra reviewer columns are simply ignored on
    merge rather than causing a failure.
    """
    header = header.strip()
    return _HEBREW_TO_CANONICAL.get(header, header)


def _read_review_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix != ".csv":
        raise ValueError("The edited review file must be the CSV exported for review (.csv)")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if any(not header.strip() for header in headers) or len(set(headers)) != len(headers):
            raise ValueError("Blank or duplicate review CSV headers are not allowed")
        canonical_headers = [_canonical_header(header) for header in headers]
        if "id" not in canonical_headers:
            raise ValueError("Review file must keep the 'id' column as the join key")
        if len(set(canonical_headers)) != len(canonical_headers):
            raise ValueError("Review headers collapse to duplicate fields after language normalization")
        # Rename each row's keys to canonical field names before restoring CSV escaping.
        rows = []
        for raw in reader:
            renamed = {_canonical_header(key): value for key, value in raw.items()}
            rows.append(csv_question_row(renamed))
        return rows


def _coerce_form(value: str, current: QuestionForm) -> QuestionForm:
    value = (value or "").strip()
    return QuestionForm(value) if value else current


def _coerce_behavior(value: str, current: ExpectedBehavior) -> ExpectedBehavior:
    value = (value or "").strip()
    return ExpectedBehavior(value) if value else current


def merge_review_file(
    canonical: list[SilverQuestion],
    review_path: Path,
    *,
    diagnostics: MergeDiagnostics | None = None,
) -> list[SilverQuestion]:
    """Reapply reviewer edits onto the canonical technical question set.

    ``canonical`` is the full technical set (typically read from
    ``silver_questions.jsonl``). Returns the merged, validated question set in the review
    file's row order, dropping questions the reviewer removed.
    """
    diagnostics = diagnostics if diagnostics is not None else MergeDiagnostics()
    by_id = {question.id: question for question in canonical}
    diagnostics.canonical_total = len(canonical)

    rows = _read_review_rows(review_path)
    diagnostics.review_total = len(rows)

    seen: set[str] = set()
    merged: list[SilverQuestion] = []
    for number, row in enumerate(rows, 2):
        identifier = (row.get("id") or "").strip()
        if not identifier:
            raise ValueError(f"row {number}: review row is missing an id")
        if identifier in seen:
            diagnostics.duplicate_review_ids.append(identifier)
            raise ValueError(f"row {number}: duplicate id {identifier!r} in review file")
        seen.add(identifier)
        base = by_id.get(identifier)
        if base is None:
            diagnostics.unknown_ids.append(identifier)
            raise ValueError(
                f"row {number}: id {identifier!r} is not in the canonical set; new questions "
                "cannot be grounded from the review file. Generate them instead."
            )
        merged.append(_apply_edits(base, row, diagnostics))

    diagnostics.merged = len(merged)
    diagnostics.deleted_by_reviewer = len(by_id) - len(seen & set(by_id))
    return validate_question_set(merged)


def _apply_edits(base: SilverQuestion, row: dict, diagnostics: MergeDiagnostics) -> SilverQuestion:
    updated = base.model_copy(deep=True)
    factual_changes: list[str] = []
    content_changed = False

    if "topic" in row and row["topic"].strip():
        content_changed |= updated.topic != row["topic"].strip()
        updated.topic = row["topic"].strip()
    if "difficulty" in row and row["difficulty"].strip():
        content_changed |= updated.difficulty != row["difficulty"].strip()
        updated.difficulty = row["difficulty"].strip()
    if "answerable" in row and (row.get("answerable") or "").strip():
        new_answerable = parse_bool(row["answerable"], updated.answerable)
        content_changed |= updated.answerable != new_answerable
        updated.answerable = new_answerable
    if "question_form" in row:
        new_form = _coerce_form(row["question_form"], updated.question_form)
        content_changed |= updated.question_form != new_form
        updated.question_form = new_form
    if "expected_behavior" in row:
        new_behavior = _coerce_behavior(row["expected_behavior"], updated.expected_behavior)
        content_changed |= updated.expected_behavior != new_behavior
        updated.expected_behavior = new_behavior

    for factual in FACTUAL_FIELDS:
        if factual in row and row[factual].strip() and row[factual].strip() != getattr(updated, factual).strip():
            factual_changes.append(factual)
            setattr(updated, factual, row[factual].strip())

    # Decision fields overlay unconditionally.
    if "reviewer_notes" in row:
        updated.reviewer_notes = row["reviewer_notes"]
    if "review_status" in row and row["review_status"].strip():
        updated.review_status = row["review_status"].strip()

    if content_changed or factual_changes:
        diagnostics.content_edited += 1
    if factual_changes:
        diagnostics.factual_edits_flagged += 1
        note = (
            "עריכה אנושית שינתה שדה עובדתי ("
            + ", ".join(factual_changes)
            + ") — יש לאמת מחדש מול המקורות והטענות המאוחסנות"
        )
        updated.reviewer_notes = f"{updated.reviewer_notes}\n{note}".strip() if updated.reviewer_notes else note
        if updated.review_status.lower() == "approved":
            updated.review_status = "needs_reground"
            logger.warning(
                "review_merge_factual_edit id=%s downgraded approved->needs_reground fields=%s",
                updated.id, ",".join(factual_changes),
            )
    return updated
