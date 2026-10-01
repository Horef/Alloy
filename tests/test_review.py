import csv

import pytest

from chatbot_eval.io import write_questions
from chatbot_eval.models import (
    EvidenceQuote,
    ExpectedBehavior,
    QuestionForm,
    QuestionType,
    SilverQuestion,
    SourceRef,
)
from chatbot_eval.review import (
    REVIEW_COLUMNS,
    MergeDiagnostics,
    merge_review_file,
    write_review_file,
)


def _question(identifier="Q0001", **overrides):
    base = dict(
        id=identifier,
        topic="מדיניות",
        question="מהם התנאים?",
        expected_answer="התנאים הם א׳ ו-ב׳",
        question_type=QuestionType.TOPIC_INTEGRATION,
        reference_claims=["תנאי א׳", "תנאי ב׳"],
        supporting_quotes=[EvidenceQuote(source_id="a.md#chunk-1", quote="התנאים הם א׳ ו-ב׳")],
        sources=[SourceRef(source_id="a.md#chunk-1", file="a.md", location="chars 0-40", excerpt="התנאים הם א׳ ו-ב׳")],
        review_status="approved",
    )
    base.update(overrides)
    return SilverQuestion(**base)


def _edit_review_csv(path, mutate):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows = mutate(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_review_export_only_has_readable_columns(tmp_path):
    csv_path, md_path = write_review_file([_question()], tmp_path)
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        headers = next(csv.reader(handle))
    assert headers == REVIEW_COLUMNS
    # No raw technical JSON leaks into the reviewer file.
    assert "sources_json" not in headers
    assert "supporting_quotes" not in headers
    assert "csv_escape_version" not in headers
    text = md_path.read_text(encoding="utf-8")
    assert "Q0001" in text and "מהם התנאים?" in text


def test_review_merge_preserves_technical_columns_on_status_only_edit(tmp_path):
    original = _question(review_status="pending")
    write_questions([original], tmp_path)  # canonical jsonl/csv
    review_csv, _ = write_review_file([original], tmp_path / "review")

    def approve(rows):
        rows[0]["review_status"] = "approved"
        rows[0]["reviewer_notes"] = "נבדק ואושר"
        return rows

    _edit_review_csv(review_csv, approve)

    merged = merge_review_file([original], review_csv)
    assert len(merged) == 1
    merged_q = merged[0]
    # Human decision applied.
    assert merged_q.review_status == "approved"
    assert merged_q.reviewer_notes == "נבדק ואושר"
    # Every technical field preserved from canonical.
    assert merged_q.sources == original.sources
    assert merged_q.supporting_quotes == original.supporting_quotes
    assert merged_q.reference_claims == original.reference_claims
    assert merged_q.question_type == original.question_type


def test_review_shows_variant_context_read_only(tmp_path):
    original = _question(
        "Q0002", question_form=QuestionForm.AMBIGUOUS, parent_question_id="Q0001",
        clarification_acceptable=True, acceptable_clarification="לאיזה מסלול הכוונה?",
        closed_book_answerable=True,
    )
    review_csv, md_path = write_review_file([original], tmp_path)

    def tamper(rows):
        rows[0]["parent_question_id"] = "Q9999"
        rows[0]["acceptable_clarification"] = "אחר"
        rows[0]["closed_book_answerable"] = "false"
        return rows

    rows = list(csv.DictReader(review_csv.open(encoding="utf-8-sig", newline="")))
    assert rows[0]["parent_question_id"] == "Q0001" and rows[0]["closed_book_answerable"] == "true"
    assert "לאיזה מסלול הכוונה?" in md_path.read_text(encoding="utf-8")
    _edit_review_csv(review_csv, tamper)
    merged = merge_review_file([original], review_csv)[0]
    assert (merged.parent_question_id, merged.acceptable_clarification, merged.closed_book_answerable) == (
        "Q0001", "לאיזה מסלול הכוונה?", True,
    )


def test_review_merge_drops_questions_deleted_by_reviewer(tmp_path):
    canonical = [_question("Q0001"), _question("Q0002"), _question("Q0003")]
    review_csv, _ = write_review_file(canonical, tmp_path)

    def delete_middle(rows):
        return [row for row in rows if row["id"] != "Q0002"]

    _edit_review_csv(review_csv, delete_middle)

    diagnostics = MergeDiagnostics()
    merged = merge_review_file(canonical, review_csv, diagnostics=diagnostics)
    assert [q.id for q in merged] == ["Q0001", "Q0003"]
    assert diagnostics.deleted_by_reviewer == 1


def test_review_merge_flags_factual_edit_and_downgrades_approval(tmp_path):
    original = _question(review_status="approved")
    review_csv, _ = write_review_file([original], tmp_path)

    def edit_answer(rows):
        rows[0]["expected_answer"] = "תשובה חדשה לגמרי"
        return rows

    _edit_review_csv(review_csv, edit_answer)

    diagnostics = MergeDiagnostics()
    merged = merge_review_file([original], review_csv, diagnostics=diagnostics)
    merged_q = merged[0]
    assert merged_q.expected_answer == "תשובה חדשה לגמרי"
    # Stored grounding is kept, not discarded.
    assert merged_q.sources == original.sources
    # But approval is downgraded and a note explains why.
    assert merged_q.review_status == "needs_reground"
    assert "עובדתי" in merged_q.reviewer_notes
    assert diagnostics.factual_edits_flagged == 1


def test_review_merge_rejects_unknown_id(tmp_path):
    original = _question("Q0001")
    review_csv, _ = write_review_file([original], tmp_path)

    def rename(rows):
        rows[0]["id"] = "Q9999"
        return rows

    _edit_review_csv(review_csv, rename)

    with pytest.raises(ValueError, match="not in the canonical set"):
        merge_review_file([original], review_csv)


def test_review_merge_rejects_duplicate_ids(tmp_path):
    original = _question("Q0001")
    review_csv, _ = write_review_file([original, _question("Q0002")], tmp_path)

    def duplicate(rows):
        rows[1]["id"] = "Q0001"
        return rows

    _edit_review_csv(review_csv, duplicate)

    with pytest.raises(ValueError, match="duplicate id"):
        merge_review_file([original, _question("Q0002")], review_csv)


def test_review_merge_preserves_clarification_task_edits(tmp_path):
    clarify = _question(
        "Q0001",
        question_form=QuestionForm.AMBIGUOUS,
        expected_behavior=ExpectedBehavior.CLARIFY,
        reference_claims=[],
        supporting_quotes=[],
        expected_answer="לאיזו אוכלוסייה הכוונה?",
    )
    review_csv, _ = write_review_file([clarify], tmp_path)

    def note_only(rows):
        rows[0]["reviewer_notes"] = "בסדר"
        return rows

    _edit_review_csv(review_csv, note_only)
    merged = merge_review_file([clarify], review_csv)
    assert merged[0].expected_behavior == ExpectedBehavior.CLARIFY
    assert merged[0].reference_claims == []


def test_review_export_custom_basename_names_both_files(tmp_path):
    csv_path, md_path = write_review_file([_question()], tmp_path, basename="questions_for_review_hova")
    assert csv_path.name == "questions_for_review_hova.csv"
    assert md_path.name == "questions_for_review_hova.md"
    # The read-only markdown points reviewers at the correctly named editable CSV.
    assert "questions_for_review_hova.csv" in md_path.read_text(encoding="utf-8")


def test_review_export_sanitizes_unsafe_basename(tmp_path):
    csv_path, _ = write_review_file([_question()], tmp_path, basename="../../evil name!!")
    assert csv_path.parent == tmp_path
    assert csv_path.name.endswith(".csv") and "/" not in csv_path.name


def test_review_export_hebrew_columns_header(tmp_path):
    csv_path, _ = write_review_file([_question()], tmp_path, hebrew_columns=True)
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        headers = next(csv.reader(handle))
    # id stays English as the machine join key; the rest are Hebrew.
    assert headers[0] == "id"
    assert "שאלה" in headers
    assert "תשובה צפויה" in headers
    assert "סטטוס סקירה" in headers
    assert "question" not in headers


def test_review_merge_reads_hebrew_columns(tmp_path):
    original = _question(review_status="pending")
    review_csv, _ = write_review_file([original], tmp_path, hebrew_columns=True)

    def approve(rows):
        rows[0]["סטטוס סקירה"] = "approved"
        rows[0]["הערות סוקר"] = "אושר"
        return rows

    _edit_review_csv(review_csv, approve)

    merged = merge_review_file([original], review_csv)
    assert merged[0].review_status == "approved"
    assert merged[0].reviewer_notes == "אושר"
    # Technical columns still preserved from canonical.
    assert merged[0].sources == original.sources
    assert merged[0].reference_claims == original.reference_claims


def test_review_merge_reads_hebrew_factual_edit_and_flags(tmp_path):
    original = _question(review_status="approved")
    review_csv, _ = write_review_file([original], tmp_path, hebrew_columns=True)

    def edit(rows):
        rows[0]["תשובה צפויה"] = "תשובה חדשה"
        return rows

    _edit_review_csv(review_csv, edit)
    merged = merge_review_file([original], review_csv)
    assert merged[0].expected_answer == "תשובה חדשה"
    assert merged[0].review_status == "needs_reground"
