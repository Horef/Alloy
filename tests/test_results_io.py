from openpyxl import Workbook
import pytest

from chatbot_eval.results_io import ResultColumns, read_premade_results


def test_reads_hebrew_xlsx_and_marks_api_fault(tmp_path):
    path = tmp_path / "results.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["שאלה", "תשובה צפויה", "תשובה מהמודל", "מקור", "messageId"])
    sheet.append(["מה הנוהל?", "יש להגיש טופס.", "יש להגיש טופס.", "מסמך נוהל", "abc"])
    sheet.append(["שאלה שנייה", "תשובה", "{'fault': {'faultstring': 'Spike arrest violation'}}", None, None])
    workbook.save(path)

    pairs = read_premade_results(path)
    assert len(pairs) == 2
    assert pairs[0][0].question == "מה הנוהל?"
    assert pairs[0][1].error == ""
    assert "Spike Arrest" in pairs[1][1].error
    assert pairs[0][1].metadata["original_id"] == "abc"


def test_explicit_column_overrides_for_csv(tmp_path):
    path = tmp_path / "custom.csv"
    path.write_text("ask,gold,bot\nQuestion,Reference,Candidate\n", encoding="utf-8")
    pairs = read_premade_results(
        path,
        columns=ResultColumns(question="ask", expected_answer="gold", answer="bot"),
    )
    assert pairs[0][0].expected_answer == "Reference"
    assert pairs[0][1].answer == "Candidate"


def test_ambiguous_alias_columns_require_an_explicit_mapping(tmp_path):
    path = tmp_path / "ambiguous.csv"
    path.write_text(
        "question,expected_answer,answer,response\nQ,Reference,First,Second\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Multiple columns match 'answer'"):
        read_premade_results(path)

    pairs = read_premade_results(path, columns=ResultColumns(answer="response"))
    assert pairs[0][1].answer == "Second"


def test_invalid_rows_can_be_skipped_or_rejected_strictly(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text(
        "question,expected_answer,answer\n"
        "Valid,Reference,Candidate\n"
        "Missing reference,,Candidate\n",
        encoding="utf-8",
    )

    assert len(read_premade_results(path)) == 1
    with pytest.raises(ValueError, match=r"Strict import rejected 1 invalid row.*row 3"):
        read_premade_results(path, strict=True)


def test_explicit_errors_are_bounded_and_normalized(tmp_path):
    path = tmp_path / "error.csv"
    path.write_text(
        "question,expected_answer,answer,error\nQ,Reference,," + "secret " * 100 + "\n",
        encoding="utf-8",
    )

    error = read_premade_results(path)[0][1].error
    assert error.startswith("stored_error:")
    assert len(error) <= 316
