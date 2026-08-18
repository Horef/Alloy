from openpyxl import Workbook

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
    assert "Spike arrest" in pairs[1][1].error
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
