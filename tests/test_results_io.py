import json

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


def test_summary_generation_placeholder_is_an_infrastructure_error(tmp_path):
    path = tmp_path / "placeholder.csv"
    path.write_text(
        "question,expected_answer,answer\n"
        'Q,Reference,"לא הצלחנו ליצור סיכום לשאילתת החיפוש שלך, אבל כן מצאנו כמה תוצאות."\n',
        encoding="utf-8",
    )

    result = read_premade_results(path)[0][1]

    assert result.error.startswith("answer_generation_failed:")


def test_preserves_behavior_atomic_claims_provenance_and_parents(tmp_path):
    from chatbot_eval.models import ExpectedBehavior, QuestionForm
    from chatbot_eval.results_io import ImportDiagnostics
    path = tmp_path / "results.jsonl"
    rows = [dict(id="parent", question="Q", expected_answer="A and B", answer="A",
                 reference_claims=["A", "B"], source="runtime", reference_sources=[dict(file="gold", location="1", excerpt="A and B")]),
            dict(id="child", question="Which?", expected_answer="Specify", answer="Which?",
                 expected_behavior="clarify", parent_question_id="parent"),
            dict(id="orphan", question="Q?", expected_answer="No", answer="No", answerable=False,
                 parent_question_id="absent")]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    diagnostics = ImportDiagnostics()
    pairs = read_premade_results(path, diagnostics=diagnostics)
    assert pairs[0][0].reference_claims == ["A", "B"]
    assert pairs[0][0].sources[0].file == "gold"
    assert pairs[0][1].retrieved_context == "runtime"
    assert pairs[1][0].expected_behavior == ExpectedBehavior.CLARIFY
    assert pairs[1][0].question_form == QuestionForm.AMBIGUOUS
    assert pairs[1][0].parent_question_id == pairs[0][0].id
    assert pairs[2][0].parent_question_id == ""
    assert pairs[2][1].metadata["unresolved_parent_question_id"] == "absent"
    assert diagnostics.as_dict() == dict(total=3, accepted=3, skipped=0, errors=[])


def test_auto_detects_behavioral_metadata_aliases_and_keeps_evidence_roles_separate(tmp_path):
    from chatbot_eval.models import ExpectedBehavior, QuestionForm, QuestionType

    path = tmp_path / "aliased.jsonl"
    rows = [
        {
            "external_id": "parent", "user_prompt": "Question", "gold_answer": "Reference",
            "bot_response": "Candidate", "retrieved_chunks": "runtime retrieval",
            "target_behavior": "answer", "form": "canonical", "kind": "topic_integration",
            "complexity": "hard", "gold_claims": ["Reference"],
            "evidence_quotes": [{"source_id": "doc-1", "quote": "Reference evidence"}],
            "gold_sources": [{"source_id": "doc-1", "file": "gold.pdf", "location": "p. 1", "excerpt": "Reference evidence"}],
        },
        {
            "external_id": "child", "user_prompt": "Which one?", "gold_answer": "Specify the option",
            "bot_response": "Which option?", "target_behavior": "clarify", "form": "ambiguous",
            "source_question_id": "parent",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    pairs = read_premade_results(path)

    parent, parent_result = pairs[0]
    child, child_result = pairs[1]
    assert parent.expected_behavior == ExpectedBehavior.ANSWER
    assert parent.question_form == QuestionForm.CANONICAL
    assert parent.question_type == QuestionType.TOPIC_INTEGRATION
    assert parent.difficulty == "hard"
    assert parent.reference_claims == ["Reference"]
    assert parent.supporting_quotes[0].source_id == "doc-1"
    assert parent.sources[0].file == "gold.pdf"
    assert parent_result.retrieved_context == "runtime retrieval"
    assert parent_result.metadata["reference_evidence_origin"] == "explicit_reference"
    assert child.expected_behavior == ExpectedBehavior.CLARIFY
    assert child.question_form == QuestionForm.AMBIGUOUS
    assert child.parent_question_id == parent.id
    assert child_result.metadata["original_parent_question_id"] == "parent"


def test_explicit_behavioral_metadata_overrides_support_nonstandard_headers(tmp_path):
    path = tmp_path / "custom.jsonl"
    row = {
        "q": "Question", "gold": "Reference", "bot": "Candidate", "mode": "answer",
        "shape": "canonical", "parent": "", "category_kind": "document_wide", "level": "advanced",
        "claims": ["Reference"], "quotes": [{"source_id": "s1", "quote": "Reference quote"}],
        "gold_docs": [{"source_id": "s1", "file": "doc", "location": "1", "excerpt": "Reference quote"}],
    }
    path.write_text(json.dumps(row), encoding="utf-8")

    question, result = read_premade_results(path, columns=ResultColumns(
        question="q", expected_answer="gold", answer="bot", expected_behavior="mode",
        question_form="shape", parent_question_id="parent", question_type="category_kind",
        difficulty="level", reference_claims="claims", supporting_quotes="quotes",
        reference_sources="gold_docs",
    ))[0]

    assert question.question_type.value == "document_wide"
    assert question.difficulty == "advanced"
    assert question.reference_claims == ["Reference"]
    assert question.sources[0].source_id == "s1"
    assert result.retrieved_context == ""


def test_ambiguous_behavioral_aliases_require_an_explicit_mapping(tmp_path):
    path = tmp_path / "ambiguous-metadata.jsonl"
    path.write_text(json.dumps({
        "question": "Q", "expected_answer": "A", "answer": "A",
        "expected_behavior": "answer", "target_behavior": "answer",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="Multiple columns match 'expected_behavior'"):
        read_premade_results(path)

    question, _ = read_premade_results(
        path, columns=ResultColumns(expected_behavior="target_behavior"),
    )[0]
    assert question.expected_behavior.value == "answer"


def test_invalid_metadata_diagnostics_and_no_usable_rows(tmp_path):
    from chatbot_eval.results_io import ImportDiagnostics
    path = tmp_path / "results.csv"
    path.write_text("question,expected_answer,answer,answerable,expected_behavior,question_form\n"
                    "Q,A,A, true ,answer,canonical\n"
                    "Q,A,A,truue,answer,canonical\n"
                    "Q,A,A,true,clarify,canonical\n")
    diagnostics = ImportDiagnostics()
    pairs = read_premade_results(path, diagnostics=diagnostics)
    assert len(pairs) == 1
    assert diagnostics.skipped == 2
    assert [error["row"] for error in diagnostics.errors] == [3, 4]
    assert pairs[0][0].sources == []
    path.write_text("question,expected_answer,answer\nQ,,A\n")
    with pytest.raises(ValueError, match="No usable rows"):
        read_premade_results(path)


@pytest.mark.parametrize("text", ["question,expected_answer,answer,answer\nQ,A,A,A\n", "question,expected_answer,\nQ,A,A\n"])
def test_malformed_headers_rejected(tmp_path, text):
    path = tmp_path / "results.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match="headers"):
        read_premade_results(path)


def test_fixing_one_row_does_not_change_other_rows_resume_fingerprints(tmp_path):
    from chatbot_eval.artifacts import evaluation_fingerprint

    path = tmp_path / "results.csv"

    def fingerprint_of_first_row(second_question):
        path.write_text(
            "question,expected_answer,answer\nכמה ימים?,שלושה ימים,שלושה\n"
            f"{second_question},תשובה,תשובה\n",
            encoding="utf-8",
        )
        pairs = read_premade_results(path)
        question, result = pairs[0]
        return evaluation_fingerprint(question, result)

    # The second row is skipped (no question) and then fixed; the first row's identity is unaffected.
    assert fingerprint_of_first_row("") == fingerprint_of_first_row("מה הנוהל?")
