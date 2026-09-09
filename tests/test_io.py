import pytest

from chatbot_eval.io import read_questions, write_evaluations, write_questions
from chatbot_eval.models import (
    ChatbotResult,
    EvaluationRecord,
    EvidenceQuote,
    ExpectedBehavior,
    Outcome,
    QuestionForm,
    QuestionType,
    SilverQuestion,
    SourceRef,
)


def test_question_csv_round_trip(tmp_path):
    original = SilverQuestion(
        id="Q0001", topic="Policy", question="When?", expected_answer="Tomorrow",
        question_type=QuestionType.TOPIC_INTEGRATION,
        question_form=QuestionForm.AMBIGUOUS,
        expected_behavior=ExpectedBehavior.CLARIFY,
        parent_question_id="Q0000",
        supporting_quotes=[EvidenceQuote(source_id="a.md#chunk-1", quote="Tomorrow")],
        sources=[SourceRef(source_id="a.md#chunk-1", file="a.md", location="chars 0-20", excerpt="Tomorrow")],
    )
    csv_path, _ = write_questions([original], tmp_path)
    loaded = read_questions(csv_path)
    assert loaded[0].question == original.question
    assert loaded[0].sources[0].file == "a.md"
    assert loaded[0].sources[0].source_id == "a.md#chunk-1"
    assert loaded[0].question_type == QuestionType.TOPIC_INTEGRATION
    assert loaded[0].supporting_quotes[0].quote == "Tomorrow"
    assert loaded[0].question_form == QuestionForm.AMBIGUOUS
    assert loaded[0].expected_behavior == ExpectedBehavior.CLARIFY
    assert loaded[0].parent_question_id == "Q0000"


def test_question_csv_neutralizes_formulas_without_changing_round_trip(tmp_path):
    original = SilverQuestion(
        id="Q0001", topic="Policy", question="  =HYPERLINK(\"bad\")",
        expected_answer="+SUM(1,1)",
    )

    csv_path, _ = write_questions([original], tmp_path)
    raw = csv_path.read_text(encoding="utf-8-sig")
    loaded = read_questions(csv_path)[0]

    assert "'  =HYPERLINK" in raw
    assert "'+SUM" in raw
    assert loaded.question == original.question
    assert loaded.expected_answer == original.expected_answer


def test_lossless_sources_and_literal_apostrophes(tmp_path):
    original = SilverQuestion(
        id="Q", topic="'literal", question="'=literal formula text", expected_answer="שלום\nתשובה",
        sources=[SourceRef(source_id="a | b", file="a | b.md", location="page | 2",
                           excerpt='תא | תא\n"ציטוט"'),
                 SourceRef(source_id="second", file="'file", location="", excerpt="=formula")],
    )
    csv_path, jsonl_path = write_questions([original], tmp_path)
    assert read_questions(csv_path) == read_questions(jsonl_path)
    assert read_questions(csv_path)[0] == original


def test_canonical_sources_conflicts_and_malformed_json(tmp_path):
    import csv
    import pytest
    original = SilverQuestion(id="Q", topic="T", question="Q?", expected_answer="A",
                              sources=[SourceRef(file="a", location="b", excerpt="c")])
    path, _ = write_questions([original], tmp_path)
    with path.open(encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        row = next(reader)
    def save():
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerow(row)
    row["source_excerpts"] = "edited"
    save()
    with pytest.raises(ValueError, match="conflicts"):
        read_questions(path)
    row["sources_json"] = "invalid"
    save()
    with pytest.raises(ValueError, match="Invalid sources_json"):
        read_questions(path)


def test_csv_jsonl_legacy_normalization(tmp_path):
    import json
    import csv
    rows = [dict(id="Q1", topic="T", question="Q", expected_answer="No", answerable=" false "),
            dict(id="Q2", topic="T", question="Which?", expected_answer="Clarify", expected_behavior="clarify")]
    jp = tmp_path / "questions.jsonl"
    jp.write_text("\n".join(json.dumps(row) for row in rows))
    cp = tmp_path / "questions.csv"
    with cp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    assert read_questions(cp) == read_questions(jp)
    assert read_questions(cp)[1].question_form == QuestionForm.AMBIGUOUS


def test_question_outputs_preserve_existing_pair_when_serialization_fails(tmp_path, monkeypatch):
    csv_path = tmp_path / "silver_questions.csv"
    jsonl_path = tmp_path / "silver_questions.jsonl"
    csv_path.write_bytes(b"existing csv")
    jsonl_path.write_bytes(b"existing jsonl")
    question = SilverQuestion(id="Q1", topic="T", question="Q?", expected_answer="A")

    def fail_json_serialization(self, *args, **kwargs):
        raise RuntimeError("synthetic serialization failure")

    monkeypatch.setattr(SilverQuestion, "model_dump_json", fail_json_serialization)
    with pytest.raises(RuntimeError, match="synthetic serialization failure"):
        write_questions([question], tmp_path)

    assert csv_path.read_bytes() == b"existing csv"
    assert jsonl_path.read_bytes() == b"existing jsonl"
    assert not list(tmp_path.glob(".silver_questions.*"))


def test_evaluation_outputs_preserve_existing_pair_when_serialization_fails(tmp_path, monkeypatch):
    csv_path = tmp_path / "evaluation_details.csv"
    jsonl_path = tmp_path / "evaluation_details.jsonl"
    csv_path.write_bytes(b"existing csv")
    jsonl_path.write_bytes(b"existing jsonl")
    question = SilverQuestion(id="Q1", topic="T", question="Q?", expected_answer="A")
    record = EvaluationRecord(
        question=question,
        result=ChatbotResult(question_id="Q1", answer="A"),
        outcome=Outcome.CORRECT_ANSWER,
    )

    def fail_json_serialization(self, *args, **kwargs):
        raise RuntimeError("synthetic serialization failure")

    monkeypatch.setattr(EvaluationRecord, "model_dump_json", fail_json_serialization)
    with pytest.raises(RuntimeError, match="synthetic serialization failure"):
        write_evaluations([record], tmp_path)

    assert csv_path.read_bytes() == b"existing csv"
    assert jsonl_path.read_bytes() == b"existing jsonl"
    assert not list(tmp_path.glob(".evaluation_details.*"))
