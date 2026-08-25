from chatbot_eval.io import read_questions, write_questions
from chatbot_eval.models import EvidenceQuote, ExpectedBehavior, QuestionForm, QuestionType, SilverQuestion, SourceRef


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
