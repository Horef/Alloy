from chatbot_eval.insights import generate_insights, write_insights
from chatbot_eval.models import (
    ChatbotResult, EvaluationInsights, EvaluationRecord, InsightIssue,
    JudgeScores, Outcome, SilverQuestion,
)
from chatbot_eval.report import write_report


def test_insights_are_validated_saved_and_embedded(tmp_path):
    scores = JudgeScores(
        claim_assessments=[],
        required_points_total=2, answer_points_addressed=1, answer_points_correct=0,
        answer_false_claims=1, answer_unsupported_claims=0, answer_extraneous_claims=0,
        retrieval_points_found=2, retrieved_chunks_total=1,
        retrieved_chunks_relevant=1, retrieved_chunks_contradictory=0,
        answer_scope="exact", incorrect_type="hallucination", response_is_abstention=False,
        explanation="הסבר", missing_or_wrong="פרט שגוי", retrieval_explanation="האחזור הצליח",
    )
    record = EvaluationRecord(
        question=SilverQuestion(id="Q1", topic="מידע אישי", question="שאלה", expected_answer="תשובה"),
        result=ChatbotResult(question_id="Q1", answer="תשובה שגויה", retrieved_context="מקור"),
        outcome=Outcome.MISLEADING_HALLUCINATION, scores=scores,
    )

    class FakeLLM:
        def generate(self, prompt, schema, model):
            assert "Q1" in prompt
            return EvaluationInsights(
                executive_summary="זוהה כשל חוזר בשלב יצירת התשובה.",
                strengths=["האחזור מצא את המידע הנדרש."],
                issues=[InsightIssue(
                    title="התעלמות ממידע שנמצא", priority="high", confidence="medium",
                    evidence_count=1, affected_topics=["מידע אישי"], observed_pattern="האחזור הצליח אך התשובה שגויה.",
                    likely_cause_hypothesis="ייתכן שהנחיות המענה אינן מדגישות שימוש במקור.",
                    recommendation="להוסיף בדיקת התאמה למקור לפני החזרת תשובה.",
                    example_question_ids=["Q1", "NOT_REAL"],
                )], methodology_note="הניתוח מבוסס על דפוסים בתוצאות.",
            )

    insights = generate_insights([record], FakeLLM(), "fake")
    assert insights.issues[0].example_question_ids == ["Q1"]
    assert insights.issues[0].evidence_count == 1
    assert insights.issues[0].affected_topics == ["מידע אישי"]
    assert insights.issues[0].priority == "medium"
    assert insights.issues[0].confidence == "low"
    assert write_insights(insights, tmp_path).exists()
    _, report_path = write_report([record], tmp_path, insights)
    report = report_path.read_text(encoding="utf-8")
    assert "תובנות והמלצות" in report
    assert "אינם מוכיחים סיבתיות" in report


def test_insights_without_valid_evidence_are_discarded():
    record = EvaluationRecord(
        question=SilverQuestion(id="Q1", topic="נושא", question="שאלה", expected_answer="תשובה"),
        result=ChatbotResult(question_id="Q1", answer="שגוי"),
        outcome=Outcome.UNRELATED_ANSWER,
    )

    class FakeLLM:
        def generate(self, prompt, schema, model):
            return EvaluationInsights(
                executive_summary="סיכום",
                issues=[InsightIssue(
                    title="בעיה", priority="medium", confidence="medium", evidence_count=1,
                    affected_topics=["נושא"], observed_pattern="דפוס",
                    likely_cause_hypothesis="השערה", recommendation="המלצה",
                    example_question_ids=["NOT_REAL"],
                )],
                strengths=[], methodology_note="שיטה",
            )

    assert generate_insights([record], FakeLLM(), "fake").issues == []


def test_insights_budget_is_strict_and_references_only_included_records():
    import json
    import pytest
    records = [EvaluationRecord(question=SilverQuestion(id=str(i), topic="x", question="q" * 700,
        expected_answer="a" * 900), result=ChatbotResult(question_id=str(i), answer="a" * 900),
        outcome=Outcome.UNRELATED_ANSWER) for i in range(5)]
    class Fake:
        def generate(self, prompt, schema, model):
            assert len(prompt) <= 11000
            payload = json.loads(prompt.split("COMPACT QUESTION RECORDS:\n")[1])
            assert 0 < payload["included_records"] < 5
            assert payload["omitted_records"] == 5 - payload["included_records"]
            assert "context_available" in payload["records"][0]
            omitted = next(str(i) for i in range(5) if str(i) not in {r["id"] for r in payload["records"]})
            return EvaluationInsights(executive_summary="x", strengths=[], methodology_note="x", issues=[InsightIssue(
                title="x", priority="medium", confidence="medium", evidence_count=1, affected_topics=["x"],
                observed_pattern="x", likely_cause_hypothesis="x", recommendation="x", example_question_ids=[omitted])])
    assert generate_insights(records, Fake(), "fake", max_prompt_chars=11000).issues == []
    with pytest.raises(ValueError, match="cannot fit"):
        generate_insights(records, object(), "fake", max_prompt_chars=10)
