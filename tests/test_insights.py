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
