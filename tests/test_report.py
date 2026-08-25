from chatbot_eval.models import ChatbotResult, EvaluationRecord, ExpectedBehavior, JudgeScores, Outcome, QuestionForm, SilverQuestion
from chatbot_eval.report import build_summary, write_report


def test_hebrew_report_and_pipeline_statistics(tmp_path):
    scores = JudgeScores(
        claim_assessments=[],
        required_points_total=2, answer_points_addressed=1, answer_points_correct=0,
        answer_false_claims=1, answer_unsupported_claims=0, answer_extraneous_claims=0,
        retrieval_points_found=2, retrieved_chunks_total=1,
        retrieved_chunks_relevant=1, retrieved_chunks_contradictory=0,
        answer_scope="exact", incorrect_type="hallucination",
        response_is_abstention=False, explanation="התשובה נשמעת סבירה אך שגויה.",
        missing_or_wrong="המספר שגוי.", retrieval_explanation="המידע הנכון נמצא במקטעים.",
    )
    record = EvaluationRecord(
        question=SilverQuestion(id="Q1", topic="זכויות", question="מה הסכום?", expected_answer="100"),
        result=ChatbotResult(question_id="Q1", answer="200", retrieved_context="הסכום הוא 100"),
        outcome=Outcome.MISLEADING_HALLUCINATION, scores=scores,
    )
    summary = build_summary([record])
    assert summary["generation_failures_despite_good_retrieval"] == 1
    assert summary["risky_misinformation_rate"] == 1
    _, report_path = write_report([record], tmp_path)
    report = report_path.read_text(encoding="utf-8")
    assert 'lang="he" dir="rtl"' in report
    assert "הזיה מטעה" in report
    assert "המידע הנכון נמצא במקטעים" in report


def test_report_can_hide_correct_answer_metrics(tmp_path):
    record = EvaluationRecord(
        question=SilverQuestion(id="Q1", topic="זכויות", question="מה הסכום?", expected_answer="100"),
        result=ChatbotResult(question_id="Q1", answer="100"),
        outcome=Outcome.CORRECT_ANSWER,
    )

    _, report_path = write_report(
        [record], tmp_path, show_correct_answer_metrics=False,
    )
    report = report_path.read_text(encoding="utf-8")

    assert "שיעור תשובות נכונות" not in report
    assert "<th>תשובות נכונות</th>" not in report
    assert "שיעור תשובות שימושיות" in report
    assert "<th>תשובות שימושיות</th>" in report


def test_clarification_is_not_in_factual_answer_denominator():
    answer = EvaluationRecord(
        question=SilverQuestion(id="Q1", topic="נושא", question="שאלה", expected_answer="תשובה"),
        result=ChatbotResult(question_id="Q1", answer="תשובה"), outcome=Outcome.CORRECT_ANSWER,
    )
    clarification = EvaluationRecord(
        question=SilverQuestion(
            id="Q2", topic="נושא", question="איזה?", expected_answer="לאיזו אוכלוסייה?",
            question_form=QuestionForm.AMBIGUOUS, expected_behavior=ExpectedBehavior.CLARIFY,
        ),
        result=ChatbotResult(question_id="Q2", answer="לאיזו אוכלוסייה?"),
        outcome=Outcome.CORRECT_CLARIFICATION,
    )

    summary = build_summary([answer, clarification])

    assert summary["correct_answer_rate_on_answerable"] == 1
    assert summary["clarification_success_rate"] == 1
    assert summary["answer_tasks"] == 1
    assert summary["clarification_tasks"] == 1
