import pytest

from chatbot_eval.models import ExpectedBehavior, QuestionForm, SilverQuestion
from chatbot_eval.validation import derive_reference_claims, validate_question_set


def test_legacy_answer_gets_one_stable_reference_claim():
    question = SilverQuestion(id="Q1", topic="", question="מה הנוהל?", expected_answer="יש להגיש טופס")
    validate_question_set([question])
    assert question.reference_claims == ["יש להגיש טופס"]


def test_answer_without_claims_is_split_into_atomic_claims():
    answer = "משרתים ברמת פעילות א' זכאים ל-4 לילות.\n- משרתים ברמת פעילות א'+ זכאים ל-5 לילות.\n- ההטבה מתחדשת כל שנה."
    question = SilverQuestion(id="Q1", topic="", question="מה?", expected_answer=answer)
    validate_question_set([question])
    assert question.reference_claims == [
        "משרתים ברמת פעילות א' זכאים ל-4 לילות",
        "משרתים ברמת פעילות א'+ זכאים ל-5 לילות",
        "ההטבה מתחדשת כל שנה",
    ]


def test_claim_derivation_does_not_split_inside_numbers():
    # Periods/commas inside numbers and percentages must not create spurious claims.
    claims = derive_reference_claims("התקרה היא 250,000 ש\"ח או 5.5% מהשכר")
    assert claims == ["התקרה היא 250,000 ש\"ח או 5.5% מהשכר"]


def test_explicit_reference_claims_are_preserved_not_re_split():
    question = SilverQuestion(
        id="Q1", topic="", question="מה?", expected_answer="לא רלוונטי",
        reference_claims=["טענה ראשונה. עם נקודה בפנים", "טענה שנייה"],
    )
    validate_question_set([question])
    assert question.reference_claims == ["טענה ראשונה. עם נקודה בפנים", "טענה שנייה"]


def test_duplicate_ids_and_inconsistent_behavior_are_rejected():
    first = SilverQuestion(id="Q1", topic="", question="שאלה", expected_answer="תשובה")
    duplicate = first.model_copy()
    with pytest.raises(ValueError, match="Duplicate question IDs"):
        validate_question_set([first, duplicate])

    invalid = SilverQuestion(
        id="Q2", topic="", question="שאלה", expected_answer="בירור",
        question_form=QuestionForm.CANONICAL, expected_behavior=ExpectedBehavior.CLARIFY,
    )
    with pytest.raises(ValueError, match="not an answerable ambiguous question"):
        validate_question_set([invalid])
