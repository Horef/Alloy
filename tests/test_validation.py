import pytest

from chatbot_eval.models import ExpectedBehavior, QuestionForm, SilverQuestion
from chatbot_eval.validation import validate_question_set


def test_legacy_answer_gets_one_stable_reference_claim():
    question = SilverQuestion(id="Q1", topic="", question="מה הנוהל?", expected_answer="יש להגיש טופס")
    validate_question_set([question])
    assert question.reference_claims == ["יש להגיש טופס"]


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
