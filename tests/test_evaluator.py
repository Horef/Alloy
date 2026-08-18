from chatbot_eval.evaluator import classify, looks_like_abstention
from chatbot_eval.models import JudgeScores, Outcome, SilverQuestion


def scores(correctness=4, completeness=4, abstention=False, scope="exact", incorrect_type="not_applicable"):
    return JudgeScores(
        correctness=correctness, completeness=completeness, relevance=4, groundedness=4,
        answer_scope=scope, incorrect_type=incorrect_type,
        retrieval_relevance=4, retrieval_correctness=4, retrieval_completeness=4,
        response_is_abstention=abstention, explanation="הסבר", missing_or_wrong="",
        retrieval_explanation="הסבר אחזור",
    )


def question(answerable=True):
    return SilverQuestion(id="Q1", topic="x", question="q", expected_answer="a", answerable=answerable)


def test_outcome_matrix():
    assert classify(question(True), scores(abstention=True)) == Outcome.INCORRECT_ABSTENTION
    assert classify(question(False), scores(abstention=True)) == Outcome.CORRECT_ABSTENTION
    assert classify(question(False), scores(abstention=False)) == Outcome.SHOULD_HAVE_ABSTAINED
    assert classify(question(True), scores()) == Outcome.CORRECT_ANSWER
    assert classify(question(True), scores(3, 2, scope="too_little")) == Outcome.PARTIAL_TOO_LITTLE
    assert classify(question(True), scores(4, 4, scope="too_much")) == Outcome.PARTIAL_TOO_MUCH
    assert classify(question(True), scores(1, 1, incorrect_type="unrelated")) == Outcome.UNRELATED_ANSWER
    assert classify(question(True), scores(1, 1, incorrect_type="hallucination")) == Outcome.MISLEADING_HALLUCINATION


def test_abstention_detection_in_english_and_hebrew():
    assert looks_like_abstention("I don't have enough information to answer.")
    assert looks_like_abstention("אין לי מספיק מידע")
    assert not looks_like_abstention("The policy allows 30 days.")
