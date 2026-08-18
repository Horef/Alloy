from chatbot_eval.evaluator import classify, looks_like_abstention
from chatbot_eval.models import JudgeScores, Outcome, SilverQuestion


def scores(required=2, addressed=2, correct=2, abstention=False, scope="exact", incorrect_type="not_applicable", false_claims=0):
    return JudgeScores(
        required_points_total=required, answer_points_addressed=addressed,
        answer_points_correct=correct, answer_false_claims=false_claims,
        answer_unsupported_claims=0, answer_extraneous_claims=0,
        retrieval_points_found=required, retrieved_chunks_total=2,
        retrieved_chunks_relevant=2, retrieved_chunks_contradictory=0,
        answer_scope=scope, incorrect_type=incorrect_type,
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
    assert classify(question(True), scores(addressed=1, correct=1, scope="too_little")) == Outcome.PARTIAL_TOO_LITTLE
    assert classify(question(True), scores(scope="too_much")) == Outcome.PARTIAL_TOO_MUCH
    assert classify(question(True), scores(addressed=0, correct=0, incorrect_type="unrelated")) == Outcome.UNRELATED_ANSWER
    assert classify(question(True), scores(addressed=1, correct=0, incorrect_type="hallucination", false_claims=1)) == Outcome.MISLEADING_HALLUCINATION


def test_abstention_detection_in_english_and_hebrew():
    assert looks_like_abstention("I don't have enough information to answer.")
    assert looks_like_abstention("אין לי מספיק מידע")
    assert not looks_like_abstention("The policy allows 30 days.")
