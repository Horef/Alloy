from chatbot_eval.evaluator import Evaluator, apply_fixed_claims, classify
from chatbot_eval.models import ChatbotResult, ClaimAssessment, ExpectedBehavior, JudgeScores, Outcome, SilverQuestion


def scores(
    required=2, addressed=2, correct=2, abstention=False, clarification=False,
    scope="exact", incorrect_type="not_applicable", false_claims=0, unsupported_claims=0,
):
    return JudgeScores(
        claim_assessments=[],
        required_points_total=required, answer_points_addressed=addressed,
        answer_points_correct=correct, answer_false_claims=false_claims,
        answer_unsupported_claims=unsupported_claims, answer_extraneous_claims=0,
        retrieval_points_found=required, retrieved_chunks_total=2,
        retrieved_chunks_relevant=2, retrieved_chunks_contradictory=0,
        answer_scope=scope, incorrect_type=incorrect_type,
        response_is_abstention=abstention, explanation="הסבר", missing_or_wrong="",
        response_is_clarification=clarification, retrieval_explanation="הסבר אחזור",
    )


def question(answerable=True):
    return SilverQuestion(id="Q1", topic="x", question="q", expected_answer="a", answerable=answerable, expected_behavior=ExpectedBehavior.ANSWER if answerable else ExpectedBehavior.ABSTAIN)


def test_outcome_matrix():
    assert classify(question(True), scores(abstention=True)) == Outcome.INCORRECT_ABSTENTION
    assert classify(question(False), scores(abstention=True)) == Outcome.CORRECT_ABSTENTION
    assert classify(question(False), scores(abstention=False)) == Outcome.SHOULD_HAVE_ABSTAINED
    assert classify(question(True), scores()) == Outcome.CORRECT_ANSWER
    assert classify(question(True), scores(addressed=1, correct=1, scope="too_little")) == Outcome.PARTIAL_TOO_LITTLE
    assert classify(question(True), scores(scope="too_much")) == Outcome.PARTIAL_TOO_MUCH
    assert classify(question(True), scores(addressed=0, correct=0, incorrect_type="unrelated")) == Outcome.UNRELATED_ANSWER
    assert classify(question(True), scores(addressed=1, correct=0, incorrect_type="hallucination", false_claims=1)) == Outcome.MISLEADING_HALLUCINATION


def test_clarification_is_a_distinct_expected_behavior():
    item = question(True)
    item.expected_behavior = ExpectedBehavior.CLARIFY
    clarification = scores()
    clarification.response_is_clarification = True

    assert classify(item, clarification) == Outcome.CORRECT_CLARIFICATION
    assert classify(item, scores()) == Outcome.MISSING_CLARIFICATION


def test_response_behavior_truth_table_requires_matching_safe_whole_response():
    import pytest

    answer = question(True)
    clarify = question(True)
    clarify.expected_behavior = ExpectedBehavior.CLARIFY
    abstain = question(False)

    cases = [
        (answer, scores(), Outcome.CORRECT_ANSWER),
        (answer, scores(abstention=True), Outcome.INCORRECT_ABSTENTION),
        # A whole-response clarification cannot earn full answer credit, even when the
        # judge also reports that every supplied reference claim was correct.
        (answer, scores(clarification=True), Outcome.PARTIAL_TOO_LITTLE),
        (answer, scores(addressed=0, correct=0, clarification=True), Outcome.INCORRECT_ABSTENTION),
        (clarify, scores(clarification=True), Outcome.CORRECT_CLARIFICATION),
        (clarify, scores(abstention=True), Outcome.MISSING_CLARIFICATION),
        (abstain, scores(abstention=True), Outcome.CORRECT_ABSTENTION),
        (abstain, scores(clarification=True), Outcome.SHOULD_HAVE_ABSTAINED),
    ]
    for item, judged, expected in cases:
        assert classify(item, judged) == expected

    # A contradicted (false) claim is genuine misinformation on any task and stays a hallucination.
    for item, flag in ((clarify, "clarification"), (abstain, "abstention")):
        kwargs = {flag: True, "false_claims": 1}
        assert classify(item, scores(**kwargs)) == Outcome.MISLEADING_HALLUCINATION

    # An unsupported (not contradicted) claim is NOT misinformation: it must not override the
    # behavior verdict. A clarify task that instead answered is a missing clarification; an abstain
    # task that correctly declined is a correct abstention even if it added harmless extra detail.
    assert classify(clarify, scores(clarification=False, unsupported_claims=1)) == Outcome.MISSING_CLARIFICATION
    assert classify(abstain, scores(abstention=True, unsupported_claims=1)) == Outcome.CORRECT_ABSTENTION

    with pytest.raises(ValueError, match="flags conflict"):
        classify(answer, scores(abstention=True, clarification=True))


def test_fixed_claims_determine_answer_counts():
    item = question(True)
    item.reference_claims = ["הוגש טופס", "התקבל אישור"]
    judged = scores(required=99, addressed=99, correct=99)
    judged.retrieval_points_found = 2
    judged.claim_assessments = [
        ClaimAssessment(claim_id="C002", addressed=True, correct=False),
        ClaimAssessment(claim_id="C001", addressed=True, correct=True),
    ]

    result = apply_fixed_claims(item, judged)

    assert result.required_points_total == 2
    assert result.answer_points_addressed == 2
    assert result.answer_points_correct == 1
    assert [claim.claim_id for claim in result.claim_assessments] == ["C001", "C002"]


def test_evaluator_persists_callback_after_fixed_claim_judging():
    item = question(True)
    item.reference_claims = ["תשובת הייחוס"]

    class Adapter:
        def ask(self, current):
            return ChatbotResult(question_id=current.id, answer="תשובת הייחוס")

    class Judge:
        def generate(self, prompt, schema, model):
            judged = scores(required=1, addressed=1, correct=1)
            judged.claim_assessments = [
                ClaimAssessment(claim_id="C001", addressed=True, correct=True),
            ]
            return judged

    persisted = []
    records = Evaluator(Adapter(), Judge(), "judge").evaluate([item], on_record=persisted.append)

    assert records[0].outcome == Outcome.CORRECT_ANSWER
    assert persisted == records


def test_evaluator_bounds_large_answer_and_context():
    item = question(True)
    item.reference_claims = ["תשובת הייחוס"]

    class Adapter:
        def ask(self, current):
            return ChatbotResult(
                question_id=current.id, answer="A" * 500, retrieved_context="C" * 500,
            )

    class Judge:
        def generate(self, prompt, schema, model):
            assert "content omitted by Alloy input limit" in prompt
            judged = scores(required=1, addressed=1, correct=1)
            judged.claim_assessments = [
                ClaimAssessment(claim_id="C001", addressed=True, correct=True),
            ]
            return judged

    record = Evaluator(
        Adapter(), Judge(), "judge", max_answer_chars=120, max_context_chars=130,
    ).evaluate([item])[0]

    assert record.result.metadata["judge_input_truncation"]["answer_chars_omitted"] > 0
    assert record.result.metadata["judge_input_truncation"]["context_chars_omitted"] > 0


def test_concurrent_evaluation_preserves_input_order():
    items = [SilverQuestion(id=f"Q{i}", topic="x", question="q", expected_answer="a") for i in range(4)]

    class Adapter:
        def ask(self, current):
            return ChatbotResult(question_id=current.id, answer="error", error="stored error")

    records = Evaluator(Adapter(), object(), "judge", max_concurrency=2).evaluate(items)

    assert [record.question.id for record in records] == [item.id for item in items]


def test_nonanswer_false_claims_are_risky_and_conflicting_flags_fail():
    import pytest
    for behavior in (ExpectedBehavior.CLARIFY, ExpectedBehavior.ABSTAIN):
        item = question(behavior != ExpectedBehavior.ABSTAIN)
        item.expected_behavior = behavior
        judged = scores(false_claims=1)
        judged.response_is_clarification = behavior == ExpectedBehavior.CLARIFY
        judged.response_is_abstention = behavior == ExpectedBehavior.ABSTAIN
        assert classify(item, judged) == Outcome.MISLEADING_HALLUCINATION
    judged.response_is_clarification = judged.response_is_abstention = True
    with pytest.raises(ValueError, match="flags conflict"):
        classify(item, judged)


def test_unsupported_elaboration_does_not_become_hallucination():
    """A correct answer with one extra true-but-unsupported detail is not misinformation.

    This guards against the regression where any unsupported claim forced
    MISLEADING_HALLUCINATION, which mislabeled ~70% of reported hallucinations in real runs.
    """
    item = question(True)
    # All required points correct, one harmless unsupported elaboration, judge says not a hallucination.
    assert classify(item, scores(unsupported_claims=1, incorrect_type="not_applicable")) == Outcome.CORRECT_ANSWER
    # Same, but the extra detail also broadened the scope: still not a hallucination, just partial.
    assert classify(item, scores(unsupported_claims=1, scope="too_much")) == Outcome.PARTIAL_TOO_MUCH
    # Genuine misinformation is still caught: a contradicted claim, or the judge's own verdict.
    assert classify(item, scores(false_claims=1)) == Outcome.MISLEADING_HALLUCINATION
    assert classify(item, scores(addressed=1, correct=0, incorrect_type="hallucination")) == Outcome.MISLEADING_HALLUCINATION


def test_judge_results_boundary_validation_and_no_adapter_mutation():
    import pytest
    from chatbot_eval.response_errors import SUMMARY_GENERATION_FAILURE
    adapter = object()
    evaluator = Evaluator(adapter, object(), "judge")
    item = question()
    for answer in ("   ", SUMMARY_GENERATION_FAILURE):
        result = ChatbotResult(question_id=item.id, answer=answer)
        assert evaluator.judge_results([(item, result)])[0].outcome == Outcome.CHATBOT_ERROR
        assert result.error == ""
        assert evaluator.chatbot is adapter
    with pytest.raises(ValueError, match="does not match"):
        evaluator.judge_results([(item, ChatbotResult(question_id="wrong", answer="a"))])
    with pytest.raises(ValueError, match="Duplicate"):
        evaluator.judge_results([(item, ChatbotResult(question_id=item.id, answer="a"))] * 2)


def test_limited_gap_language_does_not_override_judge():
    class Judge:
        def generate(self, prompt, schema, model):
            judged = scores(required=1, addressed=1, correct=1)
            judged.claim_assessments = [ClaimAssessment(claim_id="C001", addressed=True, correct=True)]
            return judged
    evaluator = Evaluator(object(), Judge(), "judge")
    for answer in ('30 days; attachment not provided', '30 ימים; אין מספיק מידע על הנספח', 'Policy quotes "cannot answer"; 30 days.'):
        assert evaluator.judge_results([(question(), ChatbotResult(question_id="Q1", answer=answer))])[0].outcome == Outcome.CORRECT_ANSWER
