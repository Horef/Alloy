"""Tests for evaluator: the classify outcome matrix, fixed-claim recounting, and Evaluator runs."""
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


def test_answer_or_clarify_accepts_either_comprehensive_answer_or_clarification():
    """An 'answer_or_clarify' ambiguous variant is an ANSWER task (its reference is the parent's
    comprehensive answer), but a clarifying question is an equally valid response."""
    item = question(True)  # ANSWER task
    item.clarification_acceptable = True
    # A correct comprehensive answer succeeds as usual.
    assert classify(item, scores()) == Outcome.CORRECT_ANSWER
    # A clarifying question is ALSO accepted (not a missing-answer failure).
    assert classify(item, scores(clarification=True)) == Outcome.CORRECT_CLARIFICATION
    # But genuine misinformation is still caught first.
    assert classify(item, scores(clarification=True, false_claims=1)) == Outcome.MISLEADING_HALLUCINATION
    # Without the flag, a clarification on an answer task is not a full success.
    plain = question(True)
    assert classify(plain, scores(clarification=True)) != Outcome.CORRECT_CLARIFICATION


def test_broad_question_needs_only_min_key_points_and_tolerates_long_overviews():
    from chatbot_eval.models import QuestionForm

    item = SilverQuestion(
        id="B1", topic="x", question="מה מגיע לי?", expected_answer="a", reference_claims=["א", "ב", "ג", "ד"],
        question_form=QuestionForm.BROAD, clarification_acceptable=True, min_key_points=2,
    )
    assert classify(item, scores(required=4, addressed=2, correct=2, scope="too_much")) == Outcome.CORRECT_ANSWER
    assert classify(item, scores(required=4, addressed=2, correct=1, scope="too_little")) == Outcome.PARTIAL_TOO_LITTLE
    assert classify(item, scores(required=4, correct=0, addressed=0, clarification=True)) == Outcome.CORRECT_CLARIFICATION
    assert classify(item, scores(required=4, false_claims=1)) == Outcome.MISLEADING_HALLUCINATION


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
    # Genuine misinformation is a contradicted claim only; the judge's own label does not decide it.
    assert classify(item, scores(false_claims=1)) == Outcome.MISLEADING_HALLUCINATION
    assert classify(item, scores(addressed=1, correct=0, incorrect_type="hallucination")) == Outcome.UNRELATED_ANSWER
    assert classify(item, scores(incorrect_type="hallucination", unsupported_claims=2)) == Outcome.CORRECT_ANSWER


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


def test_nonresponsive_answer_is_not_counted_as_misinformation():
    item = question(True)
    # Nothing addressed, nothing asserted ("contact your HR office"): nonresponsive, not misleading.
    assert classify(item, scores(addressed=0, correct=0)) == Outcome.UNRELATED_ANSWER
    # A required point answered wrongly or a likely fabricated assertion is still not misinformation
    # unless a claim is contradicted (answer_false_claims counts only those).
    assert classify(item, scores(addressed=1, correct=0)) == Outcome.UNRELATED_ANSWER
    assert classify(item, scores(addressed=0, correct=0, unsupported_claims=1)) == Outcome.UNRELATED_ANSWER
    assert classify(item, scores(addressed=1, correct=0, false_claims=1)) == Outcome.MISLEADING_HALLUCINATION


def judged_with(**changes):
    judged = scores(**{key: changes.pop(key) for key in list(changes) if key in scores.__code__.co_varnames})
    for key, value in changes.items():
        setattr(judged, key, value)
    return judged


def test_false_claim_without_cited_evidence_is_demoted_to_unsupported():
    from chatbot_eval.models import FalseClaim

    item = question(False)
    judged = judged_with(
        abstention=False, false_claims=2,
        false_claim_details=[
            FalseClaim(claim="ס", contradicting_evidence="ציטוט מהמקור"),
            FalseClaim(claim="ע", contradicting_evidence="  "),
        ],
        incorrect_type="hallucination",
    )
    result = apply_fixed_claims(item, judged)
    assert (result.answer_false_claims, result.answer_unsupported_claims, result.demoted_false_claims) == (1, 1, 1)
    assert classify(item, result) == Outcome.MISLEADING_HALLUCINATION

    uncited = judged_with(abstention=False, false_claims=1, incorrect_type="hallucination")
    result = apply_fixed_claims(item, uncited)
    assert (result.answer_false_claims, result.answer_unsupported_claims) == (0, 1)
    assert result.incorrect_type == "unrelated"
    # An abstain task with only unverified specifics is judged on behavior, not as a hallucination.
    assert classify(item, result) == Outcome.SHOULD_HAVE_ABSTAINED


def test_single_cited_false_claim_on_correct_answer_stays_hallucination():
    from chatbot_eval.models import FalseClaim

    item = question(True)
    item.reference_claims = ["א"]
    judged = judged_with(
        required=1, addressed=1, correct=1, false_claims=1,
        false_claim_details=[FalseClaim(claim="x", contradicting_evidence="y")],
        claim_assessments=[ClaimAssessment(claim_id="C001", addressed=True, correct=True)],
    )
    assert classify(item, apply_fixed_claims(item, judged)) == Outcome.MISLEADING_HALLUCINATION


def test_unsupported_only_answer_task_with_nothing_correct_is_unrelated_not_hallucination():
    item = question(True)
    item.reference_claims = ["א"]
    judged = judged_with(
        required=1, addressed=0, correct=0, unsupported_claims=3, incorrect_type="hallucination",
        claim_assessments=[ClaimAssessment(claim_id="C001", addressed=False, correct=False)],
    )
    result = apply_fixed_claims(item, judged)
    assert classify(item, result) == Outcome.UNRELATED_ANSWER


class _Kb:
    def __init__(self, chunks):
        self.chunks = {chunk.id: chunk for chunk in chunks}
        self.queries = []

    def get(self, source_id):
        return self.chunks.get(source_id)

    def search(self, query, limit):
        self.queries.append(query)
        return list(self.chunks.values())[:limit]


def test_judge_receives_full_source_chunks_passages_and_flags_flow_into_details():
    from chatbot_eval.documents import Chunk
    from chatbot_eval.models import SourceRef

    full = "ראש " + "ט" * 900 + " סוף-המקטע-המלא"
    chunks = [Chunk("d.md#chunk-1", "d.md", "ל1", full), Chunk("e.md#chunk-1", "e.md", "ל2", "פסקה-מהמאגר")]
    item = question(False)
    item.sources = [SourceRef(source_id="d.md#chunk-1", file="d.md", location="ל1", excerpt=full[:500])]
    prompts = []

    class Judge:
        def generate(self, prompt, schema, model):
            prompts.append(prompt)
            judged = scores(required=0, addressed=0, correct=0, abstention=False)
            judged.requires_verification = ["סכום 773 ש״ח", "  "]
            return judged

    kb = _Kb(chunks)
    record = Evaluator(
        object(), Judge(), "judge", knowledge_base=kb, max_evidence_chars=5000, kb_passages=3,
    ).judge_results([(item, ChatbotResult(question_id="Q1", answer="תשובה עם 773"))])[0]

    assert "סוף-המקטע-המלא" in prompts[0]
    assert "KNOWLEDGE BASE PASSAGES (retrieved" in prompts[0] and "פסקה-מהמאגר" in prompts[0]
    assert "TASK NOTE: this question was written to be UNANSWERABLE" in prompts[0]
    assert record.result.metadata["judge_evidence"]["source_chunks_full"] == 1
    assert record.result.metadata["judge_evidence"]["kb_passage_ids"] == ["e.md#chunk-1"]
    assert record.scores.requires_verification == ["סכום 773 ש״ח"]
    assert record.outcome == Outcome.SHOULD_HAVE_ABSTAINED

    # Without a knowledge base the stored excerpt is all the judge gets.
    prompts.clear()
    Evaluator(object(), Judge(), "judge").judge_results([(item, ChatbotResult(question_id="Q1", answer="תשובה"))])
    assert "סוף-המקטע-המלא" not in prompts[0] and "retrieved from the document base" not in prompts[0]


def test_judge_schema_requires_the_flag_and_false_claim_lists_but_records_stay_loadable():
    from chatbot_eval.models import JudgeResponse

    required = set(JudgeResponse.model_json_schema()["required"])
    assert {"requires_verification", "false_claim_details"} <= required
    assert not {"requires_verification", "false_claim_details"} & set(JudgeScores.model_json_schema()["required"])
    seen = []

    class Judge:
        def generate(self, prompt, schema, model):
            seen.append(schema)
            return JudgeResponse.model_validate({**scores(required=0, addressed=0, correct=0, abstention=False).model_dump(),
                                                 "requires_verification": ["x"]})

    item = question(False)
    record = Evaluator(object(), Judge(), "judge").judge_results([(item, ChatbotResult(question_id="Q1", answer="תשובה"))])[0]
    assert seen == [JudgeResponse] and type(record.scores) is JudgeScores and record.scores.requires_verification == ["x"]


def _verification_setup(quotes, verdicts, *, verify=True, answer="תשובה"):
    """An unanswerable-task row whose first judge reports one false claim per quote; the second pass
    answers from ``verdicts`` (claim index -> verdict). Returns (record, prompts of verify calls)."""
    from chatbot_eval.documents import Chunk
    from chatbot_eval.models import ClaimVerdict, ClaimVerificationResponse, FalseClaim, SourceRef

    text = "בסעיף זה נקבע כי מענק השחרור הוא 773 שקלים לחייל מסיים שירות. תנאי נוסף: ההגשה בתוך 30 יום."
    chunk = Chunk("d.md#chunk-1", "d.md", "ל1", text)
    item = question(False)
    item.sources = [SourceRef(source_id="d.md#chunk-1", file="d.md", location="ל1", excerpt=text)]
    verify_prompts = []

    class Judge:
        def generate(self, prompt, schema, model):
            if schema is ClaimVerificationResponse:
                verify_prompts.append(prompt)
                return ClaimVerificationResponse(verdicts=[
                    ClaimVerdict(claim_index=index, verdict=verdict, reason="סיבה") for index, verdict in verdicts.items()
                ])
            judged = scores(required=0, addressed=0, correct=0, abstention=False, false_claims=len(quotes), incorrect_type="hallucination")
            judged.false_claim_details = [FalseClaim(claim=f"טענה {i}", contradicting_evidence=q) for i, q in enumerate(quotes, 1)]
            return judged

    record = Evaluator(
        object(), Judge(), "judge", knowledge_base=_Kb([chunk]), verify_false_claims=verify,
    ).judge_results([(item, ChatbotResult(question_id="Q1", answer=answer))])[0]
    return record, verify_prompts


def test_second_pass_supports_verdict_overturns_the_false_claim_without_counting_it_unsupported():
    record, prompts = _verification_setup(["מענק השחרור הוא 773 שקלים"], {1: "supports"})
    assert len(prompts) == 1 and "773" in prompts[0] and "טענה 1" in prompts[0]
    assert record.outcome == Outcome.SHOULD_HAVE_ABSTAINED
    assert (record.scores.answer_false_claims, record.scores.answer_unsupported_claims) == (0, 0)
    assert record.scores.false_claim_details == [] and record.scores.incorrect_type == "unrelated"
    assert [item.verdict for item in record.scores.false_claim_verifications] == ["supports"]
    assert record.scores.verification_calls == 1


def test_second_pass_unrelated_verdict_demotes_to_unsupported_and_contradicts_keeps_the_claim():
    record, _ = _verification_setup(["מענק השחרור הוא 773 שקלים", "ההגשה בתוך 30 יום"], {1: "unrelated", 2: "contradicts"})
    assert record.outcome == Outcome.MISLEADING_HALLUCINATION
    assert (record.scores.answer_false_claims, record.scores.answer_unsupported_claims) == (1, 1)
    assert [claim.claim for claim in record.scores.false_claim_details] == ["טענה 2"]
    assert [item.verdict for item in record.scores.false_claim_verifications] == ["unrelated", "contradicts"]
    assert record.scores.verification_calls == 1


def test_second_pass_skips_the_call_for_a_quote_not_in_the_evidence_and_demotes_it():
    record, prompts = _verification_setup(["ציטוט שלא מופיע בשום מקום"], {})
    assert prompts == []
    assert record.scores.verification_calls == 0
    assert (record.scores.answer_false_claims, record.scores.answer_unsupported_claims) == (0, 1)
    assert [item.verdict for item in record.scores.false_claim_verifications] == ["quote_not_found"]
    # Only the claim whose quote is present is sent; indexes in the call restart at 1.
    record, prompts = _verification_setup(["ציטוט שלא מופיע", "מענק השחרור... 773 שקלים"], {1: "contradicts"})
    assert len(prompts) == 1 and "ציטוט שלא מופיע" not in prompts[0]
    assert [item.verdict for item in record.scores.false_claim_verifications] == ["quote_not_found", "contradicts"]
    assert record.outcome == Outcome.MISLEADING_HALLUCINATION


def test_second_pass_can_be_switched_off_and_rejects_a_wrong_verdict_list():
    record, prompts = _verification_setup(["ציטוט שלא מופיע"], {}, verify=False)
    assert prompts == [] and record.outcome == Outcome.MISLEADING_HALLUCINATION
    assert record.scores.false_claim_verifications == [] and record.scores.answer_false_claims == 1
    record, _ = _verification_setup(["מענק השחרור הוא 773 שקלים"], {2: "supports"})
    assert record.outcome == Outcome.JUDGE_ERROR and "claim indexes" in record.judge_error


def test_claim_targeted_retrieval_adds_the_passage_that_holds_the_answers_number():
    from chatbot_eval.documents import Chunk
    from chatbot_eval.retrieval import PassageLookup

    chunks = [
        Chunk("a.md#chunk-1", "a.md", "ל1", "מענק השחרור לחייל משוחרר מוגדר בסעיף כללי של הנוהל ומשולם בסוף השירות"),
        Chunk("b.md#chunk-1", "b.md", "ל2", "הנחיה לגבי הגשת בקשות: ההגשה בתוך 45 יום מתאריך השחרור הרשמי בלבד"),
        Chunk("c.md#chunk-1", "c.md", "ל3", "נושא אחר לגמרי הנוגע לאגרת רישוי ורכב"),
    ]
    item = question(False)
    item.question = "מה מענק השחרור לחייל משוחרר"
    prompts = []

    class Judge:
        def generate(self, prompt, schema, model):
            prompts.append(prompt)
            return scores(required=0, addressed=0, correct=0, abstention=False)

    answer = "מענק השחרור משולם בסוף השירות. ההגשה בתוך 30 יום מתאריך השחרור."
    result = ChatbotResult(question_id="Q1", answer=answer)

    def run(claim_passages):
        prompts.clear()
        record = Evaluator(
            object(), Judge(), "judge", knowledge_base=PassageLookup(chunks), kb_passages=1,
            claim_passages=claim_passages,
        ).judge_results([(item, result)])[0]
        return record.result.metadata["judge_evidence"], prompts[0]

    info, prompt = run(0)
    assert info["kb_passage_ids"] == ["a.md#chunk-1"] and info["claim_passage_ids"] == []
    assert "בתוך 45 יום" not in prompt
    info, prompt = run(2)
    assert info["claim_passage_ids"] == ["b.md#chunk-1"]
    assert "בתוך 45 יום" in prompt and "specific details" in prompt
