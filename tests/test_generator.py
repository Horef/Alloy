from chatbot_eval.documents import Chunk
from chatbot_eval.generator import (
    GenerationOptions, SilverSetGenerator, _assign_stable_ids, _is_duplicate, _render_chunks,
    allocate_quotas, allocate_type_targets, validate_candidate,
)
from chatbot_eval.graph import GraphBundle, GraphNode, GraphTopic, KnowledgeGraph, build_edges
from chatbot_eval.graph_build import GraphBuilder
from chatbot_eval.models import EvidenceQuote, ExpectedBehavior, GeneratedQuestion, GeneratedVariation, NodeSignals, QuestionBatch, QuestionForm, QuestionType, SilverQuestion, SourceRef, TopicCandidate, TopicMap, VariationBatch


def _bundle(chunks, topics):
    """Build a GraphBundle directly from chunks and (name, importance, node_ids) topic tuples.

    Signal extraction/labeling are exercised separately; generation tests only need a graph whose
    topics point at the right seed nodes, so this constructs one deterministically without an LLM.
    """
    from chatbot_eval.graph import GraphNode

    nodes = [GraphNode(c.id, c.file, c.location, c.text) for c in chunks]
    graph = KnowledgeGraph(nodes=nodes, edges=build_edges(nodes))
    graph_topics = [
        GraphTopic(name=name, description="", importance=importance, node_ids=list(node_ids))
        for name, importance, node_ids in topics
    ]
    return GraphBundle(graph, graph_topics)


def test_quota_allocation_respects_total_and_cap():
    topics = [
        TopicCandidate(name="A", description="", importance=5, source_ids=[]),
        TopicCandidate(name="B", description="", importance=3, source_ids=[]),
        TopicCandidate(name="C", description="", importance=1, source_ids=[]),
    ]
    quotas = allocate_quotas(topics, total=10, minimum=1, max_share=0.4)
    assert sum(quotas.values()) == 10
    assert max(quotas.values()) <= 4
    assert quotas["A"] >= quotas["B"] >= quotas["C"]


def test_quota_does_not_overallocate_when_fewer_questions_than_topics():
    topics = [TopicCandidate(name=str(i), description="", importance=1, source_ids=[]) for i in range(5)]
    assert sum(allocate_quotas(topics, total=2, minimum=1, max_share=0.5).values()) == 2


def test_quota_never_overallocates_when_minimum_exceeds_budget():
    topics = [TopicCandidate(name=str(i), description="", importance=1, source_ids=[]) for i in range(5)]
    quotas = allocate_quotas(topics, total=2, minimum=3, max_share=0.5)
    assert sum(quotas.values()) == 2


def test_quota_preserves_share_cap_when_minimum_is_infeasible():
    topics = [
        TopicCandidate(name="A", description="", importance=5, source_ids=[]),
        TopicCandidate(name="B", description="", importance=4, source_ids=[]),
    ]

    quotas = allocate_quotas(topics, total=10, minimum=6, max_share=0.4)

    assert quotas == {"A": 4, "B": 4}
    assert sum(quotas.values()) == 8


def test_render_chunks_truncates_oversized_first_chunk_with_provenance():
    rendered = _render_chunks(
        [Chunk("large#1", "large.md", "document", "x" * 500)],
        max_chars=100,
    )

    assert len(rendered) <= 100
    assert "[SOURCE_ID: large#1]" in rendered
    assert "[PROMPT_EXCERPT_TRUNCATED]" in rendered
    assert "x" in rendered


def test_generation_diagnostics_expose_capacity_blocked_by_topic_cap():
    class EmptyLLM:
        def generate(self, prompt, schema, model):
            assert schema is QuestionBatch
            return QuestionBatch(questions=[])

    chunks = [
        Chunk("a#1", "a.md", "document", "evidence A"),
        Chunk("b#1", "b.md", "document", "evidence B"),
    ]
    bundle = _bundle(chunks, [("A", 5, ["a#1"]), ("B", 4, ["b#1"])])
    generator = SilverSetGenerator(EmptyLLM(), "test")

    generator.generate(
        chunks,
        GenerationOptions(
            max_questions=10,
            batch_chunks=2,
            min_topic_questions=6,
            max_topic_share=0.4,
            unanswerable_ratio=0,
            max_candidate_rounds=1,
        ),
        bundle=bundle,
    )

    assert generator.last_generation_diagnostics["topic_quotas"] == {"A": 4, "B": 4}
    assert generator.last_generation_diagnostics["unallocated_reason"] == "topic_cap_capacity"
    assert generator.last_generation_diagnostics["rejected"]["topic_cap_capacity"] == 2


def test_type_targets_redistribute_types_the_corpus_cannot_support():
    targets = (
        (QuestionType.BASIC_KNOWLEDGE.value, 0.5),
        (QuestionType.DOCUMENT_WIDE.value, 0.25),
        (QuestionType.CROSS_DOCUMENT.value, 0.25),
    )
    counts = allocate_type_targets(8, targets, [Chunk("a#1", "a.md", "document", "text")])

    assert counts == {QuestionType.BASIC_KNOWLEDGE.value: 8}

    unsupported_only = ((QuestionType.CROSS_DOCUMENT.value, 1.0),)
    assert allocate_type_targets(3, unsupported_only, [Chunk("a#1", "a.md", "document", "text")]) == {
        QuestionType.BASIC_KNOWLEDGE.value: 3,
    }


def test_stable_ids_are_deterministic_and_update_variant_parent():
    def questions():
        return [
            SilverQuestion(id="Q0001", topic="נושא", question="מה המדיניות?", expected_answer="מחר"),
            SilverQuestion(
                id="Q0002", topic="נושא", question="אז מתי זה קורה?", expected_answer="מחר",
                question_form=QuestionForm.NATURAL_USER, parent_question_id="Q0001",
            ),
        ]

    first, second = questions(), questions()
    _assign_stable_ids(first)
    _assign_stable_ids(second)

    assert [item.id for item in first] == [item.id for item in second]
    assert first[0].id.startswith("Q-")
    assert first[1].id.startswith("V-")
    assert first[1].parent_question_id == first[0].id


def _candidate(question_type=QuestionType.BASIC_KNOWLEDGE, source_ids=None, quotes=None):
    return GeneratedQuestion(
        question="מה המדיניות?", expected_answer="המדיניות חלה מחר", answerable=True,
        difficulty="medium", rationale="בדיקת מדיניות", question_type=question_type,
        source_ids=source_ids or ["a#1"],
        reference_claims=["המדיניות חלה מחר"],
        supporting_quotes=quotes or [EvidenceQuote(source_id="a#1", quote="חלה מחר")],
    )


def test_candidate_validation_requires_exact_and_complete_provenance():
    chunks = {"a#1": Chunk("a#1", "a.md", "document", "המדיניות חלה מחר על כולם")}
    valid, reason = validate_candidate(_candidate(), chunks)
    assert reason is None
    assert valid == [chunks["a#1"]]

    _, reason = validate_candidate(_candidate(source_ids=["missing"]), chunks)
    assert reason == "unknown_source_id"


def test_cross_document_question_requires_two_documents():
    chunks = {
        "a#1": Chunk("a#1", "a.md", "document", "המדיניות חלה מחר"),
        "a#2": Chunk("a#2", "a.md", "document", "יש להגיש בקשה"),
    }
    candidate = _candidate(
        question_type=QuestionType.CROSS_DOCUMENT,
        source_ids=["a#1", "a#2"],
        quotes=[
            EvidenceQuote(source_id="a#1", quote="חלה מחר"),
            EvidenceQuote(source_id="a#2", quote="להגיש בקשה"),
        ],
    )
    _, reason = validate_candidate(candidate, chunks)
    assert reason == "invalid_cross_document_evidence"


def test_duplicate_detection_includes_prior_review_sets():
    accepted = [SilverQuestion(id="Q1", topic="", question="כמה ימי חופש מגיעים לחייל?", expected_answer="")]
    assert _is_duplicate("כמה ימי חופש מגיעים לחייל", accepted)
    assert _is_duplicate("מהם תנאי הפטור מתורנות?", [], ("מהם תנאי הפטור מתורנות",))


def test_ambiguous_variation_expects_a_follow_up_and_preserves_provenance():
    parent = SilverQuestion(
        id="Q0001", topic="תנאי שירות", question="מהם התנאים לקבלת התש 3?",
        expected_answer="תשובת ייחוס", sources=[],
    )
    variation = GeneratedVariation(
        source_question_id="Q0001", question="איך אני מקבל התש 3?",
        question_form="ambiguous", required_clarification="האם מדובר בחייל חובה או קבע?",
        rationale="חסר סוג השירות",
    )

    result = SilverSetGenerator._to_variation(variation, parent, 2)

    assert result.question_form == QuestionForm.AMBIGUOUS
    assert result.expected_behavior == ExpectedBehavior.CLARIFY
    assert result.expected_answer == "האם מדובר בחייל חובה או קבע?"
    assert result.parent_question_id == "Q0001"
    assert result.clarification_acceptable is False


def test_answer_or_clarify_variant_is_answer_task_with_acceptable_clarification():
    parent = SilverQuestion(
        id="Q0001", topic="חופשות", question="מהם סוגי החופשה המיוחדת ותנאיהם?",
        expected_answer="חופשה אישית, כלכלית, ולימודים — כל אחת בתנאים שלה.",
        reference_claims=["חופשה אישית", "חופשה כלכלית", "חופשה ללימודים"],
    )
    variation = GeneratedVariation(
        source_question_id="Q0001", question="מה התנאים לחופשה מיוחדת?",
        question_form="ambiguous", required_clarification="לאיזה סוג חופשה מיוחדת כוונתך?",
        ambiguity_kind="answer_or_clarify", rationale="ניתן לענות מקיף או להבהיר",
    )

    result = SilverSetGenerator._to_variation(variation, parent, 2)

    # It is an ANSWER task carrying the parent's comprehensive reference, but a clarification is OK.
    assert result.expected_behavior == ExpectedBehavior.ANSWER
    assert result.reference_claims == parent.reference_claims
    assert result.expected_answer == parent.expected_answer
    assert result.clarification_acceptable is True
    assert result.acceptable_clarification == "לאיזה סוג חופשה מיוחדת כוונתך?"


def test_requested_topic_seeds_from_relevant_nodes():
    from chatbot_eval.generator import _nodes_matching_topic

    nodes = [
        GraphNode("pay#1", "salary.md", "d", "טבלת שכר", entities=["תוספת שכר"], keyphrases=["רמות שכר"],
                  summary="הסבר על שכר ותוספות"),
        GraphNode("leave#1", "leave.md", "d", "ימי חופשה", entities=["חופשה שנתית"], keyphrases=["מכסת חופשה"],
                  summary="הסבר על חופשות"),
        GraphNode("pay#2", "family.md", "d", "תשלומי משפחה", entities=["תשמ\"ש"], keyphrases=["תשלום למשפחה"],
                  summary="תשלומי משפחה ושכר דירה"),
    ]
    graph = KnowledgeGraph(nodes, build_edges(nodes))
    seeds = _nodes_matching_topic(graph, "שכר", limit=12)
    # Salary-relevant nodes rank ahead of the leave node; the leave node should not lead.
    assert "pay#1" in seeds
    assert seeds[0] != "leave#1"


def test_merge_question_sets_appends_and_dedups():
    from chatbot_eval.generator import merge_question_sets

    existing = [
        SilverQuestion(id="Q-a", topic="t", question="כמה ימי חופשה מגיעים לחייל?", expected_answer="20",
                       reference_claims=["20"]),
        SilverQuestion(id="Q-b", topic="t", question="מי מאשר יציאה לחופשה?", expected_answer="המפקד",
                       reference_claims=["המפקד"]),
    ]
    new = [
        # A genuinely new question about salary.
        SilverQuestion(id="Q-c", topic="שכר", question="כיצד מחושב התמריץ הכספי?", expected_answer="לפי קבוצה",
                       reference_claims=["לפי קבוצה"]),
        # A near-duplicate of an existing question -> dropped.
        SilverQuestion(id="Q-d", topic="t", question="כמה ימי חופשה מגיעים לחייל", expected_answer="20",
                       reference_claims=["20"]),
    ]
    merged, diag = merge_question_sets(existing, new, stable_question_ids=True)

    assert diag["existing_kept"] == 2
    assert diag["new_generated"] == 2
    assert diag["new_added"] == 1
    assert diag["new_dropped_duplicate"] == 1
    assert diag["merged_total"] == 3
    # The new salary question survived; the near-duplicate did not add a row.
    assert any("התמריץ" in q.question for q in merged)
    # Stable IDs were recomputed across the combined set and are unique.
    assert len({q.id for q in merged}) == 3


def test_regenerate_variations_replaces_variants_keeps_canonical_and_boundary():
    canonical = SilverQuestion(id="Q1", topic="t", question="שאלה קנונית", expected_answer="תשובה",
                               reference_claims=["תשובה"])
    boundary = SilverQuestion(id="U1", topic="t", question="שאלה גבולית", expected_answer="חסר מידע",
                              answerable=False, expected_behavior=ExpectedBehavior.ABSTAIN,
                              question_type=QuestionType.UNANSWERABLE)
    old_variant = SilverQuestion(id="V1", topic="t", question="ניסוח ישן", expected_answer="תשובה",
                                 question_form=QuestionForm.NATURAL_USER, parent_question_id="Q1")

    class VarLLM:
        def generate(self, prompt, schema, model):
            return VariationBatch(variations=[
                GeneratedVariation(source_question_id="Q1", question="ניסוח חדש וטבעי",
                                   question_form="natural_user", rationale="r"),
                GeneratedVariation(source_question_id="Q1", question="שאלה מעורפלת",
                                   question_form="ambiguous", required_clarification="לאיזה מקרה?",
                                   ambiguity_kind="answer_or_clarify", rationale="r"),
            ])

    merged, diag = SilverSetGenerator(VarLLM(), "m").regenerate_variations(
        [canonical, boundary, old_variant], variation_budget=2, ambiguous_variation_share=0.5,
        max_candidate_rounds=1, stable_question_ids=False,
    )
    forms = [q.question_form for q in merged]
    # Canonical and boundary kept; old variant dropped; two fresh variants added.
    assert any(q.question == "שאלה קנונית" for q in merged)
    assert any(q.question == "שאלה גבולית" for q in merged)
    assert not any(q.question == "ניסוח ישן" for q in merged)
    assert diag["canonical_kept"] == 2  # canonical + boundary are both non-variant "kept"
    assert diag["previous_variants_dropped"] == 1
    assert diag["new_variants"] == 2
    assert QuestionForm.NATURAL_USER in forms and QuestionForm.AMBIGUOUS in forms


def test_generation_budget_includes_natural_and_ambiguous_variants():
    class FakeLLM:
        def generate(self, prompt, schema, model):
            if schema is QuestionBatch:
                return QuestionBatch(questions=[_candidate()])
            if schema is VariationBatch:
                return VariationBatch(variations=[
                    GeneratedVariation(
                        source_question_id="Q0001", question="איך מקבלים את זה?",
                        question_form="ambiguous", required_clarification="לאיזו אוכלוסייה הכוונה?",
                        rationale="האוכלוסייה חסרה",
                    ),
                    GeneratedVariation(
                        source_question_id="Q0001", question="איך מקבלים את המדיניות?",
                        question_form="natural_user", rationale="ניסוח טבעי",
                    ),
                ])
            raise AssertionError(schema)

    chunks = [Chunk("a#1", "a.md", "document", "המדיניות חלה מחר על כולם")]
    questions, _ = SilverSetGenerator(FakeLLM(), "test").generate(
        chunks,
        GenerationOptions(
            max_questions=3, batch_chunks=1, min_topic_questions=1, max_topic_share=1,
            unanswerable_ratio=0, user_variation_ratio=2 / 3, ambiguous_variation_share=0.5,
        ),
        bundle=_bundle(chunks, [("תנאי שירות", 5, ["a#1"])]),
    )

    assert [question.question_form for question in questions] == [
        QuestionForm.CANONICAL, QuestionForm.AMBIGUOUS, QuestionForm.NATURAL_USER,
    ]


def test_generation_refills_rejected_candidates_within_bound():
    class FakeLLM:
        question_calls = 0

        def generate(self, prompt, schema, model):
            if schema is QuestionBatch:
                self.question_calls += 1
                if self.question_calls == 1:
                    return QuestionBatch(questions=[_candidate(quotes=[EvidenceQuote(source_id="a#1", quote="לא קיים")])])
                return QuestionBatch(questions=[_candidate()])
            raise AssertionError(schema)

    fake = FakeLLM()
    chunks = [Chunk("a#1", "a.md", "document", "המדיניות חלה מחר על כולם")]
    questions, _ = SilverSetGenerator(fake, "test").generate(
        chunks,
        GenerationOptions(
            max_questions=1, batch_chunks=1, min_topic_questions=1, max_topic_share=1,
            unanswerable_ratio=0, max_candidate_rounds=2,
        ),
        bundle=_bundle(chunks, [("נושא", 5, ["a#1"])]),
    )

    assert len(questions) == 1
    assert fake.question_calls == 2


def test_generation_retry_prompt_explains_prior_rejection():
    prompts = []

    class FakeLLM:
        def generate(self, prompt, schema, model):
            prompts.append(prompt)
            if len(prompts) == 1:
                return QuestionBatch(questions=[_candidate(
                    quotes=[EvidenceQuote(source_id="a#1", quote="לא קיים")],
                )])
            assert "quote_not_verbatim" in prompt
            return QuestionBatch(questions=[_candidate()])

    chunks = [Chunk("a#1", "a.md", "document", "המדיניות חלה מחר על כולם")]
    SilverSetGenerator(FakeLLM(), "test").generate(
        chunks,
        GenerationOptions(
            max_questions=1, batch_chunks=1, min_topic_questions=1, max_topic_share=1,
            unanswerable_ratio=0, max_candidate_rounds=2,
        ),
        bundle=_bundle(chunks, [("נושא", 5, ["a#1"])]),
    )

    assert len(prompts) == 2


def test_requested_topic_also_scopes_boundary_questions():
    class FakeLLM:
        def generate(self, prompt, schema, model):
            if schema is QuestionBatch and "boundary questions" in prompt:
                assert "TOPIC: נושא א" in prompt
                return QuestionBatch(questions=[GeneratedQuestion(
                    question="מה לגבי מקרה שאין במסמך?", expected_answer="המידע חסר", answerable=False,
                    difficulty="medium", rationale="גבול", source_ids=["a#1"],
                    reference_claims=[],
                    question_type=QuestionType.UNANSWERABLE,
                )])
            if schema is QuestionBatch:
                return QuestionBatch(questions=[_candidate()])
            raise AssertionError(schema)

    chunks = [
        Chunk("a#1", "a.md", "document", "המדיניות חלה מחר על כולם"),
        Chunk("b#1", "b.md", "document", "תוכן אחר שקיים במסמך השני ונועד רק לבדיקה"),
    ]
    bundle = _bundle(chunks, [("נושא א", 5, ["a#1"]), ("נושא ב", 4, ["b#1"])])
    questions, _ = SilverSetGenerator(FakeLLM(), "test").generate(
        chunks,
        GenerationOptions(
            max_questions=2, batch_chunks=2, min_topic_questions=1, max_topic_share=1,
            unanswerable_ratio=0.5, requested_topic="נושא א", max_candidate_rounds=1,
        ),
        bundle=bundle,
    )

    assert {question.topic for question in questions} == {"נושא א"}


def _answerable_question():
    return SilverQuestion(
        id="Q0001", topic="נושא", question="מהו הסכום?", expected_answer="הסכום הוא 100 שקלים",
        question_type=QuestionType.BASIC_KNOWLEDGE,
        reference_claims=["ישן"],
        supporting_quotes=[EvidenceQuote(source_id="stale#1", quote="ישן")],
        sources=[SourceRef(source_id="stale#1", file="stale.md", location="document", excerpt="ישן")],
        review_status="needs_reground",
    )


def test_reground_one_regenerates_grounding_and_resets_status():
    chunks = [Chunk("a.md#chunk-1", "a.md", "document", "הסכום הוא 100 שקלים לכל חייל.")]

    class RegroundLLM:
        def generate(self, prompt, schema, model, *, required_fields=None):
            assert schema is QuestionBatch
            assert "מהו הסכום?" in prompt  # fixed question is injected
            return QuestionBatch(questions=[GeneratedQuestion(
                question="מהו הסכום?", expected_answer="הסכום הוא 100 שקלים",
                answerable=True, difficulty="medium", rationale="",
                source_ids=["a.md#chunk-1"],
                question_type=QuestionType.BASIC_KNOWLEDGE,
                reference_claims=["הסכום הוא 100 שקלים"],
                supporting_quotes=[EvidenceQuote(source_id="a.md#chunk-1", quote="הסכום הוא 100 שקלים")],
            )])

    generator = SilverSetGenerator(RegroundLLM(), "model")
    updated, reason = generator.reground_one(_answerable_question(), chunks)
    assert reason is None
    assert updated.sources[0].source_id == "a.md#chunk-1"
    assert updated.reference_claims == ["הסכום הוא 100 שקלים"]
    assert updated.supporting_quotes[0].source_id == "a.md#chunk-1"
    # Regrounded rows require a fresh human approval.
    assert updated.review_status == "pending"
    # Fixed identity is preserved.
    assert updated.id == "Q0001"
    assert updated.question == "מהו הסכום?"


def test_reground_one_rejects_non_verbatim_quote():
    chunks = [Chunk("a.md#chunk-1", "a.md", "document", "הסכום הוא 100 שקלים.")]

    class BadQuoteLLM:
        def generate(self, prompt, schema, model, *, required_fields=None):
            return QuestionBatch(questions=[GeneratedQuestion(
                question="מהו הסכום?", expected_answer="הסכום הוא 100 שקלים",
                answerable=True, difficulty="medium", rationale="",
                source_ids=["a.md#chunk-1"],
                question_type=QuestionType.BASIC_KNOWLEDGE,
                reference_claims=["הסכום הוא 100 שקלים"],
                supporting_quotes=[EvidenceQuote(source_id="a.md#chunk-1", quote="ציטוט שלא קיים במקור")],
            )])

    generator = SilverSetGenerator(BadQuoteLLM(), "model")
    updated, reason = generator.reground_one(_answerable_question(), chunks)
    assert updated is None
    assert reason == "quote_not_verbatim"


def test_reground_one_rejects_altered_question_text():
    chunks = [Chunk("a.md#chunk-1", "a.md", "document", "הסכום הוא 100 שקלים.")]

    class AlteringLLM:
        def generate(self, prompt, schema, model, *, required_fields=None):
            return QuestionBatch(questions=[GeneratedQuestion(
                question="שאלה אחרת לגמרי", expected_answer="הסכום הוא 100 שקלים",
                answerable=True, difficulty="medium", rationale="",
                source_ids=["a.md#chunk-1"],
                question_type=QuestionType.BASIC_KNOWLEDGE,
                reference_claims=["הסכום הוא 100 שקלים"],
                supporting_quotes=[EvidenceQuote(source_id="a.md#chunk-1", quote="הסכום הוא 100 שקלים")],
            )])

    generator = SilverSetGenerator(AlteringLLM(), "model")
    updated, reason = generator.reground_one(_answerable_question(), chunks)
    assert updated is None
    assert reason == "model_altered_question"


def test_reground_one_skips_clarification_task():
    chunks = [Chunk("a.md#chunk-1", "a.md", "document", "טקסט")]

    class NeverCalledLLM:
        def generate(self, prompt, schema, model, *, required_fields=None):
            raise AssertionError("clarification tasks must not reach the model")

    clarify = SilverQuestion(
        id="Q0002", topic="נושא", question="איזה?", expected_answer="לאיזו אוכלוסייה?",
        question_form=QuestionForm.AMBIGUOUS, expected_behavior=ExpectedBehavior.CLARIFY,
        review_status="needs_reground",
    )
    generator = SilverSetGenerator(NeverCalledLLM(), "model")
    updated, reason = generator.reground_one(clarify, chunks)
    assert updated is None
    assert reason == "not_an_answerable_answer_task"


def test_completeness_verification_accepts_complete_answer():
    from chatbot_eval.models import AnswerCompletenessReview

    class ReviewLLM:
        def generate(self, prompt, schema, model):
            assert schema is AnswerCompletenessReview
            return AnswerCompletenessReview(verdict="complete")

    chunks = [Chunk("a#1", "a.md", "document", "המדיניות חלה מחר על כולם")]
    candidate = _candidate()
    updated, valid, verdict = SilverSetGenerator(ReviewLLM(), "model").verify_candidate_completeness(
        candidate, chunks, chunks,
    )
    assert verdict == "complete"
    assert updated is candidate
    assert valid == chunks


def test_completeness_verification_replaces_incomplete_answer_with_validated_correction():
    from chatbot_eval.models import AnswerCompletenessReview, EvidenceQuote as EQ

    class ReviewLLM:
        def generate(self, prompt, schema, model):
            return AnswerCompletenessReview(
                verdict="incomplete",
                reasoning="הראיה כוללת גם את מועד הסיום",
                corrected_answer="המדיניות חלה מחר ומסתיימת בעוד שבוע",
                corrected_reference_claims=["המדיניות חלה מחר", "המדיניות מסתיימת בעוד שבוע"],
                corrected_source_ids=["a#1"],
                corrected_supporting_quotes=[EQ(source_id="a#1", quote="חלה מחר ומסתיימת בעוד שבוע")],
            )

    chunks = [Chunk("a#1", "a.md", "document", "המדיניות חלה מחר ומסתיימת בעוד שבוע על כולם")]
    candidate = _candidate()
    updated, valid, verdict = SilverSetGenerator(ReviewLLM(), "model").verify_candidate_completeness(
        candidate, [chunks[0]], chunks,
    )
    assert verdict == "corrected"
    assert updated.expected_answer == "המדיניות חלה מחר ומסתיימת בעוד שבוע"
    assert updated.reference_claims == ["המדיניות חלה מחר", "המדיניות מסתיימת בעוד שבוע"]
    assert valid == [chunks[0]]


def test_completeness_verification_rejects_when_correction_is_missing_or_invalid():
    from chatbot_eval.models import AnswerCompletenessReview, EvidenceQuote as EQ

    class NoCorrectionLLM:
        def generate(self, prompt, schema, model):
            return AnswerCompletenessReview(verdict="contradicted", reasoning="סותר")

    class BadQuoteLLM:
        def generate(self, prompt, schema, model):
            return AnswerCompletenessReview(
                verdict="incomplete", corrected_answer="תשובה מתוקנת",
                corrected_reference_claims=["טענה"], corrected_source_ids=["a#1"],
                corrected_supporting_quotes=[EQ(source_id="a#1", quote="ציטוט שלא קיים במקור")],
            )

    chunks = [Chunk("a#1", "a.md", "document", "המדיניות חלה מחר על כולם")]
    candidate = _candidate()

    updated, valid, verdict = SilverSetGenerator(NoCorrectionLLM(), "model").verify_candidate_completeness(
        candidate, chunks, chunks,
    )
    assert verdict == "rejected:completeness_contradicted_no_correction"
    assert updated is candidate

    updated, valid, verdict = SilverSetGenerator(BadQuoteLLM(), "model").verify_candidate_completeness(
        candidate, chunks, chunks,
    )
    assert verdict == "rejected:completeness_incomplete_invalid_correction"


def test_generate_runs_completeness_pass_and_records_diagnostics():
    from chatbot_eval.models import AnswerCompletenessReview

    class FakeLLM:
        def generate(self, prompt, schema, model):
            if schema is QuestionBatch:
                return QuestionBatch(questions=[_candidate()])
            if schema is AnswerCompletenessReview:
                return AnswerCompletenessReview(
                    verdict="incomplete",
                    corrected_answer="המדיניות חלה מחר על כל המשרתים",
                    corrected_reference_claims=["המדיניות חלה מחר על כל המשרתים"],
                    corrected_source_ids=["a#1"],
                    corrected_supporting_quotes=[EvidenceQuote(source_id="a#1", quote="חלה מחר על כל המשרתים")],
                )
            raise AssertionError(schema)

    generator = SilverSetGenerator(FakeLLM(), "model")
    chunks = [Chunk("a#1", "a.md", "document", "המדיניות חלה מחר על כל המשרתים בשירות")]
    questions, _ = generator.generate(
        chunks,
        GenerationOptions(
            max_questions=1, batch_chunks=1, min_topic_questions=1, max_topic_share=1,
            unanswerable_ratio=0, verify_answer_completeness=True,
        ),
        bundle=_bundle(chunks, [("נושא", 5, ["a#1"])]),
    )

    assert len(questions) == 1
    assert questions[0].expected_answer == "המדיניות חלה מחר על כל המשרתים"
    diagnostics = generator.last_generation_diagnostics["answer_completeness_verification"]
    assert diagnostics["enabled"] is True
    assert diagnostics["outcomes"].get("corrected") == 1


def test_generate_completeness_pass_off_by_default_makes_no_review_call():
    class FakeLLM:
        def generate(self, prompt, schema, model):
            if schema is QuestionBatch:
                return QuestionBatch(questions=[_candidate()])
            raise AssertionError(f"unexpected schema {schema}")

    generator = SilverSetGenerator(FakeLLM(), "model")
    chunks = [Chunk("a#1", "a.md", "document", "המדיניות חלה מחר על כולם")]
    questions, _ = generator.generate(
        chunks,
        GenerationOptions(
            max_questions=1, batch_chunks=1, min_topic_questions=1, max_topic_share=1,
            unanswerable_ratio=0,
        ),
        bundle=_bundle(chunks, [("נושא", 5, ["a#1"])]),
    )
    assert len(questions) == 1
    assert generator.last_generation_diagnostics["answer_completeness_verification"] == {"enabled": False}
