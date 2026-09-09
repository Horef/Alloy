from chatbot_eval.documents import Chunk
from chatbot_eval.generator import (
    GenerationOptions, SilverSetGenerator, _assign_stable_ids, _is_duplicate, _render_chunks,
    allocate_quotas, allocate_type_targets, validate_candidate,
)
from chatbot_eval.models import EvidenceQuote, ExpectedBehavior, GeneratedQuestion, GeneratedVariation, QuestionBatch, QuestionForm, QuestionType, SilverQuestion, TopicCandidate, TopicMap, VariationBatch


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

    topics = [
        TopicCandidate(name="A", description="", importance=5, source_ids=["a#1"]),
        TopicCandidate(name="B", description="", importance=4, source_ids=["b#1"]),
    ]
    generator = SilverSetGenerator(EmptyLLM(), "test")

    generator.generate(
        [
            Chunk("a#1", "a.md", "document", "evidence A"),
            Chunk("b#1", "b.md", "document", "evidence B"),
        ],
        GenerationOptions(
            max_questions=10,
            batch_chunks=2,
            min_topic_questions=6,
            max_topic_share=0.4,
            unanswerable_ratio=0,
            max_candidate_rounds=1,
        ),
        topics=topics,
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


def test_generation_budget_includes_natural_and_ambiguous_variants():
    class FakeLLM:
        def generate(self, prompt, schema, model):
            if schema is TopicMap:
                return TopicMap(topics=[TopicCandidate(
                    name="תנאי שירות", description="זכאות", importance=5, source_ids=["a#1"],
                )])
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

    questions, _ = SilverSetGenerator(FakeLLM(), "test").generate(
        [Chunk("a#1", "a.md", "document", "המדיניות חלה מחר על כולם")],
        GenerationOptions(
            max_questions=3, batch_chunks=1, min_topic_questions=1, max_topic_share=1,
            unanswerable_ratio=0, user_variation_ratio=2 / 3, ambiguous_variation_share=0.5,
        ),
    )

    assert [question.question_form for question in questions] == [
        QuestionForm.CANONICAL, QuestionForm.AMBIGUOUS, QuestionForm.NATURAL_USER,
    ]


def test_generation_refills_rejected_candidates_within_bound():
    class FakeLLM:
        question_calls = 0

        def generate(self, prompt, schema, model):
            if schema is TopicMap:
                return TopicMap(topics=[TopicCandidate(name="נושא", description="", importance=5, source_ids=["a#1"])])
            if schema is QuestionBatch:
                self.question_calls += 1
                if self.question_calls == 1:
                    return QuestionBatch(questions=[_candidate(quotes=[EvidenceQuote(source_id="a#1", quote="לא קיים")])])
                return QuestionBatch(questions=[_candidate()])
            raise AssertionError(schema)

    fake = FakeLLM()
    questions, _ = SilverSetGenerator(fake, "test").generate(
        [Chunk("a#1", "a.md", "document", "המדיניות חלה מחר על כולם")],
        GenerationOptions(
            max_questions=1, batch_chunks=1, min_topic_questions=1, max_topic_share=1,
            unanswerable_ratio=0, max_candidate_rounds=2,
        ),
    )

    assert len(questions) == 1
    assert fake.question_calls == 2


def test_generation_retry_prompt_explains_prior_rejection():
    prompts = []

    class FakeLLM:
        def generate(self, prompt, schema, model):
            if schema is TopicMap:
                return TopicMap(topics=[TopicCandidate(
                    name="נושא", description="", importance=5, source_ids=["a#1"],
                )])
            prompts.append(prompt)
            if len(prompts) == 1:
                return QuestionBatch(questions=[_candidate(
                    quotes=[EvidenceQuote(source_id="a#1", quote="לא קיים")],
                )])
            assert "quote_not_verbatim" in prompt
            return QuestionBatch(questions=[_candidate()])

    SilverSetGenerator(FakeLLM(), "test").generate(
        [Chunk("a#1", "a.md", "document", "המדיניות חלה מחר על כולם")],
        GenerationOptions(
            max_questions=1, batch_chunks=1, min_topic_questions=1, max_topic_share=1,
            unanswerable_ratio=0, max_candidate_rounds=2,
        ),
    )

    assert len(prompts) == 2


def test_requested_topic_also_scopes_boundary_questions():
    class FakeLLM:
        def generate(self, prompt, schema, model):
            if schema is TopicMap:
                return TopicMap(topics=[
                    TopicCandidate(name="נושא א", description="", importance=5, source_ids=["a#1"]),
                    TopicCandidate(name="נושא ב", description="", importance=4, source_ids=["b#1"]),
                ])
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

    questions, _ = SilverSetGenerator(FakeLLM(), "test").generate(
        [
            Chunk("a#1", "a.md", "document", "המדיניות חלה מחר על כולם"),
            Chunk("b#1", "b.md", "document", "תוכן אחר שקיים במסמך השני ונועד רק לבדיקה"),
        ],
        GenerationOptions(
            max_questions=2, batch_chunks=2, min_topic_questions=1, max_topic_share=1,
            unanswerable_ratio=0.5, requested_topic="נושא א", max_candidate_rounds=1,
        ),
    )

    assert {question.topic for question in questions} == {"נושא א"}
