from chatbot_eval.documents import Chunk
from chatbot_eval.generator import GenerationOptions, SilverSetGenerator, _is_duplicate, allocate_quotas, validate_candidate
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


def _candidate(question_type=QuestionType.BASIC_KNOWLEDGE, source_ids=None, quotes=None):
    return GeneratedQuestion(
        question="מה המדיניות?", expected_answer="המדיניות חלה מחר", answerable=True,
        difficulty="medium", rationale="בדיקת מדיניות", question_type=question_type,
        source_ids=source_ids or ["a#1"],
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
