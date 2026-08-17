from chatbot_eval.generator import allocate_quotas
from chatbot_eval.models import TopicCandidate


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

