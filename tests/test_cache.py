from chatbot_eval.cache import CorpusAnalysisCache
from chatbot_eval.models import (
    ChatbotResult, EvaluationInsights, EvaluationRecord, Outcome, SilverQuestion, TopicCandidate,
)


def _topic() -> TopicCandidate:
    return TopicCandidate(name="זכאות", description="כללי זכאות", importance=5, source_ids=["policy.txt#chunk-1"])


def test_corpus_cache_reuses_chunks_and_topics(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "policy.txt").write_text("Eligibility policy and procedure. " * 10, encoding="utf-8")
    cache_dir = tmp_path / "cache"

    first = CorpusAnalysisCache(cache_dir)
    chunks, chunk_key = first.load_chunks(documents, 200, 20)
    calls = 0

    def discover():
        nonlocal calls
        calls += 1
        return [_topic()]

    topics = first.load_topics(
        chunks, chunk_key, model="model", transport="direct", batch_chunks=8,
        implementation_sha256="implementation", discover=discover,
    )
    second = CorpusAnalysisCache(cache_dir)
    cached_chunks, cached_key = second.load_chunks(documents, 200, 20)
    cached_topics = second.load_topics(
        cached_chunks, cached_key, model="model", transport="direct", batch_chunks=8,
        implementation_sha256="implementation", discover=discover,
    )

    assert calls == 1
    assert cached_chunks == chunks
    assert cached_topics == topics
    assert second.summary()["chunks"] == "hit"
    assert second.summary()["topics"] == "hit"


def test_corpus_cache_invalidates_when_document_content_changes(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    source = documents / "policy.txt"
    source.write_text("Original policy content. " * 10, encoding="utf-8")
    cache = CorpusAnalysisCache(tmp_path / "cache")
    original_chunks, original_key = cache.load_chunks(documents, 200, 20)

    source.write_text("Changed policy content. " * 10, encoding="utf-8")
    changed_cache = CorpusAnalysisCache(tmp_path / "cache")
    changed_chunks, changed_key = changed_cache.load_chunks(documents, 200, 20)

    assert changed_key != original_key
    assert changed_chunks != original_chunks
    assert changed_cache.summary()["chunks"] == "miss"


def test_refresh_replaces_matching_entries(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "policy.txt").write_text("Stable policy content. " * 10, encoding="utf-8")
    cache_dir = tmp_path / "cache"
    initial = CorpusAnalysisCache(cache_dir)
    chunks, key = initial.load_chunks(documents, 200, 20)
    initial.load_topics(
        chunks, key, model="model", transport="direct", batch_chunks=8,
        implementation_sha256="implementation", discover=lambda: [_topic()],
    )
    replacement = TopicCandidate(
        name="נהלים", description="נהלים מעודכנים", importance=4, source_ids=["policy.txt#chunk-1"],
    )
    refreshed = CorpusAnalysisCache(cache_dir, refresh=True)
    refreshed_chunks, refreshed_key = refreshed.load_chunks(documents, 200, 20)
    refreshed_topics = refreshed.load_topics(
        refreshed_chunks, refreshed_key, model="model", transport="direct", batch_chunks=8,
        implementation_sha256="implementation", discover=lambda: [replacement],
    )
    final = CorpusAnalysisCache(cache_dir)
    final_chunks, final_key = final.load_chunks(documents, 200, 20)
    cached_topics = final.load_topics(
        final_chunks, final_key, model="model", transport="direct", batch_chunks=8,
        implementation_sha256="implementation", discover=lambda: (_ for _ in ()).throw(AssertionError()),
    )

    assert refreshed_topics == [replacement]
    assert cached_topics == [replacement]
    assert refreshed.summary()["chunks"] == "refresh"
    assert refreshed.summary()["topics"] == "refresh"


def test_invalid_cache_entry_is_recomputed(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "policy.txt").write_text("Policy content. " * 10, encoding="utf-8")
    cache_dir = tmp_path / "cache"
    first = CorpusAnalysisCache(cache_dir)
    _, key = first.load_chunks(documents, 200, 20)
    cache_file = cache_dir / "chunks" / f"{key}.json"
    cache_file.write_text("not JSON", encoding="utf-8")

    recovered = CorpusAnalysisCache(cache_dir)
    chunks, recovered_key = recovered.load_chunks(documents, 200, 20)

    assert chunks
    assert recovered_key == key
    assert recovered.summary()["chunks"] == "miss"


def test_question_topic_assignments_are_reused(tmp_path):
    questions = [SilverQuestion(id="Q1", topic="לא סווג", question="מה הנוהל?", expected_answer="תשובה")]
    calls = 0

    def discover():
        nonlocal calls
        calls += 1
        return {"Q1": "נהלים"}

    first = CorpusAnalysisCache(tmp_path / "cache")
    assert first.load_question_topics(
        questions, model="model", transport="direct", batch_size=250,
        implementation_sha256="v1", discover=discover,
    ) == {"Q1": "נהלים"}
    second = CorpusAnalysisCache(tmp_path / "cache")
    assert second.load_question_topics(
        questions, model="model", transport="direct", batch_size=250,
        implementation_sha256="v1", discover=discover,
    ) == {"Q1": "נהלים"}
    assert calls == 1
    assert second.summary()["question_topics"] == "hit"


def test_insights_are_reused_for_identical_evaluation_records(tmp_path):
    record = EvaluationRecord(
        question=SilverQuestion(id="Q1", topic="נושא", question="שאלה", expected_answer="תשובה"),
        result=ChatbotResult(question_id="Q1", answer="תשובה"), outcome=Outcome.CORRECT_ANSWER,
    )
    expected = EvaluationInsights(
        executive_summary="סיכום", strengths=[], issues=[], methodology_note="שיטה",
    )
    calls = 0

    def discover():
        nonlocal calls
        calls += 1
        return expected

    first = CorpusAnalysisCache(tmp_path / "cache")
    first.load_insights(
        [record], model="model", transport="direct", max_prompt_chars=80_000,
        implementation_sha256="v1", discover=discover,
    )
    second = CorpusAnalysisCache(tmp_path / "cache")
    actual = second.load_insights(
        [record], model="model", transport="direct", max_prompt_chars=80_000,
        implementation_sha256="v1", discover=discover,
    )
    assert actual == expected
    assert calls == 1
    assert second.summary()["insights"] == "hit"
