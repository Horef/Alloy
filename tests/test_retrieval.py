from chatbot_eval.documents import Chunk
from chatbot_eval.llm import EmbeddingCountMismatch
from chatbot_eval.retrieval import (
    BM25, CachedEmbedder, ChunkRetriever, ModelEmbedder, dedup_tokens, index_terms, query_terms, token_similarity,
)


def test_model_embedder_falls_back_to_single_requests_for_aggregating_models():
    calls: list[int] = []

    class Aggregating:
        def embed(self, texts, model):
            calls.append(len(texts))
            if len(texts) > 1:
                raise EmbeddingCountMismatch("Gemini returned an incomplete embedding response")
            return [[float(len(texts[0]))]]

    embedder = ModelEmbedder(Aggregating(), "gemini-embedding-2", concurrency=2)
    assert embedder.embed(["א", "בב", "גגג"]) == [[1.0], [2.0], [3.0]]
    assert embedder.embed(["דדדד", "ה"]) == [[4.0], [1.0]]
    assert calls.count(3) == 1 and all(n == 1 for n in calls if n != 3)


def test_prefixed_and_bare_words_match_but_shared_stems_do_not():
    assert set(index_terms("לחייל")) & set(query_terms("חייל"))
    assert set(index_terms("חייל")) & set(query_terms("לחייל"))
    assert "צהל" in query_terms('בצה"ל')
    # הורה (parent) and מורה (teacher) share the remainder ורה but neither is the other with a prefix.
    assert not set(index_terms("מורה")) & set(query_terms("הורה"))
    assert token_similarity(dedup_tokens("השכר של החייל"), dedup_tokens("שכר של חייל")) == 1.0
    assert token_similarity(dedup_tokens("הורה"), dedup_tokens("מורה")) == 0.0


def test_bm25_downweights_terms_present_in_every_document():
    docs = [["של", "חופשה", "שנתית"], ["של", "שכר"], ["של", "מענק"]]
    scores = BM25(docs).scores(["של", "שכר"])
    assert scores[1] == max(scores)
    assert scores[0] < scores[1]


def test_retriever_matches_prefixed_words_and_omits_unrelated_chunks():
    chunks = [
        Chunk("a#1", "a.md", "d", "זכאות חייל בודד למענק חודשי"),
        Chunk("b#1", "b.md", "d", "שעות פעילות ומסדרים ביחידה"),
    ]
    ranked = ChunkRetriever(chunks).rank("מה מגיע לחייל הבודד?", 5)
    assert [c.id for c in ranked] == ["a#1"]


def test_retriever_fuses_dense_ranking_and_survives_embedding_failure():
    chunks = [
        Chunk("a#1", "a.md", "d", "מילה נדירה כאן"),
        Chunk("b#1", "b.md", "d", "טקסט אחר לגמרי"),
    ]

    class Dense:
        def embed(self, texts):
            return [[1.0, 0.0] if "אחר" in text or "שאלה" in text else [0.0, 1.0] for text in texts]

    ranked = ChunkRetriever(chunks, Dense()).rank("שאלה בלי מילים משותפות", 2)
    assert [c.id for c in ranked] == ["b#1", "a#1"]

    class Broken:
        def embed(self, texts):
            raise RuntimeError("no embeddings on this gateway")

    assert [c.id for c in ChunkRetriever(chunks, Broken()).rank("מילה נדירה", 2)] == ["a#1"]


def test_cached_embedder_persists_vectors_across_instances(tmp_path):
    calls: list[list[str]] = []

    class Counting:
        def embed(self, texts):
            calls.append(list(texts))
            return [[float(len(text)), 1.0] for text in texts]

    path = tmp_path / "embeddings" / "m.jsonl"
    first = CachedEmbedder(Counting(), path).embed(["א", "בב", "א"])
    second = CachedEmbedder(Counting(), path).embed(["בב", "א"])
    assert calls == [["א", "בב"]]
    assert first == [[1.0, 1.0], [2.0, 1.0], [1.0, 1.0]]
    assert second == [[2.0, 1.0], [1.0, 1.0]]


def test_retriever_does_not_match_words_that_only_share_a_stem():
    chunks = [
        Chunk("a#1", "a.md", "d", "זכויות מורה בבית הספר"),
        Chunk("b#1", "b.md", "d", "זכויות הורה לילד"),
    ]
    assert [c.id for c in ChunkRetriever(chunks).rank("מה מגיע להורה?", 5)] == ["b#1"]
