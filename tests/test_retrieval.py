from chatbot_eval.documents import Chunk
from chatbot_eval.retrieval import BM25, CachedEmbedder, ChunkRetriever, dedup_tokens, search_tokens


def test_search_tokens_add_prefix_stripped_hebrew_variants():
    tokens = search_tokens('מה מגיע לחייל בצה"ל?')
    assert "לחייל" in tokens and "חייל" in tokens
    assert "צהל" in tokens
    assert dedup_tokens("השכר של החייל") == dedup_tokens("שכר של חייל")


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
