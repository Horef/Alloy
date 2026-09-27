import re

from chatbot_eval.documents import Chunk
from chatbot_eval.graph import derive_topic_clusters
from chatbot_eval.graph_build import GraphBuilder, group_chunks_by_document
from chatbot_eval.models import (
    GraphTopicLabel,
    GraphTopicLabelBatch,
    NodeSignals,
    NodeSignalsBatch,
    ThemeVocabulary,
)


def _ids(prompt):
    return re.findall(r"\[chunk_id: (.*?)\]", prompt)


class SignalLLM:
    def generate(self, prompt, schema, model):
        if schema is NodeSignalsBatch:
            return NodeSignalsBatch(signals=[
                NodeSignals(chunk_id=i, entities=['צה"ל'], keyphrases=["מבנה שכר"], summary=f"s {i}")
                for i in _ids(prompt)
            ])
        if schema is GraphTopicLabelBatch:
            return GraphTopicLabelBatch(labels=[GraphTopicLabel(name="שכר", description="שאלות שכר")])
        raise AssertionError(schema)


def test_extract_document_signals_returns_one_record_per_chunk_in_order():
    chunks = [Chunk("a#1", "a.md", "d", "x"), Chunk("a#2", "a.md", "d", "y")]
    signals = GraphBuilder(SignalLLM(), "m").extract_document_signals("a.md", chunks, batch_chunks=1)
    assert [s.chunk_id for s in signals] == ["a#1", "a#2"]
    assert all(s.entities == ['צה"ל'] for s in signals)


def test_extraction_degrades_gracefully_when_model_returns_nothing():
    class EmptyLLM:
        def generate(self, prompt, schema, model):
            return NodeSignalsBatch(signals=[])

    chunks = [Chunk("a#1", "a.md", "d", "x")]
    signals = GraphBuilder(EmptyLLM(), "m").extract_document_signals("a.md", chunks, batch_chunks=8)
    assert [s.chunk_id for s in signals] == ["a#1"]
    assert signals[0].entities == [] and signals[0].keyphrases == []


def test_assemble_and_label_produces_prevalence_weighted_topics():
    chunks = [
        Chunk("a#1", "a.md", "d", "טקסט"),
        Chunk("a#2", "a.md", "d", "טקסט"),
        Chunk("b#1", "b.md", "d", "טקסט"),
    ]
    builder = GraphBuilder(SignalLLM(), "m")
    signals = {}
    for f, cs in group_chunks_by_document(chunks).items():
        for s in builder.extract_document_signals(f, cs, batch_chunks=8):
            signals[s.chunk_id] = s
    graph = builder.assemble_graph(chunks, signals, keyphrase_overlap_threshold=0.3)
    assert len(graph.nodes) == 3
    clusters = derive_topic_clusters(graph)
    topics = builder.label_topics(graph, clusters)
    assert topics and topics[0].name == "שכר"
    assert all(1 <= t.importance <= 5 for t in topics)


def test_label_topics_falls_back_when_labeling_fails():
    class NoLabelLLM:
        def generate(self, prompt, schema, model):
            if schema is NodeSignalsBatch:
                return NodeSignalsBatch(signals=[
                    NodeSignals(chunk_id=i, entities=["מונח"], keyphrases=["ביטוי"], summary="s")
                    for i in _ids(prompt)
                ])
            raise RuntimeError("labeling unavailable")

    chunks = [Chunk("a#1", "a.md", "d", "x")]
    builder = GraphBuilder(NoLabelLLM(), "m")
    signals = {s.chunk_id: s for s in builder.extract_document_signals("a.md", chunks, batch_chunks=8)}
    graph = builder.assemble_graph(chunks, signals, keyphrase_overlap_threshold=0.3)
    topics = builder.label_topics(graph, derive_topic_clusters(graph))
    # Deterministic fallback name derived from the cluster signature, never a crash.
    assert topics and topics[0].name


# --- theme layer (opt-in) -----------------------------------------------------------------------


def test_build_theme_vocabulary_enforces_ceiling_and_drops_blank_names():
    class VocabLLM:
        def generate(self, prompt, schema, model):
            assert schema is ThemeVocabulary
            return ThemeVocabulary(themes=[
                GraphTopicLabel(name="שכר", description="תשלומים"),
                GraphTopicLabel(name="  ", description="ריק"),      # blank name dropped
                GraphTopicLabel(name="חופשות", description="ימי חופשה"),
                GraphTopicLabel(name="מילואים", description="שירות מילואים"),
            ])

    builder = GraphBuilder(VocabLLM(), "m")
    chunks = [Chunk("a#1", "a.md", "d", "x")]
    vocab = builder.build_theme_vocabulary(chunks, ["summary one", "summary two"], max_themes=2)
    names = [t.name for t in vocab.themes]
    assert names == ["שכר", "חופשות"]  # blank removed, then capped at 2, order preserved


def test_build_theme_vocabulary_is_a_noop_without_summaries():
    class ExplodingLLM:
        def generate(self, prompt, schema, model):
            raise AssertionError("must not call the model with no summaries")

    builder = GraphBuilder(ExplodingLLM(), "m")
    vocab = builder.build_theme_vocabulary([Chunk("a#1", "a.md", "d", "x")], ["", "   "], max_themes=20)
    assert vocab.themes == []


def test_build_theme_vocabulary_falls_back_to_empty_on_model_error():
    class FailingLLM:
        def generate(self, prompt, schema, model):
            raise RuntimeError("model down")

    builder = GraphBuilder(FailingLLM(), "m")
    vocab = builder.build_theme_vocabulary([Chunk("a#1", "a.md", "d", "x")], ["real summary"], max_themes=20)
    assert vocab.themes == []


def test_theme_aware_extraction_injects_vocabulary_and_tags_chunks():
    seen_prompts: list[str] = []

    class ThemeLLM:
        def generate(self, prompt, schema, model):
            seen_prompts.append(prompt)
            return NodeSignalsBatch(signals=[
                NodeSignals(chunk_id=i, entities=["e"], summary=f"s {i}", theme="שכר")
                for i in _ids(prompt)
            ])

    vocab = ThemeVocabulary(themes=[
        GraphTopicLabel(name="שכר", description="תשלומים"),
        GraphTopicLabel(name="חופשות", description="ימי חופשה"),
    ])
    chunks = [Chunk("a#1", "a.md", "d", "x")]
    signals = GraphBuilder(ThemeLLM(), "m").extract_document_signals(
        "a.md", chunks, batch_chunks=8, theme_vocabulary=vocab,
    )
    assert signals[0].theme == "שכר"
    # The controlled vocabulary is rendered into the extraction prompt.
    assert "שכר" in seen_prompts[0] and "חופשות" in seen_prompts[0]


def test_extraction_without_vocabulary_leaves_theme_empty():
    chunks = [Chunk("a#1", "a.md", "d", "x")]
    signals = GraphBuilder(SignalLLM(), "m").extract_document_signals("a.md", chunks, batch_chunks=8)
    assert signals[0].theme == ""


def test_failed_extraction_call_reports_incomplete_signals():
    import pytest

    from chatbot_eval.graph import IncompleteExtraction

    class FlakyLLM:
        calls = 0

        def generate(self, prompt, schema, model):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("transient")
            return SignalLLM().generate(prompt, schema, model)

    chunks = [Chunk("a#1", "a.md", "d", "x"), Chunk("a#2", "a.md", "d", "y")]
    with pytest.raises(IncompleteExtraction) as caught:
        GraphBuilder(FlakyLLM(), "m").extract_document_signals("a.md", chunks, batch_chunks=1)
    assert [s.chunk_id for s in caught.value.signals] == ["a#1", "a#2"]
    assert caught.value.signals[0].entities and not caught.value.signals[1].entities
