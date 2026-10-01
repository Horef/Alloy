import json

import pytest

from chatbot_eval import cli
from chatbot_eval.documents import Chunk
from chatbot_eval.graph import GraphBundle, GraphNode, GraphTopic, KnowledgeGraph
from chatbot_eval.planning import (
    GenerationPlan, PlanningParameters, build_plan, information_units, plan_quotas, suggest_graph_parameters,
)


def _chunk(chunk_id: str, statements: int) -> Chunk:
    text = " ".join(f"החייל זכאי להטבה מספר {i} בתנאים מסוימים." for i in range(statements))
    return Chunk(chunk_id, chunk_id.split("#")[0], "d", text)


def test_information_units_count_distinct_statements():
    assert information_units("") == 0
    assert information_units("קצר") == 1
    text = "- **חייל בודד** זכאי למענק חודשי קבוע.\n- חייל בודד זכאי למענק חודשי קבוע.\n- הזכאות נבדקת פעם בשנה בלשכה."
    assert information_units(text) == 2


def test_plan_represents_every_chunk_and_scales_with_density():
    chunks = [_chunk("a.md#1", 2), _chunk("a.md#2", 30), _chunk("b.md#1", 30), _chunk("c.md#1", 1)]
    topics = [
        GraphTopic("sparse", "", 1, ["a.md#1", "c.md#1"]),
        GraphTopic("dense", "", 1, ["a.md#2", "b.md#1"]),
    ]
    parameters = PlanningParameters(margin_of_error=0)
    plan = build_plan(chunks, topics, parameters, graph={})
    by_name = {topic.name: topic for topic in plan.topics}
    assert by_name["sparse"].canonical == 2
    assert by_name["dense"].canonical == 2 * round(30 * 0.35 / 3.2)
    assert plan.totals["canonical"] == by_name["sparse"].canonical + by_name["dense"].canonical
    assert plan.totals["total"] == plan.totals["canonical"] + plan.totals["variations"] + plan.totals["boundary"]
    assert all(topic.boundary >= 2 for topic in plan.topics)


def test_precision_floor_is_capped_by_what_the_content_supports():
    chunks = [_chunk("a.md#1", 50), _chunk("a.md#2", 50), _chunk("a.md#3", 50), _chunk("b.md#1", 40)]
    topics = [
        GraphTopic("rich", "", 1, ["a.md#1", "a.md#2", "a.md#3"]),
        GraphTopic("thin", "", 1, ["b.md#1"]),
    ]
    plan = build_plan(chunks, topics, PlanningParameters(margin_of_error=0.2), graph={})
    by_name = {topic.name: topic for topic in plan.topics}
    assert by_name["rich"].canonical == 25 and by_name["rich"].reason.startswith("per-topic")
    # 40 statements support at most floor(40 * 0.6 / 3.2) = 7 questions, below the floor of 25.
    assert by_name["thin"].canonical == 7


def test_plan_quotas_honor_edits_and_reject_a_changed_graph():
    chunks = [_chunk("a.md#1", 10), _chunk("b.md#1", 10)]
    topics = [GraphTopic("A", "", 1, ["a.md#1"]), GraphTopic("B", "", 1, ["b.md#1"])]
    plan = build_plan(chunks, topics, PlanningParameters(margin_of_error=0), graph={})
    edited = GenerationPlan.model_validate_json(plan.model_dump_json())
    edited.topics[0].canonical = 9
    budgets, canonical, boundary = plan_quotas(edited, topics)
    assert dict(canonical)["A"] == 9 and budgets[0] == 9 + edited.topics[1].canonical
    assert budgets[2] == sum(q for _, q in boundary)

    changed = [GraphTopic("A", "", 1, ["a.md#1", "b.md#1"])]
    with pytest.raises(ValueError, match="topics changed"):
        plan_quotas(plan, changed)


def test_graph_suggestions_scale_with_corpus_size():
    small, large = suggest_graph_parameters(85), suggest_graph_parameters(240)
    assert small["target_topics"] < large["target_topics"]
    assert 3 <= small["max_cluster_nodes"] <= 12


def test_generate_follows_planned_topic_quotas_exactly():
    from chatbot_eval.generator import GenerationOptions, SilverSetGenerator
    from chatbot_eval.models import EvidenceQuote, GeneratedQuestion, QuestionBatch

    class FakeLLM:
        calls = 0

        def generate(self, prompt, schema, model):
            self.calls += 1
            source = prompt.split("[SOURCE_ID: ")[1].split("]")[0]
            return QuestionBatch(questions=[GeneratedQuestion(
                question=" ".join(f"מילה{self.calls}x{i}x{j}" for j in range(5)), expected_answer="תשובה",
                answerable=True, difficulty="easy", rationale="r", source_ids=[source], reference_claims=["תשובה"],
                supporting_quotes=[EvidenceQuote(source_id=source, quote="עובדה")],
            ) for i in range(8)])

    chunks = [Chunk("a#1", "a.md", "d", "עובדה א"), Chunk("b#1", "b.md", "d", "עובדה ב")]
    nodes = [GraphNode(c.id, c.file, c.location, c.text) for c in chunks]
    bundle = GraphBundle(KnowledgeGraph(nodes, []), [
        GraphTopic("A", "", 5, ["a#1"]), GraphTopic("B", "", 1, ["b#1"]),
    ])
    generator = SilverSetGenerator(FakeLLM(), "test")
    questions, _ = generator.generate(chunks, GenerationOptions(
        max_questions=1000, batch_chunks=1, min_topic_questions=1, max_topic_share=0.35,
        unanswerable_ratio=0.5, max_candidate_rounds=1,
        planned_budgets=(5, 0, 0), topic_quotas=(("A", 1), ("B", 4)),
    ), bundle=bundle)
    assert [q.topic for q in questions] == ["A", "B", "B", "B", "B"]
    assert generator.last_generation_diagnostics["budgets"]["source"] == "plan"


def test_plan_command_writes_an_editable_plan(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('[gemini]\ntransport = "direct"\n', encoding="utf-8")
    documents = tmp_path / "docs"
    documents.mkdir()
    (documents / "a.md").write_text("x", encoding="utf-8")
    chunks = [_chunk("a.md#1", 12), _chunk("a.md#2", 4)]

    class Cache:
        enabled = True
        refresh = False

        def load_chunks(self, *args, **kwargs):
            return chunks, "chunk-key"

        def summary(self):
            return {}

    bundle = GraphBundle(KnowledgeGraph([], []), [GraphTopic("נושא", "", 3, [c.id for c in chunks])])
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "GeminiStructuredLLM", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "_analysis_cache", lambda *args, **kwargs: Cache())
    monkeypatch.setattr(cli, "_build_graph_bundle", lambda *args, **kwargs: bundle)

    output = tmp_path / "plan"
    assert cli.main(["--config", str(config), "plan", "--documents", str(documents), "--output", str(output)]) == 0
    plan = GenerationPlan.model_validate_json((output / "generation_plan.json").read_text(encoding="utf-8"))
    assert plan.corpus["chunks"] == 2 and plan.topics[0].name == "נושא"
    assert plan.graph["used"]["max_cluster_nodes"] == 12
    assert json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["status"] == "completed"
