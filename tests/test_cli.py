import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from chatbot_eval import cli
from chatbot_eval.artifacts import RunManifest
from chatbot_eval.config import load_settings
from chatbot_eval.models import ChatbotResult, EvaluationInsights, EvaluationRecord, Outcome, SilverQuestion


def test_optional_insights_returns_written_artifact(tmp_path, monkeypatch):
    expected = EvaluationInsights(
        executive_summary="סיכום", strengths=[], issues=[], methodology_note="שיטה",
    )
    monkeypatch.setattr(cli, "generate_insights", lambda records, llm, model, **kwargs: expected)
    args = Namespace(generate_insights=True, output=tmp_path)

    class Cache:
        def load_insights(self, records, **kwargs):
            return kwargs["discover"]()

    settings = SimpleNamespace(gemini_transport="direct", insights_max_prompt_chars=80_000)
    insights, path = cli._optional_insights(
        args, [], object(), "model", cache=Cache(), settings=settings,
    )

    assert insights == expected
    assert path == tmp_path / "evaluation_insights.json"
    assert path.exists()


def test_generation_cache_flags_are_explicit_and_mutually_exclusive():
    parser = cli.build_parser()
    args = parser.parse_args(["generate", "--documents", "docs", "--refresh-cache"])

    assert args.refresh_cache is True
    assert args.no_cache is False

    try:
        parser.parse_args(["generate", "--documents", "docs", "--refresh-cache", "--no-cache"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("mutually exclusive cache flags should be rejected")


def test_report_manifest_uses_report_specific_filename(tmp_path):
    settings_path = tmp_path / "config.toml"
    settings_path.write_text("", encoding="utf-8")
    settings = load_settings(settings_path, require_api_key=False)

    with RunManifest(tmp_path / "report", command="report", settings=settings, inputs=[], parameters={}):
        pass

    assert (tmp_path / "report" / "report_manifest.json").exists()
    assert not (tmp_path / "report" / "run_manifest.json").exists()


def test_live_parser_exposes_deployment_identity():
    args = cli.build_parser().parse_args([
        "evaluate", "--questions", "questions.json", "--chatbot-url", "https://chatbot.test",
        "--deployment-id", "blue-2026-09",
    ])

    assert args.deployment_id == "blue-2026-09"


def test_evaluate_file_parser_exposes_behavioral_metadata_columns():
    args = cli.build_parser().parse_args([
        "evaluate-file", "--results", "results.jsonl",
        "--expected-behavior-column", "mode", "--question-form-column", "shape",
        "--parent-question-id-column", "parent", "--question-type-column", "kind",
        "--difficulty-column", "level", "--reference-claims-column", "claims",
        "--supporting-quotes-column", "quotes", "--reference-sources-column", "gold_docs",
    ])

    assert args.expected_behavior_column == "mode"
    assert args.question_form_column == "shape"
    assert args.parent_question_id_column == "parent"
    assert args.question_type_column == "kind"
    assert args.difficulty_column == "level"
    assert args.reference_claims_column == "claims"
    assert args.supporting_quotes_column == "quotes"
    assert args.reference_sources_column == "gold_docs"


def test_report_passes_evaluation_contracts_to_comparison(tmp_path, monkeypatch):
    settings_path = tmp_path / "config.toml"
    settings_path.write_text("", encoding="utf-8")
    record = EvaluationRecord(
        question=SilverQuestion(id="Q1", topic="x", question="q", expected_answer="a"),
        result=ChatbotResult(question_id="Q1", answer="a"), outcome=Outcome.CORRECT_ANSWER,
    )
    current = tmp_path / "current"
    previous = tmp_path / "previous"
    for directory, contract in ((current, {"judge": "new"}), (previous, {"judge": "old"})):
        directory.mkdir()
        (directory / "evaluation_details.jsonl").write_text(record.model_dump_json() + "\n", encoding="utf-8")
        (directory / "run_manifest.json").write_text(
            json.dumps({"evaluation_contract": contract}), encoding="utf-8",
        )
    captured = {}

    def fake_write_report(records, output, insights, **kwargs):
        captured.update(kwargs)
        output.mkdir(parents=True, exist_ok=True)
        summary, report = output / "evaluation_summary.json", output / "evaluation_report.html"
        summary.write_text("{}", encoding="utf-8")
        report.write_text("report", encoding="utf-8")
        return summary, report

    monkeypatch.setattr(cli, "write_report", fake_write_report)
    assert cli.main([
        "--config", str(settings_path), "report", "--results", str(current),
        "--compare-with", str(previous), "--output", str(tmp_path / "report"),
    ]) == 0

    assert captured["current_contract"] == {"judge": "new"}
    assert captured["previous_contract"] == {"judge": "old"}


def test_cache_directory_cannot_be_inside_documents(tmp_path):
    documents = tmp_path / "docs"
    documents.mkdir()
    args = Namespace(
        cache_dir=documents / "cache", config=tmp_path / "config.toml", documents=documents,
        no_cache=False, refresh_cache=False,
    )
    settings = SimpleNamespace(cache_directory="unused", cache_enabled=True)

    with pytest.raises(ValueError, match="outside --documents"):
        cli._analysis_cache(args, settings)


def test_resume_can_retry_only_checkpointed_errors(tmp_path):
    question = SilverQuestion(id="Q1", topic="x", question="q", expected_answer="a")
    items = [(question, ChatbotResult(question_id="Q1", answer=""))]

    class Evaluator:
        def __init__(self, outcome, answer=""):
            self.outcome = outcome
            self.answer = answer
            self.evaluate_calls = 0
            self.judge_calls = 0

        def evaluate(self, questions, on_record):
            self.evaluate_calls += 1
            record = EvaluationRecord(
                question=questions[0], result=ChatbotResult(question_id="Q1", answer=self.answer),
                outcome=self.outcome,
            )
            on_record(record)
            return [record]

        def judge_results(self, pairs, on_record):
            self.judge_calls += 1
            question, result = pairs[0]
            assert result.answer == "captured chatbot answer"
            record = EvaluationRecord(
                question=question, result=result, outcome=self.outcome,
            )
            on_record(record)
            return [record]

    args = Namespace(
        checkpoint=None, output=tmp_path, resume=False, retry_errors=False,
    )
    cli._checkpointed_evaluation(
        args, items, Evaluator(Outcome.JUDGE_ERROR, "captured chatbot answer"),
        premade=False, contract={"judge": "v1"},
    )
    args.resume = True
    args.retry_errors = True
    retry = Evaluator(Outcome.CORRECT_ANSWER)
    records, resumed, _ = cli._checkpointed_evaluation(
        args, items, retry, premade=False, contract={"judge": "v1"},
    )

    assert records[0].outcome == Outcome.CORRECT_ANSWER
    assert records[0].result.answer == "captured chatbot answer"
    assert retry.evaluate_calls == 0
    assert retry.judge_calls == 1
    assert resumed == 0


def test_retry_errors_requires_resume():
    with pytest.raises(ValueError, match="requires --resume"):
        cli.main(["evaluate-file", "--results", "missing.csv", "--retry-errors"])


def test_generation_resume_rejects_cache_refresh():
    with pytest.raises(ValueError, match="cannot be combined"):
        cli.main([
            "generate", "--documents", "docs", "--resume", "--refresh-cache",
        ])


def test_generate_persists_diagnostics_and_inventories_them_in_manifest(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "source.txt").write_text("source", encoding="utf-8")
    output = tmp_path / "output"
    expected = {
        "planned_total": 1,
        "accepted_total": 0,
        "rejected": {"quote_not_verbatim": 1},
        "unallocated": 1,
        "boundary_evidence_scope": "cluster_excerpts",
    }

    from chatbot_eval.graph import GraphBundle, KnowledgeGraph

    class Cache:
        enabled = True
        refresh = False

        def load_chunks(self, *args, **kwargs):
            return [], "chunk-key"

        def summary(self):
            return {"hits": 0, "misses": 0}

    class Generator:
        def __init__(self, *args, **kwargs):
            self.last_generation_diagnostics = {}

        def generate(self, chunks, options, *, bundle):
            self.last_generation_diagnostics = expected
            return [], bundle.topics

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "GeminiStructuredLLM", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "_analysis_cache", lambda *args, **kwargs: Cache())
    monkeypatch.setattr(cli, "_build_graph_bundle", lambda *args, **kwargs: GraphBundle(KnowledgeGraph([], []), []))
    monkeypatch.setattr(cli, "SilverSetGenerator", Generator)

    assert cli.main([
        "--config", str(config), "generate", "--documents", str(documents),
        "--output", str(output), "--max-questions", "1",
    ]) == 0

    diagnostics_path = output / "generation_diagnostics.json"
    assert json.loads(diagnostics_path.read_text(encoding="utf-8")) == expected
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["results"]["generation_diagnostics"] == expected
    assert str(diagnostics_path.resolve()) in {
        item["path"] for item in manifest["results"]["outputs"]
    }


def test_revary_regenerates_only_variations(tmp_path, monkeypatch):
    import json
    from chatbot_eval.io import write_questions
    from chatbot_eval.models import GeneratedVariation, QuestionForm, VariationBatch

    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")

    canonical = SilverQuestion(id="Q1", topic="t", question="שאלה קנונית", expected_answer="תשובה",
                               reference_claims=["תשובה"])
    old_variant = SilverQuestion(id="V1", topic="t", question="ניסוח ישן", expected_answer="תשובה",
                                 question_form=QuestionForm.NATURAL_USER, parent_question_id="Q1")
    questions_dir = tmp_path / "in"
    csv_path, jsonl_path = write_questions([canonical, old_variant], questions_dir)
    output = tmp_path / "out"

    class VarLLM:
        def generate(self, prompt, schema, model, **kwargs):
            return VariationBatch(variations=[
                GeneratedVariation(source_question_id="Q1", question="ניסוח חדש",
                                   question_form="natural_user", rationale="r"),
            ])

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "GeminiStructuredLLM", lambda *args, **kwargs: VarLLM())

    assert cli.main([
        "--config", str(config), "revary", "--questions", str(jsonl_path),
        "--output", str(output), "--variation-count", "1",
    ]) == 0

    merged = [json.loads(line) for line in (output / "silver_questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    questions_text = {q["question"] for q in merged}
    assert "שאלה קנונית" in questions_text       # canonical kept
    assert "ניסוח ישן" not in questions_text       # old variant dropped
    assert "ניסוח חדש" in questions_text           # fresh variant added
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["results"]["revary_diagnostics"]["new_variants"] == 1


def test_generate_merge_into_appends_to_existing_set(tmp_path, monkeypatch):
    import json
    from chatbot_eval.graph import GraphBundle, KnowledgeGraph
    from chatbot_eval.io import write_questions

    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "source.txt").write_text("source", encoding="utf-8")
    output = tmp_path / "output"

    # Existing reviewed set to merge into.
    existing = [
        SilverQuestion(id="Q-old1", topic="t", question="שאלה קיימת אחת", expected_answer="א",
                       reference_claims=["א"]),
        SilverQuestion(id="Q-old2", topic="t", question="שאלה קיימת שתיים", expected_answer="ב",
                       reference_claims=["ב"]),
    ]
    _, merge_target = write_questions(existing, tmp_path / "existing")

    generated = [
        SilverQuestion(id="Q-new", topic="שכר", question="שאלה חדשה על שכר", expected_answer="ג",
                       reference_claims=["ג"]),
    ]

    class Cache:
        enabled = True
        refresh = False
        def load_chunks(self, *a, **k):
            return [], "chunk-key"
        def summary(self):
            return {}

    class Generator:
        def __init__(self, *a, **k):
            self.last_generation_diagnostics = {}
        def generate(self, chunks, options, *, bundle):
            self.last_generation_diagnostics = {"accepted_total": 1}
            return list(generated), bundle.topics

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "GeminiStructuredLLM", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_analysis_cache", lambda *a, **k: Cache())
    monkeypatch.setattr(cli, "_build_graph_bundle", lambda *a, **k: GraphBundle(KnowledgeGraph([], []), []))
    monkeypatch.setattr(cli, "SilverSetGenerator", Generator)

    assert cli.main([
        "--config", str(config), "generate", "--documents", str(documents),
        "--output", str(output), "--max-questions", "1", "--merge-into", str(merge_target),
    ]) == 0

    merged = [json.loads(l) for l in (output / "silver_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    questions_text = {q["question"] for q in merged}
    assert "שאלה קיימת אחת" in questions_text and "שאלה קיימת שתיים" in questions_text  # existing kept
    assert "שאלה חדשה על שכר" in questions_text                                          # new appended
    assert len(merged) == 3
    diag = json.loads((output / "generation_diagnostics.json").read_text(encoding="utf-8"))
    assert diag["merge"]["new_added"] == 1 and diag["merge"]["existing_kept"] == 2
