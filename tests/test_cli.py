from argparse import Namespace
from types import SimpleNamespace

import pytest

from chatbot_eval import cli
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
        def __init__(self, outcome):
            self.outcome = outcome

        def evaluate(self, questions, on_record):
            record = EvaluationRecord(
                question=questions[0], result=ChatbotResult(question_id="Q1", answer=""),
                outcome=self.outcome,
            )
            on_record(record)
            return [record]

    args = Namespace(
        checkpoint=None, output=tmp_path, resume=False, retry_errors=False,
    )
    cli._checkpointed_evaluation(
        args, items, Evaluator(Outcome.JUDGE_ERROR), premade=False, contract={"judge": "v1"},
    )
    args.resume = True
    args.retry_errors = True
    records, resumed, _ = cli._checkpointed_evaluation(
        args, items, Evaluator(Outcome.CORRECT_ANSWER), premade=False, contract={"judge": "v1"},
    )

    assert records[0].outcome == Outcome.CORRECT_ANSWER
    assert resumed == 0


def test_retry_errors_requires_resume():
    with pytest.raises(ValueError, match="requires --resume"):
        cli.main(["evaluate-file", "--results", "missing.csv", "--retry-errors"])


def test_generation_resume_rejects_cache_refresh():
    with pytest.raises(ValueError, match="cannot be combined"):
        cli.main([
            "generate", "--documents", "docs", "--resume", "--refresh-cache",
        ])
