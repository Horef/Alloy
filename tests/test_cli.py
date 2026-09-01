from argparse import Namespace
from types import SimpleNamespace

import pytest

from chatbot_eval import cli
from chatbot_eval.models import EvaluationInsights


def test_optional_insights_returns_written_artifact(tmp_path, monkeypatch):
    expected = EvaluationInsights(
        executive_summary="סיכום", strengths=[], issues=[], methodology_note="שיטה",
    )
    monkeypatch.setattr(cli, "generate_insights", lambda records, llm, model: expected)
    args = Namespace(generate_insights=True, output=tmp_path)

    insights, path = cli._optional_insights(args, [], object(), "model")

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
