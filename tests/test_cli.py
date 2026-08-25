from argparse import Namespace

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
