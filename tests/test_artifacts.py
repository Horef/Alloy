import json

import pytest

from chatbot_eval.artifacts import EvaluationCheckpoint, RunManifest, evaluation_fingerprint
from chatbot_eval.config import load_settings
from chatbot_eval.models import ChatbotResult, EvaluationRecord, Outcome, SilverQuestion


def _record(answer="תשובה"):
    question = SilverQuestion(
        id="Q1", topic="נושא", question="שאלה", expected_answer="תשובה",
        reference_claims=["תשובה"],
    )
    return EvaluationRecord(
        question=question,
        result=ChatbotResult(question_id="Q1", answer=answer),
        outcome=Outcome.CORRECT_ANSWER,
    )


def test_checkpoint_round_trip_and_input_mismatch_detection(tmp_path):
    record = _record()
    fingerprint = evaluation_fingerprint(record.question)
    path = tmp_path / "checkpoint.jsonl"
    checkpoint = EvaluationCheckpoint(path, {"Q1": fingerprint}, resume=False)
    checkpoint.append(record)

    loaded = EvaluationCheckpoint(path, {"Q1": fingerprint}, resume=True).load()
    assert loaded["Q1"].result.answer == "תשובה"

    with pytest.raises(ValueError, match="input mismatch"):
        EvaluationCheckpoint(path, {"Q1": "different"}, resume=True).load()


def test_manifest_records_failure_without_api_key(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text("[gemini]\napi_key_env='TEST_GEMINI_KEY'\n", encoding="utf-8")
    monkeypatch.setenv("TEST_GEMINI_KEY", "secret-value")
    settings = load_settings(config)
    input_path = tmp_path / "input.txt"
    input_path.write_text("input", encoding="utf-8")

    with pytest.raises(RuntimeError):
        with RunManifest(
            tmp_path / "output", command="test", settings=settings,
            inputs=[input_path], parameters={"safe": True},
        ):
            raise RuntimeError("expected failure")

    manifest = json.loads((tmp_path / "output" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["inputs"][0]["sha256"]
    assert "api_key" not in manifest["settings"]
    assert "apigee_api_key" not in manifest["settings"]
    assert "secret-value" not in json.dumps(manifest)


def test_manifest_redacts_apigee_api_key(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text(
        """
[gemini]
transport = "apigee"
apigee_api_key_env = "TEST_APIGEE_KEY"
apigee_base_url = "https://preprod.apigee.digital.idf.il/ai_gateway/v1/hr"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_APIGEE_KEY", "apigee-secret-value")
    settings = load_settings(config)
    input_path = tmp_path / "input.txt"
    input_path.write_text("input", encoding="utf-8")

    with RunManifest(
        tmp_path / "output", command="test", settings=settings,
        inputs=[input_path], parameters={},
    ):
        pass

    manifest_text = (tmp_path / "output" / "run_manifest.json").read_text(encoding="utf-8")
    assert "apigee_api_key" not in manifest_text
    assert "apigee-secret-value" not in manifest_text
