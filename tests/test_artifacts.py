import json

import pytest

from chatbot_eval.artifacts import (
    EvaluationCheckpoint, RunManifest, StructuredCallCheckpoint, evaluation_fingerprint,
)
from chatbot_eval.config import load_settings
from chatbot_eval.models import ChatbotResult, EvaluationRecord, Outcome, SilverQuestion, TopicCandidate, TopicMap


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


def test_checkpoint_rejects_changed_evaluation_contract(tmp_path):
    record = _record()
    fingerprint = evaluation_fingerprint(record.question, contract={"judge": "v1"})
    path = tmp_path / "checkpoint.jsonl"
    checkpoint = EvaluationCheckpoint(
        path, {"Q1": fingerprint}, resume=False, run_signature="contract-v1",
    )
    checkpoint.append(record)

    with pytest.raises(ValueError, match="contract mismatch"):
        EvaluationCheckpoint(
            path, {"Q1": fingerprint}, resume=True, run_signature="contract-v2",
        ).load()


def test_structured_call_checkpoint_replays_without_calling_llm(tmp_path):
    response = TopicMap(topics=[TopicCandidate(
        name="נושא", description="תיאור", importance=5, source_ids=[],
    )])

    class LLM:
        calls = 0

        def generate(self, prompt, schema, model):
            self.calls += 1
            return response

    llm = LLM()
    path = tmp_path / "generation.jsonl"
    first = StructuredCallCheckpoint(path, llm, signature="same", resume=False)
    assert first.generate("prompt", TopicMap, "model") == response
    resumed = StructuredCallCheckpoint(path, llm, signature="same", resume=True)
    assert resumed.generate("prompt", TopicMap, "model") == response
    assert llm.calls == 1


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
