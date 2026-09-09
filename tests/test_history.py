import json

import pytest

from chatbot_eval.history import (
    discover_previous_run, read_current_prompt, read_evaluation_contract, read_evaluation_records,
)
from chatbot_eval.artifacts import file_sha256
from chatbot_eval.contracts import records_hash
from chatbot_eval.models import (
    ChatbotResult, EvaluationInsights, EvaluationRecord, Outcome, SilverQuestion,
)


def _record(identifier="Q1"):
    return EvaluationRecord(
        question=SilverQuestion(id=identifier, topic="נושא", question="שאלה", expected_answer="תשובה"),
        result=ChatbotResult(question_id=identifier, answer="תשובה"),
        outcome=Outcome.CORRECT_ANSWER,
    )


def test_history_loaders_accept_standard_output_directories(tmp_path):
    (tmp_path / "evaluation_details.jsonl").write_text(_record().model_dump_json() + "\n", encoding="utf-8")
    prompt_dir = tmp_path / "prompt"
    prompt_dir.mkdir()
    (prompt_dir / "prompt_package.json").write_text(
        json.dumps({"system_prompt_hebrew": "הנחיות נוכחיות"}), encoding="utf-8",
    )

    records, insights = discover_previous_run(tmp_path)

    assert records == [_record()]
    assert insights is None
    assert read_current_prompt(prompt_dir) == "הנחיות נוכחיות"


def test_evaluation_loader_reports_line_and_duplicate_ids(tmp_path):
    path = tmp_path / "evaluation_details.jsonl"
    path.write_text(_record().model_dump_json() + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_evaluation_records(path)

    path.write_text(_record().model_dump_json() + "\n" + _record().model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate question ID"):
        read_evaluation_records(path)


def test_evaluation_loader_migrates_pre_claim_assessment_records(tmp_path):
    record = _record().model_dump(mode="json")
    record["scores"] = {
        "required_points_total": 0, "answer_points_addressed": 0, "answer_points_correct": 0,
        "answer_false_claims": 0, "answer_unsupported_claims": 0, "answer_extraneous_claims": 0,
        "retrieval_points_found": 0, "retrieved_chunks_total": 0,
        "retrieved_chunks_relevant": 0, "retrieved_chunks_contradictory": 0,
        "answer_scope": "exact", "incorrect_type": "not_applicable",
        "response_is_abstention": False, "explanation": "", "missing_or_wrong": "",
        "retrieval_explanation": "",
    }
    path = tmp_path / "evaluation_details.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    loaded = read_evaluation_records(path)

    assert loaded[0].scores.claim_assessments == []


def test_history_only_loads_insights_when_status_matches_records(tmp_path):
    records_path = tmp_path / "evaluation_details.jsonl"
    records_path.write_text(_record().model_dump_json() + "\n", encoding="utf-8")
    insights_path = tmp_path / "evaluation_insights.json"
    insights_path.write_text(
        EvaluationInsights(
            executive_summary="סיכום", strengths=[], issues=[], methodology_note="שיטה",
        ).model_dump_json(),
        encoding="utf-8",
    )
    (tmp_path / "evaluation_insights_status.json").write_text(
        json.dumps({
            "status": "generated", "records_sha256": records_hash([_record()]),
            "insights_sha256": file_sha256(insights_path),
        }),
        encoding="utf-8",
    )

    records, insights = discover_previous_run(tmp_path)

    assert records == [_record()]
    assert insights is not None

    records_path.write_text(_record("Q2").model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different evaluation records"):
        discover_previous_run(tmp_path)


def test_failed_insights_status_does_not_reuse_stale_artifact(tmp_path):
    (tmp_path / "evaluation_details.jsonl").write_text(_record().model_dump_json() + "\n", encoding="utf-8")
    (tmp_path / "evaluation_insights.json").write_text(
        EvaluationInsights(
            executive_summary="ישן", strengths=[], issues=[], methodology_note="ישן",
        ).model_dump_json(),
        encoding="utf-8",
    )
    (tmp_path / "evaluation_insights_status.json").write_text(
        json.dumps({"status": "failed", "records_sha256": records_hash([_record()])}),
        encoding="utf-8",
    )

    records, insights = discover_previous_run(tmp_path)

    assert records == [_record()]
    assert insights is None


def test_evaluation_contract_loader_uses_original_run_manifest(tmp_path):
    contract = {"judge_model": "gemini-flash", "max_answer_chars": 1000}
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"evaluation_contract": contract}), encoding="utf-8",
    )
    (tmp_path / "report_manifest.json").write_text(
        json.dumps({"evaluation_contract": {"judge_model": "wrong"}}), encoding="utf-8",
    )

    assert read_evaluation_contract(tmp_path) == contract
    legacy_file = tmp_path / "evaluation_details.jsonl"
    legacy_file.write_text(_record().model_dump_json() + "\n", encoding="utf-8")
    assert read_evaluation_contract(legacy_file) is None


def test_evaluation_contract_loader_rejects_malformed_contract(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"evaluation_contract": "not-an-object"}), encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty object"):
        read_evaluation_contract(tmp_path)
