import json

import pytest

from chatbot_eval.history import (
    discover_previous_run, read_current_prompt, read_evaluation_records,
)
from chatbot_eval.models import ChatbotResult, EvaluationRecord, Outcome, SilverQuestion


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
