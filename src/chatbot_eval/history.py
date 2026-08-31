from __future__ import annotations

import json
from pathlib import Path

from .models import EvaluationInsights, EvaluationRecord


def _resolve_artifact(path: Path, filenames: tuple[str, ...], description: str) -> Path:
    if not path.exists():
        raise ValueError(f"{description} path does not exist: {path}")
    if path.is_file():
        return path
    for filename in filenames:
        candidate = path / filename
        if candidate.is_file():
            return candidate
    expected = ", ".join(filenames)
    raise ValueError(f"No {description} found under {path}; expected one of: {expected}")


def read_evaluation_records(path: Path) -> list[EvaluationRecord]:
    artifact = _resolve_artifact(path, ("evaluation_details.jsonl",), "evaluation results")
    records: list[EvaluationRecord] = []
    seen_ids: set[str] = set()
    with artifact.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                if isinstance(payload, dict) and isinstance(payload.get("scores"), dict):
                    payload["scores"].setdefault("claim_assessments", [])
                record = EvaluationRecord.model_validate(payload)
            except Exception as exc:
                raise ValueError(
                    f"Invalid evaluation record in {artifact} at line {line_number}: {exc}"
                ) from exc
            if record.question.id in seen_ids:
                raise ValueError(f"Duplicate question ID {record.question.id!r} in {artifact}")
            seen_ids.add(record.question.id)
            records.append(record)
    if not records:
        raise ValueError(f"No evaluation records found in {artifact}")
    return records


def read_evaluation_insights(path: Path) -> EvaluationInsights:
    artifact = _resolve_artifact(path, ("evaluation_insights.json",), "evaluation insights")
    try:
        return EvaluationInsights.model_validate_json(artifact.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid evaluation insights in {artifact}: {exc}") from exc


def read_current_prompt(path: Path, max_chars: int = 60_000) -> str:
    artifact = _resolve_artifact(
        path,
        ("prompt_package.json", "generated_system_prompt.md"),
        "current prompt",
    )
    text = artifact.read_text(encoding="utf-8")
    if artifact.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid current-prompt JSON in {artifact}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("system_prompt_hebrew"), str):
            raise ValueError(f"Current-prompt JSON must contain a string system_prompt_hebrew: {artifact}")
        text = payload["system_prompt_hebrew"]
    text = text.strip()
    if not text:
        raise ValueError(f"Current prompt is empty: {artifact}")
    if len(text) > max_chars:
        raise ValueError(f"Current prompt exceeds the {max_chars}-character safety limit: {artifact}")
    return text


def discover_previous_run(path: Path) -> tuple[list[EvaluationRecord] | None, EvaluationInsights | None]:
    """Load the standard artifacts available in a previous output directory."""
    if not path.exists():
        raise ValueError(f"Previous-run path does not exist: {path}")
    if path.is_file():
        return read_evaluation_records(path), None
    results_path = path / "evaluation_details.jsonl"
    insights_path = path / "evaluation_insights.json"
    records = read_evaluation_records(results_path) if results_path.is_file() else None
    insights = read_evaluation_insights(insights_path) if insights_path.is_file() else None
    if records is None and insights is None:
        raise ValueError(
            f"No evaluation_details.jsonl or evaluation_insights.json found under previous run {path}"
        )
    return records, insights
