from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_key: str
    generation_model: str
    judge_model: str
    max_questions: int
    chunk_chars: int
    chunk_overlap_chars: int
    batch_chunks: int
    unanswerable_ratio: float
    user_variation_ratio: float
    ambiguous_variation_share: float
    max_candidate_rounds: int
    min_topic_questions: int
    max_topic_share: float
    request_timeout_seconds: int
    max_retries: int
    progress_enabled: bool
    log_file: str
    log_level: str

    def validate(self) -> "Settings":
        errors: list[str] = []
        if not self.generation_model.strip() or not self.judge_model.strip():
            errors.append("Gemini model names must not be empty")
        if self.max_questions < 1:
            errors.append("generation.max_questions must be positive")
        if self.chunk_chars < 80:
            errors.append("generation.chunk_chars must be at least 80")
        if not 0 <= self.chunk_overlap_chars < self.chunk_chars:
            errors.append("generation.chunk_overlap_chars must be in [0, chunk_chars)")
        if self.batch_chunks < 1:
            errors.append("generation.batch_chunks must be positive")
        if self.min_topic_questions < 0:
            errors.append("generation.min_topic_questions must be non-negative")
        if not 0 < self.max_topic_share <= 1:
            errors.append("generation.max_topic_share must be in (0, 1]")
        if not 0 <= self.unanswerable_ratio < 1:
            errors.append("generation.unanswerable_ratio must be in [0, 1)")
        if not 0 <= self.user_variation_ratio < 1:
            errors.append("generation.user_variation_ratio must be in [0, 1)")
        if self.unanswerable_ratio + self.user_variation_ratio >= 1:
            errors.append("unanswerable_ratio + user_variation_ratio must be below 1")
        if not 0 <= self.ambiguous_variation_share <= 1:
            errors.append("generation.ambiguous_variation_share must be in [0, 1]")
        if self.max_candidate_rounds < 1:
            errors.append("generation.max_candidate_rounds must be positive")
        if self.request_timeout_seconds <= 0:
            errors.append("evaluation.request_timeout_seconds must be positive")
        if self.max_retries < 0:
            errors.append("evaluation.max_retries must be non-negative")
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            errors.append("runtime.log_level must be DEBUG, INFO, WARNING, or ERROR")
        if errors:
            raise ValueError("Invalid configuration: " + "; ".join(errors))
        return self


def _get(data: dict[str, Any], section: str, key: str, default: Any) -> Any:
    section_value = data.get(section, {})
    if not isinstance(section_value, dict):
        raise ValueError(f"Configuration section [{section}] must be a table")
    return section_value.get(key, default)


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def load_settings(config_path: Path, require_api_key: bool = True) -> Settings:
    config_path = config_path.resolve()
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    load_dotenv(config_path.parent / ".env")
    key_name = str(_get(data, "gemini", "api_key_env", "GEMINI_API_KEY"))
    api_key = os.getenv(key_name, "")
    if require_api_key and not api_key:
        raise ValueError(
            f"Missing {key_name}. Copy .env.example to {config_path.parent / '.env'} "
            "and set the key there."
        )
    return Settings(
        api_key=api_key,
        generation_model=str(_get(data, "gemini", "generation_model", "gemini-2.5-flash")),
        judge_model=str(_get(data, "gemini", "judge_model", "gemini-2.5-flash")),
        max_questions=int(_get(data, "generation", "max_questions", 30)),
        chunk_chars=int(_get(data, "generation", "chunk_chars", 12000)),
        chunk_overlap_chars=int(_get(data, "generation", "chunk_overlap_chars", 800)),
        batch_chunks=int(_get(data, "generation", "batch_chunks", 8)),
        unanswerable_ratio=float(_get(data, "generation", "unanswerable_ratio", 0.1)),
        user_variation_ratio=float(_get(data, "generation", "user_variation_ratio", 0.3)),
        ambiguous_variation_share=float(_get(data, "generation", "ambiguous_variation_share", 0.33)),
        max_candidate_rounds=int(_get(data, "generation", "max_candidate_rounds", 3)),
        min_topic_questions=int(_get(data, "generation", "min_topic_questions", 1)),
        max_topic_share=float(_get(data, "generation", "max_topic_share", 0.35)),
        request_timeout_seconds=int(_get(data, "evaluation", "request_timeout_seconds", 60)),
        max_retries=int(_get(data, "evaluation", "max_retries", 2)),
        progress_enabled=_as_bool(_get(data, "runtime", "progress_enabled", True), "runtime.progress_enabled"),
        log_file=str(_get(data, "runtime", "log_file", "")),
        log_level=str(_get(data, "runtime", "log_level", "INFO")),
    ).validate()
