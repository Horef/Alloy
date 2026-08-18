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
    min_topic_questions: int
    max_topic_share: float
    request_timeout_seconds: int
    max_retries: int
    progress_enabled: bool
    log_file: str
    log_level: str


def _get(data: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return data.get(section, {}).get(key, default)


def load_settings(config_path: Path, require_api_key: bool = True) -> Settings:
    config_path = config_path.resolve()
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    load_dotenv(config_path.parent / ".env")
    key_name = _get(data, "gemini", "api_key_env", "GEMINI_API_KEY")
    api_key = os.getenv(key_name, "")
    if require_api_key and not api_key:
        raise ValueError(
            f"Missing {key_name}. Copy .env.example to {config_path.parent / '.env'} "
            "and set the key there."
        )
    return Settings(
        api_key=api_key,
        generation_model=_get(data, "gemini", "generation_model", "gemini-2.5-flash"),
        judge_model=_get(data, "gemini", "judge_model", "gemini-2.5-flash"),
        max_questions=int(_get(data, "generation", "max_questions", 30)),
        chunk_chars=int(_get(data, "generation", "chunk_chars", 12000)),
        chunk_overlap_chars=int(_get(data, "generation", "chunk_overlap_chars", 800)),
        batch_chunks=int(_get(data, "generation", "batch_chunks", 8)),
        unanswerable_ratio=float(_get(data, "generation", "unanswerable_ratio", 0.1)),
        min_topic_questions=int(_get(data, "generation", "min_topic_questions", 1)),
        max_topic_share=float(_get(data, "generation", "max_topic_share", 0.35)),
        request_timeout_seconds=int(_get(data, "evaluation", "request_timeout_seconds", 60)),
        max_retries=int(_get(data, "evaluation", "max_retries", 2)),
        progress_enabled=bool(_get(data, "runtime", "progress_enabled", True)),
        log_file=str(_get(data, "runtime", "log_file", "")),
        log_level=str(_get(data, "runtime", "log_level", "INFO")),
    )

