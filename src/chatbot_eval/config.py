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
    gemini_transport: str
    apigee_api_key: str
    apigee_base_url: str
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
    stable_question_ids: bool
    question_type_targets: dict[str, float]
    min_topic_questions: int
    max_topic_share: float
    cache_enabled: bool
    cache_directory: str
    request_timeout_seconds: int
    max_retries: int
    chatbot_max_retries: int
    chatbot_retry_base_seconds: float
    chatbot_pacing_seconds: float
    chatbot_max_response_bytes: int
    chatbot_require_json_content_type: bool
    progress_enabled: bool
    log_file: str
    log_level: str

    def validate(self) -> "Settings":
        errors: list[str] = []
        if self.gemini_transport not in {"direct", "apigee"}:
            errors.append("gemini.transport must be direct or apigee")
        if self.gemini_transport == "apigee" and not (
            self.apigee_base_url.startswith("https://") and "/ai_gateway/" in self.apigee_base_url
        ):
            errors.append("gemini.apigee_base_url must be an HTTPS AI Gateway client URL")
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
        if not self.cache_directory.strip():
            errors.append("cache.directory must not be empty")
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
        allowed_types = {"basic_knowledge", "topic_integration", "document_wide", "cross_document"}
        unknown_types = sorted(set(self.question_type_targets) - allowed_types)
        if unknown_types:
            errors.append(f"generation.question_type_targets has unsupported types: {unknown_types}")
        if any(value < 0 for value in self.question_type_targets.values()):
            errors.append("generation.question_type_targets values must be non-negative")
        if self.question_type_targets and abs(sum(self.question_type_targets.values()) - 1.0) > 0.001:
            errors.append("generation.question_type_targets values must sum to 1")
        if self.request_timeout_seconds <= 0:
            errors.append("evaluation.request_timeout_seconds must be positive")
        if self.max_retries < 0:
            errors.append("evaluation.max_retries must be non-negative")
        if self.chatbot_max_retries < 0:
            errors.append("evaluation.chatbot_max_retries must be non-negative")
        if self.chatbot_retry_base_seconds < 0 or self.chatbot_pacing_seconds < 0:
            errors.append("chatbot retry and pacing durations must be non-negative")
        if self.chatbot_max_response_bytes < 1024:
            errors.append("evaluation.chatbot_max_response_bytes must be at least 1024")
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
    transport = str(_get(data, "gemini", "transport", "direct")).strip().lower()
    key_name = str(_get(data, "gemini", "api_key_env", "GEMINI_API_KEY"))
    apigee_key_name = str(_get(data, "gemini", "apigee_api_key_env", "APIGEE_API_KEY"))
    api_key = os.getenv(key_name, "")
    apigee_api_key = os.getenv(apigee_key_name, "")
    selected_key_name = apigee_key_name if transport == "apigee" else key_name
    selected_key = apigee_api_key if transport == "apigee" else api_key
    if require_api_key and not selected_key:
        raise ValueError(
            f"Missing {selected_key_name}. Copy .env.example to {config_path.parent / '.env'} "
            "and set the key there."
        )
    return Settings(
        api_key=api_key,
        gemini_transport=transport,
        apigee_api_key=apigee_api_key,
        apigee_base_url=str(_get(
            data, "gemini", "apigee_base_url",
            "https://preprod.apigee.digital.idf.il/ai_gateway/v1/hr",
        )).strip().rstrip("/"),
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
        stable_question_ids=_as_bool(
            _get(data, "generation", "stable_question_ids", True),
            "generation.stable_question_ids",
        ),
        question_type_targets={
            str(key): float(value)
            for key, value in dict(_get(data, "generation", "question_type_targets", {})).items()
        },
        min_topic_questions=int(_get(data, "generation", "min_topic_questions", 1)),
        max_topic_share=float(_get(data, "generation", "max_topic_share", 0.35)),
        cache_enabled=_as_bool(_get(data, "cache", "enabled", True), "cache.enabled"),
        cache_directory=str(_get(data, "cache", "directory", ".chatbot_eval_cache")),
        request_timeout_seconds=int(_get(data, "evaluation", "request_timeout_seconds", 60)),
        max_retries=int(_get(data, "evaluation", "max_retries", 2)),
        chatbot_max_retries=int(_get(data, "evaluation", "chatbot_max_retries", 2)),
        chatbot_retry_base_seconds=float(_get(data, "evaluation", "chatbot_retry_base_seconds", 0.5)),
        chatbot_pacing_seconds=float(_get(data, "evaluation", "chatbot_pacing_seconds", 0.0)),
        chatbot_max_response_bytes=int(_get(data, "evaluation", "chatbot_max_response_bytes", 5_000_000)),
        chatbot_require_json_content_type=_as_bool(
            _get(data, "evaluation", "chatbot_require_json_content_type", True),
            "evaluation.chatbot_require_json_content_type",
        ),
        progress_enabled=_as_bool(_get(data, "runtime", "progress_enabled", True), "runtime.progress_enabled"),
        log_file=str(_get(data, "runtime", "log_file", "")),
        log_level=str(_get(data, "runtime", "log_level", "INFO")),
    ).validate()
