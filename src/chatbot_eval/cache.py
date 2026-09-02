from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

from .artifacts import atomic_write_text, file_sha256
from .documents import SUPPORTED_SUFFIXES, Chunk, load_chunks
from .models import EvaluationInsights, EvaluationRecord, PromptPackage, SilverQuestion, TopicCandidate, TopicMap

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _document_inventory(root: Path) -> list[dict[str, str]]:
    resolved = root.resolve()
    files = [
        path for path in sorted(resolved.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not files:
        raise ValueError(f"No supported documents found under {root}")
    return [
        {"path": str(path.relative_to(resolved)), "sha256": file_sha256(path)}
        for path in files
    ]


class CorpusAnalysisCache:
    """Content-addressed local cache for extracted chunks and discovered topic maps."""

    def __init__(self, directory: Path, *, enabled: bool = True, refresh: bool = False):
        self.directory = directory
        self.enabled = enabled
        self.refresh = refresh
        self.events: dict[str, str] = {}

    def load_chunks(
        self,
        root: Path,
        chunk_chars: int,
        overlap_chars: int,
        progress_enabled: bool = False,
    ) -> tuple[list[Chunk], str]:
        inventory = _document_inventory(root)
        key = _json_hash({
            "schema_version": CACHE_SCHEMA_VERSION,
            "chunker_implementation_sha256": file_sha256(Path(load_chunks.__code__.co_filename)),
            "documents": inventory,
            "chunk_chars": chunk_chars,
            "overlap_chars": overlap_chars,
        })
        path = self.directory / "chunks" / f"{key}.json"
        if self.enabled and not self.refresh:
            cached = self._read_chunks(path, key)
            if cached is not None:
                self.events["chunks"] = "hit"
                logger.info("cache_hit kind=chunks key=%s path=%s", key[:12], path)
                return cached, key

        chunks = load_chunks(root, chunk_chars, overlap_chars, progress_enabled)
        self.events["chunks"] = "refresh" if self.enabled and self.refresh else "miss"
        if self.enabled:
            payload = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "kind": "chunks",
                "key": key,
                "chunks": [chunk.__dict__ for chunk in chunks],
            }
            atomic_write_text(path, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            logger.info("cache_write kind=chunks key=%s path=%s", key[:12], path)
        else:
            self.events["chunks"] = "disabled"
        return chunks, key

    def load_topics(
        self,
        chunks: list[Chunk],
        chunk_key: str,
        *,
        model: str,
        transport: str,
        batch_chunks: int,
        implementation_sha256: str,
        discover: Callable[[], list[TopicCandidate]],
    ) -> list[TopicCandidate]:
        key = _json_hash({
            "schema_version": CACHE_SCHEMA_VERSION,
            "topic_discovery_implementation_sha256": implementation_sha256,
            "chunk_key": chunk_key,
            "chunk_ids": [chunk.id for chunk in chunks],
            "model": model,
            "transport": transport,
            "batch_chunks": batch_chunks,
        })
        path = self.directory / "topics" / f"{key}.json"
        if self.enabled and not self.refresh:
            cached = self._read_topics(path, key)
            if cached is not None:
                self.events["topics"] = "hit"
                logger.info("cache_hit kind=topics key=%s path=%s", key[:12], path)
                return cached

        topics = discover()
        self.events["topics"] = "refresh" if self.enabled and self.refresh else "miss"
        if self.enabled:
            payload = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "kind": "topics",
                "key": key,
                "topics": [topic.model_dump(mode="json") for topic in topics],
            }
            atomic_write_text(path, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            logger.info("cache_write kind=topics key=%s path=%s", key[:12], path)
        else:
            self.events["topics"] = "disabled"
        return topics

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "refresh_requested": self.refresh,
            "directory": str(self.directory.resolve()),
            **self.events,
        }

    def load_question_topics(
        self,
        questions: list[SilverQuestion],
        *,
        model: str,
        transport: str,
        batch_size: int,
        implementation_sha256: str,
        discover: Callable[[], dict[str, str]],
    ) -> dict[str, str]:
        key = _json_hash({
            "schema_version": CACHE_SCHEMA_VERSION,
            "implementation_sha256": implementation_sha256,
            "questions": [{"id": q.id, "question": q.question} for q in questions],
            "model": model, "transport": transport, "batch_size": batch_size,
        })
        path = self.directory / "question_topics" / f"{key}.json"
        if self.enabled and not self.refresh:
            payload = self._read_payload(path, "question_topics", key)
            if payload is not None and isinstance(payload.get("assignments"), dict):
                assignments = payload["assignments"]
                if all(isinstance(k, str) and isinstance(v, str) for k, v in assignments.items()):
                    self.events["question_topics"] = "hit"
                    return assignments
        assignments = discover()
        self.events["question_topics"] = "refresh" if self.enabled and self.refresh else "miss"
        if self.enabled:
            atomic_write_text(path, json.dumps({
                "schema_version": CACHE_SCHEMA_VERSION, "kind": "question_topics", "key": key,
                "assignments": assignments,
            }, ensure_ascii=False, separators=(",", ":")))
        else:
            self.events["question_topics"] = "disabled"
        return assignments

    def load_insights(
        self,
        records: list[EvaluationRecord],
        *,
        model: str,
        transport: str,
        max_prompt_chars: int,
        implementation_sha256: str,
        discover: Callable[[], EvaluationInsights],
    ) -> EvaluationInsights:
        records_hash = _json_hash([record.model_dump(mode="json") for record in records])
        key = _json_hash({
            "schema_version": CACHE_SCHEMA_VERSION, "records_sha256": records_hash,
            "implementation_sha256": implementation_sha256, "model": model,
            "transport": transport, "max_prompt_chars": max_prompt_chars,
        })
        path = self.directory / "insights" / f"{key}.json"
        if self.enabled and not self.refresh:
            payload = self._read_payload(path, "insights", key)
            if payload is not None:
                try:
                    insights = EvaluationInsights.model_validate(payload["insights"])
                    self.events["insights"] = "hit"
                    return insights
                except (KeyError, TypeError, ValueError):
                    pass
        insights = discover()
        self.events["insights"] = "refresh" if self.enabled and self.refresh else "miss"
        if self.enabled:
            atomic_write_text(path, json.dumps({
                "schema_version": CACHE_SCHEMA_VERSION, "kind": "insights", "key": key,
                "insights": insights.model_dump(mode="json"),
            }, ensure_ascii=False, separators=(",", ":")))
        else:
            self.events["insights"] = "disabled"
        return insights

    def load_prompt_package(
        self,
        *,
        chunk_key: str,
        topics: list[TopicCandidate],
        assistant_name: str,
        audience: str,
        previous_records: list[EvaluationRecord],
        previous_insights: EvaluationInsights | None,
        current_prompt: str,
        model: str,
        transport: str,
        document_context_chars: int,
        evaluation_context_chars: int,
        auxiliary_context_chars: int,
        implementation_sha256: str,
        generate: Callable[[], PromptPackage],
    ) -> PromptPackage:
        key = _json_hash({
            "schema_version": CACHE_SCHEMA_VERSION,
            "implementation_sha256": implementation_sha256,
            "chunk_key": chunk_key,
            "topics": [topic.model_dump(mode="json") for topic in topics],
            "assistant_name": assistant_name,
            "audience": audience,
            "previous_records_sha256": _json_hash([
                record.model_dump(mode="json") for record in previous_records
            ]),
            "previous_insights": previous_insights.model_dump(mode="json") if previous_insights else None,
            "current_prompt_sha256": _json_hash(current_prompt),
            "model": model,
            "transport": transport,
            "document_context_chars": document_context_chars,
            "evaluation_context_chars": evaluation_context_chars,
            "auxiliary_context_chars": auxiliary_context_chars,
        })
        path = self.directory / "prompt_packages" / f"{key}.json"
        if self.enabled and not self.refresh:
            payload = self._read_payload(path, "prompt_package", key)
            if payload is not None:
                try:
                    package = PromptPackage.model_validate(payload["prompt_package"])
                    self.events["prompt_package"] = "hit"
                    logger.info("cache_hit kind=prompt_package key=%s path=%s", key[:12], path)
                    return package
                except (KeyError, TypeError, ValueError):
                    pass

        package = generate()
        self.events["prompt_package"] = "refresh" if self.enabled and self.refresh else "miss"
        if self.enabled:
            atomic_write_text(path, json.dumps({
                "schema_version": CACHE_SCHEMA_VERSION,
                "kind": "prompt_package",
                "key": key,
                "prompt_package": package.model_dump(mode="json"),
            }, ensure_ascii=False, separators=(",", ":")))
            logger.info("cache_write kind=prompt_package key=%s path=%s", key[:12], path)
        else:
            self.events["prompt_package"] = "disabled"
        return package

    @staticmethod
    def _read_chunks(path: Path, expected_key: str) -> list[Chunk] | None:
        payload = CorpusAnalysisCache._read_payload(path, "chunks", expected_key)
        if payload is None:
            return None
        try:
            values = payload["chunks"]
            if not isinstance(values, list):
                raise TypeError("chunks must be a list")
            if any(
                not isinstance(item, dict)
                or set(item) != {"id", "file", "location", "text"}
                or any(not isinstance(item[field], str) for field in ("id", "file", "location", "text"))
                for item in values
            ):
                raise TypeError("each chunk must contain exactly four string fields")
            return [Chunk(**item) for item in values]
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("cache_invalid kind=chunks path=%s error=%s", path, exc)
            return None

    @staticmethod
    def _read_topics(path: Path, expected_key: str) -> list[TopicCandidate] | None:
        payload = CorpusAnalysisCache._read_payload(path, "topics", expected_key)
        if payload is None:
            return None
        try:
            return TopicMap(topics=payload["topics"]).topics
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("cache_invalid kind=topics path=%s error=%s", path, exc)
            return None

    @staticmethod
    def _read_payload(path: Path, kind: str, expected_key: str) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("payload must be an object")
            if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
                raise ValueError("schema version mismatch")
            if payload.get("kind") != kind or payload.get("key") != expected_key:
                raise ValueError("cache identity mismatch")
            return payload
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("cache_invalid kind=%s path=%s error=%s", kind, path, exc)
            return None
