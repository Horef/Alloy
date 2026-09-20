from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .artifacts import atomic_write_text, file_sha256
from .documents import SUPPORTED_SUFFIXES, Chunk, load_chunks
from .graph import GraphBundle
from .models import (
    EvaluationInsights,
    EvaluationRecord,
    NodeSignals,
    PromptPackage,
    SilverQuestion,
    TopicCandidate,
    TopicMap,
)
from .prompt_policy import POLICY_VERSION, assemble_prompt, strip_policy

logger = logging.getLogger(__name__)

# v2: added per-document node-signal and knowledge-graph cache kinds for KG-based generation.
CACHE_SCHEMA_VERSION = 2


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

    def __init__(self, directory: Path, *, enabled: bool = True, refresh: bool = False, model_identity: dict | None = None):
        self.model_identity = model_identity or {}
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
            "model": model, "model_identity": self.model_identity,
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

    def load_node_signals(
        self,
        chunks_by_document: dict[str, list[Chunk]],
        *,
        model: str,
        transport: str,
        batch_chunks: int,
        implementation_sha256: str,
        extract: Callable[[str, list[Chunk]], list[NodeSignals]],
        max_concurrency: int = 1,
    ) -> dict[str, NodeSignals]:
        """Return per-chunk signals, extracting only documents whose content changed.

        This is the incremental core of KG rebuilds. Each document's signals are cached under a key
        derived from that document's chunk ids and text plus the extractor implementation, model,
        transport, and batch size. Unchanged documents hit the cache and skip the LLM entirely;
        added or edited documents miss and are extracted; deleted documents simply stop being
        requested, so their stale entries are ignored (content-addressed, never overwritten).
        Returns a flat ``chunk_id -> NodeSignals`` map across all documents.

        Cache-missed documents are independent, so with ``max_concurrency > 1`` they are extracted
        in a thread pool. Results are still merged deterministically in sorted document order, and
        each document's cache file is written by its own worker, so concurrency never affects the
        cached content or the returned mapping.
        """
        keys: dict[str, str] = {}
        cache_paths: dict[str, Path] = {}
        missed: list[str] = []
        hits = 0
        cached_signals: dict[str, list[NodeSignals]] = {}
        for document, chunks in sorted(chunks_by_document.items()):
            key = _json_hash({
                "schema_version": CACHE_SCHEMA_VERSION,
                "signals_implementation_sha256": implementation_sha256,
                "document": document,
                "chunks": [{"id": c.id, "text": c.text} for c in chunks],
                "model": model, "model_identity": self.model_identity,
                "transport": transport, "batch_chunks": batch_chunks,
            })
            keys[document] = key
            cache_paths[document] = self.directory / "node_signals" / f"{key}.json"
            cached = self._read_node_signals(cache_paths[document], key) if (self.enabled and not self.refresh) else None
            if cached is not None:
                hits += 1
                cached_signals[document] = cached
            else:
                missed.append(document)

        def _extract_and_store(document: str) -> tuple[str, list[NodeSignals]]:
            document_signals = extract(document, chunks_by_document[document])
            if self.enabled:
                atomic_write_text(cache_paths[document], json.dumps({
                    "schema_version": CACHE_SCHEMA_VERSION, "kind": "node_signals", "key": keys[document],
                    "signals": [signal.model_dump(mode="json") for signal in document_signals],
                }, ensure_ascii=False, separators=(",", ":")))
            return document, document_signals

        extracted: dict[str, list[NodeSignals]] = {}
        if missed and max_concurrency > 1:
            with ThreadPoolExecutor(max_workers=min(max_concurrency, len(missed))) as executor:
                for document, document_signals in executor.map(_extract_and_store, missed):
                    extracted[document] = document_signals
        else:
            for document in missed:
                _, document_signals = _extract_and_store(document)
                extracted[document] = document_signals

        # Merge in deterministic (sorted) document order regardless of extraction order.
        signals: dict[str, NodeSignals] = {}
        for document in sorted(chunks_by_document):
            for signal in cached_signals.get(document, extracted.get(document, [])):
                signals[signal.chunk_id] = signal
        misses = len(missed)
        if not self.enabled:
            self.events["node_signals"] = "disabled"
        else:
            self.events["node_signals"] = f"documents={len(chunks_by_document)} reused={hits} extracted={misses}"
        logger.info(
            "node_signals documents=%d reused=%d extracted=%d refresh=%s",
            len(chunks_by_document), hits, misses, self.refresh,
        )
        return signals

    def load_graph(
        self,
        chunks: list[Chunk],
        signals: dict[str, NodeSignals],
        *,
        model: str,
        transport: str,
        keyphrase_overlap_threshold: float,
        max_topics: int,
        min_cluster_nodes: int,
        implementation_sha256: str,
        build: Callable[[], GraphBundle],
        clustering_params: dict | None = None,
    ) -> GraphBundle:
        """Cache the assembled graph (edges) and labeled topics.

        Edge building is deterministic, but topic labeling uses one LLM call, so the assembled
        result is cached. The key covers the ordered chunk set, every chunk's signals, the edge and
        clustering parameters, the model/transport, and the graph-build implementation. Any change
        to a document's signals (i.e. an edited or added document) changes the key and forces a
        rebuild; unchanged corpora reuse the graph and skip the labeling call.
        """
        from .graph import GraphEdge, GraphNode, GraphTopic, KnowledgeGraph

        key = _json_hash({
            "schema_version": CACHE_SCHEMA_VERSION,
            "graph_implementation_sha256": implementation_sha256,
            "chunks": [{"id": c.id, "file": c.file, "location": c.location, "text": c.text} for c in chunks],
            "signals": {
                cid: signals[cid].model_dump(mode="json")
                for cid in sorted(signals)
            },
            "model": model, "model_identity": self.model_identity, "transport": transport,
            "keyphrase_overlap_threshold": keyphrase_overlap_threshold,
            "max_topics": max_topics, "min_cluster_nodes": min_cluster_nodes,
            "clustering_params": clustering_params or {},
        })
        path = self.directory / "graph" / f"{key}.json"
        if self.enabled and not self.refresh:
            payload = self._read_payload(path, "graph", key)
            if payload is not None:
                try:
                    graph_data = payload["graph"]
                    nodes = [GraphNode(**node) for node in graph_data["nodes"]]
                    edges = [GraphEdge(source_id=e["source_id"], target_id=e["target_id"],
                                       type=e["type"], weight=e["weight"], shared=tuple(e["shared"]))
                             for e in graph_data["edges"]]
                    topics = [GraphTopic(**topic) for topic in graph_data["topics"]]
                    self.events["graph"] = "hit"
                    logger.info("cache_hit kind=graph key=%s path=%s", key[:12], path)
                    return GraphBundle(KnowledgeGraph(nodes=nodes, edges=edges), topics)
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("cache_invalid kind=graph path=%s error=%s", path, exc)

        bundle = build()
        self.events["graph"] = "refresh" if self.enabled and self.refresh else "miss"
        if self.enabled:
            atomic_write_text(path, json.dumps({
                "schema_version": CACHE_SCHEMA_VERSION, "kind": "graph", "key": key,
                "graph": {
                    "nodes": [node.__dict__ for node in bundle.graph.nodes],
                    "edges": [
                        {"source_id": e.source_id, "target_id": e.target_id, "type": e.type,
                         "weight": e.weight, "shared": list(e.shared)}
                        for e in bundle.graph.edges
                    ],
                    "topics": [topic.__dict__ for topic in bundle.topics],
                },
            }, ensure_ascii=False, separators=(",", ":")))
            logger.info("cache_write kind=graph key=%s path=%s", key[:12], path)
        return bundle

    @staticmethod
    def _read_node_signals(path: Path, expected_key: str) -> list[NodeSignals] | None:
        payload = CorpusAnalysisCache._read_payload(path, "node_signals", expected_key)
        if payload is None:
            return None
        try:
            values = payload["signals"]
            if not isinstance(values, list):
                raise TypeError("signals must be a list")
            return [NodeSignals.model_validate(item) for item in values]
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("cache_invalid kind=node_signals path=%s error=%s", path, exc)
            return None

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
            "model": model, "model_identity": self.model_identity, "transport": transport, "batch_size": batch_size,
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
            "implementation_sha256": implementation_sha256, "model": model, "model_identity": self.model_identity,
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
        instruction_profile: str = "guided", answer_policy: str = "balanced",
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
            "instruction_profile": instruction_profile, "answer_policy": answer_policy,
            "model": model, "model_identity": self.model_identity,
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
                    raw_package = payload["prompt_package"]
                    if not isinstance(raw_package, dict):
                        raise TypeError("prompt_package must be an object")
                    if raw_package.get("instruction_profile") != instruction_profile:
                        raise ValueError("instruction profile mismatch")
                    if raw_package.get("answer_policy") != answer_policy:
                        raise ValueError("answer policy mismatch")
                    if raw_package.get("policy_version") != POLICY_VERSION:
                        raise ValueError("response policy version mismatch")
                    package = PromptPackage.model_validate(raw_package)
                    domain_prompt = strip_policy(package.system_prompt_hebrew)
                    if not domain_prompt:
                        raise ValueError("domain prompt is empty after removing response policy")
                    if package.system_prompt_hebrew != assemble_prompt(
                        domain_prompt, instruction_profile, answer_policy,
                    ):
                        raise ValueError("response policy assembly mismatch")
                    self.events["prompt_package"] = "hit"
                    logger.info("cache_hit kind=prompt_package key=%s path=%s", key[:12], path)
                    return package
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("cache_invalid kind=prompt_package path=%s error=%s", path, exc)

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
