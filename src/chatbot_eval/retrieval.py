"""Corpus retrieval for question-targeted evidence and duplicate detection.

Lexical retrieval is BM25 over Hebrew-aware tokens: each token is canonicalized like graph entities
(NFKC, niqqud and gershayim removed) and an attached one-letter prefix variant is indexed alongside
it, so "לחייל" also matches "חייל". IDF weighting keeps frequent function words ("של", "את") from
dominating the ranking. When an embedding model is configured, dense cosine ranking is fused with
BM25 by reciprocal rank fusion; embedding failures degrade to lexical-only retrieval.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, Sequence

from .documents import Chunk
from .graph import _strip_one_prefix, normalize_entity
from .llm import EmbeddingCountMismatch

logger = logging.getLogger(__name__)

_HEBREW_START = re.compile(r"[\u05d0-\u05ea]")
_RRF_K = 60
_EMBED_MAX_CHARS = 8000


def search_tokens(text: str) -> list[str]:
    """Tokens for lexical search: canonical tokens plus their prefix-stripped Hebrew variants."""
    tokens: list[str] = []
    for token in normalize_entity(text).split():
        tokens.append(token)
        if _HEBREW_START.match(token):
            stripped = _strip_one_prefix(token)
            if stripped and stripped != token:
                tokens.append(stripped)
    return tokens


def dedup_tokens(text: str) -> set[str]:
    """One canonical form per token (prefix stripped when safe) for near-duplicate comparison."""
    result: set[str] = set()
    for token in normalize_entity(text).split():
        stripped = _strip_one_prefix(token) if _HEBREW_START.match(token) else None
        result.add(stripped or token)
    return result


class BM25:
    def __init__(self, documents: Sequence[list[str]], *, k1: float = 1.5, b: float = 0.75):
        self._k1, self._b = k1, b
        self._tf = [Counter(tokens) for tokens in documents]
        self._lengths = [len(tokens) for tokens in documents]
        self._average = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        document_frequency: Counter = Counter()
        for tf in self._tf:
            document_frequency.update(tf.keys())
        total = len(self._tf)
        self._idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        terms = [term for term in dict.fromkeys(query_tokens) if term in self._idf]
        results = []
        for tf, length in zip(self._tf, self._lengths):
            norm = self._k1 * (1 - self._b + self._b * length / self._average) if self._average else self._k1
            results.append(sum(
                self._idf[term] * tf[term] * (self._k1 + 1) / (tf[term] + norm)
                for term in terms if tf[term]
            ))
        return results


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class ModelEmbedder:
    """Adapts an LLM client exposing ``embed(texts, model)`` to the :class:`Embedder` protocol.

    Some models (e.g. multimodal ``gemini-embedding-2``) fold every input of one request into a
    single vector; when a batch returns the wrong count, this switches to one text per request,
    issued concurrently.
    """

    def __init__(
        self, client, model: str, batch_size: int = 100, concurrency: int = 4, dimensions: int | None = None,
    ):
        self._client, self._model, self._batch_size = client, model, batch_size
        self._concurrency = max(1, concurrency)
        self._dimensions = dimensions
        self._single = batch_size <= 1

    def _request(self, texts: list[str]) -> list[list[float]]:
        if self._dimensions:
            return self._client.embed(texts, self._model, self._dimensions)
        return self._client.embed(texts, self._model)

    def _embed_singly(self, texts: list[str]) -> list[list[float]]:
        with ThreadPoolExecutor(max_workers=min(self._concurrency, len(texts))) as pool:
            return [vectors[0] for vectors in pool.map(lambda text: self._request([text]), texts)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._single:
            return self._embed_singly(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                vectors.extend(self._request(batch))
            except EmbeddingCountMismatch:
                if len(batch) == 1:
                    raise
                logger.info("embedding_model_aggregates_inputs model=%s; switching to one text per request", self._model)
                self._single = True
                return vectors + self._embed_singly(texts[start:])
        return vectors


class CachedEmbedder:
    """Content-addressed embedding store (JSONL) so repeated and resumed runs reuse vectors."""

    def __init__(self, embedder: Embedder, path: Path | None):
        self._embedder, self._path = embedder, path
        self._vectors: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    self._vectors[item["sha256"]] = item["vector"]
                except (ValueError, KeyError, TypeError):
                    logger.warning("embedding_cache_line_invalid path=%s", path)

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        with self._lock:
            missing = list(dict.fromkeys(key for key in keys if key not in self._vectors))
        if missing:
            by_key = {key: text for key, text in zip(keys, texts)}
            # The network call runs outside the lock so concurrent callers are not serialized.
            fresh = self._embedder.embed([by_key[key] for key in missing])
            if len(fresh) != len(missing):
                raise ValueError("embedding model returned a different number of vectors than inputs")
            with self._lock:
                lines = []
                for key, vector in zip(missing, fresh):
                    if key in self._vectors:
                        continue
                    self._vectors[key] = [round(value, 6) for value in vector]
                    lines.append(json.dumps({"sha256": key, "vector": self._vectors[key]}))
                if self._path is not None and lines:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    with self._path.open("a", encoding="utf-8") as handle:
                        handle.write("\n".join(lines) + "\n")
        with self._lock:
            return [self._vectors[key] for key in keys]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class ChunkRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        self.chunks = chunks
        self._bm25 = BM25([search_tokens(chunk.text) for chunk in chunks])
        self._embedder = embedder
        self._chunk_vectors: list[list[float]] | None = None
        self._lock = threading.Lock()

    def _dense_order(self, query: str) -> list[int]:
        if self._embedder is None:
            return []
        try:
            with self._lock:
                if self._chunk_vectors is None:
                    self._chunk_vectors = self._embedder.embed([chunk.text[:_EMBED_MAX_CHARS] for chunk in self.chunks])
            query_vector = self._embedder.embed([query[:_EMBED_MAX_CHARS]])[0]
        except Exception as exc:
            logger.warning("embedding_retrieval_failed error_type=%s; using lexical ranking only", type(exc).__name__)
            self._embedder = None
            return []
        similarities = [cosine(query_vector, vector) for vector in self._chunk_vectors]
        return sorted(range(len(self.chunks)), key=lambda index: -similarities[index])

    def rank(self, query: str, limit: int) -> list[Chunk]:
        """Top ``limit`` chunks for ``query``; chunks with no lexical or dense signal are omitted."""
        if limit <= 0 or not self.chunks:
            return []
        scores = self._bm25.scores(search_tokens(query))
        lexical = [index for index in sorted(range(len(scores)), key=lambda i: -scores[i]) if scores[index] > 0]
        dense = self._dense_order(query)
        if not dense:
            return [self.chunks[index] for index in lexical[:limit]]
        fused: Counter = Counter()
        for order in (lexical, dense):
            for rank, index in enumerate(order):
                fused[index] += 1 / (_RRF_K + rank + 1)
        ranked = sorted(fused, key=lambda index: (-fused[index], index))
        return [self.chunks[index] for index in ranked[:limit]]
