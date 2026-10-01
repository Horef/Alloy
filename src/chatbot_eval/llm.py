from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Protocol, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from .response_errors import bounded_retry_delay

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

# Bump when the request configuration or response parsing changes what a structured call returns.
STRUCTURED_CALL_CONTRACT = 1


class EmbeddingCountMismatch(RuntimeError):
    """The embedding model returned a different number of vectors than inputs."""


class StructuredLLM(Protocol):
    def generate(
        self, prompt: str, schema: type[T], model: str,
        *, required_fields: dict[str, int] | None = None,
    ) -> T: ...


class CachedStructuredLLM:
    """Persistent, content-addressed store of structured responses, shared by all runs and commands.

    Only used with a fixed seed: then an identical model, schema, and prompt yield the same response,
    so serving it from disk loses nothing while re-runs, partly changed plans, and re-judging the
    same answers cost no tokens for their unchanged calls. The file name carries the transport,
    endpoint, seed, and call contract; the entry key covers model, schema, and prompt.
    """

    def __init__(self, llm, path: Path, *, read: bool = True):
        self._llm, self._path, self._read = llm, path, read
        self._entries: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.hits = self.misses = 0
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    self._entries[item["key"]] = item["response"]
                except (ValueError, KeyError, TypeError):
                    logger.warning("llm_call_cache_line_invalid path=%s", path)

    @staticmethod
    def _key(prompt: str, schema, model: str, required_fields: dict[str, int] | None) -> str:
        identity = {"model": model, "schema": schema.model_json_schema(), "prompt": prompt,
                    "required_fields": required_fields or {}}
        encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def generate(self, prompt: str, schema: type[T], model: str, *, required_fields: dict[str, int] | None = None) -> T:
        key = self._key(prompt, schema, model, required_fields)
        if self._read:
            with self._lock:
                cached = self._entries.get(key)
            if cached is not None:
                try:
                    result = schema.model_validate(cached)
                    with self._lock:
                        self.hits += 1
                    return result
                except ValidationError:
                    logger.warning("llm_call_cache_entry_invalid schema=%s", schema.__name__)
        if required_fields:
            result = self._llm.generate(prompt, schema, model, required_fields=required_fields)
        else:
            result = self._llm.generate(prompt, schema, model)
        payload = result.model_dump(mode="json")
        with self._lock:
            self.misses += 1
            self._entries[key] = payload
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "response": payload}, ensure_ascii=False) + "\n")
        return result

    def embed(self, texts: list[str], model: str, dimensions: int | None = None) -> list[list[float]]:
        if dimensions:
            return self._llm.embed(texts, model, dimensions)
        return self._llm.embed(texts, model)

    def summary(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "stored": len(self._entries)}


def _augment_required(json_schema: dict, required_fields: dict[str, int] | None) -> dict:
    """Promote conditionally-required fields to schema-level ``required``.

    Pydantic omits fields that have defaults (``default_factory=list``) from ``required``, which
    lets structured-output models drop them. When a caller knows a field must be present for this
    request (e.g. improvement-mode regression cases), it can pass the field name mapped to the
    minimum number of list items. This mutates a copy so the model class schema is untouched.
    """
    if not required_fields:
        return json_schema
    schema = copy.deepcopy(json_schema)
    properties = schema.get("properties", {})
    required = list(schema.get("required", []))
    for field, minimum in required_fields.items():
        if field not in properties:
            continue
        if field not in required:
            required.append(field)
        if minimum > 0 and properties[field].get("type") == "array":
            properties[field]["minItems"] = max(minimum, properties[field].get("minItems", 0))
    schema["required"] = required
    return schema


class GeminiStructuredLLM:
    """Bound application attempts and retry sleeps, not a total wall-clock deadline.

    The SDK enforces its request timeout; an accepted request may be retried after
    a transport failure. Application attempts are at most max_retries + 1.
    """
    def __init__(
        self,
        api_key: str,
        max_retries: int = 2,
        *,
        transport: str = "direct",
        apigee_api_key: str = "",
        apigee_base_url: str = "",
        request_timeout_seconds: int = 120,
        max_retry_delay_seconds: float = 60.0,
        seed: int | None = None,
    ):
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be finite and positive")
        if not math.isfinite(max_retry_delay_seconds) or max_retry_delay_seconds < 0:
            raise ValueError("max_retry_delay_seconds must be finite and nonnegative")
        self._max_retry_delay_seconds = max_retry_delay_seconds
        self._seed = seed
        http_options = types.HttpOptions(timeout=request_timeout_seconds * 1000)
        if transport == "direct":
            self._client = genai.Client(api_key=api_key, http_options=http_options)
        elif transport == "apigee":
            if not apigee_api_key or not apigee_base_url:
                raise ValueError("Apigee transport requires an API key and base URL")
            self._client = genai.Client(
                api_key="apigee-placeholder",
                vertexai=True,
                project="",
                location="",
                http_options=types.HttpOptions(
                    api_version="v1",
                    base_url=apigee_base_url.rstrip("/"),
                    headers={"x-apikey": apigee_api_key},
                    timeout=request_timeout_seconds * 1000,
                ),
            )
        else:
            raise ValueError(f"Unsupported Gemini transport: {transport!r}")
        self._transport = transport
        self._max_retries = max_retries

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, ValidationError):
            return False
        if isinstance(exc, errors.APIError):
            return exc.code in {408, 409, 429, 500, 502, 503, 504}
        # The optional transport backends the SDK re-exports (httpx, requests) vary by SDK version,
        # so collect only the classes that actually exist. Referencing a missing attribute directly
        # used to raise AttributeError from inside the retry check and mask the real error.
        network_errors: list[type] = [TimeoutError, ConnectionError]
        for backend, names in (
            ("httpx", ("TimeoutException", "TransportError")),
            ("requests", ("Timeout", "ConnectionError")),
        ):
            module = getattr(errors, backend, None)
            for name in names:
                exc_type = getattr(module, name, None)
                if isinstance(exc_type, type):
                    network_errors.append(exc_type)
        return isinstance(exc, tuple(network_errors))

    @staticmethod
    def _retry_delay(exc: Exception, attempt: int, cap: float = 60.0) -> float:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        return bounded_retry_delay(headers.get("Retry-After") if headers else None, attempt, 1.0, cap)

    @staticmethod
    def _log_quota(response, model: str) -> None:
        http_response = getattr(response, "sdk_http_response", None)
        headers = getattr(http_response, "headers", None)
        if not headers:
            return
        metric = headers.get("x-selected-metric")
        if not metric:
            return
        request_used = headers.get(f"x-request-{metric}-used")
        daily_used = headers.get(f"x-quota-{metric}-used")
        logger.info(
            "gemini_quota model=%s metric=%s request_used=%s daily_limit=%s daily_used=%s remaining=%s",
            model, metric, request_used, headers.get("x-quota-limit"), daily_used,
            headers.get(f"x-quota-{metric}-remaining"),
        )

    def generate(
        self, prompt: str, schema: type[T], model: str,
        *, required_fields: dict[str, int] | None = None,
    ) -> T:
        response_json_schema = _augment_required(schema.model_json_schema(), required_fields)

        def call() -> T:
            response = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=response_json_schema,
                    # A fixed seed makes identical prompts return (near-)identical responses.
                    seed=getattr(self, "_seed", None),
                    # This pipeline never exposes tools to the model. Disable AFC explicitly
                    # so the SDK does not initialize its function-calling loop or log its
                    # default maximum-remote-calls message.
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini returned an empty structured response")
            self._log_quota(response, model)
            return schema.model_validate_json(response.text)

        return self._with_retries(call, model)

    def embed(self, texts: list[str], model: str, dimensions: int | None = None) -> list[list[float]]:
        """Embed ``texts`` with an embedding model through the configured transport."""
        config = types.EmbedContentConfig(output_dimensionality=dimensions) if dimensions else None

        def call() -> list[list[float]]:
            response = self._client.models.embed_content(model=model, contents=texts, config=config)
            return [list(item.values or []) for item in (response.embeddings or [])]

        vectors = self._with_retries(call, model)
        # Checked outside the retry loop: a count mismatch is deterministic model behavior, not a failure.
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise EmbeddingCountMismatch("Gemini returned an incomplete embedding response")
        return vectors

    def _with_retries(self, call, model: str):
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return call()
            except Exception as exc:
                last_error = exc
                retryable = self._is_retryable(exc)
                if attempt < self._max_retries and retryable:
                    logger.warning(
                        "gemini_call_retry model=%s attempt=%d max_attempts=%d error_type=%s",
                        model, attempt + 1, self._max_retries + 1, type(exc).__name__,
                    )
                    time.sleep(self._retry_delay(exc, attempt, getattr(self, "_max_retry_delay_seconds", 60.0)))
                else:
                    logger.error(
                        "gemini_call_failed model=%s attempts=%d retryable=%s elapsed_seconds=%.3f error_type=%s",
                        model, attempt + 1, retryable, time.perf_counter() - started, type(exc).__name__,
                    )
                    break
        assert last_error is not None
        raise last_error
