"""Model providers beyond the Google GenAI SDK, and the factory that builds the client for a transport.

`build_llm(settings)` returns the structured-LLM client for `gemini.transport`: `GeminiStructuredLLM`
(`direct`, `apigee`), `OpenAICompatibleLLM` (`openai`: any server that implements the OpenAI
chat-completions and embeddings API, with a configurable auth header) or `BedrockStructuredLLM`
(`bedrock`: the AWS Bedrock Converse API through the optional `boto3` package, credentials from the
standard AWS chain). All three satisfy the `StructuredLLM` protocol plus `embed`, so the rest of the
package is unchanged. Retries, backoff, `Retry-After` and the retry deadline behave as in `llm.py`;
validation errors are never retried. The `openai` transport rejects file attachments and Bedrock does
not support a sampling seed (see `docs/design/model-layer.md`).
"""
from __future__ import annotations

import http.client
import json
import logging
import math
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .llm import GeminiStructuredLLM, EmbeddingCountMismatch, Media, _augment_required
from .response_errors import bounded_retry_delay

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

TRANSIENT_STATUSES = {408, 409, 429, 500, 502, 503, 504}
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class ProviderHTTPError(RuntimeError):
    """A non-success HTTP response from a provider; carries the status and Retry-After hint."""

    def __init__(self, status: int, message: str, retry_after: str | None = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status, self.retry_after = status, retry_after


def _strip_fence(text: str) -> str:
    match = _FENCE.match(text)
    return match.group(1) if match else text.strip()


def _check_limits(max_retries, request_timeout_seconds, max_retry_delay_seconds, retry_deadline_seconds) -> None:
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be finite and positive")
    if not math.isfinite(max_retry_delay_seconds) or max_retry_delay_seconds < 0:
        raise ValueError("max_retry_delay_seconds must be finite and nonnegative")
    if not math.isfinite(retry_deadline_seconds) or retry_deadline_seconds < 0:
        raise ValueError("retry_deadline_seconds must be finite and nonnegative")


class _RetryingLLM:
    """Shared retry loop: at most max_retries + 1 attempts, capped sleeps, optional overall deadline."""

    def __init__(self, max_retries, request_timeout_seconds, max_retry_delay_seconds, retry_deadline_seconds):
        _check_limits(max_retries, request_timeout_seconds, max_retry_delay_seconds, retry_deadline_seconds)
        self._max_retries = max_retries
        self._timeout = request_timeout_seconds
        self._max_retry_delay_seconds = max_retry_delay_seconds
        self._retry_deadline_seconds = retry_deadline_seconds

    def _is_retryable(self, exc: Exception) -> bool:
        raise NotImplementedError

    def _retry_after(self, exc: Exception) -> str | None:
        return None

    def _with_retries(self, call, model: str):
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return call()
            except Exception as exc:
                last_error = exc
                retryable = self._is_retryable(exc)
                delay = bounded_retry_delay(self._retry_after(exc), attempt, 1.0, self._max_retry_delay_seconds)
                deadline = self._retry_deadline_seconds
                within_deadline = not deadline or time.perf_counter() - started + delay <= deadline
                if attempt < self._max_retries and retryable and within_deadline:
                    logger.warning(
                        "llm_call_retry model=%s attempt=%d max_attempts=%d error_type=%s",
                        model, attempt + 1, self._max_retries + 1, type(exc).__name__,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "llm_call_failed model=%s attempts=%d retryable=%s deadline_reached=%s "
                        "elapsed_seconds=%.3f error_type=%s",
                        model, attempt + 1, retryable, not within_deadline, time.perf_counter() - started,
                        type(exc).__name__,
                    )
                    break
        assert last_error is not None
        raise last_error


class OpenAICompatibleLLM(_RetryingLLM):
    """Structured calls and embeddings over an OpenAI-compatible HTTP API (`/chat/completions`, `/embeddings`).

    `structured_output` is `json_schema` (the response_format carries the Pydantic JSON schema) or
    `json_object` (plain JSON mode plus the schema in the prompt, for servers without schema support).
    The key is sent in `auth_header`: as `Bearer <key>` for `Authorization`, raw for any other header
    (for example `api-key` or `x-apikey`); an empty key sends no credentials.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        auth_header: str = "Authorization",
        structured_output: str = "json_schema",
        max_retries: int = 2,
        request_timeout_seconds: int = 120,
        max_retry_delay_seconds: float = 60.0,
        seed: int | None = None,
        retry_deadline_seconds: float = 0.0,
    ):
        super().__init__(max_retries, request_timeout_seconds, max_retry_delay_seconds, retry_deadline_seconds)
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("openai base URL must be an HTTP(S) URL")
        if structured_output not in {"json_schema", "json_object"}:
            raise ValueError("structured_output must be json_schema or json_object")
        self._base_url = base_url.rstrip("/")
        self._structured_output = structured_output
        self._seed = seed
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers[auth_header] = f"Bearer {api_key}" if auth_header.lower() == "authorization" else api_key

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, ProviderHTTPError):
            return exc.status in TRANSIENT_STATUSES
        return isinstance(exc, (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, http.client.HTTPException))

    def _retry_after(self, exc: Exception) -> str | None:
        return exc.retry_after if isinstance(exc, ProviderHTTPError) else None

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self._base_url}/{path}", data=json.dumps(payload).encode("utf-8"),
            headers=self._headers, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = " ".join(exc.read().decode("utf-8", "replace").split())[:300]
            raise ProviderHTTPError(exc.code, detail, exc.headers.get("Retry-After")) from exc

    def generate(
        self, prompt: str, schema: type[T], model: str,
        *, required_fields: dict[str, int] | None = None, media: Media | None = None,
    ) -> T:
        if media:
            raise ValueError(
                "The openai transport does not support file attachments (conversion.backend = gemini); "
                "use conversion.backend = local"
            )
        json_schema = _augment_required(schema.model_json_schema(), required_fields)
        content = prompt
        if self._structured_output == "json_schema":
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {"name": re.sub(r"[^A-Za-z0-9_-]", "_", schema.__name__)[:64], "schema": json_schema},
            }
        else:
            response_format = {"type": "json_object"}
            content = f"{prompt}\n\nRespond with one JSON object that conforms to this JSON Schema:\n{json.dumps(json_schema)}"
        payload: dict[str, Any] = {
            "model": model, "messages": [{"role": "user", "content": content}], "response_format": response_format,
        }
        if self._seed is not None:
            payload["seed"] = self._seed

        def call() -> T:
            data = self._post("chat/completions", payload)
            choices = data.get("choices") or []
            message = (choices[0].get("message") if choices else None) or {}
            text = message.get("content")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("The model returned an empty structured response")
            return schema.model_validate_json(_strip_fence(text))

        return self._with_retries(call, model)

    def embed(self, texts: list[str], model: str, dimensions: int | None = None) -> list[list[float]]:
        payload: dict[str, Any] = {"model": model, "input": texts}
        if dimensions:
            payload["dimensions"] = dimensions

        def call() -> list[list[float]]:
            items = sorted(self._post("embeddings", payload).get("data") or [], key=lambda item: item.get("index", 0))
            return [list(item.get("embedding") or []) for item in items]

        vectors = self._with_retries(call, model)
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise EmbeddingCountMismatch("The endpoint returned an incomplete embedding response")
        return vectors


_BEDROCK_RETRYABLE_CODES = {
    "ThrottlingException", "TooManyRequestsException", "ModelTimeoutException", "ServiceUnavailableException",
    "InternalServerException", "ModelNotReadyException",
}
_BEDROCK_NETWORK_ERRORS = {"EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError", "ConnectionClosedError"}
_BEDROCK_TOOL = "return_structured_result"
_BEDROCK_IMAGE_FORMATS = {"image/png": "png", "image/jpeg": "jpeg", "image/gif": "gif", "image/webp": "webp"}


class BedrockStructuredLLM(_RetryingLLM):
    """Structured calls through the AWS Bedrock Converse API, with the result returned as a forced tool call.

    The Pydantic JSON schema is the input schema of one tool the model must call, which every Converse
    model with tool use supports. Credentials and region come from boto3's standard chain unless
    `region` or `profile` name them. Converse has no sampling seed, so `seed` is not sent. PDFs and
    images are sent as document or image blocks before the prompt. Embeddings support the Amazon
    Titan and Cohere Embed model families through InvokeModel.
    """

    def __init__(
        self,
        region: str = "",
        profile: str = "",
        *,
        max_tokens: int = 4096,
        max_retries: int = 2,
        request_timeout_seconds: int = 120,
        max_retry_delay_seconds: float = 60.0,
        retry_deadline_seconds: float = 0.0,
        client: Any = None,
    ):
        super().__init__(max_retries, request_timeout_seconds, max_retry_delay_seconds, retry_deadline_seconds)
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self._max_tokens = max_tokens
        if client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise RuntimeError(
                    "gemini.transport = bedrock needs the optional boto3 package: pip install 'chatbot-eval-poc[bedrock]'"
                ) from exc
            session = boto3.Session(profile_name=profile or None, region_name=region or None)
            client = session.client(
                "bedrock-runtime",
                config=Config(
                    read_timeout=request_timeout_seconds, connect_timeout=min(request_timeout_seconds, 10),
                    retries={"max_attempts": 1, "mode": "standard"},  # attempts are counted here, not by botocore
                ),
            )
        self._client = client

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, ValidationError):
            return False
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            code = (response.get("Error") or {}).get("Code")
            status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
            return code in _BEDROCK_RETRYABLE_CODES or status in TRANSIENT_STATUSES
        names = {cls.__name__ for cls in type(exc).__mro__}
        return bool(names & _BEDROCK_NETWORK_ERRORS) or isinstance(exc, (TimeoutError, ConnectionError))

    def _retry_after(self, exc: Exception) -> str | None:
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            headers = (response.get("ResponseMetadata") or {}).get("HTTPHeaders") or {}
            return headers.get("retry-after")
        return None

    @staticmethod
    def _media_blocks(media: Media) -> list[dict]:
        blocks = []
        for index, (mime, data) in enumerate(media, start=1):
            if mime == "application/pdf":
                blocks.append({"document": {"format": "pdf", "name": f"attachment {index}", "source": {"bytes": data}}})
            elif mime in _BEDROCK_IMAGE_FORMATS:
                blocks.append({"image": {"format": _BEDROCK_IMAGE_FORMATS[mime], "source": {"bytes": data}}})
            else:
                raise ValueError(f"The bedrock transport cannot attach {mime}; use a PDF or an image")
        return blocks

    def generate(
        self, prompt: str, schema: type[T], model: str,
        *, required_fields: dict[str, int] | None = None, media: Media | None = None,
    ) -> T:
        json_schema = _augment_required(schema.model_json_schema(), required_fields)
        content = [*(self._media_blocks(media) if media else []), {"text": prompt}]
        request = {
            "modelId": model,
            "messages": [{"role": "user", "content": content}],
            "toolConfig": {
                "tools": [{"toolSpec": {
                    "name": _BEDROCK_TOOL, "description": "Return the result in the required structure.",
                    "inputSchema": {"json": json_schema},
                }}],
                "toolChoice": {"tool": {"name": _BEDROCK_TOOL}},
            },
            "inferenceConfig": {"maxTokens": self._max_tokens},
        }

        def call() -> T:
            response = self._client.converse(**request)
            blocks = ((response.get("output") or {}).get("message") or {}).get("content") or []
            for block in blocks:
                tool = block.get("toolUse")
                if tool and tool.get("name") == _BEDROCK_TOOL:
                    return schema.model_validate(tool.get("input") or {})
            for block in blocks:  # models without forced tool choice may answer in plain JSON text
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    return schema.model_validate_json(_strip_fence(text))
            if response.get("stopReason") == "max_tokens":
                raise RuntimeError("The model reached bedrock_max_tokens before finishing; raise gemini.bedrock_max_tokens")
            raise RuntimeError("The model returned an empty structured response")

        return self._with_retries(call, model)

    def embed(self, texts: list[str], model: str, dimensions: int | None = None) -> list[list[float]]:
        if "titan-embed" in model:
            bodies = [{"inputText": text, **({"dimensions": dimensions, "normalize": True} if dimensions else {})}
                      for text in texts]
        elif "cohere.embed" in model:
            bodies = [{"texts": texts, "input_type": "search_document"}]
        else:
            raise ValueError(f"Embedding model {model!r} is not supported on bedrock; use an Amazon Titan or Cohere Embed model")

        def invoke(body: dict) -> dict:
            response = self._client.invoke_model(
                modelId=model, body=json.dumps(body), contentType="application/json", accept="application/json",
            )
            return json.loads(response["body"].read())

        vectors: list[list[float]] = []
        for body in bodies:
            result = self._with_retries(lambda body=body: invoke(body), model)
            if "embedding" in result:
                vectors.append(list(result["embedding"]))
            else:
                embeddings = result.get("embeddings") or []
                if isinstance(embeddings, dict):
                    embeddings = embeddings.get("float") or []
                vectors.extend(list(vector) for vector in embeddings)
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise EmbeddingCountMismatch("Bedrock returned an incomplete embedding response")
        return vectors


def build_llm(settings):
    """The structured-LLM client for `settings.gemini_transport`; secrets come from the loaded settings."""
    common = dict(
        max_retries=settings.max_retries,
        request_timeout_seconds=settings.gemini_request_timeout_seconds,
        retry_deadline_seconds=settings.retry_deadline_seconds,
    )
    transport = settings.gemini_transport
    if transport in {"direct", "apigee"}:
        return GeminiStructuredLLM(
            settings.api_key, settings.max_retries, transport=transport,
            apigee_api_key=settings.apigee_api_key, apigee_base_url=settings.apigee_base_url,
            request_timeout_seconds=settings.gemini_request_timeout_seconds, seed=settings.seed,
            retry_deadline_seconds=settings.retry_deadline_seconds,
        )
    if transport == "openai":
        return OpenAICompatibleLLM(
            settings.openai_base_url, settings.openai_api_key, auth_header=settings.openai_auth_header,
            structured_output=settings.openai_structured_output, seed=settings.seed, **common,
        )
    if transport == "bedrock":
        return BedrockStructuredLLM(
            settings.bedrock_region, settings.bedrock_profile, max_tokens=settings.bedrock_max_tokens, **common,
        )
    raise ValueError(f"Unsupported model transport: {transport!r}")
