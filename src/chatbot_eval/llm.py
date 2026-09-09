from __future__ import annotations

import logging
import math
import time
from typing import Protocol, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from .response_errors import bounded_retry_delay

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class StructuredLLM(Protocol):
    def generate(self, prompt: str, schema: type[T], model: str) -> T: ...


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
    ):
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be finite and positive")
        if not math.isfinite(max_retry_delay_seconds) or max_retry_delay_seconds < 0:
            raise ValueError("max_retry_delay_seconds must be finite and nonnegative")
        self._max_retry_delay_seconds = max_retry_delay_seconds
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
        network_errors = (
            TimeoutError, ConnectionError,
            errors.httpx.TimeoutException, errors.httpx.TransportError,
            errors.requests.Timeout, errors.requests.ConnectionError,
        )
        return isinstance(exc, network_errors)

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

    def generate(self, prompt: str, schema: type[T], model: str) -> T:
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=schema.model_json_schema(),
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
