from __future__ import annotations

import logging
import time
from typing import Protocol, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class StructuredLLM(Protocol):
    def generate(self, prompt: str, schema: type[T], model: str) -> T: ...


class GeminiStructuredLLM:
    def __init__(
        self,
        api_key: str,
        max_retries: int = 2,
        *,
        transport: str = "direct",
        apigee_api_key: str = "",
        apigee_base_url: str = "",
    ):
        if transport == "direct":
            self._client = genai.Client(api_key=api_key)
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
                ),
            )
        else:
            raise ValueError(f"Unsupported Gemini transport: {transport!r}")
        self._transport = transport
        self._max_retries = max_retries

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
                if attempt < self._max_retries:
                    logger.warning(
                        "gemini_call_retry model=%s attempt=%d max_attempts=%d error=%r",
                        model, attempt + 1, self._max_retries + 1, exc,
                    )
                    time.sleep(2**attempt)
                else:
                    logger.error("gemini_call_failed model=%s attempts=%d error=%r", model, attempt + 1, exc)
        assert last_error is not None
        raise last_error
