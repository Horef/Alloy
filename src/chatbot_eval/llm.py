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
    def __init__(self, api_key: str, max_retries: int = 2):
        self._client = genai.Client(api_key=api_key)
        self._max_retries = max_retries

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
