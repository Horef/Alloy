from __future__ import annotations

import json
import random
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Protocol

from .models import ChatbotResult, SilverQuestion
from .response_errors import placeholder_error


class ChatbotAdapter(Protocol):
    def ask(self, question: SilverQuestion) -> ChatbotResult: ...


class HttpChatbotAdapter:
    """Generic JSON-over-HTTP adapter; field paths cover most later integrations."""

    TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        url: str,
        question_field: str = "question",
        answer_field: str = "answer",
        context_field: str = "retrieved_context",
        timeout: int = 60,
        headers: dict[str, str] | None = None,
        *,
        max_retries: int = 2,
        retry_base_seconds: float = 0.5,
        pacing_seconds: float = 0.0,
        max_response_bytes: int = 5_000_000,
        require_json_content_type: bool = True,
    ):
        self.url, self.question_field, self.answer_field = url, question_field, answer_field
        self.context_field, self.timeout = context_field, timeout
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.pacing_seconds = pacing_seconds
        self.max_response_bytes = max_response_bytes
        self.require_json_content_type = require_json_content_type
        self._last_request_at: float | None = None

    @staticmethod
    def _field(data: dict, dotted_path: str, default=None):
        value = data
        for part in dotted_path.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _category(status: int | None, exc: Exception | None = None) -> str:
        if status in {401, 403}:
            return "authentication"
        if status == 429:
            return "throttled"
        if status == 408 or isinstance(exc, (TimeoutError, socket.timeout)):
            return "timeout"
        if status is not None and status >= 500:
            return "server_error"
        if status is not None:
            return "http_error"
        if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
            return "malformed_response"
        return "connection_error"

    def _retry_delay(self, attempt: int, headers=None) -> float:
        retry_after = headers.get("Retry-After") if headers else None
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                try:
                    retry_time = parsedate_to_datetime(retry_after)
                    if retry_time.tzinfo is None:
                        retry_time = retry_time.replace(tzinfo=timezone.utc)
                    return max(0.0, (retry_time - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError):
                    pass
        base = self.retry_base_seconds * (2**attempt)
        return base + random.uniform(0, base * 0.25) if base else 0.0

    def _pace(self) -> None:
        if self._last_request_at is None or self.pacing_seconds <= 0:
            return
        remaining = self.pacing_seconds - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def _error_result(
        self,
        question: SilverQuestion,
        started: float,
        *,
        category: str,
        message: str,
        attempts: int,
        status: int | None = None,
    ) -> ChatbotResult:
        metadata = {"error_category": category, "attempts": attempts}
        if status is not None:
            metadata["http_status"] = status
        return ChatbotResult(
            question_id=question.id,
            answer="",
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata=metadata,
            error=f"{category}: {' '.join(message.split())[:300]}",
        )

    def ask(self, question: SilverQuestion) -> ChatbotResult:
        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            self._pace()
            request = urllib.request.Request(
                self.url,
                data=json.dumps({self.question_field: question.question}).encode("utf-8"),
                headers=self.headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    self._last_request_at = time.monotonic()
                    content_type = response.headers.get_content_type()
                    if self.require_json_content_type and not (
                        content_type == "application/json" or content_type.endswith("+json")
                    ):
                        return self._error_result(
                            question, started, category="malformed_response",
                            message=f"expected JSON content type, received {content_type}",
                            attempts=attempt + 1, status=response.status,
                        )
                    body = response.read(self.max_response_bytes + 1)
                    if len(body) > self.max_response_bytes:
                        return self._error_result(
                            question, started, category="response_too_large",
                            message=f"response exceeded {self.max_response_bytes} bytes",
                            attempts=attempt + 1, status=response.status,
                        )
                    payload = json.loads(body.decode("utf-8"))
                    answer = self._field(payload, self.answer_field)
                    if answer is None:
                        return self._error_result(
                            question, started, category="malformed_response",
                            message=f"missing answer field {self.answer_field!r}",
                            attempts=attempt + 1, status=response.status,
                        )
                    retrieved_context = self._field(payload, self.context_field, "")
                    if error := placeholder_error(answer):
                        return ChatbotResult(
                            question_id=question.id,
                            answer=answer,
                            retrieved_context=retrieved_context,
                            latency_ms=(time.perf_counter() - started) * 1000,
                            metadata={
                                "http_status": response.status,
                                "attempts": attempt + 1,
                                "error_category": "answer_generation_failed",
                            },
                            error=error,
                        )
                    return ChatbotResult(
                        question_id=question.id,
                        answer=answer,
                        retrieved_context=retrieved_context,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        metadata={"http_status": response.status, "attempts": attempt + 1},
                    )
            except urllib.error.HTTPError as exc:
                self._last_request_at = time.monotonic()
                category = self._category(exc.code, exc)
                if exc.code in self.TRANSIENT_STATUSES and attempt < self.max_retries:
                    time.sleep(self._retry_delay(attempt, exc.headers))
                    continue
                return self._error_result(
                    question, started, category=category, message=str(exc),
                    attempts=attempt + 1, status=exc.code,
                )
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                self._last_request_at = time.monotonic()
                reason = getattr(exc, "reason", exc)
                category = self._category(None, reason if isinstance(reason, Exception) else exc)
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay(attempt))
                    continue
                return self._error_result(
                    question, started, category=category, message=str(exc), attempts=attempt + 1,
                )
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._last_request_at = time.monotonic()
                return self._error_result(
                    question, started, category="malformed_response", message=str(exc), attempts=attempt + 1,
                )
        raise AssertionError("unreachable retry loop")
