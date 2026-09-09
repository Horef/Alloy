from __future__ import annotations

import json
import math
import http.client
import socket
import time
import threading
import urllib.error
import urllib.request
import urllib.parse
from typing import Protocol

from .models import ChatbotResult, SilverQuestion
from .response_errors import placeholder_error, bounded_retry_delay


class ChatbotAdapter(Protocol):
    def ask(self, question: SilverQuestion) -> ChatbotResult: ...


class HttpChatbotAdapter:
    """Generic JSON HTTP adapter with at most max_retries + 1 attempts.

    Each retry wait is capped. Socket timeouts are not a total operation deadline;
    a timeout may happen after server acceptance, so retries are at-least-once.
    """

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
        max_retry_delay_seconds: float = 60.0,
        max_response_bytes: int = 5_000_000,
        require_json_content_type: bool = True,
    ):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
            raise ValueError("chatbot URL must be HTTP(S), with a host and without credentials or fragment")
        for name, value in (("question_field", question_field), ("answer_field", answer_field), ("context_field", context_field)):
            if not isinstance(value, str) or not value.strip() or any(not part.strip() for part in value.split(".")):
                raise ValueError(f"{name} must be a nonempty field path")
        for name, value in (("timeout", timeout), ("retry_base_seconds", retry_base_seconds), ("pacing_seconds", pacing_seconds), ("max_retry_delay_seconds", max_retry_delay_seconds)):
            if not math.isfinite(value) or value < 0 or (name == "timeout" and value == 0):
                raise ValueError(f"{name} must be finite and {'positive' if name == 'timeout' else 'nonnegative'}")
        if not isinstance(max_retries, int) or max_retries < 0 or max_response_bytes <= 0:
            raise ValueError("max_retries must be a nonnegative integer and max_response_bytes positive")
        self.max_retry_delay_seconds = max_retry_delay_seconds
        self.url, self.question_field, self.answer_field = url, question_field, answer_field
        self.context_field, self.timeout = context_field, timeout
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.pacing_seconds = pacing_seconds
        self.max_response_bytes = max_response_bytes
        self.require_json_content_type = require_json_content_type
        self._last_request_at: float | None = None
        self._pacing_lock = threading.Lock()

    @staticmethod
    def _field(data: dict, dotted_path: str, default=None):
        value = data
        for part in dotted_path.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

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
        return bounded_retry_delay(
            headers.get("Retry-After") if headers else None,
            attempt, self.retry_base_seconds, self.max_retry_delay_seconds,
        )

    def _pace(self) -> None:
        with self._pacing_lock:
            if self._last_request_at is not None and self.pacing_seconds > 0:
                remaining = self.pacing_seconds - (time.monotonic() - self._last_request_at)
                if remaining > 0:
                    time.sleep(remaining)
            self._last_request_at = time.monotonic()

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
                    if not isinstance(answer, str) or not answer.strip():
                        return self._error_result(
                            question, started, category="malformed_response",
                            message=f"answer field {self.answer_field!r} must be a nonblank string",
                            attempts=attempt + 1, status=response.status,
                        )
                    missing = object()
                    context = self._field(payload, self.context_field, missing)
                    if context is missing or context is None:
                        context_status = "missing" if context is missing else "null"
                        retrieved_context = ""
                    else:
                        context_status = "empty" if context == "" or context == [] or context == {} else "populated"
                        retrieved_context = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False)
                    if error := placeholder_error(answer):
                        return ChatbotResult(
                            question_id=question.id,
                            answer=answer,
                            retrieved_context=retrieved_context,
                            latency_ms=(time.perf_counter() - started) * 1000,
                            metadata={
                                "http_status": response.status,
                                "retrieval_context_status": context_status,
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
                        metadata={"http_status": response.status, "attempts": attempt + 1, "retrieval_context_status": context_status},
                    )
            except urllib.error.HTTPError as exc:
                category = self._category(exc.code, exc)
                if exc.code in self.TRANSIENT_STATUSES and attempt < self.max_retries:
                    time.sleep(self._retry_delay(attempt, exc.headers))
                    continue
                return self._error_result(
                    question, started, category=category, message=type(exc).__name__,
                    attempts=attempt + 1, status=exc.code,
                )
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, http.client.HTTPException) as exc:
                reason = getattr(exc, "reason", exc)
                category = self._category(None, reason if isinstance(reason, Exception) else exc)
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay(attempt))
                    continue
                return self._error_result(
                    question, started, category=category, message=type(exc).__name__, attempts=attempt + 1,
                )
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                return self._error_result(
                    question, started, category="malformed_response", message=type(exc).__name__, attempts=attempt + 1,
                )
        raise AssertionError("unreachable retry loop")
