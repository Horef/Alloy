from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Protocol

from .models import ChatbotResult, SilverQuestion


class ChatbotAdapter(Protocol):
    def ask(self, question: SilverQuestion) -> ChatbotResult: ...


class HttpChatbotAdapter:
    """Generic JSON-over-HTTP adapter; field paths cover most later integrations."""

    def __init__(self, url: str, question_field: str = "question", answer_field: str = "answer", context_field: str = "retrieved_context", timeout: int = 60, headers: dict[str, str] | None = None):
        self.url, self.question_field, self.answer_field = url, question_field, answer_field
        self.context_field, self.timeout = context_field, timeout
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    @staticmethod
    def _field(data: dict, dotted_path: str, default: str = "") -> str:
        value = data
        for part in dotted_path.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    def ask(self, question: SilverQuestion) -> ChatbotResult:
        started = time.perf_counter()
        request = urllib.request.Request(
            self.url,
            data=json.dumps({self.question_field: question.question}).encode("utf-8"),
            headers=self.headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return ChatbotResult(
                question_id=question.id,
                answer=self._field(payload, self.answer_field),
                retrieved_context=self._field(payload, self.context_field),
                latency_ms=(time.perf_counter() - started) * 1000,
                metadata={"http_status": response.status},
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return ChatbotResult(question_id=question.id, answer="", latency_ms=(time.perf_counter() - started) * 1000, error=str(exc))

