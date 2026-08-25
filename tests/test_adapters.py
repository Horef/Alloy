import urllib.error
from email.message import Message

from chatbot_eval.adapters import HttpChatbotAdapter
from chatbot_eval.models import SilverQuestion


class _Response:
    def __init__(self, body: bytes, *, status: int = 200, content_type: str = "application/json"):
        self.body = body
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit=-1):
        return self.body if limit < 0 else self.body[:limit]


def test_http_adapter_retries_throttling_and_records_attempts(monkeypatch):
    attempts = []
    headers = Message()
    headers["Retry-After"] = "0"

    def urlopen(request, timeout):
        attempts.append(request)
        if len(attempts) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "too many requests", headers, None)
        return _Response(b'{"answer":"ok","retrieved_context":"source"}')

    monkeypatch.setattr("chatbot_eval.adapters.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("chatbot_eval.adapters.time.sleep", lambda _: None)
    result = HttpChatbotAdapter("https://example.invalid", max_retries=1).ask(
        SilverQuestion(id="Q1", topic="Topic", question="Question", expected_answer="Answer"),
    )

    assert result.answer == "ok"
    assert result.retrieved_context == "source"
    assert result.metadata["attempts"] == 2
    assert len(attempts) == 2


def test_http_adapter_does_not_retry_authentication_errors(monkeypatch):
    attempts = []

    def urlopen(request, timeout):
        attempts.append(request)
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", Message(), None)

    monkeypatch.setattr("chatbot_eval.adapters.urllib.request.urlopen", urlopen)
    result = HttpChatbotAdapter("https://example.invalid", max_retries=3).ask(
        SilverQuestion(id="Q1", topic="Topic", question="Question", expected_answer="Answer"),
    )

    assert result.metadata == {"error_category": "authentication", "attempts": 1, "http_status": 401}
    assert result.error.startswith("authentication:")
    assert len(attempts) == 1


def test_http_adapter_rejects_non_json_and_oversized_responses(monkeypatch):
    responses = iter([
        _Response(b"not json", content_type="text/html"),
        _Response(b'{"answer":"too long"}'),
    ])
    monkeypatch.setattr("chatbot_eval.adapters.urllib.request.urlopen", lambda request, timeout: next(responses))
    question = SilverQuestion(id="Q1", topic="Topic", question="Question", expected_answer="Answer")

    wrong_type = HttpChatbotAdapter("https://example.invalid").ask(question)
    too_large = HttpChatbotAdapter("https://example.invalid", max_response_bytes=5).ask(question)

    assert wrong_type.metadata["error_category"] == "malformed_response"
    assert too_large.metadata["error_category"] == "response_too_large"
