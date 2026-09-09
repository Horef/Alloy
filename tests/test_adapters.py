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


def test_http_adapter_marks_summary_generation_placeholder_as_error(monkeypatch):
    body = '{"answer":"לא הצלחנו ליצור סיכום לשאילתת החיפוש שלך, אבל כן מצאנו כמה תוצאות.","retrieved_context":"source"}'
    monkeypatch.setattr(
        "chatbot_eval.adapters.urllib.request.urlopen",
        lambda request, timeout: _Response(body.encode("utf-8")),
    )
    question = SilverQuestion(id="Q1", topic="Topic", question="Question", expected_answer="Answer")

    result = HttpChatbotAdapter("https://example.invalid").ask(question)

    assert result.error.startswith("answer_generation_failed:")
    assert result.metadata["error_category"] == "answer_generation_failed"
    assert result.retrieved_context == "source"


import json
import pytest
import http.client


@pytest.mark.parametrize('answer', [None, '', '  ', {}, [], True, 42])
def test_http_invalid_answer_is_error(monkeypatch, answer):
    monkeypatch.setattr('chatbot_eval.adapters.urllib.request.urlopen', lambda *a, **kw: _Response(json.dumps({'answer': answer}).encode()))
    result = HttpChatbotAdapter('https://example.invalid').ask(SilverQuestion(id='Q1', topic='t', question='q', expected_answer='a'))
    assert result.metadata['error_category'] == 'malformed_response'


@pytest.mark.parametrize('payload,status,context', [
    ({'answer': ' ok '}, 'missing', ''),
    ({'answer': ' ok ', 'retrieved_context': None}, 'null', ''),
    ({'answer': ' ok ', 'retrieved_context': []}, 'empty', '[]'),
    ({'answer': ' ok ', 'retrieved_context': ['source']}, 'populated', '["source"]'),
])
def test_http_context_telemetry(monkeypatch, payload, status, context):
    monkeypatch.setattr('chatbot_eval.adapters.urllib.request.urlopen', lambda *a, **kw: _Response(json.dumps(payload).encode()))
    result = HttpChatbotAdapter('https://example.invalid').ask(SilverQuestion(id='Q1', topic='t', question='q', expected_answer='a'))
    assert result.answer == ' ok '
    assert result.retrieved_context == context
    assert result.metadata['retrieval_context_status'] == status


@pytest.mark.parametrize('error', [http.client.IncompleteRead(b'partial'), ConnectionResetError('reset')])
def test_http_retries_expected_read_errors(monkeypatch, error):
    calls = []
    def failing(*args, **kwargs):
        calls.append(1)
        raise error
    monkeypatch.setattr('chatbot_eval.adapters.urllib.request.urlopen', failing)
    monkeypatch.setattr('chatbot_eval.adapters.time.sleep', lambda _: None)
    result = HttpChatbotAdapter('https://example.invalid', max_retries=1).ask(SilverQuestion(id='Q1', topic='t', question='q', expected_answer='a'))
    assert len(calls) == 2
    assert result.metadata['error_category'] == 'connection_error'


@pytest.mark.parametrize('hint', ['inf', 'NaN', '-5', '999999999', 'Wed, 01 Jan 2100 00:00:00 GMT'])
def test_http_retry_delay_is_finite_and_capped(hint):
    delay = HttpChatbotAdapter('https://example.invalid', max_retry_delay_seconds=3)._retry_delay(10000, {'Retry-After': hint})
    assert 0 <= delay <= 3


@pytest.mark.parametrize('kwargs', [{'pacing_seconds': float('nan')}, {'retry_base_seconds': float('inf')}, {'answer_field': 'answer..text'}, {'max_retries': -1}])
def test_http_invalid_configuration_fails_before_request(kwargs):
    with pytest.raises(ValueError):
        HttpChatbotAdapter('https://example.invalid', **kwargs)
