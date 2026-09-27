from chatbot_eval.llm import GeminiStructuredLLM
from chatbot_eval.models import JudgeScores
from google.genai import errors
import pytest


def test_direct_client_remains_the_default(monkeypatch):
    captured = {}
    monkeypatch.setattr("chatbot_eval.llm.genai.Client", lambda **kwargs: captured.update(kwargs) or object())

    GeminiStructuredLLM("direct-secret")

    assert captured["api_key"] == "direct-secret"
    assert captured["http_options"].timeout == 120_000


def test_apigee_client_uses_gateway_sdk_configuration(monkeypatch):
    captured = {}

    def client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("chatbot_eval.llm.genai.Client", client)

    GeminiStructuredLLM(
        "", transport="apigee", apigee_api_key="secret",
        apigee_base_url="https://preprod.apigee.digital.idf.il/ai_gateway/v1/hr/",
    )

    assert captured["api_key"] == "apigee-placeholder"
    assert captured["vertexai"] is True
    assert captured["project"] == captured["location"] == ""
    assert captured["http_options"].api_version == "v1"
    assert captured["http_options"].base_url.endswith("/ai_gateway/v1/hr")
    assert captured["http_options"].headers == {"x-apikey": "secret"}


def test_quota_headers_are_logged_without_response_content(caplog):
    class HttpResponse:
        headers = {
            "x-selected-metric": "tokens", "x-request-tokens-used": "42",
            "x-quota-limit": "1000", "x-quota-tokens-used": "100",
            "x-quota-tokens-remaining": "900",
        }

    class Response:
        sdk_http_response = HttpResponse()

    with caplog.at_level("INFO"):
        GeminiStructuredLLM._log_quota(Response(), "model")

    assert "metric=tokens" in caplog.text
    assert "request_used=42" in caplog.text
    assert "remaining=900" in caplog.text


def test_structured_calls_explicitly_disable_afc():
    captured = {}

    class Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)

            class Response:
                text = JudgeScores(
                    claim_assessments=[],
                    required_points_total=2, answer_points_addressed=2, answer_points_correct=2,
                    answer_false_claims=0, answer_unsupported_claims=0, answer_extraneous_claims=0,
                    retrieval_points_found=2, retrieved_chunks_total=1,
                    retrieved_chunks_relevant=1, retrieved_chunks_contradictory=0,
                    answer_scope="exact", incorrect_type="not_applicable",
                    response_is_abstention=False, explanation="תקין", missing_or_wrong="",
                    retrieval_explanation="האחזור תקין",
                ).model_dump_json()

            return Response()

    class Client:
        models = Models()

    llm = object.__new__(GeminiStructuredLLM)
    llm._client = Client()
    llm._max_retries = 0
    llm.generate("prompt", JudgeScores, "test-model")

    assert captured["config"].automatic_function_calling.disable is True


def test_embed_returns_vectors_and_rejects_incomplete_responses():
    class Embedding:
        def __init__(self, values):
            self.values = values

    class Models:
        def __init__(self, vectors):
            self.vectors = vectors

        def embed_content(self, **kwargs):
            return type("Response", (), {"embeddings": [Embedding(v) for v in self.vectors]})()

    llm = object.__new__(GeminiStructuredLLM)
    llm._max_retries = 0
    llm._client = type("Client", (), {"models": Models([[0.1, 0.2], [0.3, 0.4]])})()
    assert llm.embed(["א", "ב"], "embedding-model") == [[0.1, 0.2], [0.3, 0.4]]

    llm._client = type("Client", (), {"models": Models([[0.1, 0.2]])})()
    with pytest.raises(RuntimeError, match="incomplete embedding"):
        llm.embed(["א", "ב"], "embedding-model")


def test_non_retryable_client_error_fails_immediately(monkeypatch):
    class Models:
        calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            raise errors.ClientError(401, {"message": "unauthorized"})

    llm = object.__new__(GeminiStructuredLLM)
    llm._client = type("Client", (), {"models": Models()})()
    llm._max_retries = 3
    monkeypatch.setattr("chatbot_eval.llm.time.sleep", lambda _: pytest.fail("must not sleep"))

    with pytest.raises(errors.ClientError):
        llm.generate("prompt", JudgeScores, "model")
    assert llm._client.models.calls == 1


def test_retryable_server_error_honors_retry_after(monkeypatch):
    class Response:
        headers = {"Retry-After": "0"}

    class Models:
        calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise errors.ServerError(503, {"message": "busy"}, Response())
            return type("Result", (), {"text": JudgeScores(
                claim_assessments=[], required_points_total=0, answer_points_addressed=0,
                answer_points_correct=0, answer_false_claims=0, answer_unsupported_claims=0,
                answer_extraneous_claims=0, retrieval_points_found=0, retrieved_chunks_total=0,
                retrieved_chunks_relevant=0, retrieved_chunks_contradictory=0, answer_scope="exact",
                incorrect_type="not_applicable", response_is_abstention=True, explanation="x",
                missing_or_wrong="", retrieval_explanation="",
            ).model_dump_json()})()

    llm = object.__new__(GeminiStructuredLLM)
    llm._client = type("Client", (), {"models": Models()})()
    llm._max_retries = 1
    delays = []
    monkeypatch.setattr("chatbot_eval.llm.time.sleep", delays.append)

    llm.generate("prompt", JudgeScores, "model")
    assert delays == [0.0]


@pytest.mark.parametrize('hint', ['inf', 'NaN', '-1', '999999999', 'Wed, 01 Jan 2100 00:00:00 GMT'])
def test_llm_retry_delay_is_capped(hint):
    error = RuntimeError('retry')
    error.response = type('Response', (), {'headers': {'Retry-After': hint}})()
    assert 0 <= GeminiStructuredLLM._retry_delay(error, 1000, cap=2) <= 2


@pytest.mark.parametrize('kwargs', [{'max_retries': -1}, {'request_timeout_seconds': float('inf')}, {'max_retry_delay_seconds': float('nan')}])
def test_llm_invalid_retry_configuration(kwargs):
    with pytest.raises(ValueError):
        GeminiStructuredLLM('unused', **kwargs)
