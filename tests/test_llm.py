from chatbot_eval.llm import GeminiStructuredLLM
from chatbot_eval.models import JudgeScores


def test_direct_client_remains_the_default(monkeypatch):
    captured = {}
    monkeypatch.setattr("chatbot_eval.llm.genai.Client", lambda **kwargs: captured.update(kwargs) or object())

    GeminiStructuredLLM("direct-secret")

    assert captured == {"api_key": "direct-secret"}


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
