"""Tests for llm_providers: the OpenAI-compatible and Bedrock clients, retries, and the transport factory."""
from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from chatbot_eval.config import load_settings
from chatbot_eval.contracts import cache_identity
from chatbot_eval.llm import EmbeddingCountMismatch, GeminiStructuredLLM
from chatbot_eval.llm_providers import BedrockStructuredLLM, OpenAICompatibleLLM, build_llm


class Answer(BaseModel):
    text: str
    items: list[str] = []


class _Server:
    """A local OpenAI-compatible server that replays scripted (status, body, headers) responses."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                owner.requests.append((self.path, dict(self.headers), json.loads(self.rfile.read(length))))
                status, body, headers = owner.script.pop(0)
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    created = []

    def make(script):
        created.append(_Server(script))
        return created[-1]

    yield make
    for item in created:
        item.close()


def _chat(content):
    return {"choices": [{"message": {"content": content}}]}


def test_openai_structured_call_sends_schema_seed_and_bearer_key(server):
    srv = server([(200, _chat('{"text": "hi"}'), {})])
    llm = OpenAICompatibleLLM(srv.url, "secret", seed=7)

    assert llm.generate("question", Answer, "some-model") == Answer(text="hi")

    path, headers, body = srv.requests[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret"
    assert body["model"] == "some-model" and body["seed"] == 7
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["properties"]["text"]


def test_openai_json_object_mode_puts_the_schema_in_the_prompt_and_strips_fences(server):
    srv = server([(200, _chat('```json\n{"text": "ok"}\n```'), {})])
    llm = OpenAICompatibleLLM(srv.url, "k", auth_header="api-key", structured_output="json_object")

    assert llm.generate("q", Answer, "m").text == "ok"

    _, headers, body = srv.requests[0]
    assert headers["Api-Key"] == "k" and "Authorization" not in headers
    assert body["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in body["messages"][0]["content"] and "seed" not in body


def test_openai_without_key_sends_no_credentials(server):
    srv = server([(200, _chat('{"text": "x"}'), {})])
    OpenAICompatibleLLM(srv.url, "").generate("q", Answer, "m")
    assert "Authorization" not in srv.requests[0][1]


def test_openai_retries_transient_errors_and_honors_retry_after(server, monkeypatch):
    sleeps = []
    monkeypatch.setattr("chatbot_eval.llm_providers.time.sleep", sleeps.append)
    srv = server([(429, {"error": "slow down"}, {"Retry-After": "3"}), (200, _chat('{"text": "ok"}'), {})])

    assert OpenAICompatibleLLM(srv.url, "k").generate("q", Answer, "m").text == "ok"
    assert sleeps == [3.0] and len(srv.requests) == 2


def test_openai_authentication_errors_are_not_retried(server):
    srv = server([(401, {"error": "bad key"}, {})])
    with pytest.raises(RuntimeError, match="HTTP 401"):
        OpenAICompatibleLLM(srv.url, "k").generate("q", Answer, "m")
    assert len(srv.requests) == 1


def test_openai_rejects_attachments_and_empty_answers(server):
    srv = server([(200, _chat(""), {})])
    llm = OpenAICompatibleLLM(srv.url, "k")
    with pytest.raises(ValueError, match="attachments"):
        llm.generate("q", Answer, "m", media=[("application/pdf", b"%PDF")])
    with pytest.raises(RuntimeError, match="empty"):
        llm.generate("q", Answer, "m")


def test_openai_embeddings_are_ordered_and_checked(server):
    data = {"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]}
    srv = server([(200, data, {}), (200, {"data": [{"index": 0, "embedding": [1.0]}]}, {})])
    llm = OpenAICompatibleLLM(srv.url, "k")

    assert llm.embed(["a", "b"], "emb", 2) == [[1.0, 0.0], [0.0, 1.0]]
    assert srv.requests[0][2]["dimensions"] == 2
    with pytest.raises(EmbeddingCountMismatch):
        llm.embed(["a", "b"], "emb")


class _ClientError(Exception):
    def __init__(self, code, status=400, headers=None):
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status, "HTTPHeaders": headers or {}}}


class _Bedrock:
    def __init__(self, *responses):
        self.responses, self.calls, self.invocations = list(responses), [], []

    def converse(self, **request):
        self.calls.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def invoke_model(self, **request):
        self.invocations.append(request)
        item = self.responses.pop(0)
        return {"body": io.BytesIO(json.dumps(item).encode())}


def _tool(value):
    return {"output": {"message": {"content": [{"toolUse": {"name": "return_structured_result", "input": value}}]}}}


def test_bedrock_forces_a_tool_call_carrying_the_schema():
    client = _Bedrock(_tool({"text": "hi", "items": ["a"]}))
    llm = BedrockStructuredLLM(client=client, max_tokens=999)

    assert llm.generate("question", Answer, "anthropic.model-v1", required_fields={"items": 1}) == Answer(text="hi", items=["a"])

    request = client.calls[0]
    assert request["modelId"] == "anthropic.model-v1"
    assert request["messages"][0]["content"] == [{"text": "question"}]
    assert request["toolConfig"]["toolChoice"] == {"tool": {"name": "return_structured_result"}}
    schema = request["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]
    assert "items" in schema["required"]
    assert request["inferenceConfig"] == {"maxTokens": 999}


def test_bedrock_sends_pdfs_before_the_prompt_and_falls_back_to_json_text():
    client = _Bedrock({"output": {"message": {"content": [{"text": '```json\n{"text": "t"}\n```'}]}}})
    llm = BedrockStructuredLLM(client=client)

    assert llm.generate("q", Answer, "m", media=[("application/pdf", b"%PDF")]).text == "t"

    blocks = client.calls[0]["messages"][0]["content"]
    assert blocks[0]["document"]["format"] == "pdf" and blocks[-1] == {"text": "q"}
    with pytest.raises(ValueError, match="cannot attach"):
        llm.generate("q", Answer, "m", media=[("text/csv", b"a")])


def test_bedrock_reports_truncated_output():
    client = _Bedrock({"output": {"message": {"content": []}}, "stopReason": "max_tokens"})
    with pytest.raises(RuntimeError, match="bedrock_max_tokens"):
        BedrockStructuredLLM(client=client).generate("q", Answer, "m")


def test_bedrock_retries_throttling_but_not_access_errors(monkeypatch):
    sleeps = []
    monkeypatch.setattr("chatbot_eval.llm_providers.time.sleep", sleeps.append)
    client = _Bedrock(_ClientError("ThrottlingException", 429, {"retry-after": "2"}), _tool({"text": "ok"}))
    assert BedrockStructuredLLM(client=client).generate("q", Answer, "m").text == "ok"
    assert sleeps == [2.0]

    denied = _Bedrock(_ClientError("AccessDeniedException", 403))
    with pytest.raises(_ClientError):
        BedrockStructuredLLM(client=denied).generate("q", Answer, "m")
    assert len(denied.calls) == 1


def test_bedrock_embeddings_for_titan_and_cohere_and_unknown_models():
    titan = _Bedrock({"embedding": [1.0]}, {"embedding": [2.0]})
    assert BedrockStructuredLLM(client=titan).embed(["a", "b"], "amazon.titan-embed-text-v2:0", 1) == [[1.0], [2.0]]
    assert json.loads(titan.invocations[0]["body"]) == {"inputText": "a", "dimensions": 1, "normalize": True}

    cohere = _Bedrock({"embeddings": {"float": [[1.0], [2.0]]}})
    assert BedrockStructuredLLM(client=cohere).embed(["a", "b"], "cohere.embed-multilingual-v3") == [[1.0], [2.0]]

    with pytest.raises(ValueError, match="not supported"):
        BedrockStructuredLLM(client=_Bedrock()).embed(["a"], "some.other-model")


def test_bedrock_without_boto3_gives_an_install_hint(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "boto3", None)
    with pytest.raises(RuntimeError, match=r"chatbot-eval-poc\[bedrock\]"):
        BedrockStructuredLLM()


def _settings(tmp_path, body, monkeypatch, **env):
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return load_settings(path)


def test_factory_builds_the_client_for_each_transport(tmp_path, monkeypatch):
    openai = _settings(
        tmp_path, '[gemini]\ntransport = "openai"\nopenai_base_url = "https://llm.example.com/v1"\n',
        monkeypatch, OPENAI_API_KEY="k",
    )
    assert isinstance(build_llm(openai), OpenAICompatibleLLM)

    monkeypatch.setattr("chatbot_eval.llm_providers.BedrockStructuredLLM.__init__", lambda self, *a, **k: None)
    bedrock = _settings(tmp_path, '[gemini]\ntransport = "bedrock"\nbedrock_region = "eu-west-1"\n', monkeypatch)
    assert isinstance(build_llm(bedrock), BedrockStructuredLLM)

    monkeypatch.setattr("chatbot_eval.llm.genai.Client", lambda **kwargs: object())
    direct = _settings(tmp_path, "", monkeypatch, GEMINI_API_KEY="k")
    assert isinstance(build_llm(direct), GeminiStructuredLLM)


def test_each_endpoint_gets_its_own_cache_identity():
    base = dict(seed=7)
    identities = {
        cache_identity(SimpleNamespace(gemini_transport="openai", openai_base_url="https://a.example/v1", **base))["endpoint_sha256"],
        cache_identity(SimpleNamespace(gemini_transport="openai", openai_base_url="https://b.example/v1", **base))["endpoint_sha256"],
        cache_identity(SimpleNamespace(gemini_transport="bedrock", bedrock_region="us-east-1", **base))["endpoint_sha256"],
        cache_identity(SimpleNamespace(gemini_transport="bedrock", bedrock_region="eu-west-1", **base))["endpoint_sha256"],
        cache_identity(SimpleNamespace(gemini_transport="direct", **base))["endpoint_sha256"],
    }
    assert len(identities) == 5
