# 5. The model layer (`llm.py`, `llm_providers.py`)
Status: current (verified 2026-11-01)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md).

The rest of the code depends on one protocol:

```python
class StructuredLLM(Protocol):
    def generate(self, prompt: str, schema: type[T], model: str, *, required_fields=None) -> T: ...
```

At run time the CLI stacks three implementations of it:

```
StructuredCallCheckpoint   (generate only: run journal for --resume)
  -> CachedStructuredLLM   (persistent cross-run call cache, seeded runs only)
    -> the provider client   (the actual API call, retries; chosen by gemini.transport)
```

`llm_providers.build_llm(settings)` picks the provider client. The `[gemini]` config section
configures all of them (the name is historical), and `Settings.gemini_transport` is one of:

| Transport | Client | Structured output | Seed | Attachments | Embeddings |
|---|---|---|---|---|---|
| `direct` | `GeminiStructuredLLM` | JSON schema in the GenAI SDK config | yes | yes | Gemini embedding models |
| `apigee` | `GeminiStructuredLLM` | same, through a gateway | yes | yes | same |
| `openai` | `OpenAICompatibleLLM` | `response_format` JSON schema (or JSON mode with the schema in the prompt) | yes (`seed`) | no | `/embeddings` |
| `bedrock` | `BedrockStructuredLLM` | the schema is the input schema of one forced tool call (Converse) | no | PDF and images | Titan, Cohere Embed |

Credentials: `direct`, `apigee` and `openai` read a key from the environment variable the config
names (`openai` can run with none, and sends it in a configurable header); `bedrock` uses boto3's
standard credential chain, so no secret passes through `Settings`. `boto3` is an optional extra
(`.[bedrock]`) imported lazily, and a `forbid` rule keeps it out of every other module.

Providers share one contract: `generate(prompt, schema, model, *, required_fields=None, media=None)`
returning a validated Pydantic object and `embed(texts, model, dimensions=None)`. The two added
providers reuse the retry loop below through `_RetryingLLM` (statuses 408/409/429/500/502/503/504 or
the Bedrock throttling and timeout codes, `Retry-After`, capped waits, optional deadline) and never
retry validation errors. Their caches and corpus-cache keys include the endpoint (URL, or region and
profile) through `contracts.model_endpoint`, so switching provider or gateway never serves another
provider's cached answers. Offline tests cover them with a local HTTP server and a fake Bedrock
client; they have **not** been run against a live endpoint, so provider-specific limits (for
example a model that rejects forced tool choice or a gateway without `json_schema`) show up
as call errors: use `gemini.openai_structured_output = "json_object"` for the latter.

**`GeminiStructuredLLM`** uses the Google GenAI SDK with `response_mime_type=application/json` and
the schema's JSON Schema, then parses the response with `schema.model_validate_json`. Two transports
exist: `direct` (Gemini API key) and `apigee` (an API gateway; the SDK runs in
Vertex-compatible mode with a placeholder key and the real credential only in the `x-apikey`
header). Retries cover timeouts, connection errors, and HTTP 408/409/429/500/502/503/504, with jittered
exponential backoff that honors `Retry-After` and is capped per wait. An optional
`evaluation.retry_deadline_seconds` stops retrying once a retry would start after that deadline, so
one call takes at most the deadline plus one request timeout (the chatbot adapter applies the same
rule). Validation errors are never retried: a malformed response is a deterministic property of the
prompt, not a transient fault.

`required_fields` exists because Pydantic omits fields with defaults from the JSON Schema
`required` list, and structured-output models then feel free to drop them. Callers that need a field
for a specific request (for example `revision_mappings` when a prompt is being revised) promote it
to required with a minimum item count on a copy of the schema.

**`CachedStructuredLLM`** is an append-only JSONL store keyed by SHA-256 of
`(model, JSON schema, prompt, required_fields)`. Its file name encodes the transport, endpoint, seed,
and `STRUCTURED_CALL_CONTRACT`. It is used only when a seed is set. *Why it is safe:* with a fixed
seed an identical request returns the same response (the README notes this was verified through the
gateway), so serving it from disk loses nothing, while re-runs, partially edited plans, and re-judging
the same answers pay only for prompts that actually changed. `--refresh-cache` bypasses reads and
appends fresh responses (the last one wins on load); `--no-cache` bypasses it entirely.

**Embeddings** go through the same client (`embed`). Some embedding models fold a multi-text request
into one vector; `ModelEmbedder` detects the count mismatch and switches to one text per request,
issued concurrently.
