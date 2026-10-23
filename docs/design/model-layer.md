# 5. The model layer (`llm.py`)

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
    -> GeminiStructuredLLM (the actual API call, retries, quota logging)
```

**`GeminiStructuredLLM`** uses the Google GenAI SDK with `response_mime_type=application/json` and
the schema's JSON Schema, then parses the response with `schema.model_validate_json`. Two transports
exist: `direct` (Gemini API key) and `apigee` (the internal AI Gateway; the SDK runs in
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
