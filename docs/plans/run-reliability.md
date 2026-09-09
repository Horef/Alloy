# Plan: reliable capture, retry, checkpoints, and artifacts

Status: implemented offline on 2026-09-07; a single total wall-clock operation deadline remains a documented limitation. Parent review: [library-review.md](../library-review.md). R1–R4 and R6 have synthetic reproductions in [review_probes.py](../review_probes.py); R5 is established by inspecting the contract dictionaries and cache keys.

## Objective

Make interruption/retry preserve the intended work, reject incompatible reuse, and prevent a completed run from accidentally incorporating stale artifacts. Keep the current files and lightweight synchronous architecture; no queue/database/orchestration framework is needed.

## 1. Repair checkpoint parsing and append boundaries

Files: `artifacts.py`, `tests/test_artifacts.py`.

Both checkpoint loaders accept a complete last JSON object without a newline, but append does not insert a separator. Generation loading also catches contract errors in the same block as torn-JSON errors, allowing a signature mismatch to be discarded.

Implementation:

1. Parse each nonblank line first. Only a JSON decoding failure at the final unterminated line qualifies as a recoverable torn tail. Valid JSON with a bad schema, missing required fields, or a wrong signature is a hard error even without a newline.
2. Validate schema and signature outside the torn-tail catch. Never silently truncate semantically incompatible records.
3. If the final entry is valid but lacks a newline, normalize the append boundary before writing the next entry. Do this once during load or in a shared journal append helper. Keep valid records byte-equivalent except for required line termination.
4. Apply the same logic to evaluation and generation journals. Preserve evaluation's latest-entry-per-ID semantics for retries and generation's occurrence semantics for repeated identical calls.
5. Use the existing atomic replacement for torn-tail repair, with a clear log entry. Do not remove unrelated files or discard valid completed work.

Desired tests: normal newline; complete final JSON without newline followed by append and another resume; torn final JSON; malformed middle line; signature mismatch with/without newline; schema mismatch on final line; repeated evaluation record; repeated generation call. Test both classes using the same cases where practical.

## 2. Separate capture retries from judge retries

Files: `cli.py`, `evaluator.py`, tests for evaluator/CLI/checkpoints.

For live evaluation, `_checkpointed_evaluation` drops all error rows from completed state and sends them back through `evaluate`, which calls `chatbot.ask`. For a judge-only failure, the original chatbot response is already stored and should be reused.

Implementation:

1. Divide retry work into missing captures / chatbot failures and saved responses with judge failures.
2. Retry saved judge failures with `judge_results` against the stored `ChatbotResult`. Never call the live adapter for those rows. Preserve original answer, context, latency, and capture metadata.
3. Retry chatbot failures through the adapter, then judge successful captures. Persist replacement records through the same checkpoint callback.
4. Restore the original input order after merging all subsets. Successful rows remain untouched.
5. Refactor `judge_results` so it does not temporarily mutate `self.chatbot`. Prefer a shared `_judge_one(question, result, callback)` helper used by both paths. This is a narrow decomposition, not a separate workflow engine.
6. Document at-least-once behavior for HTTP retries: a timeout can occur after a server accepted a request. Do not claim exactly-once execution. Support a caller-provided idempotency/session strategy only when the endpoint actually offers it.

Desired tests: judge-error retry makes zero chatbot calls and one judge call; chatbot-error retry makes one new capture; mixed failures preserve order and successful rows; a retried judge failure still retains its original result; premade chatbot errors are preserved; interruption after partial retry resumes correctly.

## 3. Validate HTTP output and bound retry waits

Files: `adapters.py`, `llm.py`, `response_errors.py`, `config.py`, tests for adapters/LLM/config.

The answer extractor currently JSON-serializes nulls, dictionaries, and lists. That behavior can be useful for context serialization but is unsafe for the answer field. Empty live responses also bypass the error handling used by premade import.

Implementation:

1. Separate answer extraction from context extraction. An answer must be a nonblank string at the configured path. Missing, null, object, list, boolean, or number values become categorized malformed-response errors. Preserve actual valid text exactly.
2. Keep context serialization explicit: string context passes through; supported arrays/objects may be serialized for the generic adapter, but null means unavailable, not the string `null`. Record whether the field was absent, explicitly empty, or populated when available.
3. Apply the same known-placeholder/blank checks at the evaluator boundary so custom adapters cannot accidentally turn infrastructure failures into abstention successes. Keep imported chatbot errors as errors.
4. Validate URL/field-path configuration before a run starts. Catch expected network/read failures, including incomplete-body reads and connection resets, as per-record transport errors; do not hide arbitrary programming exceptions with a blanket success fallback.
5. Add a finite retry-delay cap and explicit total per-operation deadline, or document a bounded alternative if a total deadline cannot be enforced by the underlying client. `Retry-After` must be finite and nonnegative; reject/ignore infinity and NaN. Clamp long future HTTP dates. Bound fallback exponential backoff too.
6. Keep transient/permanent error distinctions and existing attempt counts. Report exhausted attempts and elapsed time without full response bodies or credentials.
7. Reject nonfinite configured retry/pacing durations. Record relevant installed transport SDK versions in manifests; don't change the SDK abstraction or add a new HTTP library merely for style.

Desired tests: answer null/blank/object/list/number; missing path; valid Hebrew text; context null and arrays; known placeholder; malformed JSON; incomplete read; timeout and permanent auth failure; Retry-After integer/date/infinite/far-future; delay cap and deadline. Use patched transports/clocks/sleep—no actual network or long waits.

## 4. Bind optional artifacts to their generating run

Files: `cli.py`, `artifacts.py`, `history.py`, `insights.py`, history/CLI tests.

Disabling insights or catching an insights-generation failure does not touch an older `evaluation_insights.json` in the same directory. `discover_previous_run` later loads it by filename without checking which results it belongs to.

Implementation:

1. Assign a run ID and record artifact input hashes/status in the manifest. Optional insight status must be `generated`, `disabled`, or `failed`, with the evaluation-record hash when generated.
2. Directory-based discovery must honor the current completed manifest and validate the insight's association with the selected results. Ignore or reject stale/mismatched optional artifacts with a clear explanation. Do not delete historical insight files to solve this.
3. For legacy directories without association metadata, retain explicit-file loading but surface association as unverified. Do not silently claim legacy insight/result compatibility. A narrowly scoped compatibility warning is sufficient; do not add an approval workflow to ordinary reading.
4. Prefer fresh output directories in examples. If reusing one, ensure the manifest accurately lists only outputs produced/validated for the current run, even when optional steps fail.
5. Write final question/evaluation JSONL and CSV via temporary files and atomic replacement, as is already done for reports. Add a completed manifest only after outputs are finalized. Readers should not mistake half-written files from a failed run for current completed results.
6. Preserve the evaluation manifest when regenerating a report in the same directory; use a separate report manifest or append a report-generation record instead of replacing capture provenance with report provenance.
7. Before reading or writing, reject unsafe input/output overlap that would overwrite the currently selected input artifact. Reject generated output directories within `--documents`, which can otherwise feed generated artifacts back into the corpus.

Desired tests: first run produces insights, second disables them, third discovers only compatible current artifacts; same sequence with insight-generation failure; explicitly supplied legacy insight; wrong artifact hash; failure during final write; regeneration of a report preserves original capture provenance. Files are preserved throughout these tests.

## 5. Centralize consequential run identity

Files: `cli.py`, `cache.py`, `artifacts.py`, stage fingerprint helpers and their tests.

Current fingerprints hash individual modules, not complete stage contracts. Changes to `models.py`, validation, transport settings, or the gateway URL are not uniformly reflected. Conversely, evaluation concurrency is part of the signature even for premade judge retries. Header names alone do not distinguish routing changes.

Implementation:

1. Add a small shared contract builder returning plain dictionaries: stage name/version, relevant model, nonsecret endpoint identity, schema fingerprint, relevant prompt/validator fingerprints, input budgets, and output-affecting generation parameters.
2. Include a normalized endpoint hash for the Gemini gateway in generation, judging, and model-derived caches. Never persist keys or URL credentials.
3. For live chatbot capture, include nonsecret routing configuration or an explicit user-provided deployment/tenant identity. Do not fingerprint secret authorization token values merely to distinguish tenants; token rotation should not invalidate semantically identical work. Require a new run/deployment identity when routing semantics change.
4. Include model schemas and semantic-validation versions where changing them changes accepted or classified output. A monolithic repository hash is easy but invalidates unrelated stages; a short explicit dependency list is sufficient.
5. Keep execution-only changes such as progress/log paths outside identity. Decide concurrency relevance separately for premade judging and stateful live capture; do not relax a live-state contract without specifying session isolation.
6. Store the canonical contract itself in the manifest and only its digest in repeated checkpoint entries. This keeps mismatch explanations inspectable without duplicating large dictionaries per row.
7. Make cache hits run the same semantic checks as freshly generated values where practical. Empty/invalid topic maps or bad evidence references must not become permanently trusted merely because their JSON shape is valid.

Desired tests: gateway change invalidates; schema/validator change invalidates; nonsecret tenant routing change invalidates; API-key rotation does not expose a key; log-level change does not invalidate; unchanged contracts replay exactly; errors explain the changed contract field.

## Completion criteria

Deliver small changes in the order 1, 3, 2, 4, 5; run focused tests after each and the full army suite at integration. Update README with exact retry call behavior and manifest/cache migration semantics. Existing historical artifacts remain intact. No live gateway/chatbot call is needed to establish the offline behavior; network compatibility and timeout behavior against the real service remain separately identified live checks.
