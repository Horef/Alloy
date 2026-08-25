# Chatbot Evaluation Module

This standalone Python module builds and runs evaluation sets for internal knowledge chatbots and
helps managers bootstrap chatbot instructions. It supports four workflows:

1. `generate`: create a representative, evidence-backed silver question set from local documents.
2. `evaluate`: send reviewed questions to a JSON-over-HTTP chatbot and judge its answers.
3. `evaluate-file`: judge questions and chatbot answers saved previously in Excel, CSV, or JSONL.
4. `generate-prompt`: derive a reviewable Hebrew system prompt and safety checklist from documents.

Gemini provides structured question generation, answer judging, optional topic inference, and
optional cross-result insights. The module does not require the existing chatbot fleet; integration
is isolated behind a small chatbot adapter boundary.

Generated questions are called *silver* because they are grounded and automatically validated, but
they are not final ground truth until a domain expert reviews them.

## Architecture and data flow

```text
local documents -> chunks -> topic map -> balanced quotas
                -> typed Q&A generation -> evidence validation/deduplication
                -> canonical questions -> natural/ambiguous user variations
                -> reviewable silver CSV/JSONL -> human approval
                                                   |
                         +-------------------------+
                         v
               live chatbot or premade Q&A file
                         |
                         v
                 Gemini claim-level judge
                         |
                         v
          details CSV/JSONL + summary JSON + Hebrew HTML
                         |
                         +-> optional topic inference and insights

local documents -> chunks -> topic map -> prompt blueprint
                -> Hebrew system prompt + structured manager review package
```

The components are independent: generation needs documents and Gemini but no chatbot; evaluation
consumes reviewed questions and either a chatbot adapter or a premade results file.

## Installation

Python 3.11 or newer is required. In the development environment used for this repository:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/army/bin/python -m pip install -e '.[dev]'
cp config.example.toml config.toml
cp .env.example .env
```

Set the Gemini API key in `.env`:

```dotenv
GEMINI_API_KEY=replace-me
```

Do not put the secret in `config.toml` or commit `.env`. The environment-variable name is controlled
by `gemini.api_key_env`.

Use either the installed command or the module entry point:

```bash
chatbot-eval --help
/opt/homebrew/Caskroom/miniforge/base/envs/army/bin/python -m chatbot_eval.cli --help
```

## Configuration and precedence

Settings are read from TOML. A command-line option overrides the corresponding TOML value where an
override exists. The API key is read from the environment after loading `.env` next to the selected
config file.

Global CLI options must appear **before** `generate`, `evaluate`, or `evaluate-file`.

```toml
[gemini]
api_key_env = "GEMINI_API_KEY"
generation_model = "gemini-3.7-flash"
judge_model = "gemini-3.7-flash"

[generation]
max_questions = 30
chunk_chars = 12000
chunk_overlap_chars = 800
batch_chunks = 8
unanswerable_ratio = 0.10
user_variation_ratio = 0.30
ambiguous_variation_share = 0.33
max_candidate_rounds = 3
stable_question_ids = true
question_type_targets = { basic_knowledge = 0.50, topic_integration = 0.30, document_wide = 0.10, cross_document = 0.10 }
min_topic_questions = 1
max_topic_share = 0.35

[evaluation]
request_timeout_seconds = 60
max_retries = 2
chatbot_max_retries = 2
chatbot_retry_base_seconds = 0.5
chatbot_pacing_seconds = 0.0
chatbot_max_response_bytes = 5000000
chatbot_require_json_content_type = true

[runtime]
progress_enabled = true
log_file = ""
log_level = "INFO"
```

### TOML parameters

| Parameter | Code default | Meaning |
|---|---:|---|
| `gemini.api_key_env` | `GEMINI_API_KEY` | Environment variable containing the Gemini key. |
| `gemini.generation_model` | `gemini-2.5-flash` | Topic-discovery and question-generation model. The example config explicitly selects another model. |
| `gemini.judge_model` | `gemini-2.5-flash` | Answer-judging, topic-inference, and insights model. |
| `generation.max_questions` | `30` | Maximum requested questions. This is a ceiling, not a guaranteed count. |
| `generation.chunk_chars` | `12000` | Maximum normalized characters in one document chunk. |
| `generation.chunk_overlap_chars` | `800` | Overlap between consecutive chunks; must be smaller than `chunk_chars`. |
| `generation.batch_chunks` | `8` | Chunks per topic-discovery call. Smaller values make more calls; larger values provide more context per call. |
| `generation.unanswerable_ratio` | `0.10` | Fraction of the total budget reserved for realistic unanswerable questions; valid range `[0, 1)`. |
| `generation.user_variation_ratio` | `0.30` | Fraction of the total budget reserved for natural/ambiguous variants derived from canonical questions. Set to `0` to disable variants. The sum with `unanswerable_ratio` must be below `1`. |
| `generation.ambiguous_variation_share` | `0.33` | Fraction of the variation budget that should require clarification; valid range `[0, 1]`. The remainder is natural but answerable wording. |
| `generation.max_candidate_rounds` | `3` | Bounded attempts to refill a quota after invalid or duplicate candidates are rejected. |
| `generation.stable_question_ids` | `true` | Derive reproducible content IDs. Use `--sequential-ids` for legacy run-local IDs. |
| `generation.question_type_targets` | empty/best effort | Target proportions for answerable generated types. Unsupported document-wide or cross-document allocations are redistributed for the current corpus. Values must sum to 1. |
| `generation.min_topic_questions` | `1` | Initial minimum allocation for represented topics while budget is available. |
| `generation.max_topic_share` | `0.35` | Approximate maximum share assigned to one topic. |
| `evaluation.request_timeout_seconds` | `60` | Timeout for each live chatbot HTTP request. |
| `evaluation.max_retries` | `2` | Gemini retries after the initial attempt, with exponential backoff. |
| `evaluation.chatbot_max_retries` | `2` | Retries for transient chatbot failures (408, 429, selected 5xx, timeouts, and connection failures). Authentication and malformed responses are not retried. |
| `evaluation.chatbot_retry_base_seconds` | `0.5` | Initial chatbot exponential-backoff delay. A valid `Retry-After` header takes precedence. |
| `evaluation.chatbot_pacing_seconds` | `0.0` | Minimum delay between sequential live-chatbot requests. |
| `evaluation.chatbot_max_response_bytes` | `5000000` | Maximum accepted chatbot response size. |
| `evaluation.chatbot_require_json_content_type` | `true` | Reject live responses whose media type is not JSON or `+json`. |
| `runtime.progress_enabled` | `true` | Enables English terminal progress bars. |
| `runtime.log_file` | empty | Optional operational log path. Empty disables file logging. |
| `runtime.log_level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |

### Global CLI parameters

| Option | Default | Meaning |
|---|---|---|
| `--config PATH` | `config.toml` | TOML file. If the default is missing and `config.example.toml` exists, the command creates a copy and exits for review. |
| `--no-progress` | off | Disable progress bars for one run. |
| `--log-file PATH` | config value | Write operational logs here. |
| `--log-level LEVEL` | config value | Override logging verbosity for one run. |

```bash
chatbot-eval --config config.toml --no-progress --log-level INFO \
  evaluate-file --results ./results.xlsx
```

## Workflow 1: generate a silver question set

### Generation logic

Generation is staged rather than performed with one unconstrained prompt:

1. **Ingest documents** by recursively finding supported files and normalizing their text.
2. **Create structure-aware overlapping chunks** with stable source IDs. Paragraph/heading boundaries,
   PDF page numbers, and DOCX table locations remain visible in provenance.
3. **Discover central topics** in batches while treating document content as untrusted data.
4. **Merge topic maps** into roughly 4-12 broad, non-overlapping topics when evidence permits.
5. **Allocate quotas** by importance, minimum topic allocation, and maximum topic share.
6. **Select evidence** using topic-linked chunks followed by lexical topic overlap.
7. **Generate structured candidates** containing the question, answer, difficulty, type, rationale,
   exact source IDs, and verbatim supporting quotations.
8. **Validate provenance and enforce the configured answerable question-type targets** deterministically.
9. **Remove near-duplicates**, including matches from an optional previous silver set.
10. **Derive realistic user variations** from accepted canonical questions. Natural variants retain
    the same expected answer; deliberately ambiguous variants expect one focused follow-up question.
11. **Generate boundary cases** using the budget reserved for unanswerable questions.
12. **Assign reproducible content-derived IDs** and export for human review with
    `review_status=pending`.

The command may return fewer questions than requested. This is expected when candidates are
duplicates, evidence cannot support the requested complexity, quotations are not verbatim, source
references are invalid, or the corpus lacks enough distinct material.

The total budget is divided into canonical answerable questions, user variations, and unanswerable
questions. Variants are children of canonical questions and never replace their provenance. The
generator does not create ambiguity by adding random spelling mistakes: it removes a meaningful
discriminator such as population, status, timeframe, or requested procedure.

### Supported inputs

| Format | Behavior |
|---|---|
| `.txt`, `.md`, `.rst` | Read as UTF-8 with replacement for malformed characters. |
| `.csv` | Rows become pipe-separated blocks so row boundaries survive chunking. |
| `.json`, `.jsonl` | Read as structured or line-delimited text. |
| `.pdf` | Text extracted locally with page-aware locations. Scanned pages require OCR first. |
| `.docx` | Paragraphs and tables are extracted as distinct structural sections. |

### Question types

| Value | Meaning | Validation |
|---|---|---|
| `basic_knowledge` | Explicit fact or procedure, usually from one source. | Requires evidence and quotations. |
| `topic_integration` | Combines multiple details within a topic. | Requires evidence and quotations. |
| `document_wide` | Integrates information across a document. | At least two chunks from one file. |
| `cross_document` | Integrates information from multiple documents. | Sources from at least two files. |
| `unanswerable` | Realistic nearby question unsupported by the corpus. | `answerable=false`, no quotations. |
| `personal_basic` | Reserved for direct personal-data questions. | Rejected by document-only generation. |
| `personal_integration` | Reserved for personal-data plus policy integration. | Rejected by document-only generation. |

Personal types require a future privacy-controlled adapter. The current module will not fabricate
personal data or infer it from ordinary documents.

### Question forms and expected behavior

Question complexity and user phrasing are separate dimensions. A `cross_document` question can be
canonical or naturally phrased; an ambiguous variant inherits the parent question's type and
evidence but changes the behavior expected from the chatbot.

| `question_form` | Meaning | `expected_behavior` |
|---|---|---|
| `canonical` | Clean, explicit review question generated directly from evidence. | `answer` for supported questions, `abstain` for boundary questions. |
| `natural_user` | Shorter, conversational, first-person, or domain-shorthand version that preserves enough information. | `answer`; the parent reference answer is reused. |
| `ambiguous` | Plausible request missing a material discriminator, where a direct answer could apply the wrong rule. | `clarify`; `expected_answer` contains the ideal focused follow-up. |

Every variant stores `parent_question_id`. This makes it possible to compare whether a chatbot can
handle both the clean formulation and realistic user language without treating the variants as
independent ground truth.

### Candidate validation

An answerable candidate is retained only when:

- all source IDs exist and are unique;
- a source and supporting quotation are present;
- each quotation occurs verbatim in its declared source after whitespace normalization;
- every cited source has a quotation;
- document-wide/cross-document structure matches its declared type; and
- it is not a near-duplicate of an accepted or excluded question; and
- it contains distinct atomic `reference_claims` for stable claim-level judging.

Rejection counts are written to operational logs at `INFO` level.

### Generate parameters

| Option | Required/default | Meaning |
|---|---|---|
| `--documents DIR` | required | Root folder scanned recursively for supported documents. |
| `--output DIR` | `outputs/questions` | Destination for the silver CSV and JSONL. |
| `--max-questions N` | config value | Positive total question ceiling. |
| `--topic TEXT` | unset | Give only the requested topic a non-zero quota. A discovered exact match reuses its mapped evidence. |
| `--topic-count N` | total ceiling | Maximum requested questions for `--topic`, capped by `--max-questions`. No effect without `--topic`. |
| `--unanswerable-ratio R` | config value | Per-run boundary-question ratio in `[0, 1)`. |
| `--user-variation-ratio R` | config value | Share of the total budget used for derived user variants; `[0, 1)`. Set to `0` for canonical-only generation. |
| `--ambiguous-variation-share R` | config value | Share of variants expected to trigger clarification; `[0, 1]`. |
| `--sequential-ids` | off | Use legacy `Q0001`-style run-local IDs instead of content-derived stable IDs. |
| `--exclude-questions PATH` | unset | Existing silver CSV/JSONL whose questions participate in deduplication. |

Representative run:

```bash
chatbot-eval --config config.toml generate \
  --documents ./knowledge_base \
  --max-questions 40 \
  --user-variation-ratio 0.30 \
  --ambiguous-variation-share 0.33 \
  --output ./outputs/questions
```

Topic-specific run:

```bash
chatbot-eval --config config.toml generate \
  --documents ./knowledge_base \
  --topic "תהליך אישור חופשה" \
  --topic-count 12 \
  --max-questions 12 \
  --output ./outputs/leave-questions
```

Exclude previously reviewed, rejected, or deleted questions:

```bash
chatbot-eval --config config.toml generate \
  --documents ./knowledge_base \
  --exclude-questions ./previous/silver_questions.csv \
  --max-questions 40 \
  --output ./outputs/questions-v2
```

### Silver-set outputs and review

`silver_questions.csv` is UTF-8 with a BOM for convenient Excel use. It contains:

- identity/content: `id`, `topic`, `question`, `expected_answer`, `answerable`;
- metadata: `difficulty`, `question_type`, `question_form`, `expected_behavior`,
  `parent_question_id`, `rationale`;
- provenance: `source_ids`, `source_files`, `source_locations`, `source_excerpts`,
  `supporting_quotes`;
- review: `review_status`, `reviewer_notes`.

`supporting_quotes` is JSON inside the CSV cell. `silver_questions.jsonl` preserves nested values
without flattening.

Reviewers should confirm the user need, reference answer, sources, and quotations; edit as needed;
then set `review_status` to `approved` or `rejected`. Keep rejected/deleted questions in an exclusion
file if they must not reappear. Do not rename required CSV columns if the module will read the file.

## Workflow 2: evaluate a live chatbot

The generic adapter sends one sequential HTTP `POST` per question. Defaults:

```jsonc
// request
{"question": "the reviewed question"}

// response
{"answer": "the chatbot answer", "retrieved_context": "retrieved chunks"}
```

Nested response fields use dotted paths. For `{"data":{"message":{"answer":"..."}}}`, use
`--answer-field data.message.answer`.

### Evaluate parameters

| Option | Required/default | Meaning |
|---|---|---|
| `--questions PATH` | required | Silver CSV or JSONL. |
| `--chatbot-url URL` | required | JSON-over-HTTP endpoint. |
| `--output DIR` | `outputs/evaluation` | Output directory. |
| `--question-field NAME` | `question` | Request property containing the question; not a dotted request path. |
| `--answer-field PATH` | `answer` | Dotted response path for the answer. |
| `--context-field PATH` | `retrieved_context` | Dotted response path for retrieved context. Missing context is allowed. |
| `--header Name=Value` | none | Additional HTTP header; repeatable. Environment variables are expanded. |
| `--approved-only` | off | Evaluate only `review_status=approved` rows, case-insensitively. |
| `--generate-insights` | off | Add one cross-result Gemini analysis in Hebrew. |
| `--hide-correct-answer-metrics` | off | Hide two correctness indicators in HTML; underlying data is preserved. |
| `--resume` | off | Reuse completed records from a compatible checkpoint. |
| `--checkpoint PATH` | `<output>/evaluation_checkpoint.jsonl` | Append-only per-question checkpoint. |

```bash
chatbot-eval --config config.toml evaluate \
  --questions ./outputs/questions/silver_questions.csv \
  --approved-only \
  --chatbot-url http://localhost:8080/ask \
  --answer-field data.answer \
  --context-field data.retrieved_context \
  --header 'Authorization=Bearer $CHATBOT_TOKEN' \
  --output ./outputs/run-001
```

Timeouts, connection failures, invalid JSON, unexpected content types, oversized responses, and
missing answer fields become categorized `chatbot_error` records and are excluded from quality
denominators. The generic adapter retries only transient failures with bounded exponential backoff,
honors `Retry-After`, and supports optional pacing. Production fleet adapters should still own
authentication, idempotency, sessions, and organization-specific formats.

## Workflow 3: evaluate a premade Q&A file

This workflow never contacts the chatbot. It reads saved rows, preserves recognizable API failures
as `chatbot_error`, and sends usable rows to the same judge used for live evaluation.

Supported formats are `.xlsx`, `.csv`, and `.jsonl`. Excel uses the active sheet unless `--sheet` is
specified. The first row must contain headers.

Required fields are question, expected/reference answer, and chatbot answer. Optional fields are
retrieved source/context, external ID, topic, answerability, and explicit error text.

| Logical field | Example auto-detected aliases |
|---|---|
| Question | `question`, `prompt`, `query`, `שאלה` |
| Expected answer | `expected_answer`, `reference_answer`, `תשובה צפויה` |
| Chatbot answer | `answer`, `response`, `model_answer`, `תשובה מהמודל` |
| Retrieved source | `source`, `evidence`, `context`, `retrieved_context`, `מקור` |
| ID | `id`, `question_id`, `messageId` |
| Topic | `topic`, `category`, `נושא` |

Explicit mappings are recommended for stable production jobs.

### Evaluate-file parameters

| Option | Required/default | Meaning |
|---|---|---|
| `--results PATH` | required | Input `.xlsx`, `.csv`, or `.jsonl`. |
| `--sheet NAME` | active sheet | Excel worksheet; ignored for CSV/JSONL. |
| `--output DIR` | `outputs/file-evaluation` | Output directory. |
| `--question-column NAME` | auto | Question column. |
| `--expected-answer-column NAME` | auto | Reference-answer column. |
| `--answer-column NAME` | auto | Saved chatbot-answer column. |
| `--source-column NAME` | optional/auto | Retrieved context for retrieval diagnostics. |
| `--id-column NAME` | optional/auto | External ID retained in metadata/reviewer notes. |
| `--topic-column NAME` | optional/auto | Topic; missing values become `לא סווג`. |
| `--answerable-column NAME` | optional/auto | true/false, 1/0, or yes/no; defaults to true. |
| `--error-column NAME` | optional/auto | Stored error; such rows bypass judging. |
| `--infer-topics` | off | Infer broad Hebrew topics for unclassified rows, in batches up to 250. |
| `--generate-insights` | off | Write and embed optional cross-result insights. |
| `--hide-correct-answer-metrics` | off | Hide correctness presentation metrics but retain data. |
| `--strict` | off | Reject the entire import if any row is invalid instead of logging and skipping invalid rows. |
| `--resume` | off | Reuse completed rows whose imported inputs still match. |
| `--checkpoint PATH` | `<output>/evaluation_checkpoint.jsonl` | Append-only per-question checkpoint. |

```bash
chatbot-eval --config config.toml evaluate-file \
  --results ./custom-results.xlsx \
  --sheet Sheet1 \
  --question-column prompt \
  --expected-answer-column gold_answer \
  --answer-column bot_response \
  --source-column retrieved_chunks \
  --infer-topics \
  --generate-insights \
  --hide-correct-answer-metrics \
  --output ./outputs/custom-evaluation
```

Rows missing a question or reference are skipped with a warning unless `--strict` is used. Empty answers, explicit errors,
fault payloads, and Spike Arrest messages are infrastructure errors and are not sent to Gemini. The
importer does not retry stored errors. Imported fault details are categorized and bounded so raw
gateway payloads are not propagated into reports.

The expected-answer column is the judging reference. The optional source column is treated as the
chatbot's retrieved context for diagnostics, not as a separately reviewed gold reference.

## Workflow 4: generate a system prompt from documents

This optional component helps a chatbot manager create a strong starting prompt without copying
domain rules by hand. It reuses the document loader and topic discovery, then asks Gemini for a
structured `PromptPackage`.

The package is checked deterministically for Hebrew content, configured assistant identity,
grounding, clarification, abstention, privacy, prompt-injection guidance, distinct non-empty lists,
application guardrails, review checks, and test coverage. One bounded repair call is made when the
first package fails; a still-invalid package is rejected rather than written.

The prompt generator separates:

- **corpus-derived scope**: broad supported topics and terminology;
- **stable behavioral guidance**: politeness, concise Hebrew answers, closed-world grounding,
  focused clarification, abstention, privacy minimization, and treating retrieved content as data;
- **manager decisions**: uncertain audience, authority, escalation, conflict, and fallback choices;
- **application controls**: authorization, retrieval ACLs, tool allowlists, DLP/PII filtering,
  validation, monitoring, rate limiting, and human approval for high-risk actions.

This separation is intentional. A prompt can guide behavior but cannot guarantee protection from
jailbreaks or replace deterministic security controls. The generator does not invent tools,
permissions, approval authorities, escalation contacts, document-precedence rules, or fixed domain
facts that the corpus does not support.

### Generate-prompt parameters

| Option | Required/default | Meaning |
|---|---|---|
| `--documents DIR` | required | Document base used for topic discovery and scope evidence. |
| `--output DIR` | `outputs/prompt` | Destination directory. |
| `--assistant-name TEXT` | `העוזר הדיגיטלי` | Manager-provided chatbot name. |
| `--audience TEXT` | `משתמשי הארגון` | Intended users; this is explicit configuration rather than an inferred fact. |

```bash
chatbot-eval --config config.toml generate-prompt \
  --documents ./knowledge_base \
  --assistant-name "תומי" \
  --audience "סגלי משאבי אנוש ותנאי שירות" \
  --output ./outputs/tomi-prompt
```

Outputs:

| File | Contents |
|---|---|
| `generated_system_prompt.md` | Ready-to-review Hebrew system prompt. |
| `prompt_package.json` | Corpus scope, assumptions requiring approval, application guardrails, manager checklist, and suggested tests. |

Managers must review both files before deployment. In particular, verify every described scope,
remove unsupported rules, decide approved fallback/escalation behavior, and implement the listed
application guardrails outside the model. The prompt generator itself does not deploy or modify a
chatbot.

## Judging logic and outcomes

Gemini receives the question, reference answer, available source evidence, candidate answer, and
retrieved context. It returns structured `JudgeScores`; automatic function calling is disabled.

Every factual-answer question stores stable, independently checkable `reference_claims`. The judge
reports one assessment per fixed claim ID; application code validates those IDs and derives the
aggregate counts. Older silver files remain readable by treating their full expected answer as one
coarse claim until a reviewer decomposes it. Clarification and abstention tasks have no factual
claims and are excluded from factual-answer and retrieval denominators.

The judge reports:

- required, addressed, and correctly answered details;
- false, unsupported, and unnecessary claims;
- required details found in retrieval;
- total, relevant, and contradictory retrieved chunks;
- abstention, clarification, answer scope, and incorrect-answer type; and
- Hebrew explanations of answer and retrieval failures.

Pydantic enforces invariants such as:

```text
correct <= addressed <= required
retrieved details <= required
relevant or contradictory chunks <= total chunks
```

The final outcome is deterministic from those scores:

| Outcome | Meaning |
|---|---|
| `correct_answer` | All required details correct, with no false/unsupported claims or excessive scope. |
| `partial_too_little` | Useful information is present, but necessary details are missing. |
| `partial_too_much` | Core answer is present, but excessive unrelated information reduces quality. |
| `unrelated_answer` | Clearly off-topic/nonresponsive. |
| `misleading_hallucination` | Plausible/assertive but materially false or unsupported. |
| `incorrect_abstention` | Declined an answerable question. |
| `correct_abstention` | Correctly declined an unanswerable question. |
| `should_have_abstained` | Attempted an answer to an unanswerable question. |
| `correct_clarification` | Asked an appropriate focused follow-up for an ambiguous question. |
| `missing_clarification` | Answered or otherwise proceeded although a material discriminator was missing. |
| `chatbot_error` | Live/imported chatbot failure; excluded from quality scores. |
| `judge_error` | Gemini judging failure after retries; excluded from quality scores. |

Abstention is detected both by the judge and deterministic English/Hebrew phrase patterns.

## Evaluation outputs and report logic

Both evaluation workflows write:

| File | Contents |
|---|---|
| `evaluation_details.csv` | Hebrew row-level results and claim/retrieval counts. |
| `evaluation_details.jsonl` | Lossless `EvaluationRecord` objects. |
| `evaluation_summary.json` | Aggregate outcomes, topic metrics, answer/retrieval metrics, and pipeline diagnostics. |
| `evaluation_report.html` | Interactive Hebrew RTL report. |
| `evaluation_insights.json` | Written only when optional insights succeed. |
| `evaluation_checkpoint.jsonl` | Durable per-question records used by `--resume`. |
| `run_manifest.json` | Redacted settings, input and implementation hashes, versions, timing, status, and result counts. |

The HTML includes KPI cards, outcome distribution, retrieval-versus-answer diagnostics, information
metrics, topic performance as `percentage (count/denominator)` with 95% Wilson intervals and small-
sample warnings, parent/variant robustness metrics, optional insights before per-question details,
and searchable/filterable question drill-down.

“Good retrieval” means all required details were found and no retrieved chunk contradicted the
reference. Missing context is missing telemetry, not bad retrieval, and is excluded from retrieval
denominators. “Good retrieval + bad answer” isolates a generation-stage failure.

`--hide-correct-answer-metrics` only removes `שיעור תשובות נכונות` from the top cards and
`תשובות נכונות` from the topic table. Outcomes and JSON metrics remain available.

### Optional insights

`--generate-insights` performs one additional Gemini call after judging. It detects recurring
patterns across topics, wording, answer errors, retrieval, personal-data-like questions, and
numbers/dates. Each issue has evidence counts, example IDs, confidence, a cause hypothesis, and a
recommendation. Application code removes unknown evidence IDs, recomputes counts and topics, drops
unsupported issues, and reduces single-example high-priority claims. Correlation is not presented as
proven causation, personal values are not repeated, and insight failure does not prevent normal
report generation.

## Progress, logging, retries, and rate limits

Terminal progress and status text are in English. Disable progress globally with
`runtime.progress_enabled=false` or per run with `--no-progress`.

```bash
chatbot-eval --config config.toml \
  --log-file ./outputs/logs/evaluation.log \
  --log-level INFO \
  evaluate-file --results ./results.xlsx
```

Logs include command state, counts, models, IDs, outcomes, latency, retries, candidate rejection
reasons, and errors. They avoid complete questions/answers, API keys, and authentication headers.

Gemini calls retry according to `evaluation.max_retries` with exponential backoff. Live chatbot
calls remain sequential and use the separately configured transient retry and pacing policy.
Authentication failures and malformed responses fail immediately; rate-limit responses honor
`Retry-After` when supplied.

Evaluation writes a checkpoint after every completed chatbot/judge result. `--resume` validates
input fingerprints before reusing records and rejects mismatched checkpoints. Final JSON and HTML,
prompt packages, insights, and manifests use atomic replacement. Manifests omit API keys and record
only a hash of the live chatbot URL.

## Code map and extension points

| Module | Responsibility |
|---|---|
| `cli.py` | CLI parsing, configuration wiring, workflow orchestration. |
| `config.py` | TOML and `.env` loading. |
| `documents.py` | File discovery, extraction, chunking, provenance. |
| `generator.py` | Topics, quotas, generation, validation, deduplication. |
| `prompt_generator.py` | Document-grounded system-prompt package generation and export. |
| `models.py` | Pydantic contracts. |
| `io.py` | Silver/evaluation CSV and JSONL serialization. |
| `results_io.py` | Premade file import, column resolution, stored-error detection. |
| `adapters.py` | Chatbot protocol and generic HTTP adapter. |
| `llm.py` | Structured LLM protocol and Gemini implementation. |
| `evaluator.py` | Claim-level judging and outcome classification. |
| `topics.py` | Optional Hebrew topic inference. |
| `insights.py` | Cross-result diagnosis. |
| `report.py` | Aggregation and Hebrew HTML reporting. |
| `progress.py` | Disableable progress wrapper. |
| `logging_utils.py` | Console and optional file logging. |

### Custom chatbot adapter

```python
class ChatbotAdapter(Protocol):
    def ask(self, question: SilverQuestion) -> ChatbotResult: ...
```

Return answer, retrieved context, latency, metadata, and a non-empty `error` for expected failures.
Keep authentication, throttling, retries, sessions, and fleet-specific payloads inside the adapter.

### Custom structured LLM

```python
class StructuredLLM(Protocol):
    def generate(self, prompt: str, schema: type[T], model: str) -> T: ...
```

Alternative providers must reliably return the supplied Pydantic schema. Preserve prompt-injection
protections: documents, questions, answers, and retrieved chunks are untrusted data.

## Testing

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/army/bin/python -m pytest -q
```

Tests use fake structured LLMs and do not require a Gemini key. They cover quotas, evidence
validation, deduplication, provenance, serialization and CSV safety, strict premade imports, HTTP
retry behavior, topic inference/harmonization, classification, insight validation, prompt-package
repair, paired variant metrics, confidence intervals, and reporting options.

## Recommended production process

1. Prefer human-authored and production-derived questions when available.
2. Use silver generation to expand coverage.
3. Require domain review of every generated reference and citation.
4. Keep rejected questions in an exclusion set.
5. Freeze an approved regression set and maintain a rotating set separately.
6. Include realistic unanswerable questions according to product policy.
7. Calibrate the judge against a double-reviewed human sample before using it as a release gate.
8. Record configs, prompts, models, input versions, and raw outputs for each run.
9. Treat chatbot/judge errors as infrastructure failures, not incorrect answers.
10. Diagnose retrieval and generation separately.

## Current limitations and next maturation steps

- Silver references still require human review.
- Topic evidence selection uses mapped sources plus lexical overlap, not embeddings or a graph.
- User-variation and ambiguity budgets are configurable, but the linguistic quality of each variant
  still requires human review.
- Run manifests hash input documents and implementation files; incremental regeneration is not yet implemented.
- Exclusion is file-based rather than persisted in a review database.
- Personal types are disabled pending an allowlisted, masked, auditable data interface.
- LLM judges can have style, verbosity, and model-family biases and require human audits.
- Generated system prompts are starting points and are neither deployed automatically nor security
  controls by themselves.

Likely next steps are incremental regeneration, a second-pass quality grader, judge calibration,
persistent review workflows, and the deferred engineering-quality baseline.

## Research basis

Structured Gemini output and document processing:

- https://ai.google.dev/gemini-api/docs/structured-output
- https://github.com/googleapis/python-genai
- https://ai.google.dev/gemini-api/docs/document-processing

Document-grounded synthetic generation, multi-hop diversity, filtering, and review:

- https://docs.ragas.io/en/latest/getstarted/rag_testset_generation/
- https://deepeval.com/docs/synthetic-data-generation-introduction
- https://deepeval.com/guides/guides-using-synthesizer
- https://arxiv.org/abs/1906.05416

Separate retrieval/groundedness/answer evaluation and claim-level scoring:

- https://arxiv.org/abs/2309.15217
- https://arxiv.org/abs/2305.14251
- https://arxiv.org/abs/2408.08067

The module is moving toward production use but is not yet a statistically validated benchmark.
