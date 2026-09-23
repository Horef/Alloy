# Chatbot Evaluation Module

This standalone Python module builds and runs evaluation sets for internal knowledge chatbots and
helps managers bootstrap chatbot instructions. It supports these workflows:

1. `generate`: create a representative, evidence-backed silver question set from local documents.
2. `review-export` / `review-merge`: export a small, human-readable review file and merge the
   reviewer's edits back onto the full technical set without losing provenance.
3. `reground`: regenerate the grounding (sources, quotes, reference claims) for questions whose text
   a reviewer edited, so their evidence matches the new wording.
3a. `revary`: regenerate only the derived user variations (natural-user and ambiguous) on an existing
   silver set, keeping the reviewed canonical and boundary questions untouched.
4. `evaluate`: send reviewed questions to a JSON-over-HTTP chatbot and judge its answers.
5. `evaluate-file`: judge questions and chatbot answers saved previously in Excel, CSV, or JSONL.
6. `generate-prompt`: derive a reviewable Hebrew system prompt and safety checklist from documents.
7. `report`: regenerate a report from completed results, optionally compared with a previous run.

Gemini provides structured question generation, answer judging, optional topic inference, and
optional cross-result insights. The module does not require the existing chatbot fleet; integration
is isolated behind a small chatbot adapter boundary.

Generated questions are called *silver* because they are grounded and automatically validated, but
they are not final ground truth until a domain expert reviews them.

## Architecture and data flow

Since v1.0, question generation is built on a **corpus knowledge graph**. Chunks become graph nodes
enriched with extracted signals (entities, keyphrases, a one-line summary); deterministic typed
edges connect nodes that share entities, overlap on keyphrases, or sit adjacently in the same
document. Topics are derived from graph structure (clusters of related nodes) rather than lexical
overlap with a topic label, and the evidence for each question is a connected *cluster* of nodes, so
facts spread across chunks are presented together. This raises the recall and completeness of the
generated reference answers. Topic prevalence still drives quotas: a cluster's corpus coverage sets
its importance, preserving the original generation philosophy.

An optional **theme layer** (`generation.topic_mode = "theme"`) sits above the entity graph: one
corpus-level call derives a small controlled vocabulary of broad themes, each chunk is tagged with a
single theme from it, and topics are then grouped by theme rather than by entity overlap. This makes
a subject spread thinly across many documents (the motivating example was pay/שכר) cohere into its
own topic instead of scattering, so topic coverage self-organizes and a reviewer only has to create
and read. Themes are kept as a separate layer and never create graph edges, so theme mode cannot
cause the entity hairball. Entity mode remains the default; see `docs/theme-layer-ab-results.md`.

```text
local documents -> chunks -> node signals (entities/keyphrases/summary, extracted per document)
                -> knowledge graph (typed edges) -> prevalence-weighted topic clusters -> quotas
                -> typed Q&A generation from cluster evidence -> validation/deduplication
                -> optional answer-completeness verification
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
                         +-> optional topic inference, insights, and previous-run comparison

local documents -> chunks -> knowledge graph topics -> prompt blueprint
                + current prompt + prior evaluation evidence
                -> revised Hebrew system prompt + structured manager review package
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

For direct Gemini access, set the Gemini API key in `.env`:

```dotenv
GEMINI_API_KEY=replace-me
```

For the internal Apigee AI Gateway, set `APIGEE_API_KEY` instead and select the `apigee` transport
as described below. Do not put either secret in `config.toml` or commit `.env`.

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
transport = "direct"
api_key_env = "GEMINI_API_KEY"
apigee_api_key_env = "APIGEE_API_KEY"
apigee_base_url = "https://preprod.apigee.digital.idf.il/ai_gateway/v1/hr"
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
verify_answer_completeness = false
completeness_evidence_limit = 16

[cache]
enabled = true
directory = ".chatbot_eval_cache"

[evaluation]
request_timeout_seconds = 60
max_retries = 2
chatbot_max_retries = 2
chatbot_retry_base_seconds = 0.5
chatbot_pacing_seconds = 0.0
chatbot_max_response_bytes = 5000000
chatbot_require_json_content_type = true
judge_max_answer_chars = 20000
judge_max_context_chars = 60000
insights_max_prompt_chars = 80000
gemini_request_timeout_seconds = 120
max_concurrency = 1

[runtime]
progress_enabled = true
log_file = ""
log_level = "INFO"
```

### TOML parameters

| Parameter | Code default | Meaning |
|---|---:|---|
| `gemini.transport` | `direct` | `direct` uses Google Gemini; `apigee` routes every structured Gemini call through the internal AI Gateway. |
| `gemini.api_key_env` | `GEMINI_API_KEY` | Environment variable containing the Gemini key. |
| `gemini.apigee_api_key_env` | `APIGEE_API_KEY` | Environment variable containing the Apigee API key. The key itself is never written to manifests. |
| `gemini.apigee_base_url` | preprod `/ai_gateway/v1/hr` | HTTPS client-specific AI Gateway base URL. Change the environment/client segment only according to registered access. |
| `gemini.generation_model` | `gemini-2.5-flash` | Topic-discovery and question-generation model. The example config explicitly selects another model. |
| `gemini.judge_model` | `gemini-2.5-flash` | Answer-judging, topic-inference, and insights model. |
| `generation.max_questions` | `30` | Maximum requested questions. This is a ceiling, not a guaranteed count. |
| `generation.chunk_chars` | `12000` | Maximum normalized characters in one document chunk. The shipped example and corpus configs use `2000` (~500 tokens): smaller chunks give each document multiple knowledge-graph nodes (enabling `document_wide` questions and finer edges) and are easier for smaller models to process, while staying large enough to keep a typical rule intact. |
| `generation.chunk_overlap_chars` | `800` | Overlap between consecutive chunks; must be smaller than `chunk_chars`. The example/corpus configs use `200` to match the smaller chunk size. |
| `generation.batch_chunks` | `8` | Legacy batching hint retained for compatibility; knowledge-graph signal extraction uses `graph_extraction_batch_chunks`. |
| `generation.unanswerable_ratio` | `0.10` | Fraction of the total budget reserved for realistic unanswerable questions; valid range `[0, 1)`. |
| `generation.user_variation_ratio` | `0.30` | Fraction of the total budget reserved for natural/ambiguous variants derived from canonical questions. Set to `0` to disable variants. The sum with `unanswerable_ratio` must be below `1`. |
| `generation.ambiguous_variation_share` | `0.33` | Fraction of the variation budget that should require clarification; valid range `[0, 1]`. The remainder is natural but answerable wording. |
| `generation.max_candidate_rounds` | `3` | Bounded attempts to refill a quota after invalid or duplicate candidates are rejected. |
| `generation.stable_question_ids` | `true` | Derive reproducible content IDs. Use `--sequential-ids` for legacy run-local IDs. |
| `generation.question_type_targets` | empty/best effort | Target proportions for answerable generated types. Unsupported document-wide or cross-document allocations are redistributed for the current corpus. Values must sum to 1. |
| `generation.min_topic_questions` | `1` | Initial minimum allocation for represented topics while budget is available. |
| `generation.max_topic_share` | `0.35` | Approximate maximum share assigned to one topic. |
| `generation.graph_extraction_batch_chunks` | `8` | Chunks per knowledge-graph signal-extraction call. |
| `generation.keyphrase_overlap_threshold` | `0.3` | Jaccard threshold (in `(0, 1]`) for creating a keyphrase-overlap edge between two nodes. |
| `generation.max_graph_topics` | `40` | Maximum graph-derived topics. Beyond this, the largest (most prevalent) clusters stay distinct and the remaining small ones are bin-packed into bounded "other" buckets. A higher cap yields finer topics on dense corpora; it does not create a topic the graph structure does not support (a theme spread thinly across documents may still not cohere into its own cluster — use `generate --topic` to force a focused subset). |
| `generation.min_cluster_nodes` | `1` | Smallest standalone cluster kept before merging into "other". |
| `generation.max_cluster_nodes` | `12` | Maximum graph nodes assembled as evidence for one topic's question batch. |
| `generation.min_cluster_edge_weight` | `2.0` | Minimum shared-entity strength for two nodes to merge into one topic cluster. Higher values prevent a single hub entity from fusing the whole corpus into one topic (the knowledge-graph "hairball"). |
| `generation.max_cluster_size` | `40` | Clusters larger than this are split by weakest-edge removal so no single topic dominates. |
| `generation.topic_mode` | `entity` | How topics are formed. `entity` (default) clusters chunks by shared entities. `theme` tags each chunk with one theme from a small controlled vocabulary and groups by theme, so a subject spread thinly across the corpus (e.g. pay) becomes its own topic even when its chunks share few entities — coverage self-organizes without a hand-written topic list. Themes are a separate layer from entities and never create graph edges, so theme mode does not risk the entity hairball. Theme mode adds one corpus-level vocabulary call and re-extracts node signals once (a one-time cost). See `docs/theme-layer-ab-results.md` for the hova/keva validation. |
| `generation.extract_themes` | `false` | Tag chunks with a theme without switching clustering (useful for inspection). Implied by `topic_mode = "theme"`. |
| `generation.max_theme_vocabulary` | `20` | Ceiling on the controlled theme-vocabulary size derived for the corpus. |
| `generation.verify_answer_completeness` | `false` | When `true`, each accepted answerable canonical question is re-checked against evidence re-selected for that specific question (not just its topic). Catches reference answers left incomplete or wrong by narrow topic-driven chunk selection; the answer is corrected against the broader evidence or the candidate is rejected. Costs one extra judge/generation model call per answerable canonical candidate. |
| `generation.completeness_evidence_limit` | `16` | Maximum candidate chunks re-selected per question during answer-completeness verification. Larger values widen the recall check at higher cost. |
| `generation.prompt_max_document_chars` | `100000` | Maximum document-excerpt characters supplied to system-prompt generation; topic coverage is balanced before extra excerpts are added. |
| `generation.prompt_max_evaluation_chars` | `60000` | Maximum prior-evaluation evidence characters supplied to prompt revision. |
| `generation.prompt_max_auxiliary_chars` | `30000` | Independent maximum for the current prompt and generated-insights context. |
| `generation.prompt_instruction_profile` | `guided` | Code-assembled policy profile. `guided` includes illustrative examples; `compact` omits them. |
| `generation.prompt_answer_policy` | `balanced` | Code-assembled partial-answer policy. `balanced` permits a clearly separated supported part; `conservative` avoids an unsupported conclusion. |
| `cache.enabled` | `true` | Reuse content-addressed chunks, document topics, premade-question topics, and optional evaluation insights. |
| `cache.directory` | `.chatbot_eval_cache` | Shared local cache directory. Relative paths are resolved next to the selected config file. |
| `evaluation.request_timeout_seconds` | `60` | Timeout for each live chatbot HTTP request. |
| `evaluation.max_retries` | `2` | Gemini retries after the initial attempt, with exponential backoff. |
| `evaluation.chatbot_max_retries` | `2` | Retries for transient chatbot failures (408, 429, selected 5xx, timeouts, and connection failures). Authentication and malformed responses are not retried. |
| `evaluation.chatbot_retry_base_seconds` | `0.5` | Initial chatbot exponential-backoff delay. A valid `Retry-After` header takes precedence. |
| `evaluation.chatbot_pacing_seconds` | `0.0` | Minimum delay between sequential live-chatbot requests. |
| `evaluation.chatbot_max_response_bytes` | `5000000` | Maximum accepted chatbot response size. |
| `evaluation.chatbot_require_json_content_type` | `true` | Reject live responses whose media type is not JSON or `+json`. |
| `evaluation.judge_max_answer_chars` | `20000` | Maximum chatbot-answer characters placed in one judge prompt. Longer values retain the beginning and end and record the omitted count. |
| `evaluation.judge_max_context_chars` | `60000` | Maximum retrieved-context characters placed in one judge prompt, with the same bounded truncation metadata. |
| `evaluation.insights_max_prompt_chars` | `80000` | Approximate total character budget for optional insight evidence. Risky and failed results are prioritized. |
| `evaluation.gemini_request_timeout_seconds` | `120` | Per-call timeout for direct and Apigee Gemini SDK requests. |
| `evaluation.max_concurrency` | `1` | Chatbot/judge workers. Keep `1` for session-sensitive endpoints; values above `1` are opt-in. |
| `runtime.progress_enabled` | `true` | Enables English terminal progress bars. |
| `runtime.log_file` | empty | Optional operational log path. Empty disables file logging. |
| `runtime.log_level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |

### Apigee AI Gateway transport

To route question generation, judging, topic inference, insights, and prompt generation through the
proprietary endpoint, change the transport and provide the key through `.env`:

```toml
[gemini]
transport = "apigee"
apigee_api_key_env = "APIGEE_API_KEY"
apigee_base_url = "https://preprod.apigee.digital.idf.il/ai_gateway/v1/hr"
generation_model = "gemini-3.7-flash"
judge_model = "gemini-3.7-flash"
```

```dotenv
APIGEE_API_KEY=replace-me
```

Alloy configures the Google GenAI SDK in Vertex-compatible mode with an SDK placeholder key and
sends the real credential only in the `x-apikey` header. The base URL must use HTTPS and include the
AI Gateway client path. Unified quota response headers are recorded at `INFO` level as metric,
request usage, daily usage, limit, and remaining allowance; no API key or response content is logged.
The offline `report`, `review-export`, and `review-merge` commands do not initialize either Gemini
transport and need no API key.

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
3. **Extract node signals** per document (entities, keyphrases, one-line summary) while treating
   document content as untrusted data. Extraction is per-document and cached, so a later run only
   re-extracts documents that were added or edited.
4. **Build the knowledge graph**: deterministic typed edges connect nodes that share entities
   (matched with Hebrew-aware, prefix- and gershayim-tolerant normalization), overlap on keyphrases,
   or are adjacent in the same document.
5. **Derive prevalence-weighted topic clusters** (roughly 4-12) from strong semantic edges, splitting
   oversized clusters and merging tiny ones so no single hub entity fuses the whole corpus.
6. **Allocate quotas** by cluster importance (corpus coverage), minimum topic allocation, and maximum
   topic share.
7. **Assemble per-question evidence** as a bounded connected cluster of graph nodes (a topic's seed
   nodes plus their strongest neighbors), so related facts across chunks are presented together.
8. **Generate structured candidates** containing the question, answer, difficulty, type, rationale,
   exact source IDs, and verbatim supporting quotations.
9. **Validate provenance and enforce the configured answerable question-type targets** deterministically.
10. **Optionally verify answer completeness**: when `verify_answer_completeness` is enabled, each
    accepted answerable canonical candidate is re-checked against evidence re-selected for that
    specific question, so a reference answer that is still incomplete or wrong is corrected against the
    broader evidence or rejected. Verification outcomes are recorded in the generation diagnostics.
11. **Remove near-duplicates**, including matches from an optional previous silver set.
12. **Derive realistic user variations** from accepted canonical questions. Natural variants retain
    the same expected answer; deliberately ambiguous variants expect one focused follow-up question.
13. **Generate boundary cases** using the budget reserved for unanswerable questions.
14. **Assign reproducible content-derived IDs** and export for human review with
    `review_status=pending`.

The command may return fewer questions than requested. This is expected when candidates are
duplicates, evidence cannot support the requested complexity, quotations are not verbatim, source
references are invalid, or the corpus lacks enough distinct material.

The total budget is divided into canonical answerable questions, user variations, and unanswerable
questions. Variants are children of canonical questions and never replace their provenance. The
generator does not create ambiguity by adding random spelling mistakes: it removes a meaningful
discriminator such as population, status, timeframe, or requested procedure.

### Corpus-analysis cache

All Gemini-backed workflows share a persistent, content-addressed cache. `generate` and
`generate-prompt` cache local document extraction/chunking and the corpus knowledge graph;
`evaluate-file` can cache inferred question topics; both evaluation workflows can cache optional
insights. Final prompt packages are also cached from the full corpus, configuration, prior-run
evidence, current prompt, model, transport, budgets, and generator implementation. A normal repeated
run skips matching reusable stages. Question generation and judging are not ordinary cache entries:
generation has an explicit interruption checkpoint and judging has its explicit `--resume` checkpoint.

The knowledge graph is rebuilt **incrementally**. Node signals (entities, keyphrases, summary) are
cached per document under that document's content hash, so adding, editing, or deleting a document
only re-extracts the documents that actually changed; unchanged documents reuse their cached signals
with no model calls. The assembled graph (deterministic edges plus labeled topics) is cached as a
unit keyed by the full node-signal set, so it is rebuilt only when the corpus or graph parameters
change. This keeps re-runs on a large, mostly-stable corpus cheap.

Cache invalidation is based on supported document relative paths and SHA-256 content hashes,
chunk size/overlap, per-document node signals, graph and clustering parameters, Gemini model and
transport, cache schema, and the relevant implementation code. File modification timestamps are not
trusted. Changing any keyed input creates a new entry automatically; unchanged inputs reuse the
prior result.

Use `--refresh-cache` to ignore and atomically replace every matching workflow entry, including a
final prompt package, for
example after deciding an upstream model alias should be sampled again. Use `--no-cache` for a run
that must neither read nor write cache data. `--cache-dir` overrides the configured shared location.
The two switches are mutually exclusive.

The cache is local and may contain extracted document text. Protect it like the source knowledge
base, do not commit it, and choose a suitably protected directory in production. The cache directory
must be outside `--documents`, preventing cache JSON from becoming corpus input. Corrupt or
schema-incompatible entries are logged and recomputed. Old content-addressed entries are retained;
there is no automatic deletion policy.

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
| `natural_user` | How a real user actually asks: first-person, everyday words, common shorthand, and deliberately spanning a range of specificity. It may drop expert discriminators the canonical question spells out (specific level numbers, named categories) as long as the parent reference answer is still a correct response. | `answer`; the parent reference answer is reused. |
| `ambiguous` (`must_clarify`) | Plausible request missing a material discriminator, where a direct single-interpretation answer could be materially wrong or harmful. | `clarify`; `expected_answer` holds the ideal focused follow-up (only clarifying succeeds). |
| `ambiguous` (`answer_or_clarify`) | Plausible request missing a discriminator, but where the interpretations are all safe to present together, so a comprehensive answer covering every interpretation is just as good as clarifying. | `answer` against the parent's comprehensive reference, **and** a clarifying question is accepted too (`clarification_acceptable=true`; the focused follow-up is kept in `acceptable_clarification`). |

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

Rejection counts are written to operational logs at `INFO` level and to the reviewable
`generation_diagnostics.json` artifact.

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
| `--verify-answer-completeness` / `--no-verify-answer-completeness` | config value | Enable or disable the per-question answer-completeness verification pass for this run, overriding `generation.verify_answer_completeness`. |
| `--merge-into PATH` | unset | Append the newly generated questions onto an existing silver CSV/JSONL. The existing questions are kept verbatim (reviewer decisions and grounding preserved), near-duplicate new questions are dropped, and content-derived stable IDs are recomputed across the combined set. Pair with `--topic` to add a focused subset (e.g. a missing topic) to an already-reviewed set without regenerating it. |
| `--exclude-questions PATH` | unset | Existing silver CSV/JSONL whose questions participate in deduplication. |
| `--resume` | off | Replay successful structured calls from a compatible generation checkpoint, then continue after the interrupted call. |
| `--checkpoint PATH` | `<output>/generation_checkpoint.jsonl` | Durable structured-call journal used by generation resume. |
| `--cache-dir DIR` | config value | Override the shared local corpus-analysis cache directory. |
| `--refresh-cache` | off | Recompute and atomically replace matching chunk/topic entries. |
| `--no-cache` | off | Disable cache reads and writes for this run. |

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

Add a focused topic to an already-reviewed set (generate only that topic and merge it in, keeping
the existing questions and their reviewer decisions):

```bash
chatbot-eval --config config.toml generate \
  --documents ./knowledge_base \
  --topic "שכר" \
  --topic-count 15 \
  --max-questions 15 \
  --merge-into ./outputs/questions/silver_questions.jsonl \
  --output ./outputs/questions-with-salary
```

Exclude previously reviewed, rejected, or deleted questions:

```bash
chatbot-eval --config config.toml generate \
  --documents ./knowledge_base \
  --exclude-questions ./previous/silver_questions.csv \
  --max-questions 40 \
  --output ./outputs/questions-v2
```

To continue an interrupted generation without paying again for successful structured calls, repeat
the same command and add `--resume`. Alloy verifies document content, generation options, model,
transport, and implementation before replaying responses. Without `--resume`, the checkpoint is
replaced and the run remains fresh. Retry rounds now receive bounded rejection reasons so they can
avoid repeating candidates with invalid quotations, sources, types, or duplicates. Generation
`--resume` and `--refresh-cache` are mutually exclusive because refreshed topics could change every
downstream prompt.

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

`generation_diagnostics.json` records the requested and accepted totals, accepted distributions,
candidate rejection counts, unallocated capacity and its reason, topic quotas, and the source IDs
actually rendered for each topic. The same object is embedded in `run_manifest.json`, and the
artifact is listed in the manifest output inventory with its size and SHA-256 hash. A
`boundary_evidence_scope` value of `selected_excerpts` means boundary candidates were checked against
the excerpts selected and rendered for their generation calls. It does not claim that every passage
in the corpus was exhaustively searched for an answer. When no boundary cases were requested, the
value is `not_requested`.

Reviewers should confirm the user need, reference answer, sources, and quotations; edit as needed;
then set `review_status` to `approved` or `rejected`. Keep rejected/deleted questions in an exclusion
file if they must not reappear. Do not rename required CSV columns if the module will read the file.

Editing `silver_questions.csv` directly is possible but noisy: it carries provenance JSON,
verbatim supporting quotes, atomic reference claims, and CSV-escape metadata that a reviewer does
not need and can accidentally break. The review workflow below is the recommended path when a human
will read and edit the set.

## Workflow 2: human review round-trip

This workflow separates *reviewing* from *storing*. `review-export` writes a small, readable file
with only the columns a reviewer changes; `review-merge` reapplies those edits onto the original
technical set so no provenance is lost.

```text
silver_questions.jsonl (canonical, all technical columns)
        |
        v   review-export
questions_for_review.csv   (editable: readable columns only)
questions_for_review.md    (read-only pretty view)
        |   humans edit, delete rows, set review_status/reviewer_notes
        v   review-merge  (canonical + edited review file)
silver_questions.csv/.jsonl (full technical set, reviewer decisions applied)  -> evaluate
```

### Why a sidecar instead of editing the canonical file

The canonical JSONL stays the single source of truth for every technical field, so the review file
never has to reconstruct provenance. Rows are matched by the stable `id`, which the reviewer must
not edit. On merge, the canonical record is copied and only the reviewer-editable fields are
overlaid.

### review-export

```bash
chatbot-eval --config config.toml review-export \
  --questions ./outputs/questions/silver_questions.jsonl \
  --output ./outputs/review
```

`questions_for_review.csv` contains `id`, `topic`, `question`, `expected_answer`, `answerable`,
`question_form`, `expected_behavior`, `difficulty`, two read-only reference columns
(`sources_readable`, `supporting_quotes_readable`), and the decision columns `review_status` and
`reviewer_notes`. `questions_for_review.md` is a formatted, read-only view for reading only.

| Option | Required/default | Meaning |
|---|---|---|
| `--questions PATH` | required | Silver CSV or JSONL to export for review. |
| `--output DIR` | `outputs/review` | Destination for the review CSV and `.md`. |
| `--name STEM` | `questions_for_review` | Base filename (stem) for the exported CSV/`.md`, e.g. `questions_for_review_hova`, so per-corpus exports do not need manual renaming. Unsafe characters are stripped. |
| `--hebrew-columns` | off | Write Hebrew column headers for Hebrew-speaking reviewers. The `id` column keeps its English name because it is the machine join key. `review-merge` reads either language, so this choice is purely cosmetic. |

### review-merge

```bash
chatbot-eval --config config.toml review-merge \
  --canonical ./outputs/questions/silver_questions.jsonl \
  --review ./outputs/review/questions_for_review.csv \
  --output ./outputs/questions-reviewed
```

| Option | Required/default | Meaning |
|---|---|---|
| `--canonical PATH` | required | Original silver JSONL/CSV that holds all technical columns. |
| `--review PATH` | required | The edited `questions_for_review.csv`. |
| `--output DIR` | `outputs/questions-reviewed` | Destination for the merged silver CSV/JSONL. |

Merge rules, chosen to preserve as much technical information as possible:

- **Status/notes and light content edits** (`topic`, `question`, `expected_answer`, `answerable`,
  `question_form`, `expected_behavior`, `difficulty`) are overlaid onto the canonical record. All
  other technical columns are kept verbatim.
- **Rows the reviewer removed** from the review file are treated as intentional deletions and
  dropped from the merged set.
- **Rows whose `question` or `expected_answer` text was edited** keep their stored sources, quotes,
  and claims (nothing is discarded), but the row is flagged: a note is appended and an `approved`
  status is downgraded to `needs_reground`, because the stored grounding may no longer match the new
  text.

`needs_reground` is a signal, not an automatic action during evaluation. `review_status` affects
evaluation in exactly one way: `evaluate --approved-only` keeps only rows whose status is `approved`,
so a `needs_reground` row is excluded there and included (with its stale grounding) in a plain
`evaluate` run. Nothing in the evaluation pipeline regenerates grounding on its own. To refresh the
grounding of flagged questions, use the dedicated `reground` command (Workflow 3 below), which
regenerates their sources, supporting quotes, and reference claims from the documents and resets them
to `pending` for a fresh approval.
- **Rows with an `id` not in the canonical set** are rejected. New questions cannot be grounded from
  the review file; generate them instead. Duplicate `id`s in the review file are also rejected.

A `merge_diagnostics` block (removed / edited / factual-edit counts, unknown IDs) is printed and
recorded in `run_manifest.json`. The merged `silver_questions.csv`/`.jsonl` flows directly into
`evaluate` or `evaluate --approved-only`.

## Workflow 3: re-ground human-edited questions

When a reviewer edits a question or its expected answer, `review-merge` flags the row
`needs_reground` because the stored sources, supporting quotes, and reference claims may no longer
match the new text. `reground` refreshes only that grounding — it never rewrites the question or
answer. It reuses the same evidence selection and the same deterministic validation as `generate`
(verbatim-quote checks, source-ID existence, per-type document/cross-document constraints), so a
regrounded question is grounded to the same standard as a freshly generated one.

```bash
chatbot-eval --config config.toml reground \
  --questions ./outputs/questions-reviewed/silver_questions.jsonl \
  --documents ./knowledge_base \
  --output ./outputs/questions-regrounded
```

For each targeted question the model receives the fixed question and expected answer plus candidate
document chunks (ranked by overlap with the edited text, and always including any previously cited
sources), and must return fresh `source_ids`, verbatim `supporting_quotes`, and `reference_claims`
for that exact question. The model is required to echo the fixed text unchanged; if it alters the
question or answer, or if the grounding fails validation, the row is left untouched and the reason is
recorded. Successfully regrounded rows are reset to `review_status=pending` so a human re-approves
them before evaluation.

| Option | Required/default | Meaning |
|---|---|---|
| `--questions PATH` | required | Silver CSV/JSONL containing the questions to re-ground. |
| `--documents DIR` | required | Document root the questions were generated from. |
| `--output DIR` | `outputs/questions-regrounded` | Destination for the regrounded silver CSV/JSONL. |
| `--all` | off | Re-ground every answerable answer-task question, not only those marked `needs_reground`. |
| `--evidence-limit N` | `12` | Maximum candidate chunks offered to the model per question. Raise it when the true evidence is not ranked into the default window. |
| `--cache-dir DIR` / `--refresh-cache` / `--no-cache` | config value | Shared document-analysis cache controls (chunking is cached and reused). |

Only answerable answer-task questions can be regrounded. Clarification, abstention, and unanswerable
questions have no factual claims and are skipped with a recorded reason. A `run_manifest.json`
records how many were regrounded, how many failed, and why. Questions that could not be regrounded
keep their `needs_reground` status so they remain visible.

## Workflow 3b: regenerate user variations only

`revary` refreshes only the derived user variations on an existing silver set. It keeps the reviewed
canonical questions and the boundary/unanswerable questions untouched, drops the existing
natural-user and ambiguous variants, and generates fresh ones from the canonical questions. Use it
after improving variation quality (or to re-roll variants) without paying to rebuild the knowledge
graph or regenerate the reviewed canonical set.

```bash
chatbot-eval --config config.toml revary \
  --questions ./outputs/questions/silver_questions.jsonl \
  --output ./outputs/questions-revaried
```

| Option | Required/default | Meaning |
|---|---|---|
| `--questions PATH` | required | Silver CSV/JSONL whose variations should be regenerated. |
| `--output DIR` | `outputs/questions-revaried` | Destination for the refreshed silver CSV/JSONL. |
| `--variation-count N` | previous count | Total variants to generate; defaults to the number previously present, preserving the mix size. |
| `--ambiguous-variation-share R` | config value | Share of variants that should be ambiguous; `[0, 1]`. |
| `--exclude-questions PATH` | unset | Existing silver CSV/JSONL whose questions must not be reproduced as variants. |

New variants are written with `review_status=pending` and (for ambiguous ones) classified as
`must_clarify` or `answer_or_clarify`; review and approve them before evaluating. A
`revary_diagnostics` block records how many canonical questions were kept, how many old variants
were dropped, and the new variant mix.

## Workflow 4: evaluate a live chatbot

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
| `--deployment-id ID` | empty | Optional nonsecret deployment/model/tenant identity included in the evaluation contract and resume fingerprint. Change it when routing or deployed behavior changes. |
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
| `--compare-with PATH` | unset | Previous Alloy output directory or `evaluation_details.jsonl`; adds aggregate and matched-question comparison to the report. |
| `--soft-compare` | off | Show approximate deltas on matched questions even when the benchmark or evaluation contract changed. Warnings still appear but are framed as informational. Requires `--compare-with`. |
| `--retry-errors` | off | With `--resume`, retry prior `chatbot_error` and `judge_error` rows while preserving successful rows. |
| `--max-concurrency N` | config value (`1`) | Opt-in concurrent chatbot/judge workers; output order remains the question-file order. |
| `--cache-dir DIR` | config value | Cache location used for optional insights. |
| `--refresh-cache` | off | Recompute matching cached insights. |
| `--no-cache` | off | Do not read or write cached insights. |

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

To continue an interrupted run, repeat the same command with `--resume` and the same output
directory (or the same explicit `--checkpoint` path):

```bash
chatbot-eval --config config.toml evaluate \
  --questions ./outputs/questions/silver_questions.csv \
  --approved-only \
  --chatbot-url http://localhost:8080/ask \
  --output ./outputs/run-001 \
  --resume
```

Resume compatibility includes the judge model and implementation, transport, judge input limits,
endpoint hash, response-field mapping, header names, chatbot retry/size settings, and concurrency.
Changed contracts are rejected rather than silently mixing results. Authentication header values
are never written to checkpoints or manifests. `--retry-errors` deliberately reprocesses only prior
infrastructure/judge failures; ordinary completed results remain unchanged.
`--retry-errors` is rejected unless `--resume` is also present.
Evaluation checkpoints created before version 0.7 do not contain a contract signature and therefore
cannot be resumed under 0.7; start one fresh run to create the strengthened format.

Timeouts, connection failures, invalid JSON, unexpected content types, oversized responses, and
missing answer fields become categorized `chatbot_error` records and are excluded from quality
denominators. The generic adapter retries only transient failures with bounded exponential backoff,
honors `Retry-After`, and supports optional pacing. Production fleet adapters should still own
authentication, idempotency, sessions, and organization-specific formats.

## Workflow 5: evaluate a premade Q&A file

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
| Expected behavior | `expected_behavior`, `target_behavior` |
| Question form | `question_form`, `form` |
| Parent question ID | `parent_question_id`, `parent_id`, `source_question_id` |
| Question type | `question_type`, `type`, `kind` |
| Difficulty | `difficulty`, `complexity` |
| Reference claims | `reference_claims`, `gold_claims`, `required_claims` |
| Supporting quotes | `supporting_quotes`, `evidence_quotes` |
| Explicit reference sources | `reference_sources`, `reference_sources_json`, `gold_sources`, `gold_evidence`, `sources_json` |

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
| `--expected-behavior-column NAME` | optional/auto | Expected `answer`, `clarify`, or `abstain` behavior. |
| `--question-form-column NAME` | optional/auto | `canonical`, `natural_user`, or `ambiguous`. |
| `--parent-question-id-column NAME` | optional/auto | External parent ID; resolved to the imported parent row's Alloy ID. |
| `--question-type-column NAME` | optional/auto | Silver-question type such as `basic_knowledge` or `document_wide`. |
| `--difficulty-column NAME` | optional/auto | Difficulty label retained on the imported question. |
| `--reference-claims-column NAME` | optional/auto | JSON array of atomic reference claims. |
| `--supporting-quotes-column NAME` | optional/auto | JSON array of `{source_id, quote}` objects. |
| `--reference-sources-column NAME` | optional/auto | JSON array of explicit gold `{source_id, file, location, excerpt}` objects. |
| `--infer-topics` | off | Infer broad Hebrew topics for unclassified rows, bounded by both 250 rows and 60,000 rendered characters per batch. |
| `--generate-insights` | off | Write and embed optional cross-result insights. |
| `--hide-correct-answer-metrics` | off | Hide correctness presentation metrics but retain data. |
| `--strict` | off | Reject the entire import if any row is invalid instead of logging and skipping invalid rows. |
| `--resume` | off | Reuse completed rows whose imported inputs still match. |
| `--checkpoint PATH` | `<output>/evaluation_checkpoint.jsonl` | Append-only per-question checkpoint. |
| `--compare-with PATH` | unset | Previous Alloy output directory or `evaluation_details.jsonl`; adds comparison to the report. |
| `--soft-compare` | off | Show approximate deltas on matched questions even when the benchmark or evaluation contract changed. Warnings still appear but are framed as informational. Requires `--compare-with`. |
| `--retry-errors` | off | With `--resume`, retry prior judge errors while preserving imported chatbot errors and successful rows. |
| `--max-concurrency N` | config value (`1`) | Opt-in concurrent judge workers; final row order remains stable. |
| `--cache-dir DIR` | config value | Shared cache for inferred topics and optional insights. |
| `--refresh-cache` | off | Replace matching topic-inference and insight cache entries. |
| `--no-cache` | off | Disable those cache reads and writes. |

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
fault payloads, Spike Arrest messages, and the known answer-generation placeholder (`לא הצלחנו ליצור סיכום
לשאילתת החיפוש שלך, אבל כן מצאנו כמה תוצאות.`) are infrastructure errors and are not sent to Gemini. The
importer does not retry stored errors. Imported fault details are categorized and bounded so raw
gateway payloads are not propagated into reports.

The expected-answer column is the judging reference. The optional source column is treated as the
chatbot's retrieved context for diagnostics, not as a separately reviewed gold reference. Gold
document evidence must use an explicit reference-source column (or Alloy's canonical `sources_json`);
it remains separate from runtime retrieval. If multiple aliases for one logical field occur, the
import fails until the corresponding explicit `--*-column` mapping selects one, so metadata is not
chosen silently. Parent values refer to external IDs in the imported file and unresolved parents are
cleared while the original value is retained in result metadata.

## Workflow 6: generate a system prompt from documents

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
| `--previous-run PATH` | unset | Previous Alloy output directory or evaluation JSONL. Standard results and insights are auto-discovered in a directory. |
| `--insights PATH` | unset | Explicit `evaluation_insights.json`; overrides insights discovered through `--previous-run`. |
| `--current-prompt PATH` | unset | Current Markdown/text prompt, `prompt_package.json`, or prompt output directory to revise conservatively. |
| `--cache-dir DIR` | config value | Override the shared local corpus-analysis cache directory. |
| `--refresh-cache` | off | Recompute and atomically replace matching chunk, topic, and final prompt-package entries. |
| `--no-cache` | off | Disable cache reads and writes for this run. |

```bash
chatbot-eval --config config.toml generate-prompt \
  --documents ./knowledge_base \
  --assistant-name "תומי" \
  --audience "סגלי משאבי אנוש ותנאי שירות" \
  --output ./outputs/tomi-prompt
```

For an evidence-guided revision, keep the current prompt explicit and point to a completed run:

```bash
chatbot-eval --config config.toml generate-prompt \
  --documents ./knowledge_base \
  --assistant-name "תומי" \
  --audience "סגלי משאבי אנוש ותנאי שירות" \
  --previous-run ./outputs/run-002 \
  --current-prompt ./outputs/tomi-prompt \
  --output ./outputs/tomi-prompt-v2
```

All records contribute to aggregate diagnostics. Detailed model context is bounded and prioritizes
failed, evaluable questions; it excludes raw answers and expected answers. The generator is told to
change only prompt-addressable behavior and to route retrieval, corpus, infrastructure, permission,
and application-security problems to guardrails or manager review instead of hiding them in wording.
Document evidence is selected round-robin across discovered topics before extra chunks are added, so
a large high-priority topic cannot silently consume the entire prompt. Current-prompt, prior-run, and
insight inputs have separate limits. Identical prompt-generation inputs reuse the cached validated
package; use `--refresh-cache` when a deliberately new sample is wanted.

Outputs:

| File | Contents |
|---|---|
| `generated_system_prompt.md` | Ready-to-review Hebrew system prompt. |
| `prompt_package.json` | Corpus scope, assumptions, guardrails, manager checklist, suggested tests, revision summary, and evidence question IDs. |

Managers must review both files before deployment. In particular, verify every described scope,
remove unsupported rules, decide approved fallback/escalation behavior, and implement the listed
application guardrails outside the model. The prompt generator itself does not deploy or modify a
chatbot.

## Workflow 7: regenerate and compare reports

The `report` command creates JSON/HTML from completed Alloy evaluation JSONL without contacting the
chatbot or Gemini. It accepts either an output directory or its `evaluation_details.jsonl` file and
can also load older pre-claim-assessment artifacts through a read-time compatibility migration.

```bash
chatbot-eval --config config.toml report \
  --results ./outputs/run-002 \
  --compare-with ./outputs/run-001 \
  --insights ./outputs/run-002 \
  --output ./outputs/run-002-comparison
```

The comparison presents whole-run values and a strict like-for-like view. Questions are matched only
when the ID and every material silver-question field agree after whitespace normalization, including
wording, reference answer and claims, evidence, answerability, topic, type, form, provenance, and
expected behavior. Changed rows are flagged rather than compared, and duplicate IDs are excluded
instead of silently overwriting one another. Aggregate KPI deltas are calculated only when both
runs contain the same unique benchmark; otherwise the two absolute values remain visible but the
delta is marked non-comparable. Matched outcomes still show improvements, regressions, persistent
successes/failures, and topic-level movement.

### Why the default is strict, and when to relax it

The strictness is deliberate. Suppressing deltas when the question set or the evaluation contract
changed prevents a composition change (added, removed, or edited questions) or a configuration
change (a different judge model, transport, or judge input limits) from being read as a change in
chatbot performance. This is the right default, and it matters more once the review round-trip is in
use, because human review routinely adds, edits, or removes questions between runs.

But strictness is sometimes too conservative. After a deliberate review pass or a model upgrade, you
may still want a directional read of how the chatbot moved on the questions that stayed the same.
`--soft-compare` provides that. It keeps every warning visible but frames them as informational, and
it computes an **approximate** delta for each metric over the questions that are matched (same ID and
identical material fields) **and eligible for that metric in both runs**. Using that shared
intersection as the denominator means the soft delta is defined even when the strict delta was
suppressed only because a few questions errored in one run but not the other (the common
`eligible_population_changed` case). Soft deltas are rendered with a `≈` marker and the count of
matched questions they are based on, so they read as an estimate rather than a measured KPI change.
A metric still shows no soft delta only when there is genuinely no matched question eligible for it in
both runs. When the benchmark and contract are identical and no error populations shifted,
`--soft-compare` changes nothing: the exact deltas are already shown.

| Option | Required/default | Meaning |
|---|---|---|
| `--results PATH` | required | Current output directory or `evaluation_details.jsonl`. |
| `--compare-with PATH` | unset | Previous output directory or `evaluation_details.jsonl`. |
| `--insights PATH` | unset | Optional `evaluation_insights.json` or its output directory. |
| `--hide-correct-answer-metrics` | off | Hide correctness indicators in the HTML; underlying data is preserved. |
| `--soft-compare` | off | Show approximate matched-question deltas even when the benchmark or contract changed; warnings stay but are informational. Requires `--compare-with`. |

```bash
chatbot-eval --config config.toml report \
  --results ./outputs/run-002 \
  --compare-with ./outputs/run-001 \
  --soft-compare \
  --output ./outputs/run-002-comparison
```

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
| `evaluation_insights_status.json` | Always written; records whether insights were disabled, generated, or failed and the evaluation-record fingerprint. |
| `evaluation_checkpoint.jsonl` | Durable per-question records used by `--resume`. |
| `run_manifest.json` | Redacted settings, input and implementation hashes, versions, timing, status, and result counts. |

The `report` command writes `report_manifest.json` so report-generation metadata cannot be confused
with an evaluation run manifest. A previous-run directory only supplies `evaluation_insights.json`
when its status file matches both the evaluation-record hash and the insights-artifact hash; failed
or disabled status therefore cannot accidentally reuse a stale artifact.

The HTML includes KPI cards, outcome distribution, retrieval-versus-answer diagnostics, information
metrics, topic performance as `percentage (count/denominator)` with 95% Wilson intervals and small-
sample warnings, parent/variant robustness metrics, optional previous-run comparison, optional
insights before per-question details, and searchable/filterable question drill-down.

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

Insight evidence is globally bounded by `evaluation.insights_max_prompt_chars`; risky and failed
outcomes are considered before successes. Identical completed records reuse a content-addressed
insight result unless `--refresh-cache` or `--no-cache` is selected. Likewise,
`evaluate-file --infer-topics` caches assignments by question ID/text, model, transport, batching,
and implementation. Prompt-package cache entries also include the selected instruction profile,
answer policy, and configured model/deployment identity, so changing any of these cannot reuse an
incompatible package.

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

Gemini calls—direct or through Apigee—retry only transient timeouts, connection problems, throttling,
and selected server errors according to `evaluation.max_retries`, with jittered exponential backoff
and `Retry-After` support. Authentication, configuration, and schema-validation errors fail
immediately. Every call uses `evaluation.gemini_request_timeout_seconds`. When Apigee returns unified quota headers, Alloy logs the selected metric, request usage,
daily limit/usage, and remaining allowance without logging credentials. Live chatbot calls remain
sequential and use the separately configured transient retry and pacing policy.
Authentication failures and malformed responses fail immediately; rate-limit responses honor
`Retry-After` when supplied.

Evaluation writes a checkpoint after every completed chatbot/judge result. `--resume` validates
input and evaluation-contract fingerprints before reusing records and rejects mismatched checkpoints.
One torn final checkpoint line is discarded safely; earlier corruption remains a hard failure. Final JSON and HTML,
prompt packages, insights, and manifests use atomic replacement. Manifests omit API keys and record
only a hash of the live chatbot URL.

## Code map and extension points

| Module | Responsibility |
|---|---|
| `cli.py` | CLI parsing, configuration wiring, workflow orchestration. |
| `config.py` | TOML and `.env` loading. |
| `cache.py` | Content-addressed chunk/topic caching, invalidation, and atomic replacement. |
| `documents.py` | File discovery, extraction, chunking, provenance. |
| `generator.py` | Topics, quotas, generation, validation, deduplication. |
| `history.py` | Compatible loading of previous evaluations, insights, and current prompts. |
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
repair/revision, historical artifact compatibility, paired variant metrics, run comparisons,
confidence intervals, and reporting options.

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
- Corpus analysis is cached as whole content-addressed snapshots; per-file incremental extraction within a changed corpus is not yet implemented.
- Generation resume replays successful structured responses; it does not attempt to continue from a partially returned model response.
- Concurrent evaluation is intended only for stateless endpoints and available Gemini/chatbot quota; session-sensitive chatbots should keep the default of one worker.
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
