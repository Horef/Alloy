# Chatbot Evaluation POC

Two standalone Python components for a fleet of internal knowledge chatbots:

1. **Silver-set generation** reads local documents, discovers central topics, allocates a balanced
   question budget, generates evidence-backed questions with Gemini, removes near-duplicates, and
   exports a human-reviewable CSV plus lossless JSONL.
2. **Evaluation** sends approved questions to any JSON-over-HTTP chatbot, judges answers against the
   reviewed reference and source evidence, and creates CSV/JSONL details plus a simple HTML report.
3. **Premade-result evaluation** reads existing `.xlsx`, `.csv`, or `.jsonl` question/answer exports
   and runs the same judge without contacting the chatbot again.

The package does not import or require the existing chatbot platform. Integration is isolated in the
`ChatbotAdapter` protocol and the included generic HTTP adapter.

## Why this design

A one-shot question-generation prompt tends to over-sample salient details and under-sample broad
coverage. This POC uses a staged pipeline:

```text
local files -> text chunks -> topic map -> importance-weighted quotas
            -> evidence-scoped generation -> validation/dedup -> human review
```

The quota cap prevents one large topic from dominating. A requested topic can override the general
allocation, and the generator intentionally returns fewer than requested when the evidence cannot
support distinct questions. Optional, realistic unanswerable questions measure whether a chatbot
knows when to abstain.

The evaluator does not collapse everything into one opaque score. It reports these operational
outcomes:

| Outcome | Meaning |
|---|---|
| `correct_answer` | Answerable and correctly answered |
| `partial_too_little` | Useful core answer, but necessary information is missing |
| `partial_too_much` | Core answer is present, but excessive unrelated detail reduces quality |
| `unrelated_answer` | Clearly off-topic/nonresponsive and unlikely to be mistaken as correct |
| `misleading_hallucination` | Plausible or assertive but materially false/unsupported |
| `incorrect_abstention` | Said “I don't know” although the corpus contains the answer |
| `correct_abstention` | Correctly declined an unanswerable question |
| `should_have_abstained` | Invented/attempted an answer to an unanswerable question |
| `chatbot_error` / `judge_error` | Infrastructure failure, kept out of quality scores |

Scores for correctness, completeness, relevance, and groundedness remain available for diagnosis.
The deterministic abstention pass reduces unnecessary judge calls and makes the answerability
confusion matrix explicit.

## Setup

Use the requested environment:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/army/bin/python -m pip install -e '.[dev]'
cp config.example.toml config.toml
cp .env.example .env
```

Put the Gemini key in `.env`; `.env` is gitignored. `config.toml` contains model names and tuning
parameters, but no secret. The default is the stable `gemini-2.5-flash`; model IDs are configurable.

Supported local inputs are `.txt`, `.md`, `.rst`, `.csv`, `.json`, `.jsonl`, `.pdf`, and `.docx`.
Scanned PDFs require OCR before this POC can extract their text.

## Generate questions

General representative set:

```bash
chatbot-eval --config config.toml generate \
  --documents ./knowledge_base \
  --max-questions 40 \
  --output ./outputs/questions
```

Up to 12 questions about a user-selected topic:

```bash
chatbot-eval --config config.toml generate \
  --documents ./knowledge_base \
  --topic "leave approval process" \
  --topic-count 12 \
  --max-questions 12 \
  --output ./outputs/leave_questions
```

Open `silver_questions.csv` in Excel. Reviewers should change `review_status` from `pending` to
`approved` or `rejected`, edit the question/reference answer if needed, and leave notes. Keep the
UTF-8 CSV columns intact. JSONL retains structured source provenance for programmatic workflows.

## Evaluate a chatbot

For an endpoint accepting `{"question": "..."}` and returning
`{"answer": "...", "retrieved_context": "..."}`:

```bash
chatbot-eval --config config.toml evaluate \
  --questions ./outputs/questions/silver_questions.csv \
  --approved-only \
  --chatbot-url http://localhost:8080/ask \
  --output ./outputs/run-001
```

Nested response fields are supported with dotted paths, for example
`--answer-field data.message.answer`. Authentication headers can refer to environment variables:

```bash
chatbot-eval ... --header 'Authorization=Bearer $CHATBOT_TOKEN'
```

For fleet integration, implement `ChatbotAdapter.ask(SilverQuestion) -> ChatbotResult`; the
generation and judging code remains unchanged.

### Evaluate an existing results file

The bundled importer recognizes common English column names and the Hebrew headers in
`qa-miluim-5_8-results.xlsx`: `שאלה`, `תשובה צפויה`, `תשובה מהמודל`, `מקור`, and `messageId`.

```bash
chatbot-eval --config config.toml evaluate-file \
  --results ./qa-miluim-5_8-results.xlsx \
  --sheet Sheet1 \
  --output ./outputs/qa-miluim-evaluation
```

For a different export schema, pass explicit mappings such as:

```bash
chatbot-eval --config config.toml evaluate-file \
  --results ./custom-results.xlsx \
  --question-column prompt \
  --expected-answer-column gold_answer \
  --answer-column bot_response \
  --source-column retrieved_chunks
```

The required fields are question, expected/reference answer, and chatbot answer. Source, ID, topic,
answerability, and explicit error columns are optional. Empty answers and recognizable API fault
payloads are reported as `chatbot_error` and are not sent to the judge.

The importer does not retry errors already stored in a results file: they describe the earlier
chatbot run, not the current Gemini judging run. Such rows remain visible as infrastructure failures
but are logged at `INFO`, not as new runtime warnings. Automatic function calling is explicitly
disabled because generation and judging use structured output only and expose no tools.

If the file has no topic column, add `--infer-topics` to make an additional Gemini clustering call
(one call per 250 questions) and assign concise Hebrew topic labels before evaluation:

```bash
chatbot-eval --config config.toml evaluate-file \
  --results ./qa-miluim-5_8-results.xlsx \
  --infer-topics \
  --output ./outputs/qa-miluim-evaluation
```

## Hebrew reports and retrieval diagnostics

Judge explanations, result labels, CSV headers, and the interactive HTML report are in Hebrew. The
HTML report is right-to-left and includes KPI cards, outcome and score charts, a retrieval-versus-
generation pipeline view, per-topic statistics, and searchable/filterable question drill-down with
the original question, reference answer, chatbot answer, retrieved chunks, and judge explanations.

Retrieved chunks are scored independently from 0 to 4 for:

- relevance to the question;
- correctness against the reviewed reference answer;
- completeness/coverage of the facts needed to answer.

`0` means the export did not include retrieved context; it is treated as missing telemetry and is
excluded from retrieval averages. “Good retrieval + incorrect answer” is reported separately so it
is easy to identify cases where retrieval found the right information but generation failed to use
or summarize it.

## Progress and operational logs

Progress bars cover document loading, topic discovery, question generation, premade-file ingestion,
and judging. Disable them in production either globally in `config.toml`:

```toml
[runtime]
progress_enabled = false
```

or for one command with `--no-progress` (global options go before the subcommand):

```bash
chatbot-eval --config config.toml --no-progress evaluate-file --results ./results.xlsx
```

File logging is opt-in and records run counts, row/question IDs, outcomes, latency, and errors without
logging complete questions, answers, API keys, or authentication headers:

```bash
chatbot-eval --config config.toml \
  --log-file ./outputs/logs/evaluation.log \
  --log-level INFO \
  evaluate-file --results ./results.xlsx
```

The same defaults can be set with `runtime.log_file` and `runtime.log_level` in `config.toml`.

## Recommended evaluation process

- Human-review every silver reference answer and its evidence before using it as a release gate.
- Start with mostly answerable, representative questions plus 5-15% plausible unanswerable cases.
- Freeze a reviewed regression set; generate a separate rotating set to reduce overfitting.
- Calibrate judge thresholds on a small double-reviewed human sample and track agreement.
- Keep judge prompts/model versions and raw outputs with every run. Never count transport or judge
  failures as chatbot failures.
- If retrieval context becomes available, separately inspect retrieval coverage and answer
  groundedness; an end-to-end score alone cannot locate whether retrieval or generation failed.

## Research basis and limitations

Gemini's official Python SDK supports JSON-schema-constrained output, which this POC uses to avoid
fragile JSON parsing. Google's Files API is preferable for repeatedly processing very large PDFs;
this local-first POC instead extracts text locally to minimize uploaded data and maintain exact
file/chunk provenance:

- https://ai.google.dev/gemini-api/docs/structured-output
- https://github.com/googleapis/python-genai
- https://ai.google.dev/gemini-api/docs/document-processing

The metric split follows the RAGAS observation that retrieval quality, groundedness/faithfulness,
and answer quality are distinct dimensions. The explicit answerable/unanswerable outcome matrix is
inspired by Abstain-QA. LLM judges can exhibit position, verbosity, style, and self-family biases;
this POC uses pointwise rubric judging with a reference and evidence, but it still requires human
calibration and periodic audit:

- https://arxiv.org/abs/2309.15217
- https://aclanthology.org/2025.coling-main.627/
- https://arxiv.org/abs/2306.05685

This is a POC, not a statistically validated benchmark. Generated references are “silver,” not
ground truth, until a domain reviewer approves them.
