# Chatbot Evaluation POC

Two standalone Python components for a fleet of internal knowledge chatbots:

1. **Silver-set generation** reads local documents, discovers central topics, allocates a balanced
   question budget, generates evidence-backed questions with Gemini, removes near-duplicates, and
   exports a human-reviewable CSV plus lossless JSONL.
2. **Evaluation** sends approved questions to any JSON-over-HTTP chatbot, judges answers against the
   reviewed reference and source evidence, and creates CSV/JSONL details plus a simple HTML report.

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
| `partial_answer` | Useful, but materially incomplete or partly wrong |
| `incorrect_answer` | Answerable, but answer is wrong |
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
