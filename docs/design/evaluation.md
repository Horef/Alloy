# 13. Evaluation (`adapters.py`, `results_io.py`, `evaluator.py`)

Status: current (verified 2026-10-27)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md). Citations such as [R1] are listed in [references.md](references.md).

## 13.1 Getting answers

- **Live** (`evaluate`): `HttpChatbotAdapter` POSTs `{question_field: question}` and reads the answer
  and retrieved context from dotted response paths. Transient failures (408, 429, 5xx, timeouts,
  connection errors) are retried with backoff that honors `Retry-After`; authentication errors and
  malformed responses are not. Responses must be JSON, under a size limit, with a non-blank string
  answer. A known "summary could not be generated" placeholder is classified as an infrastructure
  error, not an answer. Concurrency defaults to 1 because many chatbots are session-sensitive.
- **Saved** (`evaluate-file`): `.xlsx`, `.csv`, or `.jsonl`, with columns found by alias (English or
  Hebrew) or explicit `--*-column` flags. Two columns matching one field is an error rather than a
  silent choice. Stored errors, empty answers, gateway fault payloads, and throttling messages become
  `chatbot_error`. The retrieved-context column is kept strictly separate from gold reference
  evidence.

## 13.2 Input-boundary validation (`validation.py`)

Every question set is normalized and checked when it enters the system (CSV, JSONL, premade import):
legacy defaults are applied explicitly, booleans are parsed strictly (a typo must not silently turn a
question into an abstention task), and task invariants are enforced (an answer task is answerable; a
clarify task is an answerable ambiguous question without claims; an abstain task is unanswerable
without claims; anchors are canonical; broad questions are answer tasks). Answer tasks with no claims
get claims derived deterministically from the reference answer, one per sentence or line.

## 13.3 The judge

One judge call per question receives the question, expected behavior, reference answer, the fixed
reference claims (`C001`, `C002`, ...), the gold source evidence and verified quotes, the candidate
answer, and the chatbot's retrieved context. Long answers and contexts are cut to configured limits,
keeping the beginning and end, and the omitted character count is recorded.

The judge returns **counts and per-claim assessments, not a grade**:

- for each fixed claim: addressed? correct?;
- false claims (contradicted by the reference), unsupported claims (likely fabricated and risky),
  extraneous claims (true but unnecessary);
- retrieval: required points present in the context, chunks total / relevant / contradictory;
- whole-response flags: abstention, clarification; answer scope; incorrect type.

Several rules in the judge prompt deserve emphasis because they are frequent sources of bad judges:

- **The retrieved context is not the correctness oracle.** A claim is never false just because the
  chatbot's retrieval did not contain it; correctness is judged only against the reference and gold
  evidence. Retrieval is scored separately.
- **A true elaboration is not a hallucination.** Consistent extra detail counts as extraneous, not
  unsupported.
- **No outside knowledge.**

`apply_fixed_claims` then validates in code that the judge returned exactly one assessment per claim
ID, and **recomputes** required/addressed/correct totals from those assessments, so the aggregate
counts cannot disagree with the per-claim evidence. Pydantic enforces count invariants
(`correct ≤ addressed ≤ required`, `relevant ≤ total`, and so on). Any violation becomes a
`judge_error`.

*Why this design:* holistic 1-10 judge scores are known to be noisy and biased toward length and
style [R28]; counting checkable units against fixed claims is more stable and auditable [R2, R3, R4].
Fixing the claims in advance, at generation and review time, means the judge cannot quietly change
what "complete" means from run to run.

## 13.4 From scores to an outcome: `classify`

The outcome is a deterministic function of the scores and the question's task, evaluated in this
order:

1. Abstention and clarification both set: inconsistent judge output, so `judge_error`.
2. Any false claim, or the judge called it a hallucination: **`misleading_hallucination`**. Safety is
   checked first, so a refusal or clarifying question that also asserts something false is not a
   success.
3. Clarify task: `correct_clarification` if the response is a clarification, else
   `missing_clarification`.
4. `answer_or_clarify` question and the response is a clarification: `correct_clarification`.
5. Answer task and the response is an abstention: `incorrect_abstention`.
6. Abstain task: `correct_abstention` if it abstained, else `should_have_abstained`.
7. Answer task and the response is a clarification: `partial_too_little` if any claim was correct,
   else `incorrect_abstention` (asking instead of answering defers the answer).
8. Off-topic with no correct claim: `unrelated_answer`.
9. All required claims correct (for broad questions, `min_key_points`) and not too long (broad
   overviews are exempt): `correct_answer`.
10. Some claims correct: `partial_too_much` if the scope is too long, else `partial_too_little`.
11. Nothing correct: `misleading_hallucination` if a required point was answered wrongly
    (addressed but not correct) or a likely fabricated claim was asserted; otherwise the reply is
    nonresponsive ("contact your HR office") and is `unrelated_answer`, so it does not inflate the
    misinformation rate.

Unsupported claims alone never override an otherwise correct answer; they stay visible in the
`factual_risk` metric. An earlier version also forced "abstention" whenever the answer contained a
phrase like "not provided"; that override was removed because it turned correct answers that mention
one missing attachment into failures (finding E1 in [library-review.md](../archive/library-review.md)).

## 13.5 Checkpoints and contracts

Each judged record is appended to `evaluation_checkpoint.jsonl` with a per-question input fingerprint
and a run signature over the **evaluation contract**: judge model, judge implementation hash, model
route, input limits, and for live runs the endpoint hash, field mapping, header *names*, retry,
deadline, and pacing settings, deployment ID, and concurrency. Each premade row's fingerprint covers
only that row, so fixing one bad row in the source file does not invalidate the others. `--resume` rejects a checkpoint from a different
contract instead of mixing results; `--retry-errors` re-runs only infrastructure and judge errors
(judge-only retries reuse the saved chatbot answer).
