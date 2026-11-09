# 13. Evaluation (`adapters.py`, `results_io.py`, `evaluator.py`)
Status: current (verified 2026-11-09)

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

**Evidence the judge reads (D18).** Stored source excerpts are only the first 500 characters of a
chunk, so a true added detail further into the chunk looked unsupported. With a document root
(`--documents`) the judge gets the full text of each cited chunk (matched by `source_id`, and only
when the excerpt is the start of that chunk) and `evaluation.judge_kb_passages` extra
knowledge-base passages, found lexically from the question plus the start of the answer and excluding
the cited chunks. Each block is bounded by `evaluation.judge_max_evidence_chars`; what was read and
omitted is recorded in `result.metadata["judge_evidence"]`. Without `--documents` behavior is the
previous excerpt-only evidence.

**Claim-targeted evidence (D19).** The question-based passages are found from the question and the
first 1500 characters of the answer, so a detail late in a long answer is poorly represented in the
query. `evaluation.judge_claim_passages` more passages are therefore retrieved with each of the
answer's *specific sentences* alone (those with a digit or a quoted name, at most eight, three
tokens or more), round-robin by rank, skipping cited chunks and passages already found; their ids
are in `judge_evidence.claim_passage_ids`. Lexical retrieval only, no model call, inside the same
`judge_max_evidence_chars` cap.

**Second pass on contradictions (D19).** The first judge over-reported contradictions (comparing a
value for one entitlement with another's). After `apply_fixed_claims`, `Evaluator._verify_false_claims`
checks every claim in `false_claim_details`: `locate_quote` looks for the quote, token by token (the
retrieval tokenizer: case, niqqud, punctuation and spacing ignored; "..." parts must all occur in
order), in the reference answer, cited chunks, supporting quotes and retrieved passages. Not found:
the claim is demoted to unsupported with verdict `quote_not_found` and no call is made. The rest go
in **one** `VERIFY_PROMPT` call per row with the question, the claims, their quotes and, per quote,
the block it was found in, windowed to `evaluation.judge_verify_evidence_chars` around the quote.
Per claim the verifier returns `contradicts` (stays false), `supports` (no longer false and *not*
unsupported: the claim is true) or `unrelated` (demoted to unsupported) with a Hebrew reason; a
response that does not return exactly one verdict per index is a `judge_error`. Decisions are stored
in `scores.false_claim_verifications` and the call count in `scores.verification_calls`.
`evaluation.judge_verify_false_claims = false` skips the pass. The pass runs before `classify`, so
the outcome rules are unchanged: they only see the surviving false claims. A limit: the verifier is
the same model family reading the same evidence, so it removes slips of comparison, not gaps in the
knowledge base.

**Targeted second vote (D20).** The noise study (the same v4 judge run twice on the corpus B set) showed
the first judge's own variance: real hallucinations losing their false claim, rows swapping between
`correct_answer` and `partial_too_much`. `Evaluator._judge_pass` is one complete vote (judge call,
`apply_fixed_claims`, the D19 pass, `classify`). After the first vote, `needs_second_vote` selects
uncertain rows: `answer_false_claims > 0`, `demoted_false_claims > 0`, a verification verdict other than
`contradicts`, a `requires_verification` item, or the outcome `misleading_hallucination` or
`unrelated_answer`. For those, `Evaluator._second_vote` runs `_judge_pass` again with `VOTE_NOTE` (vote
index 2, "assess from scratch") appended to the prompt, so the call-cache key differs from the first
vote's. `evaluation.second_vote` decides: `harsher` keeps the outcome with the higher
`models.OUTCOME_SEVERITY` rank, `lenient` the lower, `off` skips the call; equal rank keeps the first
vote. The severity order, most severe first, is `misleading_hallucination`, `should_have_abstained`,
`missing_clarification`, `incorrect_abstention`, `unrelated_answer`, `partial_too_little`,
`partial_too_much`, then the success outcomes (`correct_answer`, `correct_abstention`,
`correct_clarification`) sharing one rank. It agrees with `report.RISKY` (the first three) and
`BEHAVIOR_FAILURE_OUTCOMES` (the first four, with hallucination outside it). The kept vote's scores
become the record's scores; `vote_outcomes` holds both outcomes, `votes_disagree` is true when they
differ, `second_vote_calls` is 1 and `verification_calls` sums both votes. A second vote that raises is
logged, recorded in `second_vote_error` and leaves the first vote. In the study about 42 of 842 rows were
uncertain (about 0.55M tokens). The setting is part of the evaluation contract, so changing it makes an
older checkpoint non-resumable; records written before D20 load with the new fields at their defaults.

The judge returns **counts and per-claim assessments, not a grade**:

- for each fixed claim: addressed? correct?;
- false claims (contradicted by the reference or evidence, each with a verbatim quote of the
  contradicting text in `false_claim_details`), unsupported claims (not found in the evidence, but
  not contradicted), extraneous claims (true but unnecessary);
- `requires_verification`: specific values (numbers, dates, amounts, conditions) the evidence neither
  supports nor contradicts. It is stored per row, shown in the report with a count, rate and filter,
  and **never changes the outcome**: it is a to-do list for a human reviewer. The judge is asked for
  it through `JudgeResponse` (`JudgeScores` with this list and `false_claim_details` marked required
  in the JSON schema); a defaulted field is otherwise dropped by structured-output models, which
  left the flag empty on every row of the first live run;
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
counts cannot disagree with the per-claim evidence. `_reconcile_false_claims` applies the same
principle to false claims: only those with a non-blank `contradicting_evidence` quote count; the
rest are demoted to unsupported (`demoted_false_claims` records how many), and an
`incorrect_type` of hallucination without a cited claim is rewritten. Pydantic enforces count invariants
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
2. Any contradicted (cited) false claim: **`misleading_hallucination`**. Safety is
   checked first, so a refusal or clarifying question that also asserts something false is not a
   success, and one false claim on an otherwise correct answer still counts. The judge's
   `incorrect_type` label no longer decides this, and unsupported claims never do (D18).
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
11. Nothing correct and no contradicted claim: `unrelated_answer` (a wrong or fabricated answer
    that contradicts nothing, and a nonresponsive reply such as "contact your HR office", are both
    kept out of the misinformation rate; an unsupported claim stays visible in `factual_risk`).

Unsupported claims alone never override an otherwise correct answer or produce a hallucination;
they stay visible in the `factual_risk` metric. Abstain and clarify tasks are judged on behavior
(the judge is told that added general detail is not a contradiction) unless a claim is contradicted.
An earlier version also forced "abstention" whenever the answer contained a
phrase like "not provided"; that override was removed because it turned correct answers that mention
one missing attachment into failures (finding E1 in [library-review.md](../archive/library-review.md)).

## 13.5 Checkpoints and contracts

Each judged record is appended to `evaluation_checkpoint.jsonl` with a per-question input fingerprint
and a run signature over the **evaluation contract**: judge model, judge implementation hash, model
route, input limits, the evidence settings (`judge_max_evidence_chars`, `judge_kb_passages`, and the
hash of the knowledge base given with `--documents`), and for live runs the endpoint hash, field mapping, header *names*, retry,
deadline, and pacing settings, deployment ID, and concurrency. Each premade row's fingerprint covers
only that row, so fixing one bad row in the source file does not invalidate the others. `--resume` rejects a checkpoint from a different
contract instead of mixing results; `--retry-errors` re-runs only infrastructure and judge errors
(judge-only retries reuse the saved chatbot answer).
