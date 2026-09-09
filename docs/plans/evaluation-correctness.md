# Plan: trustworthy scoring and comparisons

Status: implemented offline on 2026-09-07; live judge calibration remains follow-up. Parent review: [library-review.md](../library-review.md). Findings E1–E3 are reproduced by [review_probes.py](../review_probes.py).

## Objective

Make the reported outcome agree with the scored behavior and claims, and prevent changing error/telemetry coverage from appearing as chatbot improvement. Preserve historical raw artifacts and existing metric fields where practical; explicitly version changed semantics.

## 1. Remove the abstention substring override

Files: `src/chatbot_eval/evaluator.py`, `tests/test_evaluator.py`.

`looks_like_abstention` currently searches the entire answer for phrases such as `not provided`, `not found`, and `אין מספיק מידע`. `_evaluate_one` then forces `response_is_abstention=True` even if the judge found a correct substantive answer. This can also classify quoted text or a limited information gap as a complete abstention.

Implement:

1. Remove substring matches as an authoritative override. Retain them only as optional diagnostic evidence, or delete the helper if it has no remaining caller.
2. Treat empty/whitespace output as a response error before judging, consistently with the adapter/import plan. Known infrastructure placeholders remain infrastructure errors, not abstentions.
3. Expand the judge rubric: whole-response abstention must be distinguished from a supported partial answer, a quotation containing refusal language, and an answer that says one attachment is unavailable.
4. Do not add a second LLM call merely to classify abstention.

Desired tests: a correct answer containing `not provided` remains an answer; Hebrew limited-gap phrasing does not trigger a whole-response override; exact refusal is classified using judge evidence; quoted refusal does not override a correct response; empty output avoids the judge.

## 2. Validate success and risk independently

Files: `evaluator.py`, `models.py`, `report.py`, `labels.py` if adding a label, tests for those contracts.

Currently `classify` returns clarification/abstention success before inspecting harmful claim counts. The report derives risk only from the outcome enum, and totals factual claim violations only for answer tasks. Thus contradictory-but-valid judge objects can produce a green success and zero visible false claims.

Implement the smallest consistent contract:

1. Require no false or unsupported factual claims for `CORRECT_ABSTENTION` and `CORRECT_CLARIFICATION`. Judge flags and claim counters must not silently disagree.
2. Reject mutually incompatible whole-response flags or other impossible combinations as judge-contract failures, with an actionable explanation. Do not silently coerce them into a success. A response with a clarification question plus an asserted false answer is not a successful clarification.
3. For scored false/unsupported content, prefer the existing `MISLEADING_HALLUCINATION` outcome where appropriate, including on non-answer tasks. Document precedence with a truth table; make abstention/clarification success depend on the task and safe behavior together.
4. Preserve the existing coarse `risky_misinformation_rate` for compatibility if needed, but label its scope clearly: it currently includes *all* missing clarifications, including harmless refusals. Add separate behavior-failure and factual-risk counts/rates instead of pretending these are identical concepts.
5. Keep `answer_claim_metrics` limited to answer tasks as its name implies. Add an all-scored-responses violation summary for false/unsupported claims so non-answer-task problems are visible. Include denominators and do not sum mutually overlapping categories into an invented total.
6. Use expected behavior consistently; avoid having one branch use `expected_behavior` and another implicitly infer it from `answerable` without validating the pair.
7. Ensure public evaluator entry points validate questions and result-ID alignment, just as CLI import does. Duplicate premade IDs must not be silently replaced by `PremadeAdapter`'s dictionary.

Update fake score fixtures to represent internally consistent tasks. Existing tests sometimes create `answerable=false` with the default `expected_behavior=answer` or clarification tasks with factual counts; those fixtures should not define the intended contract.

Desired tests: full outcome truth table for answer/clarify/abstain; false claims alongside each positive flag; both response flags true; correct partial answer; harmless refusal on a clarify task; known infrastructure error; result-ID mismatch; duplicate IDs; all-task violation totals. Add tests at both `classify` and full `Evaluator` levels so an upstream override cannot undo the fix.

## 3. Make comparisons metric-specific

Files: `report.py`, `history.py`, optionally CLI manifest loading; `tests/test_report.py` and history tests.

Keep the existing full-question identity check. It is useful and already prevents many invalid comparisons. Add *eligible population* checks per metric rather than weakening identity to ID/text alone.

Implement:

1. Derive eligible ID sets for factual answers, clarification, abstention, usefulness, factual risk, and retrieval metrics. Infrastructure-error rate uses the full benchmark denominator.
2. If the eligible sets differ, keep absolute values but suppress that aggregate quality delta and its favorable/unfavorable coloring. Return a machine-readable reason such as `eligible_population_changed`. Do not reuse the global benchmark flag as proof that every metric is comparable.
3. Optionally add a separately named paired-subset delta using only IDs eligible in both runs. Always show excluded counts; never relabel it as the whole-run result. Prefer this simple subset approach over a statistical framework.
4. Preserve `matched_outcomes` and its exclusion of infrastructure errors. The reproduction must show zero paired improvements and no green whole-run accuracy gain when only a failed row becomes a judge error.
5. Display numerator and denominator beside each comparison metric. Explain that retrieval metrics with absent telemetry cannot diagnose failed retrieval.
6. Add an optional evaluation-contract compatibility check from manifests: judge model/rubric/schema and input-limit changes should be disclosed. For historical JSONL without that provenance, mark contract compatibility unknown; don't invent it or reject all historical reports outright.
7. Align the topic usefulness denominator with the headline's answerable population, or rename the topic metric to clearly state that it includes abstention tasks in its denominator. Preserve old JSON fields with documented migration if downstream users rely on them.
8. Clarify the scope of `--hide-correct-answer-metrics` in README/CLI; if it is intended to apply throughout the report, pass it into comparison rendering too.

Desired tests: identical benchmark/identical eligible sets; newly failed judge; recovered judge; same count but different excluded IDs; changing context availability; added/removed/duplicate/conflicting question identities; fully matched self-comparison; no eligible rows; legacy provenance absent. Confirm absolute values remain visible when deltas are suppressed.

## 4. Make insights consume the corrected contract

Files: `insights.py`, associated tests; coordinate with the existing prompt-improvement plan.

Pass expected behavior, question form, all-task violations, and evidence-availability flags. Missing context is not evidence of a retrieval failure. Validate issue references against the IDs actually sent, not the full input list. Keep the existing one insights call, with honest coverage/omission counts; do not add an implicit repair/analysis loop. Insights may hypothesize a cause but cannot establish that a particular prompt instruction was absent without seeing the prompt.

## Compatibility and validation

Increment an explicit evaluation-contract version and include it in fresh checkpoints/manifests. Regenerating HTML from historical records should not silently rejudge or rewrite their old outcome labels. Rejudging corrected imports must use fresh compatible checkpoints and new output directories.

Run focused evaluator/report/insight tests, then the full army suite. Add a small synthetic end-to-end import → judge → report test with a mixed task set. Offline completion means the metric contract is correct for supplied judge evidence, not that an LLM judge is calibrated; calibration requires a separately reviewed set and live judging.

Handoff: implement sections 1–3 first, including tests and documentation, then coordinate section 4 with bounded-evidence work. No live calls or deployment are needed.
