# 14. Metrics and reporting (`report.py`)
Status: current (verified 2026-11-09)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md). Citations such as [R1] are listed in [references.md](references.md).

## 14.1 Populations

Every metric states its denominator, because mixing task types is the most common way chatbot
metrics mislead:

| Metric                             | Numerator                                                          | Denominator                         |
| ---------------------------------- | ------------------------------------------------------------------ | ----------------------------------- |
| `factual_answer_success_rate`      | `correct_answer`                                                   | answer tasks                        |
| `useful_answer_rate_on_answerable` | correct, partial, or appropriate clarification                     | answerable questions                |
| `clarification_success_rate`       | `correct_clarification`                                            | clarify tasks                       |
| `abstention_success_rate`          | `correct_abstention`                                               | abstain tasks                       |
| `misinformation_rate`              | `misleading_hallucination`                                         | evaluable records                   |
| `factual_risk_rate`                | at least one false or unsupported claim                            | scored records, all task types      |
| `behavior_failure_rate`            | missing clarification, incorrect abstention, should have abstained | evaluable records                   |
| `good_retrieval_rate`              | all required points retrieved, no contradictory chunk              | answer tasks with retrieval context |
| `infrastructure_error_rate`        | chatbot or judge errors                                            | all records                         |
| `requires_verification_rate`       | at least one value the judge flagged `requires_verification`       | scored records, all task types      |

Infrastructure errors are excluded from every quality denominator. **Missing retrieval context is
missing telemetry, not bad retrieval**: records without context (including `"[]"`, `"{}"`, `null`)
are excluded from retrieval metrics rather than counted as failures.

`requires_verification_*` (the `דורש אימות` card and filter; summary fields count, rate, denominator,
total values and question IDs) is a review aid: the values are shown in each row's details, and the
flag never changes an outcome or any other metric. A row's details also list its false claims with
the contradicting evidence the judge quoted and the second-pass verdict on each (D19). The
second pass is reported as a card ("טענות שגויות שנסתרו בבדיקה שנייה") and summary fields
`false_claim_verification_calls`, `false_claims_checked`, `false_claims_confirmed`,
`false_claims_overturned`, `false_claims_overturned_by_verdict` (`supports`, `unrelated`,
`quote_not_found`) and `false_claims_overturned_question_ids`; they are counts of claims, not a rate
over rows, and the hallucination metrics already use the post-verification outcome.

The second vote (D20) adds `second_vote_rows` (rows judged twice), `second_vote_calls`,
`second_vote_errors`, `votes_disagree_count` and `votes_disagree_question_ids`, shown as a card
("הצבעות שופט שלא תאמו", only when some row had a second vote). A row's details list both outcomes, and
`evaluation_details.csv` has the columns "הצבעות השופט לא תאמו" and "סיווגי ההצבעות". Outcome metrics use
the kept vote's outcome; `review-export` does not export evaluation rows, so it carries no vote data.

## 14.2 Diagnosing the pipeline

Answer tasks with retrieval telemetry fall into a 2 × 2 matrix: retrieval good or poor × answer good
or poor. "Good retrieval, bad answer" isolates a generation-stage failure, the separation RAGAS and
RAGChecker argue for [R2, R4]. Useful partial answers despite good retrieval are reported separately so
they do not inflate the failure count.

## 14.3 Uncertainty

Topic-level rates come with 95% Wilson score intervals [R29], which, unlike the textbook Wald interval,
behave correctly for small samples and rates near 0 or 1 [R30]. Topics with fewer than ten evaluable
results are flagged as small samples. Planning's precision rule (section 10) is the same idea used in
advance.

## 14.4 Variants

For each variant whose parent was also judged, the pair is classified as both succeed, parent only,
variant only, or both fail. `robustness_when_parent_succeeds` and `degradation_rate` measure how much
realistic phrasing costs. In `user_facing` sets, the phrasing gap compares anchors with their natural
phrasings. `evaluation.canonical_scoring` decides whether anchors count in headline metrics (`gap`,
the default: no; `include`: yes; `exclude`: not even sent to the chatbot).

## 14.5 Comparing runs

Two runs are compared strictly by default. Questions are matched only when the ID **and every
material field** agree after whitespace normalization (review status and notes are ignored).
Aggregate deltas are shown only when both runs contain exactly the same benchmark *and* the same
eligible population for that metric; otherwise both absolute values are shown and the delta is marked
non-comparable. The evaluation contracts of the two runs are compared field by field (names only,
never values).

*Why so strict:* human review routinely adds, edits, and removes questions between runs, and judge
models get upgraded. Without these rules a composition change, or a question that errored in one run
only, appears as chatbot improvement. The earlier review reproduced exactly that: a single failed
answer turning into a judge error produced a favorable 50-point delta with zero real improvements
(finding E3 in [library-review.md](../archive/library-review.md)).

`--soft-compare` relaxes this for directional reads: it computes an approximate delta over questions
that are matched *and* eligible in both runs, marked `≈` with its sample size.
