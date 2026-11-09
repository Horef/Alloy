# Work log
Newest first. Each entry: what changed, how it was verified, what's next. Never edit or delete past entries.
Earlier history: `git log` (commit bodies explain each change) and `docs/archive/implementation-progress.md` (the 2026-09-07 implementation).

Earlier months: `docs/worklog/YYYY-MM.md`.

## 2026-11-09 - Judge second pass (D19) and live v4 re-run on corpora A and B
Changed (branch `feat/judge-verify`, commit `11dd2bb`): every contradicted claim is re-checked in a second call that sees the claim, its quoted text and the passages (verdict `contradicts`, `supports` or `unrelated`); a quote not found in the evidence demotes the claim without a call; claim-targeted KB passages are added to the judge evidence. Settings `judge_verify_false_claims`, `judge_verify_evidence_chars`, `judge_claim_passages`. See D19.

| Verification | Result |
|---|---|
| `python -m pytest -q` | 454 passed (offline, fake LLMs) |
| `python tools/docmap.py check` | 0 errors; warnings: docmap version, no `AGENTS.local.md` in the worktree |
| Live `evaluate-file --documents ... --soft-compare` v4 on the two golden sets, same parameters as v3 except `--compare-with` v3 (v3 had compared with v2) | no judge errors; hova 328 rows, keva 514 rows |
| Second pass | hova: 3 calls, 3 claims checked, 3 confirmed. keva: 8 calls, 11 claims checked, 9 confirmed, 2 overturned (`supports`, one row, keva ROW00164, a hand-confirmed false positive) |
| Claim-targeted evidence | hova: 173 of 284 scored rows got claim passages (692 in total); keva: 326 of 448 (1300). Mean `request_used` per Gemini call in the console logs: 12.5k (hova), 13.3k (keva) |
| Audit agreement (`calibration_check.md`) | hova: 7 of 7 audited rows agree (v3 also 7 of 7). keva: 19 of 22 (v3 20 of 22); ROW00164 now agrees; ROW00048 and ROW00257 (hand-confirmed real) lost their false claim in the first pass and moved to `unrelated_answer` and `partial_too_much` |
| Hallucinations v3 → v4 | hova 3 → 3 (ROW00298 left, ROW00078 entered); keva 9 → 7 |

Findings: the second pass confirmed 12 of 14 contradicted claims; the one overturn is the targeted false positive. It cannot recover the two real rows lost in the first pass; that change comes from the new passages or run-to-run variance (not measured). Hova ROW00243 (hand-reviewed as a false positive in the earlier review) is confirmed as contradicted by the verifier; needs the owner's look. New hallucination hova ROW00078: the chatbot said the documents have no rule although the KB passage found by the claim retrieval states it, so it is arguably right. Keva ROW00407 (real fabrication) is still not flagged. About 10 rows per corpus swap between `correct_answer` and `partial_too_much` without a second-pass call, so use v3 → v4 outcome deltas with care.

Next: owner reviews the remaining hallucination rows; consider one more v4 run to measure noise; decide on merging the branch.

## 2026-11-09 - Judge calibration (D18) and live re-run on corpora A and B
Changed: `misleading_hallucination` now needs a contradicted claim with a quoted contradicting text (the judge's `incorrect_type` label and the nothing-correct fallback no longer decide it; nothing correct and nothing contradicted is `unrelated_answer`); uncited false claims are demoted to unsupported. New `--documents` gives the judge full cited chunks plus top-k knowledge-base passages (`judge_max_evidence_chars`, `judge_kb_passages`). New `requires_verification` flag (דורש אימות) per row, in the CSV, JSONL, summary and HTML (card, details, filter); it never changes an outcome. The judge asks for `JudgeResponse` so the two list fields are schema-required (a first live run left the flag empty on every row because defaulted fields are dropped by structured output). Decision D18.

| Verification | Result |
|---|---|
| `python -m pytest -q` | 366 passed (offline, fake LLMs) |
| `python tools/docmap.py check` | 0 errors; warnings: docmap 5.6.0 vs skill 5.11.0, no `AGENTS.local.md` in the worktree |
| Live `evaluate-file --documents ... --soft-compare` on the two golden sets (run ids `61e642d5-4cca-4fa9-bfd8-16b77d5b0683`, `b699e3e4-82fe-45b7-aadd-6f4b32da98f3`; commit recorded in each manifest is the data checkout's, not this branch) | no judge errors; hallucinations 7 -> 3 (328 rows) and 22 -> 9 (514 rows); correct answers 160 -> 168 and 205 -> 230 |
| Hand-audit agreement (29 audited v2 hallucinations) | 27 of 29 agree; details in each corpus's `calibration_check.md` |

Findings: two audited rows disagree. One kept hallucination although the judge's quoted "contradiction" supports the claim (same pattern in one hova row), so the quote is not machine-checked (limitations item 1). One real row moved to `should_have_abstained` and is flagged for verification, because the KB neither supports nor contradicts the claim (unsupported never gives hallucination by the owner's rule). `requires_verification` flagged 0 of 284 rows on corpus A (its answers were KB-grounded) and 26 of 448 on corpus B.

Next: owner reviews the flagged and remaining hallucination rows; consider a second-pass check that a quoted contradiction really contradicts.

## 2026-11-09 - Personal pilot fixes: exclusive rules, quote check, dedupe, clarification
Changed (branch `fix/personal-pilot`), after a live pilot of `generate-personal` on one corpus (10 rules, 39 questions) showed: an invented "does not apply" answer for list-style rules, quotes that do not support the predicate, "men and women in role X" read as gender=male, a clarification asking for fields the persona held, and 3 of 10 kept rules being cross-chunk duplicates.
1. `exclusive` on `RuleCandidate`/`ProfileRule` (prompt explains it with an example; default false). Non-exclusive rules get no violating personas and no not-side answer (`non_exclusive_rule_persona_outside` counts the skipped ones).
2. `condition_not_in_quote`: `predicate_stated` requires the quote to name every condition (label words or listed value, threshold number, year; compared by word stem). Replay on the pilot's 44 cached candidates (older prompt): 20 rejected, 23 kept, 1 merged. Known false rejections: synonyms not in the schema (text "גברים" vs value `זכר`).
3. Clarification built in code from the missing fields' labels (`clarification_for`); the model no longer drafts one.
4. Duplicates across chunks merged (canonical `when` plus similar consequence), first kept, others in `other_sources`; `max_rules` is round-robin by file, seeded. `duplicate_rule` rejection replaced by the `rules_merged_duplicates` diagnostic.
5. Gender case covered by 1 and 2 with a test.
Rule and question prompts changed, so cached rules and questions are not reused. Offline only: no live run after the fix.

| Verification command | Result |
|---|---|
| `PYTHONPATH=src python -m pytest -q` | 442 passed |
| `python tools/docmap.py update && python tools/docmap.py check` | 0 errors; 2 warnings (no `AGENTS.local.md` in this worktree; skill has 5.11.0) |

Next: re-run the pilot live with the fixed code, review, then the full sets.

## 2026-11-09 - Personalized questions: rule extraction and `generate-personal` (plan steps 3-4)
Changed: new `conditions.py` (rule model and validation, prefilter, evaluation of a rule against a profile, cached per-chunk extraction), `cache.load_profile_rules` (`profile_rules` kind), `personas.rule_personas` (satisfying, violating, missing and boundary personas), new `personal_questions.py` (code picks each persona's verdict; clarify when the decisive field is missing), `generate-personal` command, `[personalization]` settings, `SilverQuestion.persona_id/profile_keys/user_info` (stable ids unchanged for other questions), `persona_readable` review column, README Workflow 8, design 22.4, caching 9.1, plan steps 3-4. Offline only: not run against a live model or real corpus. Model calls: 1 per chunk the prefilter keeps, 1 per rule, plus 1 per distinct field for `personal_basic`.

| Verification command | Result |
|---|---|
| `PYTHONPATH=src python -m pytest -q` | 436 passed |
| `python tools/docmap.py update && python tools/docmap.py check` | 0 errors; 2 warnings (no `AGENTS.local.md` in this worktree; skill has 5.11.0) |

Next: run on a small real slice and review; plan step 5 (persona delivery to the chatbot, judge input).

## 2026-11-09 - Personal-questions branch rebased onto main
Changed: rebased the personalized-questions work (plan, profile schema, personas, new-document recipe) onto main. Dropped the branch's docmap upgrade (its Status lines, `docs/FACTS.md` and `lifecycle_strict` were already on main via 5.3 to 5.6) and its D13; renumbered the branch's decisions from D14, D15 to D16, D17 because main already uses D13 to D15. `lifecycle_exempt = ["docs/plans/*"]` in `pyproject.toml`: plans keep their own `Status: proposed | in progress | done` line (docs/plans/README.md). Design section 22 is unchanged (free on main). The two entries below date from before the rebase and cite the new numbers.

| Verification command | Result |
|---|---|
| `PYTHONPATH=src python -m pytest -q` (the shared environment's editable install points at another checkout) | 408 passed |
| `python tools/docmap.py update && python tools/docmap.py check` | 0 errors; 2 warnings (this worktree has no `AGENTS.local.md`; skill has 5.11.0) |

Next: the owner reviews and merges; then plan step 3 (rule extraction).

## 2026-11-06 - Upgrade to project-foundations 5.6.0 (docmap 5.6.0)
Changed: replaced `tools/docmap.py` 5.5.1 with 5.6.0 from the skill; `AGENTS.md` gained the "Design for what will vary next" convention (new in 5.6, template Conventions section). No code changes. Decisions: none.

| Verification command | Result |
|---|---|
| `python tools/docmap.py update && python tools/docmap.py check` | 0 errors, 0 warnings (the "skill has 5.6.0" warning is gone) |
| `python -m pytest -q` | 360 passed |
| `python tools/docmap.py stats` | session-start read 8.5k tokens in total; WORKLOG is the largest part (2.5k), so rotate it when `check` warns |

Next: none.

## 2026-11-06 - Chatbot answers merged, hallucination audit, team abstention rows relabelled
Changed (outputs only, no code): merged the chatbot's answers for corpora A and B with the question metadata (`human-only` sets of `questions-final-20261101`; 330 / 515 rows matched on question and reference text) into `golden_question_results_nov_2026.csv` and ran `evaluate-file`. Corpus B showed 10.9% hallucination (49 of 449). Audited all 49 (and corpus A's 11) against the KBs: 31 of 49 were the judge penalising a correct answer on a team question that had been relabelled "abstain"; only 5 were true fabrications. Audited all 89 team rows (51 B, 38 A): 40 are not in the KB (abstention is right and the team answer is a KB gap), 40 are answerable from the KB, 3 were unusable. Why `reground` failed on the answerable ones: not retrieval (evidence limit 12 → 30 or question-only ranking regrounded 0-3 of 89) but the all-or-nothing check on free-text team answers, which carry extra details the KB lacks (41 rows), wording and number mismatches (21 / 10), and sometimes an unrelated old source (cross-document errors). With a KB-corrected reference the same `reground_one` grounded 40 of 40 (two needed a different source hint because the model re-typed the answer). Relabelled those 40 to answer questions, dropped 3 ill-formed ones, kept 46 as abstentions. Corpus B hallucination 10.9% → 4.9% (22; 12 audited as real errors), abstention success 39.5% → 61.1%; corpus A 3.8% → 2.5%. Decisions: none (owner asked for the relabel).

| Verification command | Result |
|---|---|
| `evaluate-file` on both v2 files with `--resume` (unchanged rows reused from a trimmed copy of the old checkpoint), `--compare-with` the first run | 328 (A) and 514 (B) rows evaluated; the 66 + 44 `chatbot_error` rows (answer generation failed) are unchanged and excluded from rates |
| `reground_one` on the 40 corrected references | 40 of 40 grounded |

Not verified: the audit verdicts are LLM judgements read against the KB, spot-checked by quote only; 5 answers may come from a newer KB than the local copy. Next: re-run the chatbot on the 110 error rows; give the KB owners `team_rows_audit.csv` (46 KB gaps and 3 internal contradictions); consider a reground mode that rewrites the reference from the KB instead of refusing on partly unsupported team answers.

## 2026-11-03 - Plans sized by the user-facing set (D15)

Changed: new `planning.user_facing_target`. Content sizing gave corpus C (1,698 documents, 3,575 chunks) a 151-topic, ~12,000-question plan because every chunk gets a question; reviewers only see the natural-wording questions, so the owner wants to state that number. The planner now scales every topic's canonical quota by one factor (at least one each) to the largest total whose user-visible set fits the target. Corpus C was re-planned with `max_cluster_size = 80` and `max_graph_topics = 45` (theme vocabulary reused with `--themes-from`, so no re-tagging). Also: the 38 / 51 ungrounded team rows of corpora A and B, archived on 2026-11-01, were put back as `missing_detail` abstention questions at the owner's request (outputs only, no code). Decisions: D15.

| Verification command | Result |
|---|---|
| `python -m pytest -q` | 360 passed |
| `python tools/docmap.py update && python tools/docmap.py check` | 0 errors, 0 warnings |
| `plan` on corpus C, `user_facing_target = 700` | 52 topics, 458 canonical + 104 boundary, user_facing 700, ~2,088 model calls estimated; graph extraction reused 950 of 1,698 cached documents |

Next: owner runs `generate --plan` for corpus C, then `review-export`.

## 2026-11-03 - Upgrade to project-foundations 5.5.1 (docmap code graph)

Changed: replaced `tools/docmap.py` 5.4.0 with 5.5.1 from the skill (code graph: `calls NAME`, `deps [PATH]`, `map --budget N`; MAP.md intro names them). `AGENTS.md` "Finding your way" step 4 now lists the three commands, as in the skill's template. No code changes. Decisions: none.

| Verification command | Result |
|---|---|
| `python tools/docmap.py update && python tools/docmap.py check` | 0 errors, 0 warnings (only `docs/MAP.md` intro regenerated) |
| `python -m pytest -q` | 359 passed |
| `python tools/docmap.py map --budget 400` | ranked symbols print; `calls`, `deps` run |

Next: use `calls` and `deps` before refactors.

## 2026-11-01 - Strict doc lifecycle, provider-neutral model transports, organisation names removed
Changed: `lifecycle_strict = true` and `tools/docmap.py` 5.4.0 from the skill. New `llm_providers.py`: `gemini.transport` also accepts `openai` (any OpenAI-compatible chat-completions and embeddings endpoint, key in a configurable header, optional JSON-object mode) and `bedrock` (AWS Converse API, forced tool call carries the schema, boto3 as the optional `[bedrock]` extra, PDF/image attachments, Titan and Cohere embeddings); `build_llm` picks the client and `contracts.model_endpoint` puts the endpoint in every cache identity. The Apigee transport lost its built-in URL and the `/ai_gateway/` check. Removed organisation, corpus and personal names from tracked files: corpora became "corpus A/B/C", Hebrew test fixtures and acronym examples are neutral, `.gitignore` uses generic patterns (this machine's folders moved to `.git/info/exclude`), `docs/RUNS.md` is git-ignored. Kept on the owner's instruction: the generation prompts and the docs that describe them (soldier persona and examples). Decisions: D14.

| Verification command | Result |
|---|---|
| `python -m pytest -q` | 359 passed (19 new: 15 in `tests/test_llm_providers.py` with real HTTP round trips, retries, a fake Bedrock client, the factory and cache identity, plus config tests) |
| `python tools/docmap.py update && python tools/docmap.py check` | 0 errors, 0 warnings, with `lifecycle_strict` on |
| `git ls-files \| xargs grep -rIniE 'hova\|keva\|meitav\|digital\.idf\|\barmy\b'` | no matches outside `tools/docmap.py` |
| `git check-ignore -v hova keva meitav docs/RUNS.md` | ignored by `.git/info/exclude` and `.gitignore` |

Not verified: both new transports against a live endpoint or a real boto3 client. Next: try them live; decide on a neutral prompt variant.

## 2026-11-01 - Prompt rules from the corpus A/B team review; optional audience
Changed: merged the closed corpora A and B team reviews (`review-merge`, then `reground` of team-edited rows; rows the KB cannot ground were dropped from the sets and archived in `dropped_ungrounded_team_questions.jsonl`). Compared every reviewer edit before/after and every rejection (corpus B rejections from the domain the owner asked to skip were excluded). Lessons now in the generation prompts: answers speak to the user and never cite "the document"; answer only what is asked (reviewers cut facts appended from a second document); state the apply channel and approver when the evidence has them; skip calendar/list/table trivia and compound questions; unanswerable answers refer the user to the responsible channel. New optional `generation.audience` keeps questions to one population (corpus B had corpus A-population questions). Intent duplicates reviewers flagged sit at cosine 0.82-0.91, inside the range of distinct variants, so the duplicate threshold is unchanged. Decisions: D13.

| Verification command | Result |
|---|---|
| `python -m pytest -q` | 342 passed |
| `python tools/docmap.py update && python tools/docmap.py check` | 0 errors; 1 warning (skill 5.4.0 available, upgrade not in scope) |

Next: generate the corpus C set with these prompts and watch the review for document references.

## 2026-11-01 - Upgrade to project-foundations 5.3 (docmap 5.3.0)
Changed: replaced `tools/docmap.py` with the skill's 5.3.0 copy (adds `find`, `get`, `stats`, `docs`, the doc-lifecycle and FACTS checks, `rotate`). Gave every doc in `docs/` a Status line (the "verified" dates are the docs' last-edit date, 2026-10-23, not a fresh read; `theme-layer-ab-results.md` is `historical`). Rotated the work log (six entries moved to `docs/worklog/2026-10.md`). Added `[tool.docmap.facts]` (seven shipped defaults from `config.example.toml`, generated into `docs/FACTS.md`; existing docs still copy those values, to be replaced as they are touched) and raised `agents_max_lines` to 120. AGENTS.md gained the Scaffolding line, `find`/`get` at session start and five maintenance rows (Status lines, FACTS, superseding decisions, scaffolding feedback, milestone audit). Decisions: none.

| Verification command | Result |
|---|---|
| `python tools/docmap.py check` with the new script, before the changes | 0 errors, 24 warnings (23 missing Status lines, un-rotated work log) |
| `python tools/docmap.py update && python tools/docmap.py check` after | 0 errors, 0 warnings |
| `python tools/docmap.py stats` | session-start read about 5.3k tokens (was 6.7k before rotation), under the 8k limit |
| `python -m pytest -q` | 340 passed |

Next: run `python tools/docmap.py docs` at the next milestone and re-verify the design docs against the code; set `lifecycle_strict = true` once that is done.
