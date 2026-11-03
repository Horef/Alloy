# Work log
Newest first. Each entry: what changed, how it was verified, what's next. Never edit or delete past entries.
Earlier history: `git log` (commit bodies explain each change) and `docs/archive/implementation-progress.md` (the 2026-09-07 implementation).

Earlier months: `docs/worklog/YYYY-MM.md`.

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
