# Status
Updated 2026-11-09. Snapshot only - overwrite each session; history is in `docs/WORKLOG.md` and `git log`.

## Works (verified)
| Capability | How verified |
|---|---|
| All CLI workflows (`convert-pdf`, `plan`, `generate`, `review-export`/`review-merge`, `reground`, `revary`, `add-broad`, `evaluate`, `evaluate-file`, `generate-prompt`, `report`) | Offline suite with fake LLMs: `python -m pytest -q` → **342 passed** in 2.2 s, 2026-11-01 (`python` from `AGENTS.local.md`, Python 3.13) |
| Project map and doc checks (`tools/docmap.py`, skill 5.6.0; includes the `calls`, `deps` and `map` code-graph commands) | `python tools/docmap.py check` → 0 errors, 0 warnings, 2026-11-06; every doc in `docs/` has a Status line (verified dates are last-edit dates, not a fresh read) |
| Shareable checkout: machine facts only in the git-ignored `AGENTS.local.md` | `git check-ignore -v AGENTS.local.md` → `*.local.md`; `docmap.py check` finds no home paths in AGENTS.md, shims, READMEs or code headers, 2026-10-23 |
| Import boundaries (only `llm.py` imports the Gemini SDK, only `llm_providers.py` imports `boto3`, nothing imports `cli.py`) | `docmap.py check` (forbid rules), 2026-11-01 |
| Model transports `openai` and `bedrock` (D14) | `python -m pytest -q` → 359 passed, 2026-11-01: a local HTTP server stands in for an OpenAI-compatible endpoint and a fake client for Bedrock. **Not run against a live endpoint or a real boto3 client** |
| Plan sizing by `planning.user_facing_target` (D15) | Offline test, 360 passed, 2026-11-03; live `plan` on corpus C (3,575 chunks): target 700 → 52 topics, 458 canonical, user_facing 700 (content sizing had proposed ~12,000) |
| Run index | `python tools/docmap.py runs` → 16 local runs in the git-ignored `docs/RUNS.md`, 2026-10-23 |
| CI (`.github/workflows/tests.yml`: tests and `docmap.py check` on Python 3.11 and 3.13) | GitHub Actions run for `f7faf4c` green, reported by the owner, 2026-10-23 |
| Team-review round trip (`review-merge` → `reground` of team rows) on corpora A and B | 2026-11-01: corpus A 651 / corpus B 1059 merged; 38 / 51 team rows the KB cannot ground were kept as `missing_detail` abstention questions (owner's choice, 2026-11-01; team answer in reviewer notes). 2026-11-06 audit against the KBs: 40 of the 89 are answerable and were relabelled to answer questions (`questions-final-20261106`), 3 dropped, 46 stay abstentions |
| Theme-mode generation on the pilot corpus A | Live run recorded in `docs/theme-layer-ab-results.md` (2026-09-23) |
| Profile schema for personalized questions, following the platform's keys and user line (`profiles.py`, `profile_schema.example.toml`, `profile-check`; plan step 1, D16, D17) | `tests/test_profiles.py`, offline, 2026-10-27; whole suite 408 passed after rebasing onto main, 2026-11-09 |
| Synthetic personas with pairwise coverage (`personas.py`, `personas`; plan step 2) | `tests/test_personas.py`; `chatbot-eval personas` on the example schema → 30 personas, 139/139 value pairs, 2026-10-27 |

## Known limitations (README "Current limitations"; details in `docs/design/limitations.md`)
- Personalized questions: schema and personas exist (steps 1-2 of `docs/plans/personalized-questions.md`); generation still rejects personal types. Enum value lists and the birth-date format are provisional (`open_values`). Persona delivery (D17, provisional): the platform's dataset-evaluation upload sends no user line; the proposed platform patch is not in this repository.
- Silver references and variant wording still need human review.
- LLM judge not yet calibrated against a human-reviewed sample.
- Corpus PDFs passed to `generate` directly are not OCR'd; scanned pages need `convert-pdf` (Tesseract or Gemini backend) first.
- Question-targeted retrieval is BM25 with optional embedding fusion; evidence the ranking misses can leave a reference incomplete.
- Concurrent evaluation is for stateless endpoints only.
- Exclusion lists are files, not a review database.

## Next actions
1. Generate the corpus C set from its 700-user-facing plan (owner runs `generate --plan`); then `review-export` and check whether reviewers still remove document references.
2. Re-verify the design docs against the code at the next milestone (`python tools/docmap.py docs`); `lifecycle_strict` is on, so a missing or stale Status line now fails `check`.
3. Owner: answer the remaining open questions in `docs/BRIEF.md`.
3a. Owner: fix `reground` so a partly unsupported team answer is rewritten from the KB instead of refused (limitations item 8); re-run the chatbot on the 110 `chatbot_error` rows of the 2026-11-06 evaluations; give the KB owners `team_rows_audit.csv`.
4. Try the `openai` and `bedrock` transports against a live endpoint (D14) before relying on them.
5. TBD (owner): next feature priority; whether the generation prompts (soldier persona and examples) should get a neutral variant for other teams.
6. Personalized questions: plan step 3 (rule extraction, `conditions.py`). Owner / platform team: the `פרטי משתמש` upload column (D17) and the real value strings (`profile-check --profile` on one record, locally).

## Blockers
- Personalized questions, step 5 (evaluation): the platform's dataset-evaluation upload must accept a per-row user line (platform team, D17).
