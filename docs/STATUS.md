# Status
Updated 2026-11-03. Snapshot only - overwrite each session; history is in `docs/WORKLOG.md` and `git log`.

## Works (verified)
| Capability | How verified |
|---|---|
| All CLI workflows (`convert-pdf`, `plan`, `generate`, `review-export`/`review-merge`, `reground`, `revary`, `add-broad`, `evaluate`, `evaluate-file`, `generate-prompt`, `report`) | Offline suite with fake LLMs: `python -m pytest -q` → **342 passed** in 2.2 s, 2026-11-01 (`python` from `AGENTS.local.md`, Python 3.13) |
| Project map and doc checks (`tools/docmap.py`, skill 5.5.1; includes the `calls`, `deps` and `map` code-graph commands) | `python tools/docmap.py check` → 0 errors, 0 warnings, 2026-11-03; every doc in `docs/` has a Status line (verified dates are last-edit dates, not a fresh read) |
| Shareable checkout: machine facts only in the git-ignored `AGENTS.local.md` | `git check-ignore -v AGENTS.local.md` → `*.local.md`; `docmap.py check` finds no home paths in AGENTS.md, shims, READMEs or code headers, 2026-10-23 |
| Import boundaries (only `llm.py` imports the Gemini SDK, only `llm_providers.py` imports `boto3`, nothing imports `cli.py`) | `docmap.py check` (forbid rules), 2026-11-01 |
| Model transports `openai` and `bedrock` (D14) | `python -m pytest -q` → 359 passed, 2026-11-01: a local HTTP server stands in for an OpenAI-compatible endpoint and a fake client for Bedrock. **Not run against a live endpoint or a real boto3 client** |
| Run index | `python tools/docmap.py runs` → 16 local runs in the git-ignored `docs/RUNS.md`, 2026-10-23 |
| CI (`.github/workflows/tests.yml`: tests and `docmap.py check` on Python 3.11 and 3.13) | GitHub Actions run for `f7faf4c` green, reported by the owner, 2026-10-23 |
| Team-review round trip (`review-merge` → `reground` of team rows) on corpora A and B | 2026-11-01: corpus A 651 merged / 613 kept, corpus B 1059 merged / 1008 kept; rows the KB cannot ground archived, not deleted (WORKLOG) |
| Theme-mode generation on the pilot corpus A | Live run recorded in `docs/theme-layer-ab-results.md` (2026-09-23) |

## Known limitations (README "Current limitations"; details in `docs/design/limitations.md`)
- Silver references and variant wording still need human review.
- LLM judge not yet calibrated against a human-reviewed sample.
- Corpus PDFs passed to `generate` directly are not OCR'd; scanned pages need `convert-pdf` (Tesseract or Gemini backend) first.
- Question-targeted retrieval is BM25 with optional embedding fusion; evidence the ranking misses can leave a reference incomplete.
- Concurrent evaluation is for stateless endpoints only.
- Exclusion lists are files, not a review database.

## Next actions
1. Generate the corpus C set with the review-lesson prompts and the new `generation.audience`; check whether reviewers still remove document references.
2. Re-verify the design docs against the code at the next milestone (`python tools/docmap.py docs`); `lifecycle_strict` is on, so a missing or stale Status line now fails `check`.
3. Owner: answer the remaining open questions in `docs/BRIEF.md`.
4. Try the `openai` and `bedrock` transports against a live endpoint (D14) before relying on them.
5. TBD (owner): next feature priority; whether the generation prompts (soldier persona and examples) should get a neutral variant for other teams.

## Blockers
- None recorded.
