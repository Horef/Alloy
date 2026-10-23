# Status
Updated 2026-10-23. Snapshot only - overwrite each session; history is in `docs/WORKLOG.md` and `git log`.

## Works (verified)
| Capability | How verified |
|---|---|
| All CLI workflows (`convert-pdf`, `plan`, `generate`, `review-export`/`review-merge`, `reground`, `revary`, `add-broad`, `evaluate`, `evaluate-file`, `generate-prompt`, `report`) | Offline suite with fake LLMs: `$PY -m pytest -q` → **340 passed** in 1.6 s, 2026-10-23 (`army` env, macOS) |
| Project map and doc checks (`tools/docmap.py`) | `$PY tools/docmap.py check` → 0 errors, 4 warnings (all from the untracked local `Claude outputs/` folder); in a fresh clone 0 errors, 0 warnings, 2026-10-23 |
| Import boundaries (only `llm.py` imports the Gemini SDK; nothing imports `cli.py`) | `docmap.py check` (forbid rules), 2026-10-23 |
| Run index | `$PY tools/docmap.py runs` → 16 local runs in `docs/RUNS.md`, 2026-10-23 |
| Theme-mode generation on the real hova corpus | Live run recorded in `docs/theme-layer-ab-results.md` (2026-09-23) |

## Doesn't work yet / unverified
- CI (`.github/workflows/tests.yml`, Python 3.11 and 3.13) now also runs `docmap.py check`; the first GitHub run after this push has not been checked.

## Known limitations (README "Current limitations"; details in `docs/design/limitations.md`)
- Silver references and variant wording still need human review.
- LLM judge not yet calibrated against a human-reviewed sample.
- Corpus PDFs passed to `generate` directly are not OCR'd; scanned pages need `convert-pdf` (Tesseract or Gemini backend) first.
- Question-targeted retrieval is BM25 with optional embedding fusion; evidence the ranking misses can leave a reference incomplete.
- Concurrent evaluation is for stateless endpoints only.
- Exclusion lists are files, not a review database.

## Next actions
1. Check the first GitHub Actions run with the docmap step.
2. Owner: answer the open questions in `docs/BRIEF.md` (two new ones about `docs/RUNS.md` paths and the local `Claude outputs/` folder).
3. TBD (owner): next feature priority.

## Blockers
- None recorded.
