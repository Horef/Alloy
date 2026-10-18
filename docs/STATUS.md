# Status
Updated 2026-10-18. Snapshot only - overwrite each session; history is in `docs/WORKLOG.md` and `git log`.

## Works (verified)
| Capability | How verified |
|---|---|
| All CLI workflows (`convert-pdf`, `plan`, `generate`, `review-export`/`review-merge`, `reground`, `revary`, `evaluate`, `evaluate-file`, `generate-prompt`, `report`) | Offline suite with fake LLMs: `$PY -m pytest -q` → **340 passed** in 1.7 s, 2026-10-18 (`army` env, macOS) |
| Theme-mode generation on the real hova corpus | Live run recorded in `docs/theme-layer-ab-results.md` (2026-09-23) |
| CI (`.github/workflows/tests.yml`, Python 3.11 and 3.13) | Configured; latest GitHub run not checked in this update |

## Known limitations (README "Current limitations")
- Silver references and variant wording still need human review.
- LLM judge not yet calibrated against a human-reviewed sample.
- Corpus PDFs passed to `generate` directly are not OCR'd; scanned pages need `convert-pdf` (Tesseract or Gemini backend) first.
- Question-targeted retrieval is BM25 with optional embedding fusion; evidence the ranking misses can leave a reference incomplete.
- Concurrent evaluation is for stateless endpoints only.
- Exclusion lists are files, not a review database.

## Next actions
1. Owner: answer the open questions in `docs/BRIEF.md`.
2. TBD (owner): next feature priority.

## Blockers
- None recorded.
