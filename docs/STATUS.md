# Status
Updated 2026-10-23. Snapshot only - overwrite each session; history is in `docs/WORKLOG.md` and `git log`.

## Works (verified)
| Capability | How verified |
|---|---|
| All CLI workflows (`convert-pdf`, `plan`, `generate`, `review-export`/`review-merge`, `reground`, `revary`, `add-broad`, `evaluate`, `evaluate-file`, `generate-prompt`, `report`) | Offline suite with fake LLMs: `$PY -m pytest -q` → **340 passed** in 1.6 s, 2026-10-23 (`army` env, macOS) |
| Project map and doc checks (`tools/docmap.py`) | `$PY tools/docmap.py check` → 0 errors, 0 warnings, 2026-10-23 (after `Claude outputs/` was added to the global git ignore file) |
| Import boundaries (only `llm.py` imports the Gemini SDK; nothing imports `cli.py`) | `docmap.py check` (forbid rules), 2026-10-23 |
| Run index | `$PY tools/docmap.py runs` → 16 local runs in `docs/RUNS.md`, 2026-10-23 |
| CI (`.github/workflows/tests.yml`: tests and `docmap.py check` on Python 3.11 and 3.13) | GitHub Actions run for `f7faf4c` green, reported by the owner, 2026-10-23 |
| Theme-mode generation on the real hova corpus | Live run recorded in `docs/theme-layer-ab-results.md` (2026-09-23) |

## Known limitations (README "Current limitations"; details in `docs/design/limitations.md`)
- Silver references and variant wording still need human review.
- LLM judge not yet calibrated against a human-reviewed sample.
- Corpus PDFs passed to `generate` directly are not OCR'd; scanned pages need `convert-pdf` (Tesseract or Gemini backend) first.
- Question-targeted retrieval is BM25 with optional embedding fusion; evidence the ranking misses can leave a reference incomplete.
- Concurrent evaluation is for stateless endpoints only.
- Exclusion lists are files, not a review database.

## Next actions
1. Owner: answer the remaining open questions in `docs/BRIEF.md`.
2. TBD (owner): next feature priority.

## Blockers
- None recorded.
