# Status
Updated 2026-10-13. Snapshot only - overwrite each session; history is in `docs/WORKLOG.md` and `git log`.

## Works (verified)
| Capability | How verified |
|---|---|
| All CLI workflows (`plan`, `generate`, `review-export`/`review-merge`, `reground`, `revary`, `evaluate`, `evaluate-file`, `generate-prompt`, `report`) | Offline suite with fake LLMs: `python -m pytest -q` → **304 passed** in 2.2 s, 2026-10-13 (Python 3.12 scratch env on Linux, not the `army` env) |
| Theme-mode generation on the real hova corpus | Live run recorded in `docs/theme-layer-ab-results.md` (2026-09-23) |
| CI (`.github/workflows/tests.yml`, Python 3.11 and 3.13) | Configured; latest GitHub run not checked in this update |

## Known limitations (README "Current limitations"; archived 2026-09-07 report)
- Silver references and variant wording still need human review.
- LLM judge not yet calibrated against a human-reviewed sample.
- No OCR for scanned PDFs; empty extracted pages are only warned about.
- Retries are bounded per call, but there is no total wall-clock deadline per operation.
- Concurrent evaluation is for stateless endpoints only.
- Exclusion lists are files, not a review database.

## Next actions
1. Owner: answer the open questions in `docs/BRIEF.md`.
2. TBD (owner): next feature priority.

## Blockers
- None recorded.
