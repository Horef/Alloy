# Status
Updated 2026-10-27. Snapshot only - overwrite each session; history is in `docs/WORKLOG.md` and `git log`.

## Works (verified)
| Capability | How verified |
|---|---|
| All CLI workflows (`convert-pdf`, `plan`, `generate`, `review-export`/`review-merge`, `reground`, `revary`, `add-broad`, `evaluate`, `evaluate-file`, `generate-prompt`, `report`) | Offline suite with fake LLMs: `python -m pytest -q` → **388 passed** in 1.8 s, 2026-10-27 (also on Python 3.11) (Linux cloud container, Python 3.13 venv) |
| Project map and doc checks (`tools/docmap.py`, skill version of 2026-10-27: doc lifecycle, `docs/FACTS.md`, `docs` command) | `python tools/docmap.py check` → 0 errors, 0 warnings, locally and with `CI=1`, 2026-10-27 |
| Every doc in `docs/` has a Status line, enforced (`lifecycle_strict = true`, D13) | `python tools/docmap.py docs` lists 22 docs, none without a Status line, 2026-10-27 |
| Profile schema for personalized questions, following the platform's keys and user line (`profiles.py`, `profile_schema.example.toml`, `profile-check`; plan step 1, D14, D15) | `python -m pytest -q tests/test_profiles.py` → 21 passed, 2026-10-27 |
| Synthetic personas with pairwise coverage (`personas.py`, `personas`; plan step 2) | `tests/test_personas.py` → 27 passed; `chatbot-eval personas` on the example schema → 30 personas, 139/139 value pairs, 2026-10-27 |
| Key defaults in `docs/FACTS.md` read from `config.py`, `config.example.toml`, `models.py`, `pdf_markdown.py` | `docmap.py update` regenerates it; `check` fails when it is stale, 2026-10-27 |
| Shareable checkout: machine facts only in the git-ignored `AGENTS.local.md` | `git check-ignore` → ignored; `docmap.py check` finds no machine paths, 2026-10-27 |
| Import boundaries (only `llm.py` imports the Gemini SDK; nothing imports `cli.py`) | `docmap.py check` (forbid rules), 2026-10-27 |
| Theme-mode generation on the real hova corpus | Live run recorded in `docs/theme-layer-ab-results.md` (2026-09-23) |

## Doesn't work yet / unverified
- CI for this push not yet seen (pushed to a session branch, not `main`).
- The README recipe for adding a new document's questions to an existing set (2026-10-27) was checked against `cli.py` and `documents.py`, not run live.
- Personalized questions: schema and personas exist (steps 1-2 of `docs/plans/personalized-questions.md`); generation still rejects personal types. Enum value lists and the birth-date format are provisional (`open_values`). No cross-field persona constraints yet.
- Persona delivery (D15, provisional): the platform's dataset-evaluation upload sends no user line; a proposed platform patch (not in this repository, not compiled) adds a `פרטי משתמש` column.
- Known limitations as before (README "Current limitations"; `docs/design/limitations.md`): silver references need review, judge not calibrated, no OCR for PDFs passed to `generate`, BM25 retrieval can miss evidence, concurrency for stateless endpoints only, exclusion lists are files.

## Next actions
1. Plan step 3: rule extraction (`conditions.py`) feeding rule boundary values and constraints into personas. Owner / platform team: the `פרטי משתמש` upload column (D15) and the real value strings (`profile-check --profile` on one record, locally).
2. Owner: answer the remaining open questions in `docs/BRIEF.md`.
3. Re-verify the docs `python tools/docmap.py docs` lists before 2026-12-26 (60 days), or CI fails.

## Blockers
- Personalized questions, step 5 (evaluation): the platform's dataset-evaluation upload must accept a per-row user line (platform team, D15).
