# Work log
Newest first. Each entry: what changed, how it was verified, what's next. Never edit or delete past entries.
Earlier history: `git log` (commit bodies explain each change) and `docs/archive/implementation-progress.md` (the 2026-09-07 implementation).

Earlier months: `docs/worklog/YYYY-MM.md`.

## 2026-10-23 - Upgrade project foundations: generated map, design split, CI doc check
Changed: installed `tools/docmap.py` from the project-foundations skill and configured it in `pyproject.toml` (two `forbid` import rules, AGENTS.md cap of 100 lines, run index); added module docstrings to 22 source and 24 test modules and purpose paragraphs to `src/`, `src/chatbot_eval/`, `tests/`, `docs/` and `tools/`; split `docs/architecture.md` into `docs/design/` (19 files, sections verbatim and with their numbers) plus a short `docs/ARCHITECTURE.md` (invariants and index), with section 20 moved to `docs/archive/architecture-review-fixes.md`; cut AGENTS.md from 120 to 96 lines (component rules moved to the "Rules here" sections of the src and tests READMEs); CI runs `docmap.py check`; `docs/RUNS.md` indexes the 16 local run manifests; `rotate` moved nothing (all entries are from October). Decisions: D11.

The dry-run warnings about `D01`, `D02` and `D19` in `docs/architecture.md` were not decision citations: `D01`/`D02` are the document-map labels of broad-question generation (a code span) and `D19` is part of the ACL Anthology URL of reference R24. Fixed in docmap rather than in the docs (to port back to the skill copy):
- `check`: decision IDs inside inline code, fenced code or URLs are no longer reported as citations.
- `check`: `AGENTS.md` was scanned twice for decision IDs (it is already in the Markdown file list), duplicating its warnings.

| Verification command | Result |
|---|---|
| `$PY tools/docmap.py check` before any change (verbatim script, no config) | 59 errors, 7 warnings: 46 modules without docstring, 5 folder READMEs (incl. the new `tools/`), 8 generated files never generated; `D01`/`D02`/`D19` in `docs/architecture.md`, plus `D0`/`D01`/`D02`/`D19` in the untracked `Claude outputs/` |
| `$PY tools/docmap.py check` after | 0 errors, 4 warnings (all in the untracked `Claude outputs/`); fresh clone of HEAD: 0 errors, 0 warnings |
| `$PY -m pytest -q` (`army` env, macOS), before every commit | 340 passed |
| `code_fingerprint` of the 22 modules before and after their docstrings | unchanged, so no cache entry or resume journal is invalidated |
| Throwaway script: every section of the old `architecture.md` present verbatim in `docs/design/` or the archive (heading level and link targets ignored) | 21 of 21 sections |
| `git grep` for lowercase `architecture.md` links outside `docs/archive/` | none (macOS resolves case-insensitively; Linux CI does not) |
| `$PY tools/docmap.py runs` | `docs/RUNS.md`: 16 runs |

Next: check the first CI run with the docmap step; owner answers the two new BRIEF questions.

## 2026-10-18 - Reconcile local docs commits with origin/main
Changed: rebased the six unpushed 2026-10-13 docs commits onto 16 new `origin/main` commits (PDF conversion, retry deadline, review/report/retrieval fixes, `docs/architecture.md`, docs-consistency guards). The README-cleanup commit became empty (remote made the same edits) and was dropped. AGENTS.md now combines both versions; its dated-records note and two `docs/architecture.md` links point at `docs/archive/`. A stray uncommitted README edit that reverted that cleanup was stashed (`git stash list`), not applied. The pre-rebase branch is saved as local `backup/pre-reconcile-2026-10-18`.

| Verification command | Result |
|---|---|
| `$PY -m pip install -e '.[dev]'` | installed new dependency `pdfplumber` (5 PDF tests failed without it) |
| `$PY -m pytest -q` (`army` env, macOS) | 340 passed in 1.7 s |

Next: owner reviews BRIEF open questions.

## 2026-10-13 - Project-foundations audit
Changed: `.gitignore` covers `.env.*`, keys and personal agent files; finished reviews, plans and probes moved to `docs/archive/`; two stale README limitations removed; added `docs/BRIEF.md`, `docs/STATUS.md`, this log and `docs/DECISIONS.md` (D1-D10 reconstructed from commit messages); AGENTS.md gained session-start and doc-maintenance rules. Decisions: D1-D10 (reconstructed).

| Verification command | Result |
|---|---|
| `python -m pytest -q` (Python 3.12 scratch env, Linux) | 304 passed in 2.23 s |
| `git check-ignore` on `.env.local`, `CLAUDE.local.md`, `.claude/settings.local.json`, `x.pem` | all ignored; `.env.example` still tracked |
| `git ls-files` scan for secrets, PDF/XLSX/DOCX | none tracked |

Next: owner reviews BRIEF open questions; push the commits (`git push origin main`).
