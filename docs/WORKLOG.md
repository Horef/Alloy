# Work log
Newest first. Each entry: what changed, how it was verified, what's next. Never edit or delete past entries.
Earlier history: `git log` (commit bodies explain each change) and `docs/archive/implementation-progress.md` (the 2026-09-07 implementation).

Earlier months: `docs/worklog/YYYY-MM.md`.

## 2026-10-27 - Personalized questions, step 2: personas; schema follows the platform
Changed: read the chatbot platform server's preProd code the owner shared (kept in the session scratchpad, not in this repository). The profile reaches the model as a `label: value, ...` line built on the server from the signed-in user's record, appended to the system prompt and prefixed to the retrieval query; the release-date, end-of-mandatory-service and ID toggles are never sent; the dataset-evaluation upload passes no user line. The example schema now uses the platform's keys and labels (`מין`, `סטטוס`), marks the unsent fields `excluded`, and gives age `sample_values`. `profiles.py` gains `sample_values`, enum-object descriptions and `render_user_info`. New `personas.py` and `personas` command: greedy pairwise covering array over the condition values, exact derived targets, fake identifiers. README Workflow 8, design section 22, plan (steps 1-2 done) updated. Proposed a three-file platform patch for a per-row `פרטי משתמש` column (given to the owner, not compiled). Decisions: D15 (provisional).

| Verification command | Result |
|---|---|
| `python -m pytest -q` (Python 3.13 venv; and a Python 3.11 venv in the scratchpad) | 388 passed |
| `chatbot-eval personas --schema profile_schema.example.toml --as-of 2026-10-27` | 30 personas, 139/139 value pairs (full product 540; 30 is the minimum for domains of 6 and 5) |
| `python tools/docmap.py check` | 0 errors, 0 warnings |

Next: plan step 3 (rule extraction); platform team decides on the upload column.

## 2026-10-27 - Personalized questions, step 1: profile schema and profile-check
Changed: owner approved the personalized-questions plan and answered its questions (profile reaches the chatbot as JSON in its system prompt; questions may depend on מגדר and דת; תאריך לידה is available; value lists to confirm). New `profiles.py` (schema loading and validation, profile checks that never print values, derived fields against a pinned date, schema fingerprint), `profile_schema.example.toml` (the ten fields from the owner's screenshots, four derived fields, value lists from a web search marked `open_values`), the offline `profile-check` command, `tests/test_profiles.py`, README Workflow 8 and code map, design section 22 (`docs/design/personalization.md`), security note, BRIEF non-goal superseded, `*profile*.json` git-ignored. Plan status: in progress. Decisions: D14.

| Verification command | Result |
|---|---|
| `python -m pytest -q` before the docs (guards) | 2 failed: module and `profile-check` options missing from README/design docs, as intended |
| `python -m pytest -q` after | 355 passed |
| `chatbot-eval profile-check --schema profile_schema.example.toml --profile <synthetic JSON> --as-of 2026-10-27` | 10 fields, 4 derived; reported a wrong date format and an unlisted open-enum value by field name only; exit 1 |
| `python tools/docmap.py check` | 0 errors, 0 warnings |

Next: plan step 2 (synthetic personas); owner sends a synthetic sample of the profile JSON.

## 2026-10-27 - Project foundations (doc lifecycle, FACTS.md); new-document recipe; personalized-questions plan
Changed: replaced `tools/docmap.py` verbatim with the appendix of the current project-foundations skill (extracted by script; adds Status-line lifecycle checks, the generated `docs/FACTS.md` and the `docs` command). Every doc in `docs/` (ARCHITECTURE, BRIEF, the 18 design files, the theme A/B record as historical) now opens with a Status line; `docs/README.md` and `docs/plans/*` are exempt; `lifecycle_strict = true`. `[tool.docmap.facts]` lists the defaults D2-D4 and D10 talk about. AGENTS.md gains two maintenance rows (100 lines). README Workflow 1 gains a recipe for adding a new document's questions to an existing set (`plan` + `generate --plan --merge-into` on a root holding only the new document at its corpus-relative path). New `docs/plans/personalized-questions.md` (proposed): profile schema, rule extraction, synthetic personas, metamorphic checks, persona delivery to the chatbot, with web research and owner questions; BRIEF links it. Decisions: D13.

Design docs were marked `current (verified 2026-10-27)` on this evidence: no code change since they were split from the reconciled `architecture.md` on 2026-10-23 (`git log -- src`), and a throwaway script found every backticked identifier and file name in ARCHITECTURE and `docs/design/*.md` in `src/`, README or `config.example.toml` (exceptions: example claim IDs `C001`/`C002`). Their prose was not re-read line by line against the code.

| Verification command | Result |
|---|---|
| `python tools/docmap.py check` with the new script, before the doc changes | 0 errors, 24 warnings (missing AGENTS.local.md, 23 docs without a Status line) |
| `python tools/docmap.py check` after (locally and with `CI=1`) | 0 errors, 0 warnings |
| `python tools/docmap.py docs` | 22 docs, all with a Status line |
| `python -m pytest -q` (cloud container, Python 3.13 venv) | 340 passed |

Next: owner answers the personalized-questions plan's open questions.

## 2026-10-23 - Shareable checkout: AGENTS.local.md and project-foundations v4
Changed: wrote `tools/docmap.py` verbatim from the v4 skill appendix (extracted by script; adds the `local` command, the machine-path scan and local-file key checks). Created the tracked `AGENTS.local.example.md` and this machine's git-ignored `AGENTS.local.md` (from `docmap.py local` and the owner's defaults file); `.gitignore` now ignores `*.local.md`. Removed the interpreter paths from AGENTS.md, the Copilot shim and the README (the README's Homebrew-miniforge path no longer exists); AGENTS.md says `python` and points at the local file, reads it at session start, and gains a row for machine facts (98 lines). Closed the BRIEF two-interpreter question on that evidence. Decisions: D12.

| Verification command | Result |
|---|---|
| `python tools/docmap.py check` with the new script, before the fixes | 2 errors: machine-specific path in AGENTS.md and `.github/copilot-instructions.md` |
| `git grep` for other machine facts (`/opt/`, miniforge, maintainer's machine) | README install, CLI and test commands; fixed |
| `python tools/docmap.py check` after (locally and with `CI=1`) | 0 errors, 0 warnings |
| `git check-ignore -v AGENTS.local.md CLAUDE.local.md` | both ignored by `*.local.md`; `AGENTS.local.example.md` is not |
| `python -m pytest -q` | 340 passed |

Next: check the CI run for this push.

## 2026-10-23 - Update tools/docmap.py from the skill
Changed: replaced `tools/docmap.py` verbatim with the owner's updated copy, which includes this session's two fixes plus a letter-case-exact link check (`exists_exact`, so macOS now catches links that only Linux CI used to), link rewriting when `rotate` moves entries, a decision-ID pattern without leading zeros, and a RESULTS pattern that skips subsections. No workarounds to remove: the repo has no ruff configuration, RESULTS index pattern or citation-check override. Added `Claude outputs/` to the repository `.gitignore` as well. Decisions: none.

| Verification command | Result |
|---|---|
| `cmp tools/docmap.py ~/Downloads/docmap.py` | identical |
| `$PY tools/docmap.py update` | 0 files updated |
| `$PY tools/docmap.py check` | 0 errors, 0 warnings |
| `$PY -m pytest -q` (`army` env, macOS) | 340 passed |

## 2026-10-23 - Owner answers on the run index and `Claude outputs/`
Changed: owner confirmed the CI run for `f7faf4c` (tests and `docmap.py check`) is green; `docs/RUNS.md` paths are acceptable; `Claude outputs/` is now ignored by the maintainer's global git ignore file `~/.config/git/ignore` (newly created; `core.excludesFile` is unset, so git reads this default path), which `docmap.py` honors through `--exclude-standard`. BRIEF questions closed. Decisions: none.

| Verification command | Result |
|---|---|
| `git check-ignore -v "Claude outputs/project-foundations-SKILL.md"` | matched by `~/.config/git/ignore:2` |
| `$PY tools/docmap.py check` | 0 errors, 0 warnings |

Next: owner answers the remaining BRIEF questions.

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
