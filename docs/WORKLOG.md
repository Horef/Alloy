# Work log
Newest first. Each entry: what changed, how it was verified, what's next. Never edit or delete past entries.
Earlier history: `git log` (commit bodies explain each change) and `docs/archive/implementation-progress.md` (the 2026-09-07 implementation).

Earlier months: `docs/worklog/YYYY-MM.md`.

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
