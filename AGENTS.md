# Agent instructions for Alloy

Alloy (`chatbot_eval`) generates evidence-grounded Hebrew evaluation sets from a document corpus
and judges internal chatbots against them. It is **not** a source of ground truth: silver questions
need domain review. Goal, non-goals and open questions: `docs/BRIEF.md`. User manual (workflows,
settings, outputs): [README.md](README.md).

## Start of every session

1. `git pull --rebase` and `git status`; do not discard or overwrite changes you did not make.
2. Read `docs/STATUS.md` and skim the latest entry in `docs/WORKLOG.md`.
3. Before changing a default or design choice, check `docs/DECISIONS.md` (index at the top).

## Finding your way

1. `docs/MAP.md` (generated): every folder and module with a one-line summary.
2. The folder's `README.md`: purpose, folder rules and a generated table of files and importers.
   **Before editing `src/chatbot_eval/` or `tests/`, read the "Rules here" section of its README.**
3. The module docstring; `$PY tools/docmap.py show <path>` lists its symbols and importers.
4. Before changing a function or class: `$PY tools/docmap.py where <Name>`; run the tests it lists.
5. Design and rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (invariants, index) and the
   component's file in `docs/design/`.

## Environment and commands

- Python 3.11 or newer. On the maintainer's machine `PY=/Users/sergiyhoref/Programming/Army/army/bin/python`
  (3.13, package installed editable), not the base conda interpreter. Elsewhere (cloud sessions,
  CI): `python -m venv .venv` then `.venv/bin/pip install -e '.[dev]'`.
- Tests: `$PY -m pytest -q` (offline, fake LLMs, a few seconds). Docs: `$PY tools/docmap.py check`.
  CI runs both on Python 3.11 and 3.13.
- PDF conversion uses `pdfplumber` (a core dependency); OCR shells out to the optional `tesseract`
  binary.

## Git workflow (mandatory in every session)

- The repository is under git with remote `origin` = `git@github.com:Horef/Alloy.git`, branch `main`.
- Commit after each logical, tested change. Use Conventional Commits
  (`fix(generation): …`, `feat(retrieval): …`, `docs: …`, `test: …`, `chore: …`) with a short body
  explaining why.
- Before committing: `$PY tools/docmap.py update`; then the full test suite and
  `$PY tools/docmap.py check` green, and `git diff --check` clean.
- After committing, push: `git push origin main`. Over SSH the key may prompt for its passphrase; run
  the push unpiped so the prompt is visible and let the user type it. Never ask for, echo, or store
  the passphrase. If the user is unavailable, leave the commits local and say so. In cloud sessions
  the remote may be HTTPS through a proxy; push the same way.
- Never force-push, rewrite published history, amend pushed commits, or push the local
  `backup/*` branches (they contain purged internal files). Ask the user first for anything destructive.

## Data safety

This project handles internal IDF documents. Never commit or push corpora, outputs, or caches:
`hova/`, `keva/`, `knowledge_base/`, `outputs/`, `.chatbot_eval_cache/`, `.env`, `config.toml`,
`*.pdf`, `*.xlsx`, `*.docx`. They are in `.gitignore`; check `git status` before every commit and
never use `git add -f` for them. Never paste corpus text into commit messages, docs, docstrings or
tests.

## When you change X, also update Y

Agents have repeatedly changed code without updating the files that describe it. Check your diff
against this table before committing, and update docs in the same commit as the change. Rows marked
*guarded* fail `tests/test_docs_consistency.py` or `docmap.py check` when forgotten.

| Change | Also update |
|---|---|
| Add or rename a setting | `Settings` and `validate`, `config.example.toml`, README parameter table (*guarded*) |
| Add a CLI command or option | README table for that command (*guarded*) |
| Add a module | Its docstring, README code map and a `docs/design/` file (*guarded*) |
| Add a folder with code | Its `README.md` purpose paragraph, first sentence standalone (*guarded*) |
| Add, move, rename or delete files | `$PY tools/docmap.py update` (*guarded*) |
| Change a component's behavior or design | Its file in `docs/design/`; `limitations.md` (section 19) if a limitation appears or disappears; `docs/ARCHITECTURE.md` if an invariant changes |
| Change outcomes, judging, or metrics | README "Judging logic" and outputs sections; `docs/design/evaluation.md` and `reporting.md` (sections 13 and 14) |
| Change a prompt, schema, or deterministic stage | The matching cache key or fingerprint (`docs/design/caching.md`, section 9), including every module the stage calls |
| Change PDF conversion output | Bump `pdf_markdown.CONVERTER_VERSION` (written into converted files) |
| Add an output file or manifest field | README outputs section for that workflow |
| Add or change an import boundary | `docs/ARCHITECTURE.md` and `[[tool.docmap.forbid]]` in `pyproject.toml` |
| Add a relative link in a doc | Make sure it resolves, with exact case: macOS ignores case, CI does not (*guarded*) |
| Finish a unit of work | `docs/STATUS.md` (overwrite) and a new entry at the top of `docs/WORKLOG.md` with commands and results |
| Change a default or choose between alternatives | The next D entry in `docs/DECISIONS.md`; to reverse one, add an entry that supersedes it |
| Start multi-session work | A plan in `docs/plans/` whose first line is `Status: ...`; move it to `docs/archive/` when done |

`docs/archive/` and `docs/theme-layer-ab-results.md` are dated records. Do not rewrite them; record
the current state in `docs/design/`. Never describe planned or partial work as done; say how it was
verified (offline tests use fake LLMs and do not prove live Gemini behavior).

## Keeping these instructions accurate

These instructions load into every session, so a wrong or missing line costs every future session.

- When a session reveals a gap (a step you had to discover, a file you nearly missed, a convention
  that was not written down, or an instruction that turned out wrong), fix this file or the folder
  README in the same change and mention it in the commit body.
- Prefer a guard to prose: a test in `tests/test_docs_consistency.py` or a `[tool.docmap]` rule
  instead of another sentence. `docmap.py check` warns when this file exceeds 100 lines.
- Keep entries short, specific, and checkable: a command, a path, or a rule. Folder-specific rules
  belong in that folder's README, project knowledge in `docs/`.
