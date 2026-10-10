# Agent instructions for Alloy

Alloy (`chatbot_eval`) generates evidence-grounded Hebrew evaluation sets from a document corpus
and judges internal chatbots against them. [README.md](README.md) is the user manual (workflows,
settings, outputs); [docs/architecture.md](docs/architecture.md) is the design reference. Before
changing a component, read its section there.

## Environment

- Python 3.11 or newer. On the maintainer's machine use
  `/Users/sergiyhoref/Programming/Army/army/bin/python` (3.13, package installed editable), not the
  base conda interpreter. Elsewhere (cloud sessions, CI): `python -m venv .venv` then
  `.venv/bin/pip install -e '.[dev]'`.
- PDF conversion uses `pdfplumber` (a core dependency); OCR shells out to the optional `tesseract`
  binary. Tests need neither LibreOffice, Tesseract, nor Hebrew fonts: they build PDFs at test time.
- Tests: `$PY -m pytest -q`. All offline (fake LLMs, no network), about two seconds. Run the full
  suite before every commit; CI runs it on 3.11 and 3.13.
- Layout: `src/chatbot_eval/` (one module per component; the README code map lists them all),
  `tests/` (one `test_<module>.py` per module, plus `test_docs_consistency.py`), `docs/`.

## Git workflow (mandatory in every session)

- The repository is under git with remote `origin` = `git@github.com:Horef/Alloy.git`, branch `main`.
- Start by running `git status`; do not discard or overwrite changes you did not make.
- Commit after each logical, tested change. Use Conventional Commits
  (`fix(generation): …`, `feat(retrieval): …`, `docs: …`, `test: …`, `chore: …`) with a short body
  explaining why.
- Before committing: full test suite green and `git diff --check` clean.
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
never use `git add -f` for them. Do not paste corpus text into commit messages, docs, or tests; use
short synthetic Hebrew fixtures instead. Tests that need a PDF build one at test time.

## Code conventions

- Every prompt treats document and question text as untrusted data and asks for Hebrew output.
- Every model output passes deterministic validation (`validate_candidate`, quote provenance)
  before it is accepted; add a rejection reason to the diagnostics rather than failing silently.
- New settings go in `config.Settings` (with validation), `config.example.toml`, and the README
  parameter table; wire them through `GenerationOptions` and the CLI manifest parameters.
- Cache keys must change whenever outputs would change. Fingerprint prompts and schemas, and
  fingerprint deterministic code with `contracts.code_fingerprint` (ignores comments and
  docstrings), never raw file bytes, so unrelated edits do not force expensive re-extraction.
- Keep optional, cost-adding checks off by default and report their outcomes in
  `generation_diagnostics.json` or the run manifest.
- Heavy or optional dependencies are extras in `pyproject.toml`, imported
  inside the function that needs them, with an error that names the extra to install.
- Changing outcome classification, judge inputs, or metric definitions changes what past runs mean.
  Say so in the commit body, and update the README and `docs/architecture.md` sections that define them.
- Validate behavior changes offline against real artifacts when useful (cached graphs in
  `{hova,keva}/.chatbot_eval_cache/graph/`, checkpoints in `*/outputs/*/generation_checkpoint.jsonl`),
  using throwaway scripts outside the repo.

## When you change X, also update Y

Agents have repeatedly changed code without updating the files that describe it. Check your diff
against this table before committing. Rows marked *guarded* fail `tests/test_docs_consistency.py`
when forgotten.

| Change | Also update |
|---|---|
| Add or rename a setting | `Settings` and `validate`, `config.example.toml`, README parameter table (*guarded*) |
| Add a CLI command or option | README table for that command (*guarded*) |
| Add a module | README code map and `docs/architecture.md` (*guarded*) |
| Change a component's behavior or design | Its section of `docs/architecture.md`; section 19 if a limitation appears or disappears |
| Change outcomes, judging, or metrics | README "Judging logic" and outputs sections; architecture sections 13 and 14 |
| Change a prompt, schema, or deterministic stage | The matching cache key or fingerprint (architecture section 9), including every module the stage calls |
| Change PDF conversion output | Bump `pdf_markdown.CONVERTER_VERSION` (written into converted files) |
| Add an output file or manifest field | README outputs section for that workflow |
| Add a relative link in a doc | Make sure it resolves (*guarded*) |

`docs/library-review.md`, `docs/implementation-progress.md`, `docs/plans/`, and
`docs/theme-layer-ab-results.md` are dated records. Do not rewrite them; record the current state in
`docs/architecture.md`.

## Keeping these instructions accurate

These instructions load into every session, so a wrong or missing line costs every future session.

- When a session reveals a gap (a step you had to discover, a file you nearly missed, a convention
  that was not written down, or an instruction that turned out wrong), fix this file in the same
  change and mention it in the commit body.
- Prefer a guard to prose: if a rule can be checked mechanically, add a test to
  `tests/test_docs_consistency.py` (or the relevant test file) instead of another sentence.
- Keep entries short, specific, and checkable: a command, a path, or a rule. Delete entries that no
  longer change behavior. Do not add one-off incidents, generic advice, or facts the code already
  makes obvious.
