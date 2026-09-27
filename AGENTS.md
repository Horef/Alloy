# Agent instructions for Alloy

Alloy (`chatbot_eval`) generates evidence-grounded Hebrew evaluation sets from a document corpus
and judges internal chatbots against them. See [README.md](README.md) for workflows and settings.

## Environment

- Python: `/Users/sergiyhoref/Programming/Army/army/bin/python` (3.13, package installed editable).
  Do not use the base conda interpreter.
- Tests: `$PY -m pytest -q` (all offline; fake LLMs, no network). Run the full suite before every commit.
- Layout: `src/chatbot_eval/` (generation: `documents` → `graph_build` → `graph` → `generator`,
  with `retrieval`, `cache`, `llm`; evaluation: `evaluator`, `report`), `tests/`, `docs/`.

## Git workflow (mandatory in every session)

- The repository is under git with remote `origin` = `git@github.com:Horef/Alloy.git`, branch `main`.
- Start by running `git status`; do not discard or overwrite changes you did not make.
- Commit after each logical, tested change. Use Conventional Commits
  (`fix(generation): …`, `feat(retrieval): …`, `docs: …`, `test: …`, `chore: …`) with a short body
  explaining why.
- Before committing: full test suite green and `git diff --check` clean.
- After committing, push: `git push origin main`.
- Never force-push, rewrite published history, amend pushed commits, or push the local
  `backup/*` branches (they contain purged internal files). Ask the user first for anything destructive.

## Data safety

This project handles internal IDF documents. Never commit or push corpora, outputs, or caches:
`hova/`, `keva/`, `knowledge_base/`, `outputs/`, `.chatbot_eval_cache/`, `.env`, `config.toml`,
`*.pdf`, `*.xlsx`, `*.docx`. They are in `.gitignore`; check `git status` before every commit and
never use `git add -f` for them. Do not paste corpus text into commit messages, docs, or tests; use
short synthetic Hebrew fixtures instead.

## Code conventions

- Every prompt treats document and question text as untrusted data and asks for Hebrew output.
- Every model output passes deterministic validation (`validate_candidate`, quote provenance)
  before it is accepted; add a rejection reason to the diagnostics rather than failing silently.
- New settings go in `config.Settings` (with validation), `config.example.toml`, and the README
  parameter table; wire them through `GenerationOptions` and the CLI manifest parameters.
- Cache keys must change whenever outputs would change; fingerprint prompts/schemas, not whole
  modules, so unrelated edits do not force expensive re-extraction.
- Keep optional, cost-adding checks off by default and report their outcomes in
  `generation_diagnostics.json`.
- Validate behavior changes offline against real artifacts when useful (cached graphs in
  `{hova,keva}/.chatbot_eval_cache/graph/`, checkpoints in `*/outputs/*/generation_checkpoint.jsonl`),
  using throwaway scripts outside the repo.
