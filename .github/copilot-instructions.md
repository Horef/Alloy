Follow [AGENTS.md](../AGENTS.md) at the repository root; its rules are mandatory. Read it and `AGENTS.local.md` (this machine), then `docs/STATUS.md` and `docs/MAP.md`, before any change. Most important if it is not loaded:

- Never commit corpora, outputs, caches, `.env`, `config.toml`, or PDF/XLSX/DOCX files, and never force-push.
- Before committing, run `python -m pytest -q` and `python tools/docmap.py check` (after `python tools/docmap.py update`), where `python` is the interpreter named in `AGENTS.local.md`.
- Commit each logical, tested change with a Conventional Commit message, then `git push origin main`.
