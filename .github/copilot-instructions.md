Follow [AGENTS.md](../AGENTS.md) at the repository root; its rules are mandatory. Read it, then `docs/STATUS.md` and `docs/MAP.md`, before any change. Most important if it is not loaded:

- Never commit corpora, outputs, caches, `.env`, `config.toml`, or PDF/XLSX/DOCX files, and never force-push.
- Before committing, run `/Users/sergiyhoref/Programming/Army/army/bin/python -m pytest -q` and `python tools/docmap.py check` (after `python tools/docmap.py update`).
- Commit each logical, tested change with a Conventional Commit message, then `git push origin main`.
