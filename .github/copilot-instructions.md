Follow [AGENTS.md](../AGENTS.md) at the repository root; its rules are mandatory. In particular:

- Commit each logical, tested change with a Conventional Commit message, then `git push origin main`.
- Run the full test suite with `/Users/sergiyhoref/Programming/Army/army/bin/python -m pytest -q` before committing.
- Never commit corpora, outputs, caches, `.env`, `config.toml`, or PDF/XLSX/DOCX files, and never force-push.
