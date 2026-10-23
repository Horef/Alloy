# AGENTS.local.md template

Copy to `AGENTS.local.md` (git-ignored) and fill it in for this machine; agents create and update it (`python tools/docmap.py local` prints what it can detect). One `- key: value` line per fact. Tracked files refer to these keys, never to their values. No secrets: agents read this file into their context.

- machine: <host name (OS, arch, CPUs, RAM)>
- python: <absolute path of the interpreter with Alloy installed editable (`pip install -e '.[dev]'`, Python 3.11 or newer)> - what `python` in AGENTS.md means
- memory_budget: <peak RAM that local runs may use, e.g. "6 GB, never above 10 GB">
- compute: <where heavy jobs run, or "none" (model calls go to the Gemini API or the Apigee gateway)>
- notes: <quirks agents must respect here, e.g. the SSH key asks for a passphrase, so push unpiped>
