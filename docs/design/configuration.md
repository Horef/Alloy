# 4. Configuration and entry points

Status: current (verified 2026-10-27)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md).

`config.Settings` (`config.py`) is a frozen dataclass loaded from TOML with every value validated up front
(`Settings.validate`), so a bad setting fails before any paid call. Secrets come only from
environment variables (loaded from a `.env` next to the config file) and are stripped from every
manifest. CLI flags override TOML for the current run.

`cli.py` exposes these commands:

| Command                           | Calls a model?   | Purpose                                                          |
| --------------------------------- | ---------------- | ---------------------------------------------------------------- |
| `plan`                            | yes (graph only) | Measure the corpus and write an editable `generation_plan.json`. |
| `generate`                        | yes              | Build the silver set.                                            |
| `review-export`, `review-merge`   | no               | Human review round trip.                                         |
| `reground`, `revary`, `add-broad` | yes              | Repair or extend a reviewed set without regenerating it.         |
| `evaluate`, `evaluate-file`       | yes (judge)      | Score a live chatbot or saved answers.                           |
| `report`                          | no               | Rebuild a report, optionally comparing two runs.                 |
| `generate-prompt`                 | yes              | Draft a system prompt package.                                   |

Offline commands do not even construct a model client, so they need no API key.
