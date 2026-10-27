# 18. Security and data handling

Status: current (verified 2026-10-27)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md).

- Corpora, outputs, caches, `.env`, and `config.toml` are git-ignored; caches contain extracted
  document text and must be protected like the corpus.
- Secrets live only in environment variables and never reach manifests, logs, or checkpoints.
- The model never receives tools; all inputs are framed as untrusted data; all outputs are validated.
- Personal-data question types exist in the schema but are still rejected by generation. They will be
  built only for synthetic personas from the profile schema (section 22, D14); real profiles never
  enter Alloy, and `profile-check` reports field names, never values.
