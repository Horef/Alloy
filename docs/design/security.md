# 18. Security and data handling

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md).

- Corpora, outputs, caches, `.env`, and `config.toml` are git-ignored; caches contain extracted
  document text and must be protected like the corpus.
- Secrets live only in environment variables and never reach manifests, logs, or checkpoints.
- The model never receives tools; all inputs are framed as untrusted data; all outputs are validated.
- Personal-data question types exist in the schema but are rejected by generation until a controlled,
  masked, auditable personal-data adapter exists. The generator does not fabricate personal data.
