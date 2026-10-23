# 17. Artifacts, reliability, and observability (`artifacts.py`, `io.py`)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md).

- **Atomic writes.** Every final file is written to a temporary file in the same directory, fsynced,
  and renamed, so a crash never leaves a half-written report or question set. Question CSV and JSONL
  are published together.
- **Journals.** Both checkpoints are append-only JSONL. A torn final line (crash mid-write) is
  repaired; corruption anywhere else, or a contract mismatch, is a hard error, because silently
  skipping it would mix incompatible results.
- **Run manifests.** Every command writes `run_manifest.json` (or `report_manifest.json`) with the
  redacted settings, CLI parameters, SHA-256 of every input and output, package and dependency
  versions, git commit and dirty flag, an implementation hash, timing, status, and command-specific
  diagnostics. The live chatbot URL is stored only as a hash and header values never.
- **CSV safety.** Values that start with `=`, `+`, `-`, or `@` are prefixed with an apostrophe to
  neutralize spreadsheet formula injection, and the escape is versioned (`csv_escape_version`) so it
  round-trips losslessly. Provenance in CSV is stored once as `sources_json`; the readable columns
  must agree with it.
- **Logs** record counts, IDs, models, outcomes, latencies, retries, and rejection reasons, never full
  questions, answers, keys, or header values. Apigee quota headers are logged at `INFO`.
  `logging_utils.py` configures console and file logging and quiets SDK transport chatter;
  `progress.py` wraps progress bars so production runs can disable them.
- **Supporting modules.** `history.py` loads earlier runs (evaluation records, insights only when
  their status file matches the records, and current prompts) for comparison and prompt revision.
  `response_errors.py` recognizes the known "no summary" placeholder answer and computes bounded
  retry delays shared by the Gemini client and the chatbot adapter. `labels.py` holds the Hebrew
  labels and colors used in CSV and HTML outputs.
