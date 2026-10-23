# 15. Insights and topic inference (`insights.py`, `topics.py`)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md).

**Insights** (`--generate-insights`) make one call over the aggregate summary plus compact
per-question records, prioritizing risky and failed outcomes within a character budget. Code then
drops example IDs that were not in the evidence, recomputes evidence counts and affected topics,
discards issues without evidence, and downgrades a "high priority" issue backed by a single example.
The prompt requires stating causes as hypotheses and forbids repeating personal values. A status file
binds the insights to the exact records hash, so stale insights in a reused folder are never picked up
by a later run.

**Topic inference** (`evaluate-file --infer-topics`) labels unclassified premade questions in batches
(at most 250 questions and 60,000 characters per call), then makes one harmonization call that merges
synonymous labels across batches into one taxonomy.
