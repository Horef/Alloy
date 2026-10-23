# 10. Automatic sizing (`planning.py`)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md).

Most users cannot say how many questions a corpus needs. `plan` measures it instead.

**Information units.** `information_units(text)` counts distinct statements: the text is split at
sentence terminators (not inside numbers like `5.5`), line breaks, list bullets, numbered items, and
table cell separators, and a piece counts if it has at least four words. This is a cheap, language-
agnostic proxy for "how many facts does this chunk state" *(inferred: no model call, stable across
runs)*.

**Per-topic canonical quota** is the larger of two rules:

- *Content rule:* each chunk with *u* units contributes `max(1, round(u × coverage_target /
  claims_per_question))`. Every chunk gets at least one question and dense chunks get more. With the
  defaults (0.35 coverage, 3.2 claims per question, measured on the project's corpora) a chunk with
  ten statements yields one question; thirty statements yield three.
- *Precision rule:* enough questions to estimate the topic's accuracy within ± `margin_of_error` at
  95% confidence in the worst case p = 0.5: `n = ⌈1.96² × 0.25 / E²⌉`, which is 25 for E = 0.20. The
  rule is capped at `max_topic_coverage` of the topic's units divided by claims per question, so a
  tiny topic is never asked for more questions than it has facts.

**Other budgets.** Variants are `variation_share` × canonical (with `ambiguous_share` of them
ambiguous). Boundary questions are sized so they are `boundary_share` of the final set, with at least
`min_boundary_per_topic` per topic, and are distributed by information units with the
largest-remainder (Hamilton) method. In `user_facing` mode every canonical and boundary question also
gets one natural phrasing.

**Binding.** The plan stores a fingerprint of the topic partition (names plus node IDs) and the graph
settings that produced it. `generate --plan` re-applies those settings, rebuilds the graph, and
refuses the plan if the topics no longer match, so a plan can never silently apply quotas to
different topics. In theme mode the plan also pins the theme vocabulary (section 7.5). The
per-topic numbers are meant to be edited by hand; totals are recomputed.
