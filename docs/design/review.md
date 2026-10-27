# 12. Human review and set maintenance (`review.py`)

Status: current (verified 2026-10-27)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md).

Generated questions are silver until reviewed. The review workflow is designed so a reviewer can work
in a spreadsheet without ever touching provenance JSON.

**`review-export`** writes a small CSV (optionally with Hebrew headers) with the readable content,
read-only context (parent question, boundary kind, flattened sources and quotes, broad key points),
and the decision columns `review_status`, `reviewer_notes`, and `source_hint`. A Markdown file gives a
read-only formatted view. Anchors are shown as each variant's parent question, not as separate rows.

**`review-merge`** joins the edited CSV back onto the canonical JSONL by `id`:

- reviewer-editable fields are overlaid; every technical field is copied from the canonical record;
- a deleted row is an intentional deletion; an unknown ID is an error (except reviewer-added
  `Q-manual-*` rows, which become new questions with `needs_reground`);
- editing `question` or `expected_answer` keeps the old grounding but appends a note and downgrades
  `approved` to `needs_reground`, because the stored quotes may no longer support the new text;
- changing `expected_behavior` aligns the hidden task fields (see [section 20](../archive/architecture-review-fixes.md#20-changes-made-during-this-review)).

*Why a sidecar file:* the canonical JSONL stays the single source of truth for technical fields, so the
review file never has to reconstruct provenance, and a reviewer cannot accidentally break it.

**`reground`** regenerates only grounding (sources, quotes, claims) for questions whose text changed.
Evidence is ranked by the edited question and answer, always includes the previously cited sources,
and puts first any chunks a reviewer's `source_hint` names or quotes. The model must echo the fixed
question and answer unchanged, and its output passes the same `validate_candidate`. Success resets the
row to `pending` for re-approval.

**`revary`** regenerates only variants, keeping reviewed canonical and boundary questions untouched.
**`add-broad`** adds or replaces broad questions using the cached graph.
