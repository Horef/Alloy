# 20. Changes made during this review

Moved verbatim on 2026-10-23 from section 20 of the former `docs/architecture.md`; a dated record of the fixes made while that document was written. The current design is in [../design/](../design/) and [../ARCHITECTURE.md](../ARCHITECTURE.md).

Writing this document involved reading every module against the README and tests. The following
bugs and inconsistencies were found and fixed, each with a regression test where it is behavior.

First round:

1. **Comparison populations ignored anchor scoring** (`report.build_comparison`). Headline rates
   exclude anchors by default, but comparison denominators, numerators, and soft deltas counted them.
   In `user_facing` sets a 3/4 rate was displayed as 6/8, and anchors leaked into soft comparisons.
   Eligible populations now follow `canonical_scoring`.
2. **Changing `expected_behavior` in review broke the merge** (`review._apply_edits`). Relabelling an
   answer question as a boundary question kept its hidden reference claims and quotes, so validation
   rejected the entire merge, and the reviewer could not fix that from the review file. The merge now
   aligns `answerable`, `question_type`, claims, and quotes with the new task, and flags a new answer
   task for `reground` (with a question type `reground` accepts).
3. **`generate --plan/--auto-plan --topic` was rejected only after the graph was built.** It is now
   rejected before any model call.
4. **The `generate` manifest omitted `--merge-into` and `--plan`** from its input inventory.
5. **Cache events** for the graph and theme vocabulary now read `disabled` when caching is off.
6. **README and `config.example.toml` corrections:** removed the claim that abstention is detected by
   phrase patterns, corrected the legacy-claims fallback, the `boundary_evidence_scope` values, the
   topic and quota descriptions, and `max_cluster_nodes`; replaced stale limitations; completed the
   code map; added the missing settings (`gemini.seed`, `closed_book_check`, `questions_per_call`,
   `context_nodes`, `[planning]`).

Second round (the open questions from the first round):

7. **A nonresponsive answer is no longer a hallucination** (`evaluator.classify`, step 11 in
   section 13.4).
8. **Premade resume is per row** (`results_io`): import diagnostics live in the manifest only.
9. **Prefix tolerance needs an attested bare word** (`retrieval`, section 8), removing false matches
   such as `הורה`/`מורה` in search and dedup; the `_strip_one_prefix` docstring now matches the code.
10. **Theme vocabularies are pinned in plans** (`plan --themes-from`, section 7.5).
11. **Code fingerprints ignore comments and docstrings** (`contracts.code_fingerprint`, section
    9.2), and evaluation and generation contracts no longer hash `llm.py`, whose call mechanics are
    versioned by `STRUCTURED_CALL_CONTRACT`.
12. **System-prompt validation judges the model's text,** not the code-assembled policy (section 16).
13. **Extraction diagnostics** list pages and files without text (section 6).
14. **Optional retry deadline** `evaluation.retry_deadline_seconds` (section 5).

The second round changes scoring semantics and the evaluation contract, so runs judged before it are
reported as contract-incompatible in comparisons. It does not change question IDs or the silver-set
and review file formats.
