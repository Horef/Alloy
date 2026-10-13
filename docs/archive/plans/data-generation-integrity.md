# Plan: preserve dataset meaning and generation evidence

Status: implemented offline on 2026-09-07, with corpus-wide boundary verification and a comprehensive extraction-summary artifact retained as documented limitations. Parent review: [library-review.md](../library-review.md).

## Objective

Ensure a question means the same thing across CSV/JSONL/premade workflows, and make generation's evidence limits and shortfalls visible. Preserve the existing silver-review workflow rather than attempting to make LLM-generated labels automatically authoritative.

## 1. Normalize at input boundaries without losing metadata

Files: `models.py`, `validation.py`, `io.py`, `results_io.py`, `cli.py`, their tests.

1. Extract legacy payload normalization before constructing `SilverQuestion`. Apply it consistently to CSV, JSONL, and premade imports. Keep strict validation separate from compatibility defaults.
2. Preserve explicit `expected_behavior`, `question_form`, `parent_question_id`, `question_type`, `difficulty`, `reference_claims`, supporting quotes, and reference sources where supplied. Do not replace provided atomic claims with the entire expected answer.
3. Use one strict boolean parser: accept documented true/false forms, strip whitespace, and reject unknown values. Missing values use the documented default; a typo must not silently become an abstention task.
4. For legacy rows lacking expected behavior, derive answer/abstain from answerability. For explicit clarify behavior with absent form, normalize to ambiguous; explicit contradictory form/behavior or answerability values must be rejected. Do not infer clarification from punctuation or arbitrary prose.
5. Preserve the legacy external-ID-to-ROW mapping during this focused change, but map parent references through it so relationships remain valid. Never retain external parent IDs against unrelated ROW child IDs without a mapping. Record unresolved parents as such; a parent may legitimately be absent from a filtered subset.
6. When original source provenance and actual retrieved context are both supplied, keep them distinct. The current single `source` input is runtime context by documented contract. Add an explicit reference-provenance route; do not repurpose the same column to serve both meanings.
7. Replace the evaluator's blanket `source_file` metadata shortcut (which suppresses all reference sources) with explicit reference-evidence origin. Imported runtime context is not gold; imported verified gold sources should not be thrown away merely because they came from a file.
8. Add structured import counts and row-error diagnostics: total, accepted, skipped, errors with row numbers and safe reasons. In non-strict mode, allow a partial import but make it visible in the manifest/report. If zero usable rows remain, stop before judging and never produce a completed evaluation.
9. Validate malformed CSV headers/extra fields consistently and report row numbers; don't let list-valued surplus cells reach string-only normalization and raise an obscure AttributeError.

Tests: equivalent CSV/JSONL tasks normalize identically; legacy answerability; three behaviors; missing versus contradictory form; strict booleans; atomic claims; source/reference separation; parent mapping; partial import; all invalid; malformed headers/extra cells. Include a synthetic two-claim answer and verify that its evaluation still has two fixed claims after premade import.

Acceptance for local hova: fresh imports retain 236 answer / 50 clarify / 50 abstain tasks and 50 ambiguous forms, preserve known parent relationships, and recognize the historical placeholders. Run this read-only check against the source CSVs; do not overwrite or silently relabel old evaluated JSONL.

## 2. Make CSV provenance lossless

Files: `io.py`, models only if necessary, `tests/test_io.py`, README.

`write_questions` joins source fields with ` | ` and `read_questions` splits on that delimiter. The document extractor itself uses that string between table cells. As a result, ordinary evidence can be truncated or shifted to the wrong source on a CSV round trip.

Implementation:

1. Add a canonical `sources_json` column containing the complete list of `SourceRef` objects. Prefer it when reading. Retain the human-readable source columns for compatibility and review convenience.
2. If only legacy parallel source columns exist, parse using the legacy behavior with a warning when lengths are inconsistent. Do not pretend delimiter-containing old data can always be reconstructed. Recommend the paired original JSONL as the lossless recovery source when available.
3. Validate source-ID/file/location relationships. Empty optional provenance is allowed for historical questions; present malformed JSON must fail explicitly rather than fall back silently.
4. Preserve formula neutralization and test literal apostrophe-prefixed text as well as actual formula-like text. Do not introduce another lossy escape/unescape scheme.
5. Document that editing readable source columns does not override `sources_json`; preferably provide a clear validation error if they disagree rather than silently ignoring a reviewer's change. Decide this behavior before shipping the new column.

Tests: table delimiters, multiple sources, pipes in filenames/locations/excerpts, multiline Hebrew, quotes, formula-like strings, no sources, legacy input, malformed JSON, and conflicting readable/canonical fields. Compare the entire `SourceRef` list across CSV and JSONL, not just its first filename.

## 3. Stop silently dropping document evidence

Files: `documents.py`, `cache.py`, `tests/test_documents.py`, README.

`_structured_chunks` only emits content with at least 80 non-whitespace characters. A short policy or trailing section disappears without a warning. This is particularly harmful to boundary-question generation because omitted facts may then look absent.

Implementation:

1. Emit all nonempty meaningful content. For a short trailing chunk, merge with an adjacent chunk only when the size bound and source location allow it; otherwise emit a short chunk. Keep `chunk_chars` as a maximum target, not an evidence-deletion threshold.
2. Preserve headings with their associated content when practical. Do not globally discard short pages/sections as “noise.” If filtering is later needed, make it explicit with omission diagnostics.
3. Return or separately collect an extraction summary with supported files, extracted sections, empty/unreadable sections, and omitted content counts/reasons. Warn about PDFs with no extracted text on some pages; do not claim OCR was performed.
4. Preserve DOCX paragraph/table order using the document's block order. Add a fixture with a heading, table, and following paragraph whose meaning depends on that order.
5. Validate chunk sizes/overlap at the public loader boundary, including negative values. Retain the existing cache invalidation for changed extraction code; include extractor package versions in provenance if extraction differences matter across environments.
6. Reject `--output` under `--documents` and clearly document input-root scope. Keep generated caches/artifacts out of the knowledge inventory.

Tests: a complete short document; a long section followed by a short material rule; empty section; short final PDF page via a fake extractor; block/heading association; DOCX interleaving; chunk bound and overlap; output/corpus overlap. The 26-character synthetic policy in the probe must survive.

## 4. Reject invalid candidates during bounded refill

Files: `generator.py`, `validation.py`, generator tests.

`validate_candidate` checks quotes for answerable candidates but returns early for unanswerable ones, allowing blank text and nonempty factual claims. `write_questions` then fails on the whole dataset after model calls have completed.

Implementation:

1. Validate nonblank question/reference text for every candidate before acceptance. Enforce no factual claims or supporting quotes on unanswerable candidates, matching the existing generation prompt and dataset validator.
2. Use a shared candidate-to-question semantic check where possible so generation acceptance and final writing cannot disagree. Return a rejection reason to the existing bounded refill loop rather than aborting all work at final serialization.
3. Validate variation text/form/required-clarification consistency before accepting it. Keep semantic equivalence of natural variations as a review requirement; don't claim deterministic string checks prove it.
4. Validate discovered topic source IDs against available chunks, deduplicate labels/IDs, and report empty or invalid topic coverage. Apply checks to cached and fresh topics alike.
5. Preserve bounded attempt count and record rejection/shortfall reasons in an artifact, not only log output. The last valid partial dataset should remain reviewable when a quota cannot be met.

Tests: blank answerable/unanswerable candidate; unanswerable with factual claims or quotes; unknown/duplicate sources; invalid variation; malformed topic references; a rejected candidate followed by a valid one; no valid candidate after all rounds; final writer accepts every generated item.

## 5. Make quotas and evidence scope honest

Files: `generator.py`, `cli.py`, models/artifacts as needed, tests and README.

Some shortfalls are mathematically predetermined: one topic with a 35% cap cannot consume a 30-question budget. Type targets are allocated from all chunks but enforced on candidates generated from smaller topic-specific excerpts. Boundary questions are labeled against that limited evidence, not a checked whole corpus.

Implementation:

1. Precompute feasibility of topic caps and minimums. Keep the cap by default and report the unallocated budget with reason `topic_cap_capacity`; do not silently relax the user's cap or blame discarded candidates. Report the effective allocation before model calls.
2. Allocate type targets using the evidence actually rendered for each topic, or treat global targets as soft goals with explicit shortfall reporting. Never repeatedly demand cross-document evidence when only one document was supplied to that generation call.
3. Define `--topic-count` as a total ceiling for the requested topic, including variants and boundary cases, consistent with the current help. Compute all three budgets from that ceiling. The current code only limits canonical questions while later stages use the global budget. Add a test where global max greatly exceeds the topic count.
4. Report planned, accepted, rejected, and unallocated counts by task/form/type/topic. Keep totals within the effective ceiling and retain existing ratios as targets rather than fabricated guarantees.
5. Track the source IDs actually rendered after limits; validate provenance against that set for the generation call. A valid corpus ID alone does not prove the model saw its contents.
6. For boundary questions, store an explicit absence-check scope such as `selected_excerpts` plus evaluated source IDs, and require review before interpreting them as globally unanswerable. Update documentation to match. Do not add an expensive full-corpus LLM verification pass by default.
7. If a future optional corpus-wide answerability check is added, keep it as a separately configured stage with explicit call/budget limits. Search failure alone must not be reported as proof of absence.
8. Keep quote matching described as provenance validation. It proves the text occurs in a source, not that the quoted text entails every reference claim or that a natural variation preserves intent.

Tests: one-topic cap; impossible minimum/cap combination; locally unavailable question type; topic total ceiling with variants and unanswerables enabled; deterministic shortfall reasons; source omitted by rendering limit; accepted boundary item records limited scope; unsupported candidates never force fabricated completion.

## Completion and compatibility

Implement sections 1–2 first because they directly affect the next evaluation. Follow with document completeness and generation behavior. Keep old CSV/JSONL readable through explicit migration defaults and preserve historical files. New metadata must be defaulted when loading old artifacts; old artifacts should be marked legacy/unknown where information cannot be recovered.

Run focused import/IO/document/generator tests, then the full army suite and `git diff --check`. Add one synthetic generation → CSV/JSONL → import → fake evaluation regression that checks behaviors, parent mapping, claims, and complete provenance. No live model call or rewrite of hova outputs is necessary to complete the implementation.
