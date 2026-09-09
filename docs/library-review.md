# Library review and implementation roadmap

Reviewed 2026-09-06 against the then-current `src/chatbot_eval` checkout and implemented offline 2026-09-07. This extends [the prompt-generation review](prompt-improvement-review.md). Historical artifacts were not changed; current implementation evidence is recorded in [implementation-progress.md](implementation-progress.md).

## Assessment

The library is about 4,680 Python lines, with useful separation between documents, generation, adapters, evaluation, reports, and persistence. It does not need a rewrite. Its main weakness is **inconsistent contracts between those components**: the same input can acquire different meaning depending on the import path; classification can disagree with claim evidence; checkpoint identity covers some incidental settings while omitting other consequential ones.

The baseline suite passed: **87 tests in 1.57 seconds**, using `conda run -n army python -m pytest -q`. Additional synthetic probes reproduced failures not covered by that suite. [Probe code](review_probes.py) and [observed results](review-probe-results.jsonl) are included. Run with:

```sh
conda run --no-capture-output -n army python docs/review_probes.py
```

These probes print current behavior; they are not regression tests asserting that the bugs should remain. They use temporary files and fake adapters/LLMs, make no network calls, and should be converted into desired-behavior tests during implementation.

## Prioritized findings

P1 means prioritize before relying on a new benchmark or long run. P2 means address in the corresponding focused change. These are engineering priorities, not security vulnerability ratings.

| ID | Priority | Finding and consequence | Evidence | Plan |
|---|---|---|---|---|
| E1 | P1 | Substring abstention detection overrides the judge even for a substantive correct answer. “The optional attachment is not provided. Submit the application tomorrow.” becomes `incorrect_abstention`. | Reproduced; `evaluator.py:looks_like_abstention`, `_evaluate_one` | [Evaluation](plans/evaluation-correctness.md) |
| E2 | P1 | Clarification/abstention success takes precedence over false claims. Two false claims can coexist with `correct_abstention` or `correct_clarification`; reported risky rate and aggregate false-claim count can both be zero. | Reproduced with accepted judge-score objects; `classify`, `report.py:build_summary` | Evaluation |
| E3 | P1 | Identical benchmark IDs are insufficient for metric comparability when errors change denominators. Turning the sole failed answer into a judge error yields a favorable 50-point accuracy delta, with zero paired improvements. | Reproduced; `report.py:build_comparison` | Evaluation |
| D1 | P1 | Premade imports erase clarification/form/parent metadata and atomic claim structure. The clarification-label bug is established in the prior hova review; `reference_claims` are also always rebuilt as one whole-answer claim. | Code inspection and prior CSV evidence; `results_io.py` | [Data/generation](plans/data-generation-integrity.md) |
| D2 | P1 | CSV source serialization is lossy. `Name \| Amount \| Deadline` round-trips as `Name`, while JSONL preserves it. Real table extraction uses this same delimiter. | Reproduced; `io.py:write_questions`, `read_questions` | Data/generation |
| R1 | P1 | A valid final checkpoint entry without a newline loads, but the next append concatenates JSON objects and breaks subsequent resume. Both checkpoint types are affected. | Reproduced; `artifacts.py` | [Run reliability](plans/run-reliability.md) |
| R2 | P1 | A generation-checkpoint signature mismatch on the last line without a newline is treated as a torn write, and the entry is discarded instead of rejecting the incompatible resume. | Reproduced, synthetic file becomes zero bytes; `StructuredCallCheckpoint._load` | Run reliability |
| R3 | P1 | Stale insights remain in a reused output directory when insights are disabled or fail. Later directory discovery can attach them to new results. | Disabled-insights case reproduced; failure branch has the same non-write behavior | Run reliability |
| R4 | P2 | Live `--retry-errors` retries the chatbot after a judge error, replacing the already captured answer. This changes the item being judged and adds calls. | Reproduced; `cli.py:_checkpointed_evaluation` | Run reliability |
| R5 | P2 | Cache/resume identity omits gateway endpoint and some implementation dependencies; live identity contains header names but not nonsecret routing values. Environment or routing changes can reuse old work. | Code inspection; `cache.py`, CLI contracts/fingerprints | Run reliability |
| R6 | P1 | HTTP `answer: null`, empty text, and object-valued errors are accepted as normal responses (`null` is converted to text). Live and premade paths disagree on empty-response handling. | Reproduced; `adapters.py:_field`, `ask` | Run reliability |
| D3 | P2 | Legacy JSONL with `answerable=false` and no `expected_behavior` fails validation, while the CSV path derives abstention. Invalid CSV booleans also silently become false. | JSONL failure reproduced; CSV parser inspected | Data/generation |
| D4 | P2 | All-invalid premade rows return an empty list; the CLI can proceed to write an apparently completed zero-record evaluation. Skipped rows have no structured artifact. | Empty import reproduced; CLI completion path inspected | Data/generation |
| G1 | P1 | Chunking silently drops text shorter than 80 characters, including complete short policies or final sections. The absent text then cannot inform generation. | Reproduced with a 26-character rule; `documents.py:_structured_chunks` | Data/generation |
| G2 | P2 | Candidate checks accept blank unanswerable questions with factual claims; final dataset validation rejects them only after generation has finished. | Reproduced; `generator.py:validate_candidate`, `validation.py` | Data/generation |
| G3 | P2 | A single-topic corpus with a 35% cap can allocate only 11 of 30 questions before any candidate is considered. Global type targets can also be infeasible in selected topic evidence. The final shortfall message attributes everything to bad candidates. | Quota case reproduced; type-selection path inspected | Data/generation |
| G4 | P2 | “Unanswerable” means absent from at most the selected topic excerpts, but documentation describes corpus-level absence. A rule in an omitted chunk can invalidate that label. Verbatim quotes likewise establish provenance, not semantic entailment. | Code/design limitation; `_relevant_chunks`, `UNANSWERABLE_PROMPT`, `validate_candidate` | Data/generation |
| G5 | P2 | `--topic-count` limits canonical questions but not the later variant/boundary budgets. A topic ceiling of one produces six questions in a synthetic run. | Reproduced; `generator.py:generate` | Data/generation |

## Recommended sequence

1. Repair importer fidelity (D1–D4) and evaluator classification (E1–E2), then comparison denominators (E3). These determine whether improvement measurements mean what they say.
2. Repair checkpoint loading (R1–R2), response validation (R6), and stale-artifact discovery (R3) before another expensive run.
3. Complete retry identity/routing (R4–R5), document completeness (G1), and generation validation/shortfall reporting (G2–G5).
4. Implement the earlier prompt-improvement plan against the corrected evidence. Share its bounded-evidence work with insights rather than maintaining two almost-identical implementations.
5. Re-evaluate the original saved responses into fresh directories, then run new target-chatbot experiments. Offline tests do not establish judge quality, retrieval quality, or prompt effectiveness.

## Simplification recommendations

Keep the existing dataclasses/Pydantic boundary models, lightweight adapters, and content-addressed cache. Atomic writes and per-question checkpoints have a concrete purpose; they are not gratuitous complexity.

Prefer these small simplifications after correctness tests exist:

- One question-normalization function used by both JSONL/CSV input paths and premade import. Separate legacy migration from strict validation.
- One helper for bounded text and one for bounded structured evidence, with explicit omission accounting. There are currently independent versions in evaluator, generator, prompt generator, insights, and topic inference; not all limits mean the same thing.
- One run-contract builder, with explicit stage dependencies, rather than scattered hand-built dictionaries. Include consequential configuration and exclude incidental execution settings where safe.
- Separate chatbot capture from judging in the live workflow. This makes judge-only retries possible without introducing an orchestration framework.
- Extract only the repeated evaluation finalization from `cli.py` after the preceding behavior is tested. Keep command-specific preparation explicit.
- Keep the report's escaped HTML generation unless there is an actual maintenance need for a template dependency. The larger risk today is the meaning of the numbers, not the rendering technique.

Avoid adding an agent system, vector store, optimizer, generic plugin framework, or an extra LLM call for every validation check. Model-generated silver labels still need review; no local quote check can prove an unanswerable question is absent from the entire corpus.

## Additional limitations and lower-priority follow-ups

- `Retry-After` has no finite cap or total retry deadline in either HTTP transport. Add bounded finite delays in the reliability plan; do not reinterpret a per-attempt timeout as a total-run deadline.
- Some transport/read exceptions and invalid URL construction escape `ask`; convert expected transport failures into per-record errors, while leaving programming errors visible.
- Output directories under the corpus are not rejected, while cache directories under it are. Generated Markdown/JSON/CSV can become next-run knowledge. Add output-path validation alongside document inventory checks.
- DOCX extraction moves all paragraphs ahead of all tables instead of preserving document order. PDF pages with no extracted text are silently skipped when other text exists. Add extraction warnings and structural tests, not a mandatory OCR subsystem.
- Topic inference and insight validation can accept incomplete/duplicate assignments or references to records omitted from the request. Record coverage and validate against transmitted IDs. The earlier prompt plan covers the same traceability issue.
- `--hide-correct-answer-metrics` hides the headline/topic cells but not the comparison table. Clarify its documented scope before changing it. Topic usefulness also uses all evaluable tasks as denominator, while the headline uses answerable tasks; align or label these explicitly.
- Manifests omit API-key fields, but raw exception strings are written to logs/manifests. Sanitizing configured secret values and URL credentials is a concrete follow-up; no actual credential leak was established in this review.
- Dependency versions are lower-bounded, not locked, and installed package versions are not recorded in manifests. Record relevant versions for reproducibility. No unsupported SDK-version failure was established here.

## Scope and confidence

Reviewed the source modules, associated tests, CLI paths, package configuration, and the prior hova evidence. Confirmed cases are marked as reproduced or inspected; semantic-generation concerns are explicitly limitations. No paid LLM calls, live chatbot requests, deployment, or historical data edits were performed. No claim is made that this is an exhaustive security audit or that the remaining code is defect-free.

The three plans include concrete implementation boundaries, compatibility choices, and desired tests. A smaller implementer can take one plan at a time; it should not attempt a simultaneous broad rewrite.
