# Implementation progress — 2026-09-07

User authorization: implement the reviewed prompt, evaluation, data-integrity, and reliability fixes with delegated Sol Light or weaker agents. No live Gemini/chatbot calls, deployment, commits, or historical-output rewrites were part of this offline implementation.

## Completed

- Prompt generation now assembles a versioned deterministic Hebrew response policy around model-generated domain guidance. `guided`/`compact` instruction profiles and `balanced`/`conservative` answer policies make the desired burden explicit for weaker or stronger serving models.
- Prompt revision receives bounded, whole-record behavioral evidence with actual question, reference, answer, context, evidence, score, and omission metadata. Structured revision mappings connect observed failures and included IDs to changed rules, expected behavior, non-prompt limitations, and structured regression cases.
- Cached prompt packages are bound to model/endpoint/implementation/profile/policy identity and are revalidated for policy assembly before reuse.
- Question and premade-result import preserve behavior, form, parent relationships, atomic claims, quotes, and reference sources. Alternate columns have explicit CLI overrides, and runtime retrieval remains distinct from gold evidence.
- CSV provenance is lossless through `sources_json`; conflicting readable columns fail visibly. Final question/evaluation CSV and JSONL files are staged and atomically replaced.
- Evaluator classification uses judge evidence instead of abstention substrings, rejects conflicting whole-response flags, prevents unsafe clarification/abstention successes, reports factual risk across every scored task, and keeps metric populations explicit.
- Comparisons retain strict benchmark identity, suppress metric deltas when eligible ID sets change, and disclose compatible/incompatible/unknown evaluation contracts from original run manifests.
- Judge-only retries reuse the saved chatbot response. Checkpoint parsing repairs only a torn final JSON tail, rejects semantic mismatches, and normalizes append boundaries.
- HTTP answers require a nonblank string; context missing/null/empty/populated states remain distinct. Retry waits and attempts are finite and bounded, with at-least-once semantics documented.
- Optional insight artifacts are associated with an exact evaluation-record hash and status; stale artifacts are not silently reused. Report regeneration uses `report_manifest.json` and preserves capture provenance.
- Short document evidence is retained; DOCX blocks preserve paragraph/table order; invalid chunk bounds fail early. Candidate validation occurs during bounded refill. Topic caps, requested-topic total ceilings, rendered-evidence bounds, rejection reasons, shortfalls, and selected-excerpt boundary scope are persisted in `generation_diagnostics.json` and the manifest.

## Validation

- Intended environment: `conda run -n army python -m pytest -q` — **165 passed**.
- `python -m compileall -q src` — passed.
- `git diff --check` — passed.
- Both historical `hova/silver_questions_results_*.csv` files imported read-only: 336 rows each; 236 answer / 50 clarify / 50 abstain; 186 canonical / 100 natural / 50 ambiguous; 150 mapped parents; zero skipped rows. The older export retains 33 recognized infrastructure errors and the newer export has zero.

## Deliberate limitations and live follow-up

- A prompt remains behavioral guidance rather than a security boundary; ACLs, DLP/PII filtering, output validation, monitoring, and approval controls remain application responsibilities.
- `selected_excerpts` boundary questions establish absence only within the excerpts sent to that generation call. Corpus-wide absence verification is not performed.
- PDF extraction does not perform OCR. Empty extracted pages are warned about, but there is no comprehensive extraction-summary artifact yet.
- Retries have bounded attempts, request timeouts, and wait caps, but no single total wall-clock deadline across an operation.
- Offline tests establish contracts and deterministic behavior. Actual Gemini Flash obedience, judge calibration, and prompt improvement still require a held-out live evaluation in a fresh output directory before manual promotion.

The earlier diagnostic script and JSONL in `docs/review_probes.py` and `docs/review-probe-results.jsonl` preserve pre-fix reproductions; some probes now intentionally encounter corrected validation behavior and are not the release test suite.
