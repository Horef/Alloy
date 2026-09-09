# Evaluation-driven prompt improvement: review and implementation plan

Reviewed 2026-09-06 and implemented offline 2026-09-07. Scope: `src/chatbot_eval` and local `hova` artifacts. The implementation is complete at the contract/test level; it is not a claim of improved live-model performance.

## Conclusion

The revision workflow needs better evidence and an explicit behavioral contract, not just stronger wording. Its current output repeatedly paraphrases the same grounding, clarification, and brevity rules. However, the evaluation pipeline also loses clarification labels, and the saved runs contain no retrieved context. Repair those inputs before using their metrics to optimize a prompt.

This is larger than a prompt-template edit: it spans import fidelity, evidence selection, prompt assembly, configuration/cache identity, and behavioral validation. Implement the steps below in order, as separate reviewable changes. No deployment or paid model calls are required for the offline implementation.

## Findings from this checkout

### 1. The importer erases the behavior we want to improve

Both `hova/silver_questions_results_20260826_100425.csv` and `hova/silver_questions_results_20260831_154905.csv` contain **236 answer, 50 clarify, and 50 abstain** tasks. Each also contains 50 `ambiguous` question forms. `results_io.py:read_premade_results` ignores these columns and assigns `ANSWER` to every answerable row. It also defaults the question form to canonical.

Consequently both saved evaluations contain **286 answer, zero clarify, and 50 abstain** tasks. Their clarification success rate is null. For example, `ROW00141` asks which documents are needed for a ת״ש request; the reference asks the user to distinguish between two request types. The imported task is nevertheless `answer`, and the response is classified as hallucination rather than missing clarification.

This does not establish that the response was grounded: it establishes that the behavior label and resulting diagnostic are wrong. Do not infer clarification labels from question marks when explicit metadata already exists.

### 2. The before/after results show limited progress, with important qualifications

The current strict comparison function matches all 336 question definitions across the saved runs and marks their aggregates comparable. The saved figures are:

| Saved metric | evaluation | evaluation-upd |
|---|---:|---:|
| Factual answer success | 45.8% | 46.5% |
| Risky misinformation | 15.8% | 16.4% |
| Abstention success | 84% | 88% |
| Partial answers with excess information | 74 | 55 |
| Incorrect abstentions | 23 | 36 |
| Clarification tasks | 0 | 0 |

There are 29 transitions to success and 25 transitions from success. These are descriptive results, not a controlled estimate of prompt effect. Prompt deployment/model identity for each chatbot response is not established by the prompt-generator manifests.

The older run also contains **33 responses matching the current `placeholder_error` classifier**, yet reports zero infrastructure errors. Current import code already recognizes that placeholder. Re-importing the original results will change the baseline; do not compare its corrected metrics directly against the uncorrected report.

All **672 saved results have empty `retrieved_context`**. Neither CSV exports a runtime retrieved-context column. Gold `source_excerpts` are not a replacement for the context actually seen by the chatbot. Missing telemetry does not prove retrieval returned nothing, and unsupported-by-reference detail is not automatically proof that the model invented it.

The latest `oren-prompt-upd-2` package was generated from `evaluation-upd` on September 2. No subsequent evaluation of that latest prompt was found in `hova/outputs`.

### 3. The revision generator cannot inspect the interaction

`prompt_generator.py:_evaluation_history_context` includes outcome, topic, behavior labels, scores and abbreviated judge explanations, but omits the actual question, expected answer, chatbot answer, and retrieved context. The generator must infer the failure from another model's summary.

Its selection groups all non-success outcomes together and then sorts by question ID. Earlier ordinary failures can crowd out later hallucination, abstention, and clarification cases. Adding richer records without changing selection would make this worse.

The evaluation character limit is approximate: the aggregate, first record, and JSON envelope can exceed it. Evidence IDs accepted by validation come from all supplied records/insights, including records omitted from the bounded request. Thus an accepted ID does not guarantee the generator actually saw that record.

### 4. The revised prompts lack operational examples and stopping rules

The two updated prompts contain similar eight-part policies. They say to ask one clarification question, but lack a compact decision procedure and demonstrations of when to stop without answering. They also lack examples showing when to answer directly rather than over-clarify.

The latest abstention wording is about a *topic* missing from context. A topic can be present while the requested fact or scope is absent. `ROW00301`, for example, asks about a specific operational setting, while the answer expands general guidance into that setting. The reusable rule is to check the exact claim and applicability, not mere topic similarity.

`validate_prompt_package` checks Hebrew text, keywords, list lengths, and ID membership. It does not test whether a generated prompt changes chatbot behavior. The single repair attempt fixes these structural checks, not observed behavioral failures. Insights also hypothesize absent clarification instructions even though the reviewed prompts already contain them; insights must remain hypotheses.

## Proposed behavior and configuration

Use two independent settings, rather than a single ambiguous “strictness” slider:

| Setting | Values and default | Meaning |
|---|---|---|
| `generation.prompt_instruction_profile` | `compact`, `guided` (default) | How explicitly the procedure is spelled out. Guided includes short worked examples and an output check. |
| `generation.prompt_answer_policy` | `balanced` (default), `conservative` | Whether to offer clearly separated supported partial information when part of a request lacks evidence. |

Both profiles must forbid unsupported claims and guessing user attributes. Compact does not permit weaker grounding. Conservative does not mean abstaining whenever wording is informal or the answer uses a supported paraphrase. Neither setting guarantees obedience across models. Use `guided + balanced` as the first candidate for the user's Flash deployment; test `guided + conservative` separately for the abstention/usefulness tradeoff.

The generation model is already configurable. Do not hardcode anticipated model versions or identify target-model capability from its name.

## Implementation steps

### Step 1 — Preserve evaluation metadata

Files: `results_io.py`, `cli.py`, `tests/test_results_io.py`, CLI tests, README.

1. Add optional aliases and `ResultColumns` fields for `expected_behavior`, `question_form`, and `parent_question_id`; add corresponding `evaluate-file --*-column` overrides and wire them into the reader.
2. Parse present values using the existing enums. Preserve the current defaults when values/columns are absent. If answerability is absent but explicit behavior is present, derive it as false for abstain and true otherwise. Reject contradictory explicit answerability/behavior values and invalid enum values using existing row-error/strict-mode semantics.
3. Preserve parent IDs as supplied. Keep legacy `ROW...` IDs in this change; changing identity is a separate migration. Retain the original external ID mapping for diagnosis and future dataset joins.
4. Add tests covering all three behaviors, natural/ambiguous forms, absent columns, explicit overrides, invalid values, conflicting values, and strict mode. Test that an imported clarification reaches the evaluator's clarification branch.
5. Add a small synthetic CSV fixture, not a copy of the local operational dataset.

Acceptance: the two original hova CSVs import with 236/50/50 behavior counts and 50 ambiguous forms using their explicit chatbot-answer column. The old CSV yields 33 recognized placeholders. Do not overwrite old reports. Do not reuse stale judge checkpoints for corrected question definitions.

### Step 2 — Make revision evidence concrete and genuinely bounded

Files: `prompt_generator.py`, `tests/test_prompt_generator.py`.

1. Extract a compact evidence builder that includes question ID, topic, expected behavior/form, question text, expected response, actual response, outcome, relevant score counts, and judge explanations. Include runtime retrieved context separately from gold reference evidence.
2. Mark empty runtime context as `not_available_in_record`, not `retrieval_failed`. For nonempty excerpts, include original length and truncation status; do not treat an omitted portion as evidence of absence. Carry the same distinction into the generator instructions.
3. Exclude infrastructure/judge errors from behavioral examples but retain their aggregate counts. Recognize known historical placeholders during this read-only evidence preparation without mutating old records.
4. Select deterministically by failure category and topic: first offer one example from each of missing clarification, hallucination, should-have-abstained, incorrect abstention, and excessive answers; reserve room for successful answer/clarify/abstain counterexamples; then fill remaining capacity round-robin across category/topic buckets. Use ID only as a tie-breaker. Do not guarantee every category fits a tiny budget.
5. Start with per-field caps of 700 question characters, 900 expected-answer characters, 1200 actual-answer characters, and 1500 runtime-context characters. These are engineering defaults, not empirically optimal values. Use visible truncation markers and original lengths. Do not embed full gold document collections per record.
6. Serialize whole valid JSON items. Count the entire serialized envelope against `prompt_max_evaluation_chars`. Compact/drop topic breakdowns first if necessary, then reduce optional detail; skip an oversized record and continue to smaller candidates. If the minimal required envelope cannot fit, raise an explicit budget error. Never slice serialized JSON.
7. Return included IDs and omission statistics alongside the text. Validate revision references only against IDs actually transmitted in structured records or included structured insight issues. Budget insights by whole issues, avoiding mid-JSON truncation; raw current-prompt text can retain marked head/tail truncation.
8. Include high-level warnings for inconsistent task/reference evidence; do not automatically relabel historical records or obey instructions embedded in their text. Never copy personal values from examples into the generated prompt.

Tests: hard length limit including non-ASCII JSON; long first record; deterministic selection; category/topic diversity; retained success cases; no-context caveat; placeholder exclusion; omitted IDs rejected; malicious record content remains quoted data. Fix `_bounded_context` for tiny limits so it cannot return a full tail through Python's `[-0:]` behavior.

### Step 3 — Compile an explicit response procedure

Files: `prompt_generator.py`, `models.py`, preferably a new `prompt_policy.py`, prompt tests.

Use a deterministic, versioned Hebrew core so the generator cannot “improve” away the central decision rules. Keep generated domain scope, terminology, and reviewed examples separate. Assemble the final `system_prompt_hebrew` once; never repeatedly prepend the core to a previous assembled prompt. Existing packages must remain readable with defaulted new fields.

The core should instruct the chatbot to:

1. Resolve intent using the current message and facts already supplied in the conversation. Do not ask again for known details.
2. If a missing user detail materially changes the applicable answer, ask one short question identifying that detail, then stop. Do not append a guessed answer, a list of procedures, or all possible branches.
3. Do not clarify solely because a question is short. A general definition or explicit overview can be answered generally. Do not ask the user for missing institutional facts they cannot supply.
4. Check whether the exact requested facts and their scope are supported by runtime evidence. Topic similarity is insufficient. Do not extend a rule to a different population, location, date, or process without support.
5. Answer when supported, preserving relevant conditions and exceptions. Omit unrelated conditions; “include all conditions” must not become “copy the whole document.”
6. If unsupported, state the specific information gap. Balanced mode may provide a useful supported part while clearly separating the unanswered part. Conservative mode abstains from the requested conclusion when its support is incomplete; it need not refuse independent, fully supported subquestions.
7. Handle unresolved document conflict explicitly; do not invent precedence or ask the user to resolve an institutional contradiction they cannot resolve.
8. Before sending, check for unsupported numbers, contacts, eligibility claims, invented user attributes, and an answer appended to a clarification. Remove unsupported content rather than exposing reasoning.

Keep the existing privacy, injection, role, and tone controls. Runtime context is evidence, not a source of new instructions. Never promise “immunity.”

For guided mode, include 3–5 short positive demonstrations: ambiguous personal request → one follow-up only; sufficiently specified request → direct answer; general overview → no unnecessary clarification; related context without the requested fact → precise abstention; supported partial answer under the selected policy. Each demonstration needs its own clearly isolated toy context so it does not become a source of real domain facts. Prefer neutral synthetic examples for the fixed core; domain-specific generated examples must be reviewable and supported by the supplied evidence. Do not copy benchmark questions or operational amounts into reusable policy.

Suggested Hebrew rule shape (illustrative, not a complete deployment prompt):

> לפני מענה, בדוק אם חסר פרט של המשתמש שמשנה את התשובה. אם כן, שאל שאלה ממוקדת אחת ועצור; אל תוסיף מענה משוער. אם אין פרט כזה, בדוק שהמידע המאוחזר תומך בעובדה המסוימת שנשאלה ובתחולתה. דמיון בנושא אינו מספיק. אל תבקש הבהרה רק משום שהשאלה קצרה, ואל תשאל שוב על פרט שכבר נמסר.

Google's [prompt design guidance](https://ai.google.dev/gemini-api/docs/prompting-strategies) supports clear instructions and examples to demonstrate response patterns. The specific procedure and profiles above are proposals for this application, not a Google guarantee or measured improvement.

### Step 4 — Wire profiles, traceability, and cache identity

Files: `config.py`, `config.example.toml`, `cli.py`, `cache.py`, `artifacts.py`, `models.py`, README and associated tests.

1. Add validated settings from the table and CLI overrides `--instruction-profile` / `--answer-policy`. CLI overrides config; old configs receive the defaults.
2. Include effective profiles and the policy/template fingerprint in cache keys and manifests. If policy lives outside `prompt_generator.py`, extend its fingerprint to include that file. Test that profile/template changes invalidate the cache.
3. Add defaulted review fields for effective profiles, evidence coverage, and a structured revision mapping: observed failure, included IDs, changed rule, expected observable behavior, and non-prompt limitation. Keep legacy revision-summary fields for compatibility.
4. Suggested regression cases should include question, miniature context, expected behavior, and prohibited content, not just a list of questions. Preserve the legacy list for old consumers.
5. Validate structure, included IDs, profile consistency and example completeness. Exact core preservation is deterministic; domain-example correctness and actual model obedience still need review/testing.
6. Preserve one initial package-generation call plus at most one structural-repair call. Topic discovery remains an earlier stage. Model transport retries remain separate. There is no automatic behavioral optimization loop, no hidden extra summarization call, and no call for omitted evidence. Cache hits make neither package call.

Acceptance: help/defaults/validation pass; old packages load; selected profile survives repair and artifact writing; cache separation works; serialized prompts contain the core exactly once.

### Step 5 — Correct the feedback loop and measure on the deployed model

Files: `insights.py`, relevant tests, README; runtime integration belongs to the chatbot owner.

1. Include expected behavior/form and runtime-context availability in insight records. For absent telemetry, prohibit claims that retrieval failed or that prompt absence is proven. Ask for stage-specific hypotheses and evidence limitations.
2. Re-evaluate both original CSVs using corrected labels and the current placeholder classifier into new output directories, with fresh compatible checkpoints. This requires live judging and should be a separate explicitly scheduled run. Never silently replace historical outputs.
3. Record the actual chatbot prompt hash, target model/version, generation settings, retrieval configuration, and conversation/reset conditions for future experiments. A prompt-generation manifest identifies the authoring model, not the deployed chatbot model.
4. For causal prompt testing, replay baseline and candidate against the same captured runtime contexts and inputs. If only the full RAG product can be tested, report that retrieval variability remains a confounder. Verify that the prompt reaches the intended instruction field and that the runtime allows clarification output.
5. Split development and held-out cases by parent-question family before tuning, keeping natural/ambiguous siblings together. Do not use held-out failures to generate the same candidate being evaluated. If earlier prompts saw all existing cases, create a fresh held-out set.
6. Include ambiguous and unambiguous near-neighbors, paraphrases, short general questions, missing exact facts, partial support, conflicting context, unsupported scope extrapolation, and multi-turn follow-ups where the user supplies the missing detail.
7. Report clarification recall **and unnecessary clarification on answer tasks**, misinformation, factual usefulness, abstention precision/recall, infrastructure failures, latency and token cost. Repeat target-model runs when sampling variability matters. Keep raw counts and denominators visible.
8. A candidate is only promoted after held-out testing shows improved clarification and lower misinformation without unacceptable answer/usefulness regression. Set tolerances before testing; do not choose them after seeing the result. Keep deployment manual.

If guided prompts still fail reliably, the next architectural option is a separate answer/clarify/abstain routing step with a structured output that the application enforces, followed by answer generation only for the answer branch. That is a separate runtime project with added calls, latency, and its own errors. It is not required for the first implementation.

## Validation and handoff

Run focused tests for each step, then `conda run -n army python -m pytest -q` and `git diff --check` once integration is complete. Use fake structured LLMs for offline generation/repair tests. Do not claim these tests demonstrate Flash compliance.

Suggested implementer instruction:

> Implement Steps 1–4 and the offline insight changes in Step 5 from this plan, in order. Preserve public artifact compatibility and historical hova files. Start with importer tests that reproduce the lost clarification metadata. Add bounded structured evidence and profiles with deterministic core assembly, including cache/manifests and documentation. Run the army test suite. Do not make live model calls or deploy. Report implementation results separately from the outstanding target-model experiment.

The highest priority is faithful evaluation metadata, followed by richer evidence and a guided decision procedure. Increasing the severity of the wording alone would leave both the evaluation defects and the lack of behavioral validation unresolved.
