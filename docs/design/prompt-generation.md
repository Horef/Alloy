# 16. System-prompt generation (`prompt_generator.py`, `prompt_policy.py`)

Status: current (verified 2026-10-27)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md). Citations such as [R1] are listed in [references.md](references.md).

`generate-prompt` reuses the corpus graph to draft a Hebrew system prompt and a review package
(scope summary, assumptions to review, application guardrails, a manager checklist, test questions).

Key design decisions:

- **Separation of concerns.** Corpus-derived scope and terminology come from the model. Stable
  behavioral rules (grounding, clarification, abstention, privacy, injection resistance, partial
  answers) are a versioned Hebrew policy **assembled by code** around the model's text
  (`POLICY_VERSION`, `guided` or `compact` profile, `balanced` or `conservative` answer policy), so
  they cannot drift between generations.
- **A prompt is not a security control.** Authorization, retrieval ACLs, PII filtering, rate limits,
  and human approval for risky actions go into `application_guardrails` for engineers to implement
  outside the model [R22].
- **Evidence-driven revision.** With a previous run or current prompt, the model gets bounded prior
  evidence (round-robin across outcome types, failures first in each round with successes as
  counterexamples, and across topics within each type)
  and must return auditable `revision_mappings`: observed failure → evidence IDs → changed rule →
  expected behavior → non-prompt limitation, plus `regression_cases`. Retrieval gaps, missing
  documents, and infrastructure problems are routed to guardrails rather than papered over with
  wording.
- **Deterministic validation with one repair.** The code-assembled policy block must be present and
  unmodified, and the model-written domain guidance is checked on its own (not together with the
  Hebrew policy) for Hebrew content and the assistant's name. Lists must be distinct and non-empty
  with minimum counts, and evidence IDs must exist. One repair call is made; a package that still
  fails is rejected.
