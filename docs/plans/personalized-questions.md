Status: in progress (2026-10-27) - approved by the owner (D14); step 1 (profile schema) done

# Personalized questions: profile-aware generation and evaluation

## Objective

Generate and judge questions whose correct answer depends on the asking user's personal data (the
`personal_basic` and `personal_integration` types that exist in `models.py` but are rejected by
generation today), so a chatbot that reads the user's profile is evaluated on whether it applies the
documents to *that* user. The set of profile fields is data, not code: adding a field the chatbot can
see should take a schema edit, not a release.

Profile fields visible to the chatbot as of 2026-10-27 (owner's screenshots), in two groups:

| Group | Field (Hebrew label) | Likely type | Role for evaluation |
|---|---|---|---|
| Personal details | תאריך לידה | date | condition (via derived age); available (owner, 2026-10-27) |
| Personal details | שם פרטי | text | identifier only: never a condition |
| Personal details | מגדר | enum | condition where a document conditions on it (owner); otherwise an invariance check |
| Personal details | שם משפחה | text | identifier only |
| Personal details | דת | enum | special category; condition where a document conditions on it (owner); otherwise an invariance check |
| Personal details | מצב משפחתי | enum | condition |
| Service data | סיום שירות חובה | date | condition (derived: before/after, time since) |
| Service data | מספר מזהה | id | identifier only: never generated as a real-looking value, never in prompts |
| Service data | תאריך שחרור | date | condition (derived: time since or until discharge) |
| Service data | סוג שירות | enum | condition |

## Non-goals

- No real personal data enters Alloy, its caches, outputs or prompts. Personas are synthetic, made
  by deterministic code from the schema; the BRIEF non-goal ("no personal question types until an
  allowlisted, masked, auditable data interface exists") is met by the schema (allowlist), synthetic
  values (masking) and a personas file hashed into the manifest (audit). Changing that non-goal is the
  owner's decision and needs a D entry that supersedes it.
- Not a rules engine for production eligibility decisions: extracted rules are evaluation scaffolding
  and are reviewed like silver questions.
- No inference of sensitive attributes from documents or answers.

## Research basis (2026-10-27 search; claims labelled)

- PersonaBench (Salesforce, ACL Findings 2025) evaluates RAG on questions about synthetic users'
  private data and found plain RAG insufficient for personal questions - *established* for their
  setting; supports synthetic personas instead of real profiles.
  https://arxiv.org/abs/2502.20616
- LaMP-QA reports large gains from conditioning on the user's profile for personalized long-form
  answers - *established* for their benchmark; our questions are factual entitlement questions, so
  only the "condition on profile" idea transfers. https://www.researchgate.net/publication/392334639_LaMP-QA_A_Benchmark_for_Personalized_Long-form_Question_Answering
- ProfileFoundry generates consistent synthetic person records deterministically, with cross-field
  validation rules (age gates, linked fields) - *established*; matches "deterministic code, not an
  LLM, makes personas". https://arxiv.org/abs/2606.26403
- Georgetown Beeck Center "AI-powered Rules as Code" tested LLMs on benefits-eligibility scenarios
  that vary household, income and employment, and concluded complex policy logic needs human review
  - *established*; supports extracting rules as reviewable data.
  https://digitalgovernmenthub.org/publications/ai-powered-rules-as-code
- NIST combinatorial testing (ACTS): most faults involve interactions of few parameters, so t-way
  (pairwise) covering arrays with constraints keep persona counts small - *established* for software
  testing; *plausible* for chatbot evaluation. https://csrc.nist.gov/acts
- CAFFE (ICSE 2026): counterfactual metamorphic tests - change only a protected attribute and require
  an equivalent answer - *established*; gives the invariance checks for מגדר and דת.
  https://arxiv.org/abs/2512.16816
- No published benchmark was found for profile-conditioned questions over an HR/entitlements corpus
  - *unresolved*; this plan combines the pieces above.

## Design

Each stage keeps the existing contracts: model output is untrusted and validated in code, every
stage is cached with a key that includes the schema hash (`docs/design/caching.md`), and personal
logic stays out of `cli.py` wiring except for options.

1. **Profile schema** (`profiles.py`, new; no LLM). A TOML/JSON file supplied by the user,
   e.g. `profile_schema.toml`, one entry per field: `key`, `label` (Hebrew, as the chatbot shows
   it), `group`, `type` (`enum | date | bool | number | text | id`), `values` (enum domain),
   `sensitivity` (`identifier | special_category | ordinary`) and `use`
   (`condition | invariance_only | identifier_only | excluded`). Derived fields (age from birth date,
   months since discharge) are declared in the schema with a fixed formula from a small built-in
   set and computed against a pinned `as_of` date stored in the set, so expected answers do not drift
   with the calendar. Validation rejects unknown types, enum values outside the domain, and
   identifier fields marked as conditions.
2. **Applicability rules** (`conditions.py`, new; one LLM stage). Per chunk, extract rules of the
   form "IF predicates over schema keys THEN consequence", each with a verbatim quote. Code rejects a
   rule whose key is not in the schema, whose value is outside the field's domain, or whose quote is
   not in the chunk (the same normalizer as candidate validation). The schema keys are the only
   vocabulary the model may use, which is what makes the field list dynamic. Reviewable output:
   `profile_rules.jsonl`.
3. **Personas** (`personas.py`, new; deterministic, seeded). For each rule, values that satisfy and
   values that violate each predicate (equivalence classes), plus boundary dates on both sides of
   every threshold; remaining condition fields filled by a pairwise covering array that respects
   schema constraints (e.g. discharge after mandatory-service end). Identifier fields get obviously
   synthetic values (fixed placeholder names, an ID that fails the check digit). Output:
   `personas.jsonl`, hashed into the manifest.
4. **Questions** (generator stage). `personal_integration`: (rule, persona) pairs; the expected answer
   is the rule's consequence evaluated for that persona, the reference claims are the persona-specific
   conclusion plus the general rule, and the quotes are the rule's evidence (validated as today).
   `personal_basic`: questions about the user's own profile ("מתי אני משתחרר?"), answered from the
   persona. Each question is phrased in first person without restating the profile (the chatbot must
   look it up); optional variants that state the facts in the question separate reasoning errors from
   profile-lookup errors. A persona missing a decisive field gives `expected_behavior=clarify`.
   New fields on `SilverQuestion`: `persona_id` and `profile_keys` (the fields the answer depends on).
5. **Metamorphic pairs**. For a sample of personal questions, a sibling persona that differs only in a
   decisive field (the answer must change) and one that differs only in an `invariance_only` field
   such as דת or מגדר (the answer must not change). The report adds a profile-sensitivity and an
   invariance rate next to the claim metrics.
6. **Delivering the persona to the chatbot** (adapters). The chatbot receives the profile as JSON
   inside its system prompt (owner, 2026-10-27), so the persona must replace the logged-in user's
   profile in that JSON, in the same keys and format. Strategies, chosen in config:
   - `test_accounts`: the chatbot reads the profile of the authenticated user; evaluation logs in as
     pre-provisioned test users whose staging profiles are loaded from `personas.jsonl` (most
     realistic; needs the chatbot team).
   - `payload` (preferred if the endpoint allows it): `HttpChatbotAdapter` sends the persona JSON under
     a configured dotted `profile_field`, and the chatbot puts it in the system prompt instead of the
     logged-in user's profile.
   - `in_question`: the persona is written into the question text (tests only reasoning; fallback).
   `ChatbotAdapter.ask` gains an optional persona argument; the judge receives the persona too.

## Steps (each a tested commit; the first two need no model calls)

1. `profiles.py` + schema validation + `config.example.toml` keys + tests with a synthetic schema;
   D entry superseding the BRIEF non-goal (owner).
2. `personas.py`: partitioning, boundary dates, pairwise array; property tests (every rule has a
   satisfying and a violating persona; identifiers never look real).
3. `conditions.py` with a fake LLM in tests; cache key includes the schema hash.
4. Generator stage, `SilverQuestion` fields, review-export columns (`persona_readable`), README.
5. Adapter strategies and judge input; `evaluate` options; README tables (guarded).
6. Report metrics and `docs/design/evaluation.md`, `reporting.md`.

## Verification

Offline suite with fake LLMs for every stage; a live pilot on one topic of hova with 10-20 personas,
double-reviewed, before personal questions count in any headline metric.

## Risks

- The chatbot's real profile source may format values differently from the screenshots (enum
  spellings, date formats); the schema must be confirmed against the chatbot's data contract.
- Rules spread across documents (a condition in one, an exception in another) are hard to extract per
  chunk; start with single-chunk rules and record the rest as a limitation.
- Persona counts grow with conditions; pairwise coverage and a per-topic cap keep them bounded.

## Owner answers (2026-10-27)

- Profile delivery: the profile is passed as JSON as part of the chatbot's system prompt.
- Questions may depend on מגדר and דת.
- תאריך לידה is available (it was only switched off in the screenshot).
- Value lists, as the owner remembers them and to be confirmed against the chatbot's data:
  סוג שירות = חובה, קבע, מילואים; מצב משפחתי = רווק, נשוי, אלמן and perhaps more. A web search
  (2026-10-27) found the population registry's four statuses (רווק, נשוי, גרוש, אלמן) plus פרוד and
  ידוע בציבור, which forms use but the registry does not record
  (https://he.wikipedia.org/wiki/%D7%9E%D7%A6%D7%91_%D7%9E%D7%A9%D7%A4%D7%97%D7%AA%D7%99); for service,
  חובה, קבע and מילואים, with civilian IDF employees as a separate group
  (https://he.wikipedia.org/wiki/%D7%A9%D7%99%D7%A8%D7%95%D7%AA_%D7%91%D7%A6%D7%94%22%D7%9C). The
  example schema lists these with `open_values = true`, so unlisted values pass and `profile-check`
  names the field.
- The BRIEF non-goal on personal question types is superseded by this design (D14).

## Open questions (owner)

- [ ] Can the evaluation endpoint take a profile JSON per request (or switch to a test user), so
      the chatbot puts the persona in its system prompt instead of the logged-in user's profile?
      This decides step 5.
- [ ] A synthetic example of the profile JSON exactly as the chatbot inserts it (key names, nesting,
      date format, gendered values such as רווקה), to set the schema's keys and value lists.
