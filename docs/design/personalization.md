# 22. Personalization (`profiles.py`, `personas.py`, `conditions.py`, `personal_questions.py`)

Status: current (verified 2026-11-09)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md).
The full plan, with research and the stages still to build, is
[plans/personalized-questions.md](../plans/personalized-questions.md) (D16, D17).

## 22.1 How the chatbot platform sees a profile

Read from the platform server's preProd code on 2026-10-27 (D17); recheck when the platform changes.

- A chatbot model has per-field toggles (personal: first and last name, gender, birth date,
  religion, family status; service: ID, service type, release date, end of mandatory service) and
  a user-data source. The server reads the signed-in user's record from that source, keeps the
  enabled fields, and renders them as one line, `label: value, label: value`, using its own Hebrew
  labels (for example `מין` and `סטטוס`, not the admin screen's `מגדר` and `מצב משפחתי`) and, for
  enum objects, their description.
- That line is appended to the system prompt with an instruction to tailor answers to it unless the
  question states other details, and it is also prefixed to the retrieval query.
- The three service toggles other than service type (ID, release date, end of mandatory service)
  are neither fetched nor labelled, so they never reach the prompt; the example schema marks them
  `excluded`.
- The dataset-evaluation endpoint (XLSX upload with question and expected-answer columns) calls the
  chatbot with no user line at all. Delivering a persona needs a platform change (section 22.5).

## 22.2 Profile schema (`profiles.py`)

The schema describes each field by the platform's key and printed label, so the field list can
grow without code changes. `load_profile_schema` reads TOML or JSON and rejects, with every problem
named at once:

- duplicate or malformed keys, empty labels, enums with fewer than two distinct values, `values` on
  non-enum fields, date formats that do not round-trip, `sample_values` that are not dates in the
  field's format or numbers, or that are given both on a derived field and on its source;
- an `id` field whose sensitivity is not `identifier`, and any identifier field used as a
  condition: names and ID numbers must never decide an answer;
- derived fields whose source is not a date field that the platform sends.

`condition_keys()` is the vocabulary a later rule-extraction stage may use: condition fields plus
derived fields. `schema_fingerprint` (SHA-256 of the canonical JSON) will key the caches of the
stages that read the schema (section 9).

`profile_value` reads a dotted key and turns a `{"code", "description"}` object into its
description, as the platform does. `profile_errors` checks a profile's present fields against their
types and domains; `missing_keys`, `unknown_keys` and `unlisted_value_keys` report fields absent
from the profile, keys the schema lacks, and open-enum values outside the list. Messages contain
field names only, so checking a real profile locally (the `profile-check` command) leaks nothing
into logs. `render_user_info` writes the platform's line for a profile (schema order, excluded and
absent fields skipped).

`derive` computes derived fields against a pinned `as_of` date: completed years or months
(`years_since`, `months_since`; age changes on the birthday) and signed day counts (`days_since`,
`days_until`). A question set will store its `as_of`, so expected answers do not drift with the
calendar.

## 22.3 Synthetic personas (`personas.py`)

`synthesize_personas` builds a covering array over the condition dimensions: enum and bool fields
with their whole domain, date and number fields with `sample_values`, and derived fields with
`sample_values` (the first such derived field per source date; its source date is set by
`source_date` so the derived value is exact, half a year or two weeks away from the boundary).
With strength 2 every pair of values of two dimensions appears in some persona; strength 1 only
requires every value. The search is greedy (AETG-style: each new persona starts from an uncovered
combination and fills the other dimensions with the values that cover most new combinations), uses
`random.Random(seed)` only, and stops at `max_personas`, reporting combinations left uncovered.
On the example schema, 540 combinations need 30 personas, the minimum (the two largest domains,
6 and 5 values, already need 30).

Persona IDs hash the condition values. Identifier fields get values nobody can mistake for real ones
(`בדיקה-…`, `SYNTH-…`); excluded fields are left out. Each persona is checked with `profile_errors`
and `derive` before it is returned, and stores its `user_info` line and the schema fingerprint.

Not modelled: constraints between fields (an 18-year-old widowed career soldier is a valid
persona). Boundary values taken from document rules come from `rule_personas` (section 22.4).

## 22.4 Applicability rules and personal questions (`conditions.py`, `personal_questions.py`)

**Rule extraction.** `extract_rules` shows each chunk the schema's condition fields (key, label,
type, listed values, derived fields) and asks for rules `{when: [predicate], consequence, otherwise,
quote}`; a predicate is `{key, op, values}` with ops `eq, ne, in, not_in, gt, gte, lt, lte`. One
call per chunk (`RuleBatch`), at most six rules per chunk. Everything the model returns is checked
in code (`validate_candidate`): the key must exist and be a condition (not an identifier, not
`excluded`), the operator must fit the type, values must lie in the field's domain (an `open_values`
enum may hold an unlisted value, which is kept and counted as `rules_with_unlisted_values`), the
consequence must be non-blank and the quote must occur verbatim in the chunk. Rejections are
counted by reason, never dropped silently. A rule's id is a hash of its predicates, consequence
and source, so re-runs keep ids. Duplicates inside a run are removed.

The **prefilter** skips a chunk before any call when it contains no condition field label, enum
value or derived label (`profile_terms`, normalized substring match). It is a cost cut with a known
miss (a chunk saying "בן 21" without "גיל"); `--no-prefilter` and `personalization.rule_prefilter`
turn it off. A failed call is counted (`chunks_failed`) and not cached.

**Evaluation in code.** `evaluate_rule(schema, rule, profile, as_of)` returns True, False or None
(a decisive field is missing). Derived fields are computed with `profiles.derive`. The model never
decides which side of a rule a persona is on.

**Rule personas.** `rule_personas` starts from a base persona and edits only the fields the rule
mentions: a persona that satisfies every predicate, one that violates a predicate, one with a
decisive field removed (derived predicates remove the source date), and for each numeric, date or
derived predicate the exact boundary: the first and last value that satisfy it and the value just
outside (for derived values, the first and last day found by scanning around `as_of`). Output is
deduplicated by persona id, ordered and capped by `personas_per_rule`; a rule no persona can satisfy
is skipped as `rule_unsatisfiable`.

**Questions.** One call per rule (`PersonalDraft`): the question, the answer if the rule applies,
the answer if not, claims for each, one general-rule claim and a clarification. A draft is rejected
when an answer or claims are missing or the question states the condition's value
(`question_restates_condition`, since the persona's profile should carry it). For each persona the
verdict picks the side: applies, does not apply, or `clarify` (form `ambiguous`, no claims,
the clarification as expected answer). Claims are the branch claims plus the general-rule claim, so
a judge can tell a correct conclusion from a lucky one. The quote and chunk are the question's
evidence. `personal_basic` questions need one call per distinct profile field for the question text;
the answer ("לפי הפרופיל שלך, label: value") is built from the persona and cites no document.

`SilverQuestion` gains `persona_id`, `profile_keys` (source fields the rule depends on) and
`user_info`; they join the stable-id identity only when `persona_id` is set, so existing ids do not
change. `review-export` adds a read-only `persona_readable` column.

Limits: one draft per rule, so a persona-specific conclusion is the branch text; fields are
independent in personas; as-of is not part of the rule cache key (rules do not depend on it) but
is part of the questions' content.

## 22.5 Why

- Pinning the vocabulary to the schema is what makes the field list dynamic and keeps model output
  checkable: a rule naming a key or value outside the schema can be rejected in code.
- Mirroring the platform's keys, labels and line format means a persona can be handed to the
  platform unchanged, and a question's expected answer is judged against what the model actually
  saw. Delivery is proposed as one optional per-row "user details" column in the dataset-evaluation
  upload, passed to the same function that builds the normal chat prompt (D17, provisional).
- Pairwise coverage follows combinatorial-testing practice: most failures involve few interacting
  parameters, so all pairs catch most of them at a small fraction of the full product.
- `open_values` exists because the value lists (family status, service type) are not yet confirmed
  against the platform's data; it lets a real profile pass while `profile-check` shows which lists
  need extending.
- Gender and religion are `special_category` but may be conditions (owner, D16); the plan's
  invariance checks test that answers do not change with them where no document conditions on them.
