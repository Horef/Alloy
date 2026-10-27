# 22. Personalization (`profiles.py`, `personas.py`)

Status: current (verified 2026-10-27)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md).
The full plan, with research and the stages still to build, is
[plans/personalized-questions.md](../plans/personalized-questions.md) (D14, D15).

## 22.1 How the chatbot platform sees a profile

Read from the platform server's preProd code on 2026-10-27 (D15); recheck when the platform changes.

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
  chatbot with no user line at all. Delivering a persona needs a platform change (section 22.4).

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

Not modelled yet: constraints between fields (an 18-year-old widowed career soldier is a valid
persona) and boundary values taken from document rules; the rule-extraction stage will add those as
`sample_values` and constraints.

## 22.4 Why

- Pinning the vocabulary to the schema is what makes the field list dynamic and keeps model output
  checkable: a rule naming a key or value outside the schema can be rejected in code.
- Mirroring the platform's keys, labels and line format means a persona can be handed to the
  platform unchanged, and a question's expected answer is judged against what the model actually
  saw. Delivery is proposed as one optional per-row "user details" column in the dataset-evaluation
  upload, passed to the same function that builds the normal chat prompt (D15, provisional).
- Pairwise coverage follows combinatorial-testing practice: most failures involve few interacting
  parameters, so all pairs catch most of them at a small fraction of the full product.
- `open_values` exists because the value lists (family status, service type) are not yet confirmed
  against the platform's data; it lets a real profile pass while `profile-check` shows which lists
  need extending.
- Gender and religion are `special_category` but may be conditions (owner, D14); the plan's
  invariance checks test that answers do not change with them where no document conditions on them.
