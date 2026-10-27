# 22. Personalization (`profiles.py`)

Status: current (verified 2026-10-27)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md).
The full plan, with research and the stages still to build, is
[plans/personalized-questions.md](../plans/personalized-questions.md) (D14).

## 22.1 What exists

Only the first stage: the **profile schema**. The chatbot receives its user's profile as JSON in its
system prompt; the schema describes that JSON as data, so the field list can grow without code
changes. `profiles.py` loads it from TOML or JSON (`load_profile_schema`) and rejects, with every
problem named at once:

- duplicate or malformed keys, empty labels, enums with fewer than two distinct values, `values` on
  non-enum fields, date formats that do not round-trip;
- an `id` field whose sensitivity is not `identifier`, and any identifier field used as a
  condition: names and ID numbers must never decide an answer;
- derived fields whose source is not a usable date field.

`condition_keys()` is the vocabulary a later rule-extraction stage may use: condition fields plus
derived fields. `schema_fingerprint` (SHA-256 of the canonical JSON) will key the caches of the
stages that read the schema (section 9).

`profile_errors` checks a profile's present fields against their types and domains; `missing_keys`,
`unknown_keys` and `unlisted_value_keys` report fields absent from the profile, keys the schema
lacks, and open-enum values outside the list. Messages contain field names only, so checking a real
profile locally (the `profile-check` command) leaks nothing into logs.

`derive` computes derived fields against a pinned `as_of` date: completed years or months
(`years_since`, `months_since`; age changes on the birthday) and signed day counts (`days_since`,
`days_until`). A question set will store its `as_of`, so expected answers do not drift with the
calendar.

## 22.2 Why

- Pinning the vocabulary to the schema is what makes the field list dynamic and keeps model output
  checkable: a rule naming a key or value outside the schema can be rejected in code.
- `open_values` exists because the value lists (marital status, service type) are not yet confirmed
  against the chatbot's data; it lets a real profile pass while `profile-check` shows which lists
  need extending.
- Gender and religion are `special_category` but may be conditions (owner, D14); the plan's
  invariance checks test that answers do not change with them where no document conditions on them.
