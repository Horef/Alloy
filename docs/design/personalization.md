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
- derived fields whose source is not a date field that the platform sends;
- `aliases` (enum value to other words the documents use) that name a non-value, are empty, are
  themselves a value, or belong to two values, `aliases` on a non-enum field, and empty
  `label_aliases`;
- `[[constraints]]` (below) that name unknown fields or the wrong types.

**Aliases.** The documents rarely print the platform's label or value: they say "מוסלמית" for
`Islamic` or "גברים" for `זכר`. `aliases` and `label_aliases` list those words; the rule prompt shows
them (`also_called`), the prefilter counts them as field terms and the quote check accepts them. They
are compared by stem, so "חייל" and "חיילת" (one stem) cannot separate male from female: use words
that differ ("גבר", "נשים"). A gender-neutral quote ("חייל/חיילת") rightly fails the check. The
stem comparison is loose for short words ("אישה" would match "אישור"), so prefer aliases of four
letters or fewer only when they are unambiguous ("אשה"), and review accepted rules.

**Constraints.** `[[constraints]]` state relations personas must keep. `count_matches_entries`
(`count`, `entries`): a count of 0 means no entries, an entry means a count of at least 1.
`date_gap` (`earlier`, `later`, `min_years`): the later date is at least `min_years` after the
earlier (a child's birth date after the parent's). The mechanism is data-driven, so new relations are
new schema entries. `apply_constraints(schema, profile, pinned)` repairs a profile: it changes the
side that is not pinned (fields a rule's predicates fix), drops the dependent side when both are
pinned, and never invents an absent field. Gaps apply before counts.

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

Without constraints a persona may be absurd (a child older than the parent, three children and no
child dates). With the schema's `[[constraints]]` (section 22.2) every row of the covering array and
every rule persona passes through `apply_constraints`; personas that become equal are merged. A pair
the constraints forbid (age 22 with a 12-year-old child) is thus not generated, so `pair_coverage`
stays below 100% for it. Boundary values taken from document rules come from `rule_personas`
(section 22.4); `PERSONA_VERSION` 2 marks this behavior in the cache keys.

## 22.4 Applicability rules and personal questions (`conditions.py`, `personal_questions.py`)

**Rule extraction.** `extract_rules` shows each chunk the schema's condition fields (key, label,
type, listed values, derived fields) and asks for rules `{when: [predicate], consequence, otherwise,
exclusive, quote}`; a predicate is `{key, op, values}` with ops `eq, ne, in, not_in, gt, gte, lt, lte`. One
call per chunk (`RuleBatch`), at most six rules per chunk. Everything the model returns is checked
in code (`validate_candidate`): the key must exist and be a condition (not an identifier, not
`excluded`), the operator must fit the type, values must lie in the field's domain (an `open_values`
enum may hold an unlisted value, which is kept and counted as `rules_with_unlisted_values`), the
consequence must be non-blank, the quote must occur verbatim in the chunk and must name every
condition (`condition_not_in_quote`, `predicate_stated`). Rejections are counted by reason, never
dropped silently. A rule's id is a hash of its predicates, consequence and source, so re-runs keep
ids.

`exclusive` separates a *sufficient* condition ("a lone soldier is entitled to a supplement", which
says nothing about others) from an *exclusive* one ("only lone soldiers", or "otherwise X"). The
pilot showed the model inventing a "does not apply" answer for the first kind; now only exclusive
rules get violating personas and a not-side answer. The flag defaults to false, the safe side, but
round 1 showed the model then under-used it: the prompt now lists the wording that makes a rule
exclusive ("רק", "בלבד", a closed list of the entitled, an eligibility-criteria section, an
exemptions list, bands that split everyone), and `validate_candidate` treats a non-empty `otherwise`
as exclusive, since a text that says what applies to everyone else is exclusive by definition.

The quote check is a floor, not a proof, and is cheap on purpose: an enum predicate needs the field's
distinctive label words or a listed value, a boolean the label, a number or derived predicate the
number (for an ordered operator also its neighbour, since "מעל 24" is `gte 25`; a threshold of 0 or
1 may instead be stated by the label, as in "is a parent"), a date the year. Words are compared by
stem (Hebrew prefixes and plural endings ignored). Known false rejections: a synonym the schema does
not list (the text says "גברים", the value is `זכר`) and a table row naming the value only by a
section heading. Synonyms are now covered by schema `aliases` (section 22.2). On the 44 candidates cached from the pilot (made with the older prompt, which did
not ask for a quote naming every condition) it rejects 20 and keeps 23 plus one merged duplicate;
the new prompt should reject fewer.

**Duplicates and the cap.** A rule found in several chunks (same canonical `when`, equal or
similar consequence, threshold `SAME_CONSEQUENCE` 0.6) is kept once, the first in file order, with
the other chunks in `other_sources`; `exclusive` is true only if every copy says so
(`rules_merged_duplicates`). Similarity (`consequence_similarity`) is the larger of the plain token
Jaccard and the Jaccard of stemmed tokens (Hebrew prefix and up to two suffixes stripped), so
morphology variants ("מגיע"/"זכאי", "לחודש"/"בחודש") merge; a pair whose digit tokens differ (150,000
vs 180,000) scores 0 and stays apart, so different amounts are never merged. `max_rules` then picks rules round-robin by file, in a seeded order, so
a cap no longer spends itself on the first document. Rules are extracted before the cap, so the cap
saves question calls, not rule calls.

The **prefilter** skips a chunk before any call when it contains no condition field label, enum
value or derived label (`profile_terms`, normalized substring match). It is a cost cut with a known
miss (a chunk saying "בן 21" without "גיל"); `--no-prefilter` and `personalization.rule_prefilter`
turn it off. A failed call is counted (`chunks_failed`) and not cached.

**Evaluation in code.** `evaluate_rule(schema, rule, profile, as_of)` returns True, False or None
(a decisive field is missing). Derived fields are computed with `profiles.derive`. The model never
decides which side of a rule a persona is on.

**Rule personas.** `rule_personas` starts from a base persona and edits only the fields the rule
mentions: a persona that satisfies every predicate, one with a decisive field removed (derived
predicates remove the source date), for each numeric, date or derived predicate the exact boundary
(the first and last value that satisfy it; for derived values, the first and last day found by
scanning around `as_of`), and, for an exclusive rule only, personas that violate a predicate,
including the value just outside each boundary. A non-exclusive rule gets at most one persona with
a missing field (one clarification per rule is enough). Every candidate and final profile passes
through `apply_constraints` (predicate fields pinned), and a persona is kept only if the rule's
verdict on it matches its role (a "satisfies" persona must evaluate True, and so on), so constraint
repairs cannot flip a persona to the wrong side; a rule whose satisfying persona fails is skipped. Output is
deduplicated by persona id, ordered and capped by `personas_per_rule`; a rule no persona can satisfy
is skipped as `rule_unsatisfiable`.

**Questions.** One call per rule (`PersonalDraft`): the question, the answer if the rule applies,
for an exclusive rule the answer if not, claims for each, and one general-rule claim. A draft is
rejected when an answer or claims are missing or the question states the condition's value
(`question_restates_condition`, since the persona's profile should carry it). For each persona the
verdict picks the side: applies, does not apply (exclusive rules only; a non-exclusive rule's
outside persona is skipped and counted as `non_exclusive_rule_persona_outside`), or `clarify` (form
`ambiguous`, no claims). The clarification is built in code by `clarification_for` from the labels of
the decisive fields the persona lacks, so it asks for exactly those and never for a field the persona
holds. The draft also says whether the answer `depends_on_profile` (default true); when it is false
(a phone number, a procedure anyone gets) and the rule is non-exclusive, the missing-field persona is
skipped (`clarify_skipped_generic_consequence`), because asking the user for a field the answer does
not need is a wrong expectation. The flag only removes questions; exclusive rules are unaffected. Claims are the branch claims plus the general-rule claim, so
a judge can tell a correct conclusion from a lucky one. The quote and chunk are the question's
evidence. `personal_basic` questions need one call per distinct profile field for the question text;
the answer ("לפי הפרופיל שלך, label: value") is built from the persona and cites no document.

`SilverQuestion` gains `persona_id`, `profile_keys` (source fields the rule depends on) and
`user_info`; they join the stable-id identity only when `persona_id` is set, so existing ids do not
change. `review-export` adds a read-only `persona_readable` column.

Limits: one draft per rule, so a persona-specific conclusion is the branch text; personas keep only
the relations the schema's constraints state (anything else is independent); as-of is not part of the rule cache key (rules do not depend on it) but
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
