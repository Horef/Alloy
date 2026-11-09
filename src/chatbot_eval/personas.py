"""Synthetic personas: deterministic, obviously fake profiles covering the schema's condition values.

`synthesize_personas(schema, as_of, seed=..., strength=2)` returns `Persona` records whose condition
fields form a covering array: with strength 2 every pair of values of two condition dimensions
appears in at least one persona (strength 1: every value at least once). Dimensions are enum and
bool fields (their whole domain), date and number fields with `sample_values`, and derived fields
with `sample_values`, for which the source date is set so the derived value is exactly that.
`write_personas` / `read_personas` store them as JSONL; `pair_coverage` measures a persona list.
`rule_personas(schema, rule, as_of, base_personas, seed=, limit=)` builds the few personas that
exercise one `conditions.ProfileRule`: one that satisfies it, one that violates it, one missing the
decisive field, and the exact date or number boundary on each side, with every date boundary found
by scanning the derivation itself rather than assuming calendar arithmetic.

Contracts: no LLM, no randomness beyond `random.Random(seed)`, so the same schema, date and seed give
the same personas (D16). Identifier fields get values that cannot be mistaken for real ones
("בדיקה-<id>", "SYNTH-<id>"); excluded fields are left out. Every persona passes
`profiles.profile_errors`. Constraints between fields (e.g. release after mandatory-service end)
are not modelled yet.
"""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, Field

from .conditions import ProfileRule, RulePredicate, evaluate_predicates
from .profiles import (
    ProfileField, ProfileSchema, derive, profile_errors, render_user_info, schema_fingerprint,
)

PERSONA_VERSION = 1


class Persona(BaseModel):
    id: str
    as_of: str
    profile: dict[str, Any]
    derived: dict[str, int | None] = Field(default_factory=dict)
    user_info: str = Field(default="", description="The profile line as the chatbot platform renders it")
    schema_fingerprint: str = ""


def dimensions(schema: ProfileSchema) -> list[tuple[str, list[Any]]]:
    """(key, values) per condition dimension, in schema order; a derived key stands for its source date."""
    by_source = {}
    for item in schema.derived:
        if item.sample_values and item.source not in by_source:
            by_source[item.source] = item
    dims: list[tuple[str, list[Any]]] = []
    for item in schema.fields:
        if item.use != "condition":
            continue
        if item.type == "enum":
            dims.append((item.key, list(item.values)))
        elif item.type == "bool":
            dims.append((item.key, [True, False]))
        elif item.type in {"date", "number"} and item.sample_values:
            dims.append((item.key, list(item.sample_values)))
        elif item.type == "date" and item.key in by_source:
            derived = by_source[item.key]
            dims.append((derived.key, list(derived.sample_values)))
    return dims


def _add_months(day: dt.date, months: int) -> dt.date:
    year, month = divmod(day.year * 12 + day.month - 1 + months, 12)
    return dt.date(year, month + 1, min(day.day, calendar.monthrange(year, month + 1)[1]))


def source_date(derivation: str, target: int, as_of: dt.date) -> dt.date:
    """A source date for which `derivation` at `as_of` is exactly `target`, away from the boundary."""
    if derivation == "years_since":
        return _add_months(as_of, -12 * target) - dt.timedelta(days=182)
    if derivation == "months_since":
        return _add_months(as_of, -target) - dt.timedelta(days=14)
    if derivation == "days_since":
        return as_of - dt.timedelta(days=target)
    return as_of + dt.timedelta(days=target)


def _covering_rows(dims: list[tuple[str, list[Any]]], strength: int, rng: random.Random, limit: int) -> tuple[list[tuple[int, ...]], int]:
    """Greedy covering array over value indices; returns (rows, tuples left uncovered at `limit`)."""
    if not dims:
        return [], 0
    sizes = [len(values) for _, values in dims]
    strength = max(1, min(strength, len(dims)))
    uncovered = {
        (combo, values)
        for combo in combinations(range(len(dims)), strength)
        for values in _product([range(sizes[i]) for i in combo])
    }
    rows: list[tuple[int, ...]] = []
    while uncovered and len(rows) < limit:
        best, best_gain = None, -1
        for _ in range(24):
            combo, values = rng.choice(sorted(uncovered))
            row: dict[int, int] = dict(zip(combo, values))
            order = [i for i in range(len(dims)) if i not in row]
            rng.shuffle(order)
            for dim in order:
                gains = [(_gain(row, dim, value, strength, uncovered), -value) for value in range(sizes[dim])]
                top = max(gains)[0]
                row[dim] = rng.choice([value for value, gain in enumerate(gains) if gain[0] == top])
            full = tuple(row[i] for i in range(len(dims)))
            gain = sum(1 for item in _tuples(full, strength) if item in uncovered)
            if gain > best_gain:
                best, best_gain = full, gain
        rows.append(best)
        uncovered -= set(_tuples(best, strength))
    return rows, len(uncovered)


def _product(ranges):
    out = [()]
    for values in ranges:
        out = [prefix + (value,) for prefix in out for value in values]
    return out


def _tuples(row: tuple[int, ...], strength: int):
    for combo in combinations(range(len(row)), strength):
        yield combo, tuple(row[i] for i in combo)


def _gain(row: dict[int, int], dim: int, value: int, strength: int, uncovered: set) -> int:
    fixed = sorted(row)
    gain = 0
    for others in combinations(fixed, strength - 1):
        combo = tuple(sorted(others + (dim,)))
        values = tuple(value if i == dim else row[i] for i in combo)
        gain += (combo, values) in uncovered
    return gain


def _finish_persona(
    schema: ProfileSchema, profile: dict[str, Any], as_of: dt.date, fingerprint: str,
    expected: dict[str, int] | None = None,
) -> Persona:
    """A `Persona` from a profile without identifier fields: its id hashes the profile, identifiers follow."""
    profile = dict(profile)
    digest = hashlib.sha256(json.dumps(profile, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
    persona_id = "P-" + digest.hexdigest()[:10]
    for item in schema.fields:
        if item.use == "identifier_only":
            profile[item.key] = f"SYNTH-{persona_id[2:]}" if item.type == "id" else f"בדיקה-{persona_id[2:6]}"
    derived_values = derive(schema, profile, as_of)
    wrong = [key for key, value in (expected or {}).items() if derived_values.get(key) != value]
    if problems := profile_errors(schema, profile) + wrong:  # a bug here, not bad input
        raise AssertionError(f"synthetic persona {persona_id} is invalid: {problems}")
    return Persona(
        id=persona_id, as_of=as_of.isoformat(), profile=profile, derived=derived_values,
        user_info=render_user_info(schema, profile) or "", schema_fingerprint=fingerprint,
    )


def synthesize_personas(
    schema: ProfileSchema, as_of: dt.date, *, seed: int = 0, strength: int = 2, max_personas: int = 200,
) -> tuple[list[Persona], dict]:
    """Personas covering every `strength`-way combination of condition values, plus diagnostics."""
    dims = dimensions(schema)
    rows, left = _covering_rows(dims, strength, random.Random(seed), max_personas)
    derived_by_key = {item.key: item for item in schema.derived}
    fingerprint = schema_fingerprint(schema)
    personas = []
    for row in rows:
        chosen = {key: values[index] for (key, values), index in zip(dims, row)}
        profile: dict[str, Any] = {}
        for key, value in chosen.items():
            if key in derived_by_key:
                derived = derived_by_key[key]
                source = schema.field(derived.source)
                profile[derived.source] = source_date(derived.derivation, value, as_of).strftime(source.date_format)
            else:
                profile[key] = value
        personas.append(_finish_persona(
            schema, profile, as_of, fingerprint, {key: v for key, v in chosen.items() if key in derived_by_key},
        ))
    diagnostics = {
        "persona_version": PERSONA_VERSION,
        "as_of": as_of.isoformat(),
        "seed": seed,
        "strength": max(1, min(strength, len(dims))) if dims else 0,
        "dimensions": {key: len(values) for key, values in dims},
        "personas": len(personas),
        "uncovered_combinations": left,
        "schema_fingerprint": fingerprint,
    }
    return personas, diagnostics


class RulePersona(NamedTuple):
    persona: Persona
    role: str  # satisfies, violates, missing, boundary_satisfies, boundary_violates
    variable: str = ""  # the profile field the role is about; empty for "satisfies"


_UNSET = object()  # an update value meaning "remove the field from the profile"


def _variables(schema: ProfileSchema, rule: ProfileRule) -> dict[str, list[RulePredicate]]:
    """Predicates grouped by the profile field they read (a derived key reads its source date), in rule order."""
    derived_by_key = {item.key: item for item in schema.derived}
    groups: dict[str, list[RulePredicate]] = {}
    for predicate in rule.when:
        key = predicate.key.strip()
        groups.setdefault(derived_by_key[key].source if key in derived_by_key else key, []).append(predicate)
    return groups


def _number(value: str) -> int | float:
    number = float(value)
    return int(number) if number == int(number) else number


def _candidate_updates(
    schema: ProfileSchema, variable: str, predicates: list[RulePredicate], as_of: dt.date,
) -> list[tuple[dict[str, Any], bool]]:
    """(profile update, is_edge) candidates for one variable: values around every threshold the rule uses.

    Edge candidates sit exactly on a date boundary (the first or last day a derived value holds, and
    the day just outside), where an off-by-one in the chatbot's own date arithmetic would show.
    """
    item = schema.field(variable)
    derived_by_key = {d.key: d for d in schema.derived}
    out: list[tuple[dict[str, Any], bool]] = []
    if item.type == "enum":
        values = list(item.values) + [v.strip() for p in predicates for v in p.values]
        out = [({variable: value}, False) for value in values]
    elif item.type == "bool":
        out = [({variable: True}, False), ({variable: False}, False)]
    elif item.type == "number":
        for predicate in predicates:
            for value in predicate.values:
                base = _number(value)
                out += [({variable: base + step}, False) for step in (0, -1, 1)]
    elif item.type == "date":
        for predicate in predicates:
            key = predicate.key.strip()
            if key in derived_by_key:
                out += _derived_candidates(schema, derived_by_key[key], item, _number(predicate.values[0]), as_of)
                continue
            for value in predicate.values:
                day = dt.date.fromisoformat(value.strip())
                out += [({variable: (day + dt.timedelta(days=step)).strftime(item.date_format)}, False) for step in (0, -1, 1)]
    seen: set[str] = set()
    unique = []
    for update, edge in out:
        marker = json.dumps(update, sort_keys=True, default=str)
        if marker not in seen:
            seen.add(marker)
            unique.append((update, edge))
    return sorted(unique, key=lambda entry: entry[1])


def _derived_candidates(
    schema: ProfileSchema, derived, source: ProfileField, target: int | float, as_of: dt.date,
) -> list[tuple[dict[str, Any], bool]]:
    out: list[tuple[dict[str, Any], bool]] = []
    for step in (0, -1, 1):
        day = source_date(derived.derivation, int(target) + step, as_of)
        out.append(({source.key: day.strftime(source.date_format)}, False))
    center = source_date(derived.derivation, int(target), as_of)
    window = [center + dt.timedelta(days=offset) for offset in range(-400, 401)]
    held = [
        day for day in window
        if derive(schema, {source.key: day.strftime(source.date_format)}, as_of)[derived.key] == int(target)
    ]
    if held:
        first, last = held[0], held[-1]
        for day in (first, last, first - dt.timedelta(days=1), last + dt.timedelta(days=1)):
            out.append(({source.key: day.strftime(source.date_format)}, True))
    return out


def _apply(profile: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(profile)
    for key, value in updates.items():
        if value is _UNSET:
            out.pop(key, None)
        else:
            out[key] = value
    return out


def rule_personas(
    schema: ProfileSchema, rule: ProfileRule, as_of: dt.date, base_personas: list[Persona], *,
    seed: int = 0, limit: int = 4,
) -> tuple[list[RulePersona], str | None]:
    """Personas that exercise `rule`, at most `limit`, in priority order, and a skip reason if none can.

    Order: one that satisfies every condition; one that violates the first condition; one missing the
    first condition's field (the chatbot should ask); the exact date or number boundary on each side
    for conditions that have one; then violations and missing fields of the remaining conditions.
    A rule that is not `exclusive` gets no violating personas (the evidence never says what holds for
    users outside the conditions), only satisfying, boundary-satisfying and missing-field ones.
    Other fields come from a base persona picked by seed and rule id, so the set varies between
    rules but not between runs. Which side of the rule a persona is on is classified here by the same
    evaluator the question stage uses, never assumed from how the persona was built.
    """
    groups = _variables(schema, rule)
    rng = random.Random(f"{seed}:{rule.id}")
    base = {
        key: value for key, value in (rng.choice(base_personas).profile if base_personas else {}).items()
        if (field := schema.field(key)) is None or field.use != "identifier_only"
    }
    sat: dict[str, dict[str, Any]] = {}
    viol: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, dict[str, Any]]] = {}
    for variable, predicates in groups.items():
        for update, edge in _candidate_updates(schema, variable, predicates, as_of):  # middle values come first
            verdict = evaluate_predicates(schema, predicates, _apply(base, update), as_of)
            if verdict is None:
                continue
            side = "sat" if verdict else "viol"
            (sat if verdict else viol).setdefault(variable, update)
            if edge:
                edges.setdefault(variable, {}).setdefault(side, update)
    if any(variable not in sat for variable in groups):
        return [], "rule_unsatisfiable"
    all_sat: dict[str, Any] = {}
    for variable in groups:
        all_sat.update(sat[variable])

    def missing(variable: str) -> dict[str, Any]:
        return {variable: _UNSET}

    specs: list[tuple[str, str, dict[str, Any]]] = [("satisfies", "", all_sat)]
    variables = list(groups)
    first, rest = variables[0], variables[1:]
    if first in viol and rule.exclusive:
        specs.append(("violates", first, {**all_sat, **viol[first]}))
    specs.append(("missing", first, {**all_sat, **missing(first)}))
    for variable in variables:
        if variable in edges:
            if "sat" in edges[variable]:
                specs.append(("boundary_satisfies", variable, {**all_sat, **edges[variable]["sat"]}))
            if "viol" in edges[variable] and rule.exclusive:
                specs.append(("boundary_violates", variable, {**all_sat, **edges[variable]["viol"]}))
    for variable in rest:
        if variable in viol and rule.exclusive:
            specs.append(("violates", variable, {**all_sat, **viol[variable]}))
        specs.append(("missing", variable, {**all_sat, **missing(variable)}))

    fingerprint = schema_fingerprint(schema)
    out: list[RulePersona] = []
    seen: set[str] = set()
    for role, variable, updates in specs:
        if len(out) >= limit:
            break
        persona = _finish_persona(schema, _apply(base, updates), as_of, fingerprint)
        if persona.id in seen:
            continue
        seen.add(persona.id)
        out.append(RulePersona(persona, role, variable))
    return out, None


def pair_coverage(schema: ProfileSchema, personas: list[Persona]) -> tuple[int, int]:
    """(covered, total) value pairs over the schema's condition dimensions."""
    dims = dimensions(schema)
    derived_keys = {item.key for item in schema.derived}

    def value(persona: Persona, key: str):
        return persona.derived.get(key) if key in derived_keys else persona.profile.get(key)

    total = covered = 0
    for (a, values_a), (b, values_b) in combinations(dims, 2):
        seen = {(value(p, a), value(p, b)) for p in personas}
        for pair in ((x, y) for x in values_a for y in values_b):
            total += 1
            covered += pair in seen
    return covered, total


def write_personas(personas: list[Persona], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(p.model_dump_json() + "\n" for p in personas), encoding="utf-8")
    return path


def read_personas(path: Path) -> list[Persona]:
    return [Persona.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
