"""Profile schema: the personal-data fields a chatbot sees about its user, described as data.

The schema lists each field the chatbot receives in its profile JSON (key, Hebrew label, group,
type, allowed values, sensitivity and role), plus derived fields such as age or months since
discharge that documents condition on. Adding a field is a schema edit, not a code change; later
stages (rule extraction, synthetic personas, personal questions) read only the schema.

Usage: `load_profile_schema(path)` reads TOML or JSON and raises `ValueError` listing every problem;
`profile_errors(schema, profile)` checks a profile dict without echoing its values (with
`missing_keys`, `unknown_keys` and `unlisted_value_keys` for the rest of a field-by-field check);
`derive(schema, profile, as_of)` computes the derived fields against a pinned date;
`render_user_info(schema, profile)` writes the "label: value, ..." line the chatbot platform puts in
its system prompt; `schema_fingerprint(schema)` is the cache identity of the schema.

Contracts: no LLM and no I/O beyond reading the schema file. Keys are the chatbot platform's own keys;
a dotted key (`service.type`) addresses a nested object. Identifier fields (names, ID numbers) can
never be conditions. Error messages name fields, never values, so a real profile checked locally
does not leak into logs.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

FieldType = Literal["enum", "date", "bool", "number", "text", "id"]
FieldUse = Literal["condition", "identifier_only", "excluded"]
Sensitivity = Literal["ordinary", "special_category", "identifier"]
Derivation = Literal["years_since", "months_since", "days_since", "days_until"]
_MISSING = object()


class ProfileField(BaseModel):
    key: str
    label: str
    group: str = ""
    type: FieldType
    values: list[str] = Field(default_factory=list, description="Allowed values of an enum field")
    # The chatbot may hold values outside `values` (an unconfirmed list); personas still use only
    # the listed ones, but a profile with another value is not an error.
    open_values: bool = False
    sensitivity: Sensitivity = "ordinary"
    use: FieldUse = "condition"
    date_format: str = "%Y-%m-%d"
    # Values synthetic personas may take for a date or number field (dates in `date_format`); enum
    # and bool fields use their whole domain. A field with neither is left out of personas.
    sample_values: list[str | int | float] = Field(default_factory=list)


class DerivedField(BaseModel):
    key: str
    label: str
    source: str = Field(description="Key of the date field this is computed from")
    derivation: Derivation
    # Values personas may take (e.g. ages); personas set the source date so the field has that value.
    sample_values: list[int] = Field(default_factory=list)


class ProfileSchema(BaseModel):
    version: int = 1
    fields: list[ProfileField]
    derived: list[DerivedField] = Field(default_factory=list)

    def field(self, key: str) -> ProfileField | None:
        return next((item for item in self.fields if item.key == key), None)

    def condition_keys(self) -> list[str]:
        """Keys a rule may condition on: condition fields and every derived field."""
        return [item.key for item in self.fields if item.use == "condition"] + [item.key for item in self.derived]


def schema_errors(schema: ProfileSchema) -> list[str]:
    errors: list[str] = []
    keys = [item.key for item in schema.fields] + [item.key for item in schema.derived]
    for key in sorted({key for key in keys if keys.count(key) > 1}):
        errors.append(f"{key}: key defined more than once")
    for item in schema.fields:
        name = item.key or "<empty key>"
        if not item.key.strip() or item.key != item.key.strip() or any(not part for part in item.key.split(".")):
            errors.append(f"{name}: key must be nonempty, without surrounding spaces or empty dotted parts")
        if not item.label.strip():
            errors.append(f"{name}: label is empty")
        if item.type == "enum":
            if len(set(item.values)) < 2 or any(not value.strip() for value in item.values):
                errors.append(f"{name}: an enum needs at least two distinct nonempty values")
            elif len(set(item.values)) != len(item.values):
                errors.append(f"{name}: duplicate enum values")
        elif item.values or item.open_values:
            errors.append(f"{name}: only enum fields take values/open_values")
        if item.type == "id" and item.sensitivity != "identifier":
            errors.append(f"{name}: an id field must have sensitivity 'identifier'")
        if item.sensitivity == "identifier" and item.use == "condition":
            errors.append(f"{name}: identifier fields cannot be conditions (use 'identifier_only' or 'excluded')")
        if item.type == "date":
            try:
                parsed = dt.datetime.strptime(dt.date(2001, 2, 3).strftime(item.date_format), item.date_format).date()
            except ValueError:
                parsed = None
            if parsed != dt.date(2001, 2, 3):
                errors.append(f"{name}: date_format does not round-trip a date")
        elif item.date_format != "%Y-%m-%d":
            errors.append(f"{name}: only date fields take date_format")
        if item.sample_values:
            if item.type == "date":
                if any(_parse_date(item, value) is None for value in item.sample_values):
                    errors.append(f"{name}: sample_values must be dates in date_format")
            elif item.type == "number":
                if any(isinstance(value, (str, bool)) for value in item.sample_values):
                    errors.append(f"{name}: sample_values must be numbers")
            else:
                errors.append(f"{name}: only date and number fields take sample_values")
    for item in schema.derived:
        source = schema.field(item.source)
        if not item.label.strip():
            errors.append(f"{item.key}: label is empty")
        if source is None or source.type != "date" or source.use == "excluded":
            errors.append(f"{item.key}: source {item.source!r} must be a date field that is not excluded")
        elif source.sample_values and item.sample_values:
            errors.append(f"{item.key}: give sample_values on the derived field or on {item.source!r}, not both")
        if item.derivation in {"years_since", "months_since"} and any(value < 0 for value in item.sample_values):
            errors.append(f"{item.key}: completed years/months sample_values cannot be negative")
    if not schema.fields:
        errors.append("the schema has no fields")
    return errors


def load_profile_schema(path: Path) -> ProfileSchema:
    """Read a TOML or JSON schema (`[[fields]]`, `[[derived]]`); raise ValueError naming every problem."""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if path.suffix.lower() == ".json" else tomllib.loads(text)
        schema = ProfileSchema.model_validate(data)
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"{path}: cannot parse: {exc}") from exc
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in exc.errors())
        raise ValueError(f"{path}: {problems}") from exc
    errors = schema_errors(schema)
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return schema


def schema_fingerprint(schema: ProfileSchema) -> str:
    """Stable identity of the schema, for the cache keys of stages that read it."""
    canonical = json.dumps(schema.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def profile_value(profile: dict, key: str, default: Any = None) -> Any:
    """Value at `key` (dotted keys address nested objects), or `default` when absent.

    A `{"code": ..., "description": ...}` object (how the platform's user databases return enums)
    yields its description, which is what the platform shows the model.
    """
    if key in profile:
        value = profile[key]
    else:
        value = profile
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
    if isinstance(value, dict) and "description" in value:
        return value["description"]
    return value


def _parse_date(item: ProfileField, value: Any) -> dt.date | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.datetime.strptime(value.strip(), item.date_format).date()
    except ValueError:
        return None


def profile_errors(schema: ProfileSchema, profile: dict) -> list[str]:
    """Type and domain problems of the fields present in `profile`; messages never include values."""
    errors = []
    for item in schema.fields:
        value = profile_value(profile, item.key, _MISSING)
        if value is _MISSING or value is None:
            continue
        if item.type == "enum":
            ok = isinstance(value, str) and (value in item.values or item.open_values)
        elif item.type == "date":
            ok = _parse_date(item, value) is not None
        elif item.type == "bool":
            ok = isinstance(value, bool)
        elif item.type == "number":
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:  # text, id
            ok = isinstance(value, (str, int)) and not isinstance(value, bool)
        if not ok:
            hint = {"enum": "not one of the schema's values", "date": f"not a date in format {item.date_format}"}
            errors.append(f"{item.key}: {hint.get(item.type, 'wrong type for ' + item.type)}")
    return errors


def render_user_info(schema: ProfileSchema, profile: dict) -> str | None:
    """The profile as the chatbot platform puts it in the prompt: "label: value, label: value".

    Mirrors the platform's user-info string (fields in schema order, excluded and absent fields
    skipped, booleans as JavaScript prints them); None when no field is present.
    """
    parts = []
    for item in schema.fields:
        value = profile_value(profile, item.key)
        if item.use == "excluded" or value is None:
            continue
        text = ("true" if value else "false") if isinstance(value, bool) else str(value)
        parts.append(f"{item.label}: {text}")
    return ", ".join(parts) or None


def unlisted_value_keys(schema: ProfileSchema, profile: dict) -> list[str]:
    """Open enum fields whose value is not in the listed values: the schema's list is incomplete."""
    return [
        item.key for item in schema.fields
        if item.type == "enum" and item.open_values
        and isinstance(value := profile_value(profile, item.key), str) and value not in item.values
    ]


def missing_keys(schema: ProfileSchema, profile: dict) -> list[str]:
    return [item.key for item in schema.fields if profile_value(profile, item.key) is None]


def unknown_keys(schema: ProfileSchema, profile: dict, prefix: str = "") -> list[str]:
    """Leaf keys of `profile` that no schema field names (the chatbot sends more than the schema knows)."""
    known = {item.key for item in schema.fields}
    out = []
    for key, value in profile.items():
        path = f"{prefix}{key}"
        if path in known:
            continue
        if isinstance(value, dict) and any(k.startswith(path + ".") for k in known):
            out += unknown_keys(schema, value, path + ".")
        else:
            out.append(path)
    return out


def _months_between(start: dt.date, end: dt.date) -> int:
    months = (end.year - start.year) * 12 + end.month - start.month
    return months - (end.day < start.day)


def derive(schema: ProfileSchema, profile: dict, as_of: dt.date) -> dict[str, int | None]:
    """Derived values at `as_of` (pinned per question set, so expected answers do not drift).

    `years_since`/`months_since` count completed years/months (age on a birthday is the new age);
    `days_since`/`days_until` are signed day counts. A missing or unparsable source gives None.
    """
    out: dict[str, int | None] = {}
    for item in schema.derived:
        source = schema.field(item.source)
        day = _parse_date(source, profile_value(profile, item.source)) if source else None
        if day is None:
            out[item.key] = None
        elif item.derivation == "years_since":
            out[item.key] = _months_between(day, as_of) // 12
        elif item.derivation == "months_since":
            out[item.key] = _months_between(day, as_of)
        elif item.derivation == "days_since":
            out[item.key] = (as_of - day).days
        else:
            out[item.key] = (day - as_of).days
    return out
