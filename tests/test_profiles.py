"""Tests of the profile schema: loading and validation, profile checks without values, derived fields."""
import datetime as dt
import json
from pathlib import Path

import pytest

from chatbot_eval import cli
from chatbot_eval.profiles import (
    ProfileSchema, derive, load_profile_schema, missing_keys, profile_errors, profile_value,
    schema_errors, schema_fingerprint, unknown_keys, unlisted_value_keys,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "profile_schema.example.toml"


def _schema(**extra) -> ProfileSchema:
    data = {
        "fields": [
            {"key": "service.type", "label": "סוג שירות", "type": "enum", "values": ["חובה", "קבע"]},
            {"key": "status", "label": "מצב משפחתי", "type": "enum", "values": ["רווק", "נשוי"], "open_values": True},
            {"key": "birth", "label": "תאריך לידה", "type": "date", "date_format": "%d/%m/%Y"},
            {"key": "discharge", "label": "תאריך שחרור", "type": "date"},
            {"key": "pn", "label": "מספר מזהה", "type": "id", "sensitivity": "identifier", "use": "identifier_only"},
        ],
        "derived": [
            {"key": "age", "label": "גיל", "source": "birth", "derivation": "years_since"},
            {"key": "since", "label": "חודשים", "source": "discharge", "derivation": "months_since"},
            {"key": "until", "label": "ימים", "source": "discharge", "derivation": "days_until"},
        ],
    }
    data.update(extra)
    return ProfileSchema.model_validate(data)


def test_example_schema_loads_and_keeps_identifiers_out_of_conditions():
    schema = load_profile_schema(EXAMPLE)
    identifiers = {item.key for item in schema.fields if item.sensitivity == "identifier"}
    assert identifiers == {"first_name", "last_name", "personal_number"}
    assert not identifiers & set(schema.condition_keys())
    assert {"gender", "religion", "marital_status", "service_type", "age"} <= set(schema.condition_keys())


@pytest.mark.parametrize("field, message", [
    ({"key": "pn", "label": "x", "type": "id"}, "sensitivity 'identifier'"),
    ({"key": "n", "label": "x", "type": "text", "sensitivity": "identifier"}, "cannot be conditions"),
    ({"key": "e", "label": "x", "type": "enum", "values": ["a"]}, "at least two"),
    ({"key": "t", "label": "x", "type": "text", "values": ["a", "b"]}, "only enum"),
    ({"key": "d", "label": "x", "type": "date", "date_format": "%Y"}, "round-trip"),
    ({"key": " k", "label": "x", "type": "bool"}, "surrounding spaces"),
    ({"key": "a..b", "label": "x", "type": "bool"}, "empty dotted parts"),
])
def test_schema_errors_name_each_problem(field, message):
    schema = ProfileSchema.model_validate({"fields": [field]})
    assert any(message in error for error in schema_errors(schema))


def test_duplicate_keys_and_bad_derived_sources_are_rejected(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps({
        "fields": [{"key": "a", "label": "x", "type": "bool"}, {"key": "a", "label": "y", "type": "bool"}],
        "derived": [{"key": "b", "label": "z", "source": "a", "derivation": "days_since"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError) as info:
        load_profile_schema(path)
    assert "defined more than once" in str(info.value) and "must be a date field" in str(info.value)


def test_unknown_type_is_a_load_error(tmp_path):
    path = tmp_path / "schema.toml"
    path.write_text('[[fields]]\nkey = "a"\nlabel = "x"\ntype = "color"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="fields.0.type"):
        load_profile_schema(path)


def test_profile_errors_never_include_values():
    schema = _schema()
    profile = {"service": {"type": "סודי-ערך"}, "birth": "2004-05-20", "discharge": "2026-12-01", "pn": True}
    errors = profile_errors(schema, profile)
    assert errors == [
        "service.type: not one of the schema's values",
        "birth: not a date in format %d/%m/%Y",
        "pn: wrong type for id",
    ]
    assert not any("סודי" in error or "2004" in error for error in errors)


def test_open_enum_accepts_unlisted_values_and_reports_the_field():
    schema = _schema()
    profile = {"status": "ידוע בציבור", "service": {"type": "קבע"}, "other": {"x": 1}}
    assert profile_errors(schema, profile) == []
    assert unlisted_value_keys(schema, profile) == ["status"]
    assert profile_value(profile, "service.type") == "קבע"
    assert unknown_keys(schema, profile) == ["other"]
    assert missing_keys(schema, profile) == ["birth", "discharge", "pn"]


def test_derived_fields_count_completed_units_at_the_pinned_date():
    schema = _schema()
    profile = {"birth": "21/05/2004", "discharge": "2026-12-01"}
    assert derive(schema, profile, dt.date(2026, 5, 20)) == {"age": 21, "since": -7, "until": 195}
    assert derive(schema, profile, dt.date(2026, 5, 21))["age"] == 22
    assert derive(schema, {"birth": "bad"}, dt.date(2026, 1, 1)) == {"age": None, "since": None, "until": None}


def test_fingerprint_changes_with_the_schema_only():
    assert schema_fingerprint(_schema()) == schema_fingerprint(_schema())
    changed = _schema()
    changed.fields[0].values.append("מילואים")
    assert schema_fingerprint(changed) != schema_fingerprint(_schema())


def test_profile_check_command_prints_field_names_and_fails_on_problems(tmp_path, capsys):
    profile = tmp_path / "synthetic.json"
    profile.write_text(json.dumps({"gender": "נקבה", "discharge_date": "01/12/2026", "marital_status": "רווקה"}), encoding="utf-8")
    assert cli.main(["profile-check", "--schema", str(EXAMPLE)]) == 0
    code = cli.main(["profile-check", "--schema", str(EXAMPLE), "--profile", str(profile), "--as-of", "2026-10-27"])
    out = capsys.readouterr().out
    assert code == 1
    assert "discharge_date: not a date" in out and "marital_status" in out
    assert "01/12/2026" not in out and "רווקה" not in out
