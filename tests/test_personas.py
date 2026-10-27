"""Tests of synthetic personas: pairwise coverage, determinism, derived targets, fake identifiers."""
import datetime as dt
import json
from itertools import combinations
from pathlib import Path

import pytest

from chatbot_eval import cli
from chatbot_eval.personas import (
    dimensions, pair_coverage, read_personas, source_date, synthesize_personas, write_personas,
)
from chatbot_eval.profiles import ProfileSchema, derive, load_profile_schema, profile_errors

ROOT = Path(__file__).resolve().parents[1]
AS_OF = dt.date(2026, 10, 27)


def _schema() -> ProfileSchema:
    return ProfileSchema.model_validate({
        "fields": [
            {"key": "name", "label": "שם פרטי", "type": "text", "sensitivity": "identifier", "use": "identifier_only"},
            {"key": "pn", "label": "מספר מזהה", "type": "id", "sensitivity": "identifier", "use": "identifier_only"},
            {"key": "kind", "label": "סוג שירות", "type": "enum", "values": ["חובה", "קבע", "מילואים"]},
            {"key": "status", "label": "סטטוס", "type": "enum", "values": ["רווק", "נשוי", "גרוש", "אלמן"]},
            {"key": "lone", "label": "חייל בודד", "type": "bool"},
            {"key": "birth", "label": "תאריך לידה", "type": "date", "date_format": "%d/%m/%Y"},
            {"key": "release", "label": "תאריך שחרור", "type": "date", "use": "excluded"},
            {"key": "note", "label": "הערה", "type": "text"},
        ],
        "derived": [{"key": "age", "label": "גיל", "source": "birth", "derivation": "years_since", "sample_values": [20, 21]}],
    })


def test_dimensions_cover_enums_bools_and_derived_samples_only():
    assert dimensions(_schema()) == [
        ("kind", ["חובה", "קבע", "מילואים"]), ("status", ["רווק", "נשוי", "גרוש", "אלמן"]),
        ("lone", [True, False]), ("age", [20, 21]),
    ]


def test_every_value_pair_is_covered_with_far_fewer_personas_than_the_product():
    schema = _schema()
    personas, diagnostics = synthesize_personas(schema, AS_OF)
    covered, total = pair_coverage(schema, personas)
    assert covered == total == 3 * 4 + 3 * 2 + 3 * 2 + 4 * 2 + 4 * 2 + 2 * 2
    assert diagnostics["uncovered_combinations"] == 0
    assert 12 <= len(personas) < 3 * 4 * 2 * 2


def test_strength_one_uses_every_value_once_at_least():
    schema = _schema()
    personas, _ = synthesize_personas(schema, AS_OF, strength=1)
    assert len(personas) == 4
    assert {p.profile["status"] for p in personas} == {"רווק", "נשוי", "גרוש", "אלמן"}


def test_same_inputs_give_the_same_personas_and_ids_follow_content():
    first, _ = synthesize_personas(_schema(), AS_OF, seed=3)
    again, _ = synthesize_personas(_schema(), AS_OF, seed=3)
    assert [p.model_dump() for p in first] == [p.model_dump() for p in again]
    assert len({p.id for p in first}) == len(first)


def test_personas_are_valid_synthetic_and_hit_their_derived_targets():
    schema = _schema()
    personas, _ = synthesize_personas(schema, AS_OF)
    for persona in personas:
        assert profile_errors(schema, persona.profile) == []
        assert persona.derived == derive(schema, persona.profile, AS_OF)
        assert persona.derived["age"] in {20, 21}
        assert persona.profile["name"].startswith("בדיקה-") and persona.profile["pn"].startswith("SYNTH-")
        assert "release" not in persona.profile and "note" not in persona.profile
        assert persona.user_info.startswith("שם פרטי: בדיקה-") and "תאריך שחרור" not in persona.user_info


@pytest.mark.parametrize("derivation, target", [
    ("years_since", 0), ("years_since", 21), ("months_since", 1), ("months_since", 30),
    ("days_since", 45), ("days_until", 10),
])
@pytest.mark.parametrize("as_of", [dt.date(2026, 3, 31), dt.date(2028, 2, 29), dt.date(2026, 1, 1)])
def test_source_dates_give_exactly_the_target(derivation, target, as_of):
    schema = ProfileSchema.model_validate({
        "fields": [{"key": "d", "label": "ת", "type": "date"}],
        "derived": [{"key": "x", "label": "x", "source": "d", "derivation": derivation}],
    })
    profile = {"d": source_date(derivation, target, as_of).isoformat()}
    assert derive(schema, profile, as_of)["x"] == target


def test_max_personas_reports_what_is_left_uncovered():
    personas, diagnostics = synthesize_personas(_schema(), AS_OF, max_personas=3)
    assert len(personas) == 3 and diagnostics["uncovered_combinations"] > 0


def test_round_trip_and_example_schema(tmp_path):
    schema = load_profile_schema(ROOT / "profile_schema.example.toml")
    personas, _ = synthesize_personas(schema, AS_OF)
    assert pair_coverage(schema, personas)[0] == pair_coverage(schema, personas)[1]
    path = write_personas(personas, tmp_path / "personas.jsonl")
    assert read_personas(path) == personas


def test_personas_command_writes_the_files(tmp_path, capsys):
    out = tmp_path / "personas"
    assert cli.main([
        "personas", "--schema", str(ROOT / "profile_schema.example.toml"), "--as-of", "2026-10-27", "--output", str(out),
    ]) == 0
    summary = json.loads((out / "personas_summary.json").read_text(encoding="utf-8"))
    assert summary["as_of"] == "2026-10-27" and summary["uncovered_combinations"] == 0
    assert summary["value_pairs_covered"]["covered"] == summary["value_pairs_covered"]["total"]
    assert len(read_personas(out / "personas.jsonl")) == summary["personas"]
    assert "value pairs covered" in capsys.readouterr().out


def test_pairs_are_checked_independently_of_the_helper():
    schema = _schema()
    personas, _ = synthesize_personas(schema, AS_OF, seed=7)
    dims = dimensions(schema)
    for (a, va), (b, vb) in combinations(dims, 2):
        get = lambda p, k: p.derived[k] if k == "age" else p.profile[k]
        assert {(x, y) for x in va for y in vb} <= {(get(p, a), get(p, b)) for p in personas}
