"""Tests of profile rules: candidate validation, code-side evaluation, cached extraction, rule personas."""
import datetime as dt

import pytest

from chatbot_eval.cache import CorpusAnalysisCache
from chatbot_eval.conditions import (
    ProfileRule, RuleBatch, RuleCandidate, RulePredicate, decisive_fields, evaluate_rule, extract_rules,
    validate_candidate,
)
from chatbot_eval.documents import Chunk
from chatbot_eval.personas import rule_personas, synthesize_personas
from chatbot_eval.profiles import ProfileSchema, derive

AS_OF = dt.date(2026, 10, 27)
TEXT = "חייל בגיל 21 ומעלה זכאי למענק. חייל בודד זכאי לתוספת."


def _schema(**override) -> ProfileSchema:
    data = {
        "fields": [
            {"key": "name", "label": "שם פרטי", "type": "text", "sensitivity": "identifier", "use": "identifier_only"},
            {"key": "kind", "label": "סוג שירות", "type": "enum", "values": ["חובה", "קבע", "מילואים"]},
            {"key": "lone", "label": "חייל בודד", "type": "bool"},
            {"key": "birth", "label": "תאריך לידה", "type": "date", "date_format": "%d/%m/%Y"},
            {"key": "release", "label": "תאריך שחרור", "type": "date", "use": "excluded"},
            {"key": "note", "label": "הערה", "type": "text"},
        ],
        "derived": [{"key": "age", "label": "גיל", "source": "birth", "derivation": "years_since"}],
    }
    data.update(override)
    return ProfileSchema.model_validate(data)


def _candidate(key="age", op="gte", values=("21",), quote="חייל בגיל 21 ומעלה זכאי למענק", **kw) -> RuleCandidate:
    return RuleCandidate(
        when=[RulePredicate(key=key, op=op, values=list(values))],
        consequence=kw.get("consequence", "זכאי למענק"), otherwise="", quote=quote,
    )


def _rule(*predicates: tuple[str, str, list[str]], exclusive: bool = True) -> ProfileRule:
    when = [RulePredicate(key=k, op=o, values=v) for k, o, v in predicates]
    return ProfileRule(
        id="R-1", source_id="a#1", file="a.md", location="d", when=when, consequence="c", quote="q", exclusive=exclusive,
    )


CHUNK = Chunk("a.md#chunk-1", "a.md", "document", TEXT)


def test_a_valid_candidate_becomes_a_rule_with_a_content_id():
    rule, reason, unlisted = validate_candidate(_candidate(), CHUNK, _schema(), "fp")
    again, _, _ = validate_candidate(_candidate(), CHUNK, _schema(), "fp")
    assert reason is None and not unlisted
    assert rule.id.startswith("R-") and rule.id == again.id
    assert decisive_fields(_schema(), rule) == ["birth"]  # a derived key stands for its source field


@pytest.mark.parametrize("candidate, reason", [
    (_candidate(key="nope"), "unknown_key"),
    (_candidate(key="name", op="eq", values=["דנה"]), "identifier_field"),
    (_candidate(key="release", op="gt", values=["2026-01-01"]), "field_not_condition"),
    (_candidate(key="note", op="eq", values=["x"]), "field_not_condition"),
    (_candidate(key="kind", op="gt", values=["חובה"]), "invalid_operator"),
    (_candidate(key="kind", op="eq", values=["אחר"]), "value_outside_domain"),
    (_candidate(key="lone", op="eq", values=["maybe"]), "invalid_value"),
    (_candidate(key="age", op="gte", values=["twenty"]), "invalid_value"),
    (_candidate(key="birth", op="lt", values=["01/01/2000"]), "invalid_value"),
    (_candidate(key="kind", op="eq", values=["חובה", "קבע"]), "invalid_value"),
    (RuleCandidate(when=[], consequence="c", quote="חייל בגיל 21"), "empty_predicates"),
    (_candidate(consequence="  "), "blank_consequence"),
    (_candidate(quote="משפט שאינו במסמך"), "quote_not_verbatim"),
])
def test_invalid_candidates_are_rejected_with_a_named_reason(candidate, reason):
    rule, got, _ = validate_candidate(candidate, CHUNK, _schema(), "fp")
    assert rule is None and got == reason


def test_an_open_enum_accepts_an_unlisted_value_and_reports_it():
    schema = _schema(fields=[{"key": "kind", "label": "סוג שירות", "type": "enum", "values": ["חובה"], "open_values": True}])
    chunk = Chunk("a.md#chunk-1", "a.md", "document", "חייל בקבע זכאי לתוספת.")
    rule, reason, unlisted = validate_candidate(
        _candidate(key="kind", op="eq", values=["קבע"], quote="חייל בקבע זכאי"), chunk, schema, "fp",
    )
    assert rule is not None and reason is None and unlisted


def test_every_condition_must_be_named_in_the_quote():
    # The pilot's prisoner rule quoted a sentence about "special situations" that never mentions imprisonment.
    text = "חייל בקבע בגיל 21 ומעלה זכאי. ישנם מצבים מיוחדים שבהם אין צורך לדווח."
    chunk = Chunk("a.md#chunk-1", "a.md", "document", text)
    schema = _schema()

    def check(candidate):
        return validate_candidate(candidate, chunk, schema, "fp")[1]

    assert check(_candidate(key="lone", op="eq", values=["true"], quote="ישנם מצבים מיוחדים שבהם אין צורך לדווח")) == "condition_not_in_quote"
    assert check(_candidate(key="lone", op="eq", values=["true"], quote="חייל בקבע בגיל 21 ומעלה זכאי")) == "condition_not_in_quote"
    assert check(_candidate(key="kind", op="eq", values=["חובה"], quote="חייל בקבע בגיל 21 ומעלה זכאי")) == "condition_not_in_quote"
    assert check(_candidate(key="kind", op="eq", values=["קבע"], quote="חייל בקבע בגיל 21 ומעלה זכאי")) is None  # prefix tolerated
    assert check(_candidate(key="age", op="gte", values=["21"], quote="חייל בקבע בגיל 21 ומעלה זכאי")) is None
    assert check(_candidate(key="age", op="gte", values=["30"], quote="חייל בקבע בגיל 21 ומעלה זכאי")) == "condition_not_in_quote"
    assert check(_candidate(key="age", op="gt", values=["20"], quote="חייל בקבע בגיל 21 ומעלה זכאי")) is None  # "gt 20" is "from 21"
    assert check(_candidate(key="age", op="gte", values=["99"], quote="חייל בקבע בגיל 21 ומעלה זכאי")) == "condition_not_in_quote"  # the label alone does not give the threshold
    year_chunk = Chunk("a.md#chunk-1", "a.md", "document", "מי שנולד אחרי 2004 זכאי.")
    dated = _candidate(key="birth", op="gt", values=["2004-01-01"], quote="מי שנולד אחרי 2004 זכאי")
    assert validate_candidate(dated, year_chunk, schema, "fp")[1] is None


def test_evaluation_is_exact_on_the_birthday_and_none_when_a_field_is_missing():
    rule = _rule(("age", "gte", ["21"]))
    on_birthday = {"birth": "27/10/2005"}  # turns 21 on AS_OF
    day_before = {"birth": "28/10/2005"}
    assert evaluate_rule(_schema(), rule, on_birthday, AS_OF) is True
    assert evaluate_rule(_schema(), rule, day_before, AS_OF) is False
    assert evaluate_rule(_schema(), rule, {"kind": "חובה"}, AS_OF) is None


def test_evaluation_combines_enum_bool_and_date_predicates():
    rule = _rule(("kind", "in", ["חובה", "קבע"]), ("lone", "eq", ["true"]), ("birth", "lt", ["2005-01-01"]))
    profile = {"kind": "קבע", "lone": True, "birth": "31/12/2004"}
    assert evaluate_rule(_schema(), rule, profile, AS_OF) is True
    assert evaluate_rule(_schema(), rule, {**profile, "lone": False}, AS_OF) is False
    assert evaluate_rule(_schema(), rule, {**profile, "kind": "מילואים"}, AS_OF) is False
    # One failing predicate decides even when another field is missing.
    assert evaluate_rule(_schema(), rule, {"kind": "מילואים"}, AS_OF) is False
    assert evaluate_rule(_schema(), rule, {"kind": "קבע"}, AS_OF) is None


class _FakeLLM:
    def __init__(self, candidates):
        self.candidates, self.calls = candidates, []

    def generate(self, prompt, schema, model):
        self.calls.append(prompt)
        return RuleBatch(rules=self.candidates)


def test_extraction_prefilters_caches_and_recounts_rejections(tmp_path):
    schema = _schema()
    chunks = [CHUNK, Chunk("b.md#chunk-1", "b.md", "document", "טקסט ללא קשר לפרופיל")]
    good, bad = _candidate(), _candidate(key="nope")
    llm = _FakeLLM([good, good, bad])

    result = extract_rules(chunks, schema, llm, "m", cache=CorpusAnalysisCache(tmp_path / "c"))
    assert len(llm.calls) == 1  # the second chunk names no profile field
    assert [rule.when[0].key for rule in result.rules] == ["age"]
    assert result.diagnostics["chunks_skipped_by_prefilter"] == 1
    assert result.diagnostics["rejected"] == {"unknown_key": 1}
    assert result.diagnostics["rules_merged_duplicates"] == 1
    assert result.diagnostics["llm_calls"] == 1

    again = extract_rules(chunks, schema, llm, "m", cache=CorpusAnalysisCache(tmp_path / "c"))
    assert len(llm.calls) == 1 and again.diagnostics["chunks_cache_hits"] == 1
    assert [rule.id for rule in again.rules] == [rule.id for rule in result.rules]

    # A different schema is a different prompt and key: the chunk is extracted again.
    changed = _schema(fields=_schema().model_dump()["fields"] + [
        {"key": "extra", "label": "שדה נוסף", "type": "bool"},
    ])
    extract_rules(chunks, changed, llm, "m", cache=CorpusAnalysisCache(tmp_path / "c"))
    assert len(llm.calls) == 2

    unfiltered = extract_rules(chunks, schema, llm, "m", prefilter=False)
    assert unfiltered.diagnostics["chunks_sent"] == 2


def test_a_failed_chunk_is_reported_and_not_cached(tmp_path):
    class Failing:
        def generate(self, prompt, schema, model):
            raise RuntimeError("boom")

    cache = CorpusAnalysisCache(tmp_path / "c")
    result = extract_rules([CHUNK], _schema(), Failing(), "m", cache=cache)
    assert result.rules == [] and result.diagnostics["chunks_failed"] == 1
    llm = _FakeLLM([_candidate()])
    extract_rules([CHUNK], _schema(), llm, "m", cache=CorpusAnalysisCache(tmp_path / "c"))
    assert len(llm.calls) == 1


def _personas_for(rule, schema=None, limit=8):
    schema = schema or _schema()
    base, _ = synthesize_personas(schema, AS_OF, seed=1)
    return rule_personas(schema, rule, AS_OF, base, seed=1, limit=limit)


def test_rule_personas_cover_satisfied_violated_missing_and_the_exact_birthday():
    schema, rule = _schema(), _rule(("age", "gte", ["21"]))
    personas, skipped = _personas_for(rule)
    assert skipped is None
    verdicts = {item.role: evaluate_rule(schema, rule, item.persona.profile, AS_OF) for item in personas}
    assert verdicts["satisfies"] is True and verdicts["violates"] is False and verdicts["missing"] is None
    assert verdicts["boundary_satisfies"] is True and verdicts["boundary_violates"] is False
    edge = next(item for item in personas if item.role == "boundary_satisfies").persona
    assert derive(schema, edge.profile, AS_OF)["age"] == 21
    assert edge.profile["birth"] in {"27/10/2005", "28/10/2004"}  # the first or the last day of being 21
    assert "name" in edge.profile and edge.profile["name"].startswith("בדיקה-")
    assert len({item.persona.id for item in personas}) == len(personas)


def test_rule_personas_are_deterministic_and_respect_the_limit():
    rule = _rule(("kind", "eq", ["קבע"]), ("lone", "eq", ["true"]))
    first, _ = _personas_for(rule, limit=3)
    second, _ = _personas_for(rule, limit=3)
    assert [p.persona.id for p in first] == [p.persona.id for p in second] and len(first) == 3
    assert [p.role for p in first] == ["satisfies", "violates", "missing"]


def test_an_unsatisfiable_rule_yields_no_personas():
    rule = _rule(("age", "gt", ["30"]), ("age", "lt", ["20"]))
    personas, skipped = _personas_for(rule)
    assert personas == [] and skipped == "rule_unsatisfiable"


def test_a_rule_that_is_not_exclusive_gets_no_personas_outside_its_conditions():
    schema, rule = _schema(), _rule(("age", "gte", ["21"]), ("lone", "eq", ["true"]), exclusive=False)
    personas, _ = _personas_for(rule)
    roles = {item.role for item in personas}
    assert roles == {"satisfies", "boundary_satisfies", "missing"}
    assert all(evaluate_rule(schema, rule, item.persona.profile, AS_OF) is not False for item in personas)


def _chunk(name: str, text: str) -> Chunk:
    return Chunk(f"{name}#chunk-1", name, "document", text)


def test_the_same_rule_found_in_two_chunks_is_kept_once_with_its_other_sources():
    reworded = _candidate(consequence="המענק חל על החייל זכאי למענק")
    again = _candidate(consequence="זכאי למענק")
    chunks = [_chunk("a.md", TEXT), _chunk("b.md", TEXT)]
    llm = _FakeLLM([again, reworded, _candidate(values=("30",), quote="חייל בגיל 21 ומעלה זכאי למענק. 30")])
    result = extract_rules(chunks, _schema(), llm, "m")
    ages = {rule.when[0].values[0]: rule for rule in result.rules}
    assert set(ages) == {"21"} and result.diagnostics["rules_merged_duplicates"] >= 2
    assert ages["21"].source_id == "a.md#chunk-1" and ages["21"].other_sources == ["b.md#chunk-1"]


def test_max_rules_takes_rules_round_robin_across_files_and_is_deterministic():
    schema = _schema()
    text = "חייל בגיל 21 ומעלה זכאי. חייל בקבע בגיל 30 זכאי. חייל בודד זכאי. חייל מילואים בגיל 40 זכאי."
    chunks = [_chunk(f"{name}.md", text) for name in "abc"]
    per_file = {
        "a.md": [("age", "gte", "21", "חייל בגיל 21 ומעלה"), ("lone", "eq", "true", "חייל בודד זכאי")],
        "b.md": [("age", "gte", "30", "בקבע בגיל 30"), ("kind", "eq", "קבע", "חייל בקבע")],
        "c.md": [("kind", "eq", "מילואים", "חייל מילואים"), ("age", "lt", "40", "בגיל 40")],
    }

    class ByFile:
        def generate(self, prompt, schema, model):
            name = next(name for name in per_file if f"[SOURCE_ID: {name}#chunk-1]" in prompt)
            return RuleBatch(rules=[
                RuleCandidate(
                    when=[RulePredicate(key=key, op=op, values=[value])], consequence=f"{name} {key} {value} תוצאה ייחודית",
                    quote=quote,
                )
                for key, op, value, quote in per_file[name]
            ])

    first = extract_rules(chunks, schema, ByFile(), "m", max_rules=3, seed=1)
    again = extract_rules(chunks, schema, ByFile(), "m", max_rules=3, seed=1)
    assert [rule.id for rule in first.rules] == [rule.id for rule in again.rules]
    assert sorted(rule.file for rule in first.rules) == ["a.md", "b.md", "c.md"]  # one from each file, not three from one
    assert first.diagnostics["rules_over_max_rules"] == 3
