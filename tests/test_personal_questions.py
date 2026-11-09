"""Tests of personal question building: verdict sides decided in code, clarify, rejections, round trips."""
import csv
import datetime as dt

from chatbot_eval.conditions import ProfileRule, RulePredicate, evaluate_rule
from chatbot_eval.documents import Chunk
from chatbot_eval.io import read_questions, write_questions
from chatbot_eval.models import ExpectedBehavior, QuestionForm, QuestionType
from chatbot_eval.personal_questions import BasicDraft, PersonalDraft, build_personal_questions
from chatbot_eval.personas import synthesize_personas
from chatbot_eval.profiles import ProfileSchema
from chatbot_eval.review import write_review_file
from chatbot_eval.validation import validate_question_set

AS_OF = dt.date(2026, 10, 27)
CHUNK = Chunk("a.md#chunk-1", "a.md", "document", "חייל בגיל 21 ומעלה זכאי למענק.")
SCHEMA = ProfileSchema.model_validate({
    "fields": [
        {"key": "name", "label": "שם פרטי", "type": "text", "sensitivity": "identifier", "use": "identifier_only"},
        {"key": "kind", "label": "סוג שירות", "type": "enum", "values": ["חובה", "קבע"]},
        {"key": "birth", "label": "תאריך לידה", "type": "date", "date_format": "%d/%m/%Y"},
    ],
    "derived": [{"key": "age", "label": "גיל", "source": "birth", "derivation": "years_since"}],
})
RULE = ProfileRule(
    id="R-1", source_id=CHUNK.id, file="a.md", location="document", consequence="זכאי למענק",
    quote="חייל בגיל 21 ומעלה זכאי למענק", when=[RulePredicate(key="age", op="gte", values=["21"])],
)
DRAFT = PersonalDraft(
    question="האם מגיע לי מענק?", answer_if_applies="כן, מגיע לך מענק.", claims_if_applies=["מגיע מענק"],
    answer_if_not="לא, המענק אינו חל עליך.", claims_if_not=["המענק אינו חל"],
    general_rule_claim="מענק מגיע לחייל בגיל 21 ומעלה", clarification="מה תאריך הלידה שלך?",
)


class FakeLLM:
    def __init__(self, draft=DRAFT, basic="מתי נולדתי?"):
        self.draft, self.basic, self.calls = draft, basic, []

    def generate(self, prompt, schema, model):
        self.calls.append(schema.__name__)
        return self.draft if schema is PersonalDraft else BasicDraft(question=self.basic)


def _build(llm, **kwargs):
    base, _ = synthesize_personas(SCHEMA, AS_OF, seed=1)
    return build_personal_questions([RULE], [CHUNK], SCHEMA, llm, "m", AS_OF, base, seed=1, **kwargs)


def test_each_persona_gets_the_side_its_profile_decides_and_a_missing_field_gets_clarify():
    llm = FakeLLM()
    result = _build(llm, personas_per_rule=8)
    assert llm.calls == ["PersonalDraft"]  # one call per rule, however many personas
    personas = {persona.id: persona for persona in result.personas}
    for question in result.questions:
        assert question.question_type == QuestionType.PERSONAL_INTEGRATION
        persona = personas[question.persona_id]
        verdict = evaluate_rule(SCHEMA, RULE, persona.profile, AS_OF)
        assert question.profile_keys == ["birth"] and question.user_info == persona.user_info
        assert question.supporting_quotes[0].quote == RULE.quote and question.sources[0].source_id == CHUNK.id
        if verdict is None:
            assert question.expected_behavior == ExpectedBehavior.CLARIFY
            assert question.question_form == QuestionForm.AMBIGUOUS and question.reference_claims == []
            assert question.expected_answer == DRAFT.clarification and "תאריך לידה" not in question.user_info
        else:
            assert question.expected_behavior == ExpectedBehavior.ANSWER
            assert question.expected_answer == (DRAFT.answer_if_applies if verdict else DRAFT.answer_if_not)
            assert question.reference_claims[-1] == DRAFT.general_rule_claim
    assert {"applies", "does_not_apply", "missing_field"} <= set(result.diagnostics["questions_by_side"])
    assert len({q.id for q in result.questions}) == len(result.questions)
    validate_question_set(result.questions)


def test_a_question_that_restates_a_condition_or_lacks_claims_is_rejected():
    restating = _build(FakeLLM(DRAFT.model_copy(update={"question": "אני בן 21, האם מגיע לי מענק?"})))
    assert restating.questions == [] and restating.diagnostics["rejected"] == {"question_restates_condition": 1}
    no_claims = _build(FakeLLM(DRAFT.model_copy(update={"claims_if_not": []})))
    assert no_claims.diagnostics["rejected"] == {"missing_claims": 1}


def test_a_failing_model_call_is_counted_not_raised():
    class Failing:
        def generate(self, *args):
            raise RuntimeError("boom")

    result = _build(Failing())
    assert result.questions == [] and result.diagnostics["rejected"] == {"llm_error": 1}


def test_personal_basic_answers_come_from_the_persona_with_one_call_per_field():
    llm = FakeLLM()
    result = _build(llm, personal_basic=3)
    basic = [q for q in result.questions if q.question_type == QuestionType.PERSONAL_BASIC]
    assert len(basic) == 3 and llm.calls.count("BasicDraft") == 1
    personas = {persona.id: persona for persona in result.personas}
    for question in basic:
        assert question.profile_keys == ["birth"] and question.sources == []
        assert personas[question.persona_id].profile["birth"] in question.expected_answer
    validate_question_set(result.questions)


def test_persona_fields_survive_csv_and_jsonl_round_trips_and_show_in_review_export(tmp_path):
    result = _build(FakeLLM(), personas_per_rule=3)
    csv_path, jsonl_path = write_questions(result.questions, tmp_path)
    for path in (csv_path, jsonl_path):
        loaded = {q.id: q for q in read_questions(path)}
        assert {(q.persona_id, tuple(q.profile_keys), q.user_info) for q in loaded.values()} == {
            (q.persona_id, tuple(q.profile_keys), q.user_info) for q in result.questions
        }
    review_csv, markdown = write_review_file(result.questions, tmp_path / "review")
    with review_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["persona_readable"] for row in rows} == {
        f"{q.user_info}\nשדות מכריעים: birth" for q in result.questions
    }
    assert "**פרופיל המשתמש:**" in markdown.read_text(encoding="utf-8")


def test_generate_personal_command_writes_rules_personas_questions_and_manifest(tmp_path, monkeypatch):
    import json

    from chatbot_eval import cli
    from chatbot_eval.conditions import RuleBatch, RuleCandidate

    (tmp_path / "schema.json").write_text(json.dumps(SCHEMA.model_dump(mode="json")), encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text('[personalization]\nprofile_schema = "schema.json"\nas_of = "2026-10-27"\n', encoding="utf-8")
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "a.md").write_text(CHUNK.text, encoding="utf-8")
    output = tmp_path / "out"
    calls = []

    class Router:
        def generate(self, prompt, schema, model, **kwargs):
            calls.append(schema.__name__)
            if schema is RuleBatch:
                return RuleBatch(rules=[RuleCandidate(when=RULE.when, consequence=RULE.consequence, quote=RULE.quote)])
            return DRAFT

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "build_llm", lambda *args, **kwargs: Router())
    assert cli.main(["--config", str(config), "generate-personal", "--documents", str(documents), "--output", str(output)]) == 0
    assert calls == ["RuleBatch", "PersonalDraft"]  # one call per chunk, one per rule

    questions = read_questions(output / "silver_questions.jsonl")
    assert questions and {q.question_type for q in questions} == {QuestionType.PERSONAL_INTEGRATION}
    validate_question_set(questions)
    assert len((output / "profile_rules.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    used = {json.loads(line)["id"] for line in (output / "personas.jsonl").read_text(encoding="utf-8").splitlines()}
    assert used == {q.persona_id for q in questions}
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameters"]["as_of"] == "2026-10-27"
    assert manifest["results"]["personal_diagnostics"]["rules"]["llm_calls"] == 1
    assert manifest["results"]["personal_diagnostics"]["questions"]["llm_calls"]["personal_integration"] == 1

    calls.clear()  # a second run is served from the rule and call caches
    assert cli.main(["--config", str(config), "generate-personal", "--documents", str(documents), "--output", str(output)]) == 0
    assert calls == []
