"""Offline review probes: print observed behavior, not desired regression assertions.

Run from the repository: conda run --no-capture-output -n army python docs/review_probes.py
Only synthetic data and a TemporaryDirectory are used. No network calls.
"""
from __future__ import annotations

import json
import hashlib
from email.message import Message
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from chatbot_eval.adapters import HttpChatbotAdapter
from chatbot_eval.artifacts import EvaluationCheckpoint, StructuredCallCheckpoint, evaluation_fingerprint
from chatbot_eval.documents import _structured_chunks
from chatbot_eval.cli import _checkpointed_evaluation, _optional_insights
from chatbot_eval.evaluator import Evaluator, classify, looks_like_abstention
from chatbot_eval.generator import allocate_quotas, validate_candidate
from chatbot_eval.io import read_questions, write_questions
from chatbot_eval.history import discover_previous_run
from chatbot_eval.models import (
    ChatbotResult, ClaimAssessment, EvaluationInsights, EvaluationRecord, EvidenceQuote, ExpectedBehavior, GeneratedQuestion,
    GeneratedVariation, JudgeScores, Outcome, QuestionBatch, QuestionForm, QuestionType, SilverQuestion,
    SourceRef, TopicCandidate, TopicMap, VariationBatch,
)
from chatbot_eval.report import build_comparison, build_summary
from chatbot_eval.results_io import read_premade_results
from chatbot_eval.validation import validate_question_set


def question(identifier="Q1", **changes):
    return SilverQuestion(id=identifier, topic="Policy", question="When?", expected_answer="Tomorrow",
                          reference_claims=["Tomorrow"], **changes)


def score(**changes):
    values = dict(
        claim_assessments=[ClaimAssessment(claim_id="C001", addressed=True, correct=True)],
        required_points_total=1, answer_points_addressed=1, answer_points_correct=1,
        answer_false_claims=0, answer_unsupported_claims=0, answer_extraneous_claims=0,
        retrieval_points_found=0, retrieved_chunks_total=0, retrieved_chunks_relevant=0,
        retrieved_chunks_contradictory=0, answer_scope="exact", incorrect_type="not_applicable",
        response_is_abstention=False, response_is_clarification=False,
        explanation="בדיקה", missing_or_wrong="", retrieval_explanation="אין נתוני אחזור",
    )
    values.update(changes)
    return JudgeScores(**values)


def record(q, outcome, judged=None):
    return EvaluationRecord(question=q, result=ChatbotResult(question_id=q.id, answer="Tomorrow"),
                            outcome=outcome, scores=judged)


def emit(name, **values):
    print(json.dumps({"probe": name, **values}, ensure_ascii=False))


def error_type(call):
    try:
        call()
    except Exception as exc:
        return type(exc).__name__
    return None


def main():
    with TemporaryDirectory(prefix="alloy-review-") as directory:
        root = Path(directory)
        answer = "The optional attachment is not provided. Submit the application tomorrow."

        class Adapter:
            def ask(self, q):
                return ChatbotResult(question_id=q.id, answer=answer)

        class Judge:
            def generate(self, prompt, schema, model):
                return score()

        judged = Evaluator(Adapter(), Judge(), "fake").evaluate([question()])[0]
        emit("abstention_override", detected=looks_like_abstention(answer), outcome=judged.outcome.value,
             correct_claims=judged.scores.answer_points_correct)

        abstain_q = question().model_copy(update={"answerable": False, "expected_behavior": ExpectedBehavior.ABSTAIN,
                                                   "reference_claims": []})
        clarify_q = question().model_copy(update={"expected_behavior": ExpectedBehavior.CLARIFY,
                                                   "question_form": QuestionForm.AMBIGUOUS, "reference_claims": []})
        dangerous = score(claim_assessments=[], required_points_total=0, answer_points_addressed=0,
                          answer_points_correct=0, answer_false_claims=2, incorrect_type="hallucination",
                          response_is_abstention=True)
        mixed = score(claim_assessments=[], required_points_total=0, answer_points_addressed=0,
                      answer_points_correct=0, answer_false_claims=2, incorrect_type="hallucination",
                      response_is_clarification=True)
        outcome = classify(abstain_q, dangerous)
        summary = build_summary([record(abstain_q, outcome, dangerous)])
        emit("unsafe_success", abstention=outcome.value, clarification=classify(clarify_q, mixed).value,
             reported_risky_rate=summary["risky_misinformation_rate"],
             reported_false_claims=summary["answer_claim_metrics"]["false_claims_total"])

        original = question().model_copy(update={"sources": [
            SourceRef(source_id="a#1", file="a.md", location="table", excerpt="Name | Amount | Deadline")
        ]})
        csv_path, jsonl_path = write_questions([original], root / "roundtrip")
        emit("csv_sources", original=original.sources[0].excerpt,
             csv=read_questions(csv_path)[0].sources[0].excerpt,
             jsonl=read_questions(jsonl_path)[0].sources[0].excerpt)

        legacy = root / "legacy.jsonl"
        legacy.write_text(json.dumps(dict(id="U1", topic="x", question="When?", expected_answer="Unknown",
                                         answerable=False)) + "\n", encoding="utf-8")
        emit("legacy_jsonl", error=error_type(lambda: read_questions(legacy)))

        empty = root / "invalid.csv"
        empty.write_text("question,expected_answer,answer\n,,\n", encoding="utf-8")
        emit("all_rows_invalid", imported_count=len(read_premade_results(empty)))

        a, b = question("A"), question("B")
        before = [record(a, Outcome.CORRECT_ANSWER), record(b, Outcome.MISLEADING_HALLUCINATION)]
        after = [record(a, Outcome.CORRECT_ANSWER), record(b, Outcome.JUDGE_ERROR)]
        comparison = build_comparison(after, before)
        emit("comparison_denominators", aggregate_comparable=comparison["aggregate_comparable"],
             factual_metric=comparison["metrics"]["factual_answer_success_rate"],
             matched_outcomes=comparison["matched_outcomes"])

        path = root / "evaluation.jsonl"
        r = record(question(), Outcome.CORRECT_ANSWER)
        fingerprints = {"Q1": evaluation_fingerprint(r.question)}
        checkpoint = EvaluationCheckpoint(path, fingerprints, resume=False)
        checkpoint.append(r)
        path.write_text(path.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
        resumed = EvaluationCheckpoint(path, fingerprints, resume=True)
        loaded = len(resumed.load())
        resumed.append(r)
        emit("evaluation_checkpoint_missing_newline", initial_loaded=loaded, subsequent_load_error=error_type(resumed.load))

        class TopicLLM:
            def generate(self, prompt, schema, model):
                return TopicMap(topics=[])

        path = root / "generation.jsonl"
        checkpoint = StructuredCallCheckpoint(path, TopicLLM(), signature="same", resume=False)
        checkpoint.generate("first", TopicMap, "fake")
        path.write_text(path.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
        resumed = StructuredCallCheckpoint(path, TopicLLM(), signature="same", resume=True)
        resumed.generate("second", TopicMap, "fake")
        emit("generation_checkpoint_missing_newline", subsequent_load_error=error_type(
            lambda: StructuredCallCheckpoint(path, TopicLLM(), signature="same", resume=True)))

        path = root / "mismatch.jsonl"
        checkpoint = StructuredCallCheckpoint(path, TopicLLM(), signature="old", resume=False)
        checkpoint.generate("first", TopicMap, "fake")
        path.write_text(path.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
        mismatch_error = error_type(lambda: StructuredCallCheckpoint(path, TopicLLM(), signature="new", resume=True))
        emit("generation_mismatch_without_newline", error=mismatch_error, remaining_bytes=path.stat().st_size)

        class Response:
            status = 200
            headers = Message()
            headers["Content-Type"] = "application/json"

            def __init__(self, value):
                self.value = value

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, maximum):
                return json.dumps({"answer": self.value}).encode()

        for value in (None, "", {"error": "upstream failure"}):
            with patch("chatbot_eval.adapters.urllib.request.urlopen", return_value=Response(value)):
                result = HttpChatbotAdapter("https://example.invalid", max_retries=0).ask(question())
            emit("http_answer_shape", input=value, answer=result.answer, error=result.error)

        text = "Applications close Friday."
        emit("short_document", input_chars=len(text), chunks=_structured_chunks(text, 1000, 100))

        invalid = GeneratedQuestion(question="", expected_answer="", answerable=False, difficulty="medium",
                                    rationale="", source_ids=[], question_type=QuestionType.UNANSWERABLE,
                                    reference_claims=["unexpected factual claim"])
        chunks, reason = validate_candidate(invalid, {})
        from chatbot_eval.generator import SilverSetGenerator
        silver = SilverSetGenerator._to_silver(invalid, "x", chunks, 1)
        emit("candidate_validation_gap", rejection=reason,
             final_validation_error=error_type(lambda: validate_question_set([silver])))

        quotas = allocate_quotas([TopicCandidate(name="Only topic", description="", importance=5, source_ids=[])],
                                 total=30, minimum=1, max_share=0.35)
        emit("infeasible_topic_cap", requested=30, allocated=sum(quotas.values()))

        path = root / "retry.jsonl"
        contract = {"kind": "synthetic-live-review"}
        signature = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        original_result = ChatbotResult(question_id="Q1", answer="Original captured answer")
        failed = EvaluationRecord(question=question(), result=original_result, outcome=Outcome.JUDGE_ERROR)
        checkpoint = EvaluationCheckpoint(path, {"Q1": evaluation_fingerprint(failed.question, contract=contract)},
                                          resume=False, run_signature=signature)
        checkpoint.append(failed)
        args = SimpleNamespace(checkpoint=path, output=root, resume=True, retry_errors=True)
        records, _, _ = _checkpointed_evaluation(
            args, [(failed.question, original_result)], Evaluator(Adapter(), Judge(), "fake"),
            premade=False, contract=contract,
        )
        emit("live_judge_retry_recaptures_answer", original=original_result.answer, retried=records[0].result.answer)

        stale = root / "stale"
        stale.mkdir()
        old_insights = EvaluationInsights(executive_summary="Old run", strengths=[], issues=[], methodology_note="Old run")
        (stale / "evaluation_insights.json").write_text(old_insights.model_dump_json(), encoding="utf-8")
        (stale / "evaluation_details.jsonl").write_text(r.model_dump_json() + "\n", encoding="utf-8")
        _optional_insights(SimpleNamespace(generate_insights=False, output=stale), [r], None, "fake", cache=None, settings=None)
        _, discovered = discover_previous_run(stale)
        emit("stale_insights", discovered_summary=discovered.executive_summary)

        from chatbot_eval.documents import Chunk
        from chatbot_eval.generator import GenerationOptions

        class BudgetLLM:
            def generate(self, prompt, schema, model):
                if schema is VariationBatch:
                    return VariationBatch(variations=[
                        GeneratedVariation(source_question_id="Q0001", question=text, question_form=form,
                                           required_clarification="Which category?" if form == "ambiguous" else "",
                                           rationale="synthetic")
                        for text, form in [("casual gamma", "natural_user"), ("personal delta", "natural_user"),
                                           ("vague epsilon", "ambiguous")]
                    ])
                if "boundary questions" in prompt:
                    return QuestionBatch(questions=[
                        GeneratedQuestion(question=text, expected_answer="Unknown", answerable=False,
                                          difficulty="medium", rationale="synthetic", source_ids=["a#1"],
                                          question_type=QuestionType.UNANSWERABLE, reference_claims=[])
                        for text in ["unknown alpha", "absent beta"]
                    ])
                return QuestionBatch(questions=[GeneratedQuestion(
                    question="When?", expected_answer="Tomorrow", answerable=True, difficulty="medium",
                    rationale="synthetic", source_ids=["a#1"], reference_claims=["Tomorrow"],
                    supporting_quotes=[EvidenceQuote(source_id="a#1", quote="Tomorrow")],
                )])

        items, _ = SilverSetGenerator(BudgetLLM(), "fake").generate(
            [Chunk("a#1", "a.md", "document", "Tomorrow")],
            GenerationOptions(max_questions=10, batch_chunks=1, min_topic_questions=1, max_topic_share=1,
                              unanswerable_ratio=0.2, user_variation_ratio=0.3,
                              requested_topic="Only", requested_topic_count=1),
            topics=[TopicCandidate(name="Only", description="", importance=5, source_ids=["a#1"])],
        )
        emit("topic_count_ceiling", topic_count=1, produced=len(items))


if __name__ == "__main__":
    main()
