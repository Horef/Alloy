from chatbot_eval.models import ChatbotResult, EvaluationRecord, ExpectedBehavior, JudgeScores, Outcome, QuestionForm, SilverQuestion
from chatbot_eval.report import build_comparison, build_summary, compare_evaluation_contracts, write_report


def test_hebrew_report_and_pipeline_statistics(tmp_path):
    scores = JudgeScores(
        claim_assessments=[],
        required_points_total=2, answer_points_addressed=1, answer_points_correct=0,
        answer_false_claims=1, answer_unsupported_claims=0, answer_extraneous_claims=0,
        retrieval_points_found=2, retrieved_chunks_total=1,
        retrieved_chunks_relevant=1, retrieved_chunks_contradictory=0,
        answer_scope="exact", incorrect_type="hallucination",
        response_is_abstention=False, explanation="התשובה נשמעת סבירה אך שגויה.",
        missing_or_wrong="המספר שגוי.", retrieval_explanation="המידע הנכון נמצא במקטעים.",
    )
    record = EvaluationRecord(
        question=SilverQuestion(id="Q1", topic="זכויות", question="מה הסכום?", expected_answer="100"),
        result=ChatbotResult(question_id="Q1", answer="200", retrieved_context="הסכום הוא 100"),
        outcome=Outcome.MISLEADING_HALLUCINATION, scores=scores,
    )
    summary = build_summary([record])
    assert summary["generation_failures_despite_good_retrieval"] == 1
    assert summary["risky_misinformation_rate"] == 1
    _, report_path = write_report([record], tmp_path)
    report = report_path.read_text(encoding="utf-8")
    assert 'lang="he" dir="rtl"' in report
    assert "הזיה מטעה" in report
    assert "המידע הנכון נמצא במקטעים" in report


def test_report_can_hide_correct_answer_metrics(tmp_path):
    record = EvaluationRecord(
        question=SilverQuestion(id="Q1", topic="זכויות", question="מה הסכום?", expected_answer="100"),
        result=ChatbotResult(question_id="Q1", answer="100"),
        outcome=Outcome.CORRECT_ANSWER,
    )

    _, report_path = write_report(
        [record], tmp_path, show_correct_answer_metrics=False,
    )
    report = report_path.read_text(encoding="utf-8")

    assert "שיעור תשובות נכונות" not in report
    assert "<th>תשובות נכונות</th>" not in report
    assert "שיעור תשובות שימושיות" in report
    assert "<th>תשובות שימושיות</th>" in report


def test_report_shows_infrastructure_error_count_and_percentage(tmp_path):
    records = [
        EvaluationRecord(
            question=SilverQuestion(id="Q1", topic="נושא", question="שאלה", expected_answer="תשובה"),
            result=ChatbotResult(question_id="Q1", answer="תשובה"),
            outcome=Outcome.CORRECT_ANSWER,
        ),
        EvaluationRecord(
            question=SilverQuestion(id="Q2", topic="נושא", question="שאלה", expected_answer="תשובה"),
            result=ChatbotResult(question_id="Q2", answer="", error="answer_generation_failed: placeholder"),
            outcome=Outcome.CHATBOT_ERROR,
        ),
    ]

    summary = build_summary(records)
    _, report_path = write_report(records, tmp_path)

    assert summary["infrastructure_errors"] == 1
    assert summary["infrastructure_error_rate"] == 0.5
    assert "1 (50.0%)" in report_path.read_text(encoding="utf-8")


def test_clarification_is_not_in_factual_answer_denominator():
    answer = EvaluationRecord(
        question=SilverQuestion(id="Q1", topic="נושא", question="שאלה", expected_answer="תשובה"),
        result=ChatbotResult(question_id="Q1", answer="תשובה"), outcome=Outcome.CORRECT_ANSWER,
    )
    clarification = EvaluationRecord(
        question=SilverQuestion(
            id="Q2", topic="נושא", question="איזה?", expected_answer="לאיזו אוכלוסייה?",
            question_form=QuestionForm.AMBIGUOUS, expected_behavior=ExpectedBehavior.CLARIFY,
        ),
        result=ChatbotResult(question_id="Q2", answer="לאיזו אוכלוסייה?"),
        outcome=Outcome.CORRECT_CLARIFICATION,
    )

    summary = build_summary([answer, clarification])

    assert summary["correct_answer_rate_on_answerable"] == 1
    assert summary["clarification_success_rate"] == 1
    assert summary["answer_tasks"] == 1
    assert summary["clarification_tasks"] == 1


def test_report_includes_paired_variant_robustness_and_sample_uncertainty(tmp_path):
    parent = EvaluationRecord(
        question=SilverQuestion(id="Q1", topic="נושא", question="שאלה", expected_answer="תשובה"),
        result=ChatbotResult(question_id="Q1", answer="תשובה"), outcome=Outcome.CORRECT_ANSWER,
    )
    failed_variant = EvaluationRecord(
        question=SilverQuestion(
            id="V1", topic="נושא", question="איך זה עובד?", expected_answer="תשובה",
            question_form=QuestionForm.NATURAL_USER, parent_question_id="Q1",
        ),
        result=ChatbotResult(question_id="V1", answer="לא יודע"), outcome=Outcome.UNRELATED_ANSWER,
    )
    successful_variant = EvaluationRecord(
        question=SilverQuestion(
            id="V2", topic="נושא", question="ומה התשובה?", expected_answer="תשובה",
            question_form=QuestionForm.NATURAL_USER, parent_question_id="Q1",
        ),
        result=ChatbotResult(question_id="V2", answer="תשובה"), outcome=Outcome.CORRECT_ANSWER,
    )

    summary = build_summary([parent, failed_variant, successful_variant])
    paired = summary["paired_variants"]
    assert paired["pairs"] == 2
    assert paired["both_success"] == 1
    assert paired["parent_only"] == 1
    assert paired["robustness_when_parent_succeeds"] == 0.5
    assert paired["variant_success_interval_95"] is not None
    assert summary["by_topic"]["נושא"]["small_sample_warning"] is True

    _, report_path = write_report([parent, failed_variant, successful_variant], tmp_path)
    rendered = report_path.read_text(encoding="utf-8")
    assert "עמידות לניסוחי משתמש" in rendered
    assert "מדגם קטן" in rendered
    assert "רווח סמך 95%" in rendered


def test_report_compares_aggregate_and_matched_question_results(tmp_path):
    def record(identifier, question, outcome):
        return EvaluationRecord(
            question=SilverQuestion(id=identifier, topic="נושא", question=question, expected_answer="תשובה"),
            result=ChatbotResult(question_id=identifier, answer="תשובה"), outcome=outcome,
        )

    previous = [
        record("Q1", "שאלה אחת", Outcome.UNRELATED_ANSWER),
        record("Q2", "שאלה שתיים", Outcome.CORRECT_ANSWER),
        record("Q3", "נוסח קודם", Outcome.CORRECT_ANSWER),
    ]
    current = [
        record("Q1", "שאלה אחת", Outcome.CORRECT_ANSWER),
        record("Q2", "שאלה שתיים", Outcome.UNRELATED_ANSWER),
        record("Q3", "נוסח חדש", Outcome.CORRECT_ANSWER),
        record("Q4", "שאלה חדשה", Outcome.CORRECT_ANSWER),
    ]

    comparison = build_comparison(current, previous)
    assert comparison["matched_questions"] == 2
    assert comparison["matched_outcomes"]["improved"] == 1
    assert comparison["matched_outcomes"]["regressed"] == 1
    assert comparison["id_conflicts"] == ["Q3"]
    assert comparison["current_only"] == 1
    assert comparison["aggregate_comparable"] is False
    assert comparison["metrics"]["factual_answer_success_rate"]["delta"] is None

    summary_path, report_path = write_report(current, tmp_path, previous_records=previous)
    summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
    rendered = report_path.read_text(encoding="utf-8")
    assert summary["comparison"]["matched_questions"] == 2
    assert "השוואה לריצה הקודמת" in rendered
    assert "שאלות מותאמות" in rendered
    assert "מזהים הופיעו בשני הדוחות" in rendered
    assert "לא בר השוואה" in rendered


def test_report_comparison_requires_full_question_identity_and_unique_ids():
    def record(identifier, answer, outcome=Outcome.CORRECT_ANSWER):
        return EvaluationRecord(
            question=SilverQuestion(
                id=identifier, topic="נושא", question="אותה שאלה", expected_answer=answer,
                reference_claims=[answer],
            ),
            result=ChatbotResult(question_id=identifier, answer=answer), outcome=outcome,
        )

    previous = [record("Q1", "תשובה ישנה"), record("Q2", "תשובה")]
    current = [record("Q1", "תשובה חדשה"), record("Q2", "תשובה"), record("Q2", "תשובה")]

    comparison = build_comparison(current, previous)

    assert comparison["matched_questions"] == 0
    assert comparison["id_conflicts"] == ["Q1"]
    assert comparison["current_duplicate_ids"] == ["Q2"]
    assert comparison["previous_duplicate_ids"] == []
    assert comparison["aggregate_comparable"] is False


def test_report_aggregate_delta_requires_identical_benchmark():
    previous = [EvaluationRecord(
        question=SilverQuestion(id="Q1", topic="נושא", question="שאלה", expected_answer="תשובה"),
        result=ChatbotResult(question_id="Q1", answer="לא נכון"), outcome=Outcome.UNRELATED_ANSWER,
    )]
    current = [previous[0].model_copy(update={"outcome": Outcome.CORRECT_ANSWER})]

    comparison = build_comparison(current, previous)

    assert comparison["aggregate_comparable"] is True
    assert comparison["metrics"]["factual_answer_success_rate"]["delta"] == 1.0


def test_metric_population_changes_suppress_quality_delta():
    def record(identifier, outcome):
        return EvaluationRecord(question=SilverQuestion(id=identifier, topic="x", question="q", expected_answer="a"), result=ChatbotResult(question_id=identifier, answer="a"), outcome=outcome)
    old = [record("1", Outcome.CORRECT_ANSWER), record("2", Outcome.UNRELATED_ANSWER)]
    new = [old[0], record("2", Outcome.JUDGE_ERROR)]
    comparison = build_comparison(new, old)
    metric = comparison["metrics"]["factual_answer_success_rate"]
    assert comparison["aggregate_comparable"]
    assert metric["current"] == 1 and metric["previous"] == .5
    assert metric["delta"] is None and metric["favorable"] is None
    assert metric["reason"] == "eligible_population_changed"
    assert metric["current_denominator"] == 1
    assert comparison["matched_outcomes"]["improved"] == 0
    assert comparison["metrics"]["infrastructure_error_rate"]["delta"] == .5
    swapped = [record("1", Outcome.JUDGE_ERROR), old[1]]
    assert build_comparison(swapped, new)["metrics"]["factual_answer_success_rate"]["delta"] is None


def test_all_task_false_claims_remain_visible():
    judged = JudgeScores(claim_assessments=[], required_points_total=0, answer_points_addressed=0, answer_points_correct=0,
        answer_false_claims=1, answer_unsupported_claims=2, answer_extraneous_claims=0,
        retrieval_points_found=0, retrieved_chunks_total=0, retrieved_chunks_relevant=0,
        retrieved_chunks_contradictory=0, answer_scope="exact", incorrect_type="hallucination",
        response_is_abstention=False, explanation="x", missing_or_wrong="x", retrieval_explanation="x")
    record = EvaluationRecord(question=SilverQuestion(id="1", topic="x", question="q", expected_answer="a",
        answerable=False, expected_behavior=ExpectedBehavior.ABSTAIN),
        result=ChatbotResult(question_id="1", answer="false"), scores=judged, outcome=Outcome.MISLEADING_HALLUCINATION)
    summary = build_summary([record])
    assert summary["answer_claim_metrics"]["false_claims_total"] == 0
    assert summary["all_response_claim_violations"]["false_claims_total"] == 1
    assert summary["factual_risk_rate"] == 1


def test_retrieval_population_tracks_missing_context():
    from test_evaluator import scores
    old = EvaluationRecord(question=SilverQuestion(id="1", topic="x", question="q", expected_answer="a"),
        result=ChatbotResult(question_id="1", answer="a", retrieved_context="source"),
        scores=scores(), outcome=Outcome.CORRECT_ANSWER)
    new = old.model_copy(deep=True)
    new.result.retrieved_context = ""
    comparison = build_comparison([new], [old])
    assert comparison["metrics"]["good_retrieval_rate"]["reason"] == "eligible_population_changed"
    assert comparison["metrics"]["factual_answer_success_rate"]["delta"] == 0


def test_evaluation_contract_comparison_discloses_only_changed_field_names(tmp_path):
    previous = {"judge_model": "SECRET_PREVIOUS_VALUE", "limits": {"answer": 100, "context": 200}}
    current = {"judge_model": "SECRET_CURRENT_VALUE", "limits": {"answer": 100, "context": 300}}

    contract = compare_evaluation_contracts(current, previous)

    assert contract == {
        "status": "incompatible",
        "changed_fields": ["judge_model", "limits.context"],
    }
    assert "SECRET_PREVIOUS_VALUE" not in str(contract)
    assert "SECRET_CURRENT_VALUE" not in str(contract)
    assert compare_evaluation_contracts(current, current)["status"] == "compatible"
    assert compare_evaluation_contracts(current, None) == {"status": "unknown", "changed_fields": []}

    record = EvaluationRecord(
        question=SilverQuestion(id="Q1", topic="x", question="q", expected_answer="a"),
        result=ChatbotResult(question_id="Q1", answer="a"), outcome=Outcome.CORRECT_ANSWER,
    )
    summary_path, report_path = write_report(
        [record], tmp_path, previous_records=[record],
        current_contract=current, previous_contract=previous,
    )
    summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
    assert summary["comparison"]["evaluation_contract_compatibility"] == "incompatible"
    assert summary["comparison"]["evaluation_contract_changed_fields"] == ["judge_model", "limits.context"]
    rendered = report_path.read_text(encoding="utf-8")
    assert "judge_model" in rendered and "limits.context" in rendered
    assert "SECRET_PREVIOUS_VALUE" not in rendered
    assert "SECRET_CURRENT_VALUE" not in rendered
