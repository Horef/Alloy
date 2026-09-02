from __future__ import annotations

import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .artifacts import atomic_write_text
from .labels import EXPECTED_BEHAVIOR_HEBREW, OUTCOME_COLORS, OUTCOME_HEBREW, QUESTION_FORM_HEBREW
from .models import EvaluationInsights, EvaluationRecord, ExpectedBehavior, Outcome

ANSWER_SUCCESS = {Outcome.CORRECT_ANSWER}
ANSWER_USEFUL = {Outcome.CORRECT_ANSWER, Outcome.PARTIAL_TOO_LITTLE, Outcome.PARTIAL_TOO_MUCH, Outcome.CORRECT_CLARIFICATION}
PIPELINE_SUCCESS = {Outcome.CORRECT_ANSWER, Outcome.CORRECT_ABSTENTION, Outcome.CORRECT_CLARIFICATION}
RISKY = {Outcome.MISLEADING_HALLUCINATION, Outcome.SHOULD_HAVE_ABSTAINED, Outcome.MISSING_CLARIFICATION}
MIN_TOPIC_SAMPLE = 10


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _paired_variant_metrics(records: list[EvaluationRecord]) -> dict:
    evaluable = {
        record.question.id: record
        for record in records
        if record.outcome not in {Outcome.CHATBOT_ERROR, Outcome.JUDGE_ERROR}
    }
    categories = Counter()
    by_form: dict[str, Counter] = defaultdict(Counter)
    for child in evaluable.values():
        parent_id = child.question.parent_question_id
        if not parent_id or parent_id not in evaluable:
            continue
        parent = evaluable[parent_id]
        parent_success = parent.outcome in PIPELINE_SUCCESS
        child_success = child.outcome in PIPELINE_SUCCESS
        category = (
            "both_success" if parent_success and child_success else
            "parent_only" if parent_success else
            "variant_only" if child_success else
            "both_failure"
        )
        categories[category] += 1
        by_form[child.question.question_form.value][category] += 1

    def summarize(values: Counter) -> dict:
        total = sum(values.values())
        parent_success_total = values["both_success"] + values["parent_only"]
        return {
            "pairs": total,
            **{key: values[key] for key in ("both_success", "parent_only", "variant_only", "both_failure")},
            "variant_success_rate": _rate(values["both_success"] + values["variant_only"], total),
            "robustness_when_parent_succeeds": _rate(values["both_success"], parent_success_total),
            "degradation_rate": _rate(values["parent_only"], parent_success_total),
            "variant_success_interval_95": _wilson_interval(values["both_success"] + values["variant_only"], total),
        }

    return {
        **summarize(categories),
        "by_form": {form: summarize(values) for form, values in sorted(by_form.items())},
    }


def _retrieval_good(record: EvaluationRecord) -> bool:
    return bool(
        record.scores
        and record.scores.required_points_total > 0
        and record.scores.retrieval_points_found == record.scores.required_points_total
        and record.scores.retrieved_chunks_contradictory == 0
    )


def build_summary(records: list[EvaluationRecord]) -> dict:
    counts = Counter(record.outcome.value for record in records)
    evaluable = [r for r in records if r.outcome not in {Outcome.CHATBOT_ERROR, Outcome.JUDGE_ERROR}]
    answerable = [r for r in evaluable if r.question.answerable]
    answer_tasks = [r for r in evaluable if r.question.expected_behavior == ExpectedBehavior.ANSWER]
    clarification_tasks = [r for r in evaluable if r.question.expected_behavior == ExpectedBehavior.CLARIFY]
    abstention_tasks = [r for r in evaluable if r.question.expected_behavior == ExpectedBehavior.ABSTAIN]
    retrieval_scored = [r for r in answer_tasks if r.scores and r.scores.retrieved_chunks_total > 0]
    good_retrieval = [r for r in retrieval_scored if _retrieval_good(r)]
    generation_failures = [r for r in good_retrieval if r.outcome not in PIPELINE_SUCCESS]
    pipeline = Counter()
    for record in answer_tasks:
        if not record.scores or record.scores.retrieved_chunks_total == 0:
            pipeline["retrieval_unavailable"] += 1
        else:
            retrieval = "good" if _retrieval_good(record) else "poor"
            answer = "good" if record.outcome in PIPELINE_SUCCESS else "poor"
            pipeline[f"retrieval_{retrieval}_answer_{answer}"] += 1

    by_topic_records: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        by_topic_records[record.question.topic or "לא סווג"].append(record)
    by_topic = {}
    for topic, topic_records in sorted(by_topic_records.items(), key=lambda item: (-len(item[1]), item[0])):
        topic_evaluable = [r for r in topic_records if r.outcome not in {Outcome.CHATBOT_ERROR, Outcome.JUDGE_ERROR}]
        topic_answer_tasks = [r for r in topic_evaluable if r.question.expected_behavior == ExpectedBehavior.ANSWER]
        topic_retrieval = [r for r in topic_answer_tasks if r.scores and r.scores.retrieved_chunks_total > 0]
        topic_correct = sum(r.outcome in ANSWER_SUCCESS for r in topic_records)
        topic_useful = sum(r.outcome in ANSWER_USEFUL for r in topic_records)
        topic_risky = sum(r.outcome in RISKY for r in topic_records)
        topic_good_retrieval = sum(_retrieval_good(r) for r in topic_retrieval)
        by_topic[topic] = {
            "total": len(topic_records),
            "evaluable": len(topic_evaluable), "retrieval_evaluated": len(topic_retrieval),
            "answer_tasks": len(topic_answer_tasks),
            "correct": topic_correct, "useful": topic_useful, "risky": topic_risky,
            "good_retrieval": topic_good_retrieval,
            "correct_rate": _rate(topic_correct, len(topic_answer_tasks)),
            "useful_rate": _rate(topic_useful, len(topic_evaluable)),
            "risky_rate": _rate(topic_risky, len(topic_evaluable)),
            "good_retrieval_rate": _rate(topic_good_retrieval, len(topic_retrieval)),
            "correct_interval_95": _wilson_interval(topic_correct, len(topic_answer_tasks)),
            "useful_interval_95": _wilson_interval(topic_useful, len(topic_evaluable)),
            "risky_interval_95": _wilson_interval(topic_risky, len(topic_evaluable)),
            "good_retrieval_interval_95": _wilson_interval(topic_good_retrieval, len(topic_retrieval)),
            "small_sample_warning": len(topic_evaluable) < MIN_TOPIC_SAMPLE,
            "outcomes": dict(Counter(r.outcome.value for r in topic_records)),
        }

    scored = [r for r in answer_tasks if r.scores]
    required_total = sum(r.scores.required_points_total for r in scored)
    addressed_total = sum(r.scores.answer_points_addressed for r in scored)
    correct_points = sum(r.scores.answer_points_correct for r in scored)
    false_claims = sum(r.scores.answer_false_claims for r in scored)
    unsupported_claims = sum(r.scores.answer_unsupported_claims for r in scored)
    extraneous_claims = sum(r.scores.answer_extraneous_claims for r in scored)
    retrieval_required = sum(r.scores.required_points_total for r in retrieval_scored)
    retrieval_found = sum(r.scores.retrieval_points_found for r in retrieval_scored)
    chunks_total = sum(r.scores.retrieved_chunks_total for r in retrieval_scored)
    chunks_relevant = sum(r.scores.retrieved_chunks_relevant for r in retrieval_scored)
    chunks_contradictory = sum(r.scores.retrieved_chunks_contradictory for r in retrieval_scored)
    infrastructure_errors = len(records) - len(evaluable)
    return {
        "total": len(records), "evaluable": len(evaluable),
        "answerable_questions": len(answerable),
        "answer_tasks": len(answer_tasks),
        "clarification_tasks": len(clarification_tasks),
        "abstention_tasks": len(abstention_tasks),
        "unanswerable_questions": sum(not r.question.answerable for r in evaluable),
        "infrastructure_errors": infrastructure_errors,
        "infrastructure_error_rate": _rate(infrastructure_errors, len(records)),
        "correct_answer_rate_on_answerable": _rate(sum(r.outcome in ANSWER_SUCCESS for r in answer_tasks), len(answer_tasks)),
        "factual_answer_success_rate": _rate(sum(r.outcome in ANSWER_SUCCESS for r in answer_tasks), len(answer_tasks)),
        "factual_answer_success_interval_95": _wilson_interval(
            sum(r.outcome in ANSWER_SUCCESS for r in answer_tasks), len(answer_tasks),
        ),
        "useful_answer_rate_on_answerable": _rate(sum(r.outcome in ANSWER_USEFUL for r in answerable), len(answerable)),
        "clarification_success_rate": _rate(
            sum(r.outcome == Outcome.CORRECT_CLARIFICATION for r in clarification_tasks), len(clarification_tasks),
        ),
        "abstention_success_rate": _rate(
            sum(r.outcome == Outcome.CORRECT_ABSTENTION for r in abstention_tasks), len(abstention_tasks),
        ),
        "risky_misinformation_rate": _rate(sum(r.outcome in RISKY for r in evaluable), len(evaluable)),
        "retrieval_evaluated": len(retrieval_scored),
        "good_retrieval_rate": _rate(len(good_retrieval), len(retrieval_scored)),
        "generation_failures_despite_good_retrieval": len(generation_failures),
        "generation_failure_question_ids": [r.question.id for r in generation_failures],
        "outcomes": dict(counts),
        "answer_claim_metrics": {
            "required_points_total": required_total, "addressed_points_total": addressed_total,
            "correct_points_total": correct_points,
            "required_point_coverage": _rate(correct_points, required_total),
            "addressed_point_accuracy": _rate(correct_points, addressed_total),
            "false_claims_total": false_claims, "unsupported_claims_total": unsupported_claims,
            "extraneous_claims_total": extraneous_claims,
            "false_claims_per_answer": _rate(false_claims, len(scored)),
            "unsupported_claims_per_answer": _rate(unsupported_claims, len(scored)),
            "extraneous_claims_per_answer": _rate(extraneous_claims, len(scored)),
        },
        "retrieval_claim_metrics": {
            "required_points_total": retrieval_required, "points_found_total": retrieval_found,
            "required_point_coverage": _rate(retrieval_found, retrieval_required),
            "chunks_total": chunks_total, "relevant_chunks_total": chunks_relevant,
            "irrelevant_chunks_total": chunks_total - chunks_relevant,
            "relevant_chunk_rate": _rate(chunks_relevant, chunks_total),
            "contradictory_chunks_total": chunks_contradictory,
            "contradictory_chunk_rate": _rate(chunks_contradictory, chunks_total),
        },
        "pipeline": dict(pipeline), "by_topic": by_topic,
        "pipeline_evaluable": len(answer_tasks),
        "minimum_topic_sample": MIN_TOPIC_SAMPLE,
        "paired_variants": _paired_variant_metrics(records),
    }


def build_comparison(
    current_records: list[EvaluationRecord],
    previous_records: list[EvaluationRecord],
) -> dict:
    """Compare whole-run KPIs and like-for-like question outcomes."""
    current_summary = build_summary(current_records)
    previous_summary = build_summary(previous_records)
    current_groups: dict[str, list[EvaluationRecord]] = defaultdict(list)
    previous_groups: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in current_records:
        current_groups[record.question.id].append(record)
    for record in previous_records:
        previous_groups[record.question.id].append(record)
    current_duplicate_ids = sorted(identifier for identifier, values in current_groups.items() if len(values) > 1)
    previous_duplicate_ids = sorted(identifier for identifier, values in previous_groups.items() if len(values) > 1)
    current_by_id = {identifier: values[0] for identifier, values in current_groups.items() if len(values) == 1}
    previous_by_id = {identifier: values[0] for identifier, values in previous_groups.items() if len(values) == 1}
    common_ids = sorted(set(current_by_id) & set(previous_by_id))
    id_conflicts: list[str] = []
    matched_ids: list[str] = []
    for question_id in common_ids:
        current_question = current_by_id[question_id].question.model_dump(mode="json")
        previous_question = previous_by_id[question_id].question.model_dump(mode="json")
        # Review workflow annotations do not alter the evaluation task. Every other field does.
        for value in (current_question, previous_question):
            value.pop("review_status", None)
            value.pop("reviewer_notes", None)
        if _normalized_comparison_value(current_question) == _normalized_comparison_value(previous_question):
            matched_ids.append(question_id)
        else:
            id_conflicts.append(question_id)

    current_only_ids = sorted(set(current_groups) - set(previous_groups))
    previous_only_ids = sorted(set(previous_groups) - set(current_groups))
    aggregate_comparable = not any((
        current_duplicate_ids, previous_duplicate_ids, id_conflicts,
        current_only_ids, previous_only_ids,
    ))
    metric_definitions = {
        "factual_answer_success_rate": ("תשובות עובדתיות נכונות", False),
        "useful_answer_rate_on_answerable": ("תשובות שימושיות", False),
        "clarification_success_rate": ("הבהרות מתאימות", False),
        "abstention_success_rate": ("הימנעות נכונה", False),
        "good_retrieval_rate": ("אחזור איכותי", False),
        "risky_misinformation_rate": ("סיכון למידע מטעה", True),
        "infrastructure_error_rate": ("שגיאות תשתית", True),
    }
    metrics = {}
    for key, (label, lower_is_better) in metric_definitions.items():
        current = current_summary.get(key)
        previous = previous_summary.get(key)
        delta = current - previous if aggregate_comparable and current is not None and previous is not None else None
        favorable = None
        if delta:
            favorable = delta < 0 if lower_is_better else delta > 0
        metrics[key] = {
            "label": label,
            "current": current,
            "previous": previous,
            "delta": delta,
            "lower_is_better": lower_is_better,
            "favorable": favorable,
        }

    transitions = Counter()
    states = Counter()
    by_topic: dict[str, Counter] = defaultdict(Counter)
    matched = 0
    for question_id in matched_ids:
        current = current_by_id[question_id]
        previous = previous_by_id[question_id]
        matched += 1
        transitions[f"{previous.outcome.value}->{current.outcome.value}"] += 1
        topic = current.question.topic or "לא סווג"
        if current.outcome in {Outcome.CHATBOT_ERROR, Outcome.JUDGE_ERROR} or previous.outcome in {
            Outcome.CHATBOT_ERROR, Outcome.JUDGE_ERROR,
        }:
            state = "not_comparable"
        else:
            previous_success = previous.outcome in PIPELINE_SUCCESS
            current_success = current.outcome in PIPELINE_SUCCESS
            state = (
                "improved" if current_success and not previous_success else
                "regressed" if previous_success and not current_success else
                "unchanged_success" if current_success else "unchanged_failure"
            )
        states[state] += 1
        by_topic[topic][state] += 1

    comparable = matched - states["not_comparable"]
    return {
        "current_total": len(current_records),
        "previous_total": len(previous_records),
        "matched_questions": matched,
        "comparable_questions": comparable,
        "aggregate_comparable": aggregate_comparable,
        "current_only": len(current_only_ids),
        "previous_only": len(previous_only_ids),
        "current_only_ids": current_only_ids,
        "previous_only_ids": previous_only_ids,
        "id_conflicts": id_conflicts,
        "current_duplicate_ids": current_duplicate_ids,
        "previous_duplicate_ids": previous_duplicate_ids,
        "metrics": metrics,
        "matched_outcomes": {
            **{key: states[key] for key in (
                "improved", "regressed", "unchanged_success", "unchanged_failure", "not_comparable",
            )},
            "net_improvement": states["improved"] - states["regressed"],
        },
        "outcome_transitions": dict(sorted(transitions.items())),
        "by_topic": {
            topic: {
                **{key: values[key] for key in (
                    "improved", "regressed", "unchanged_success", "unchanged_failure", "not_comparable",
                )},
                "total": sum(values.values()),
            }
            for topic, values in sorted(by_topic.items())
        },
    }


def _normalized_comparison_value(value: object) -> object:
    """Normalize insignificant text whitespace while retaining complete question semantics."""
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, list):
        return [_normalized_comparison_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalized_comparison_value(item) for key, item in sorted(value.items())}
    return value


def _pct(value: float | None) -> str:
    return "לא זמין" if value is None else f"{value:.1%}"


def _esc(value: object) -> str:
    return html.escape(str(value or ""))


def _bar(label: str, count: int, total: int, color: str) -> str:
    width = 0 if total == 0 else count / total * 100
    return f'<div class="bar-row"><span>{_esc(label)}</span><div class="track"><i style="width:{width:.2f}%;background:{color}"></i></div><b>{count}</b><small>{width:.1f}%</small></div>'


def _metric_bar(label: str, value: float | None, detail: str, color: str) -> str:
    width = 0 if value is None else value * 100
    shown = "—" if value is None else f"{value:.1%}"
    return f'<div class="score-row"><span>{_esc(label)}</span><div class="track"><i style="width:{width:.2f}%;background:{color}"></i></div><b>{shown}</b><small>{_esc(detail)}</small></div>'


def _pct_count(rate: float | None, count: int, denominator: int) -> str:
    return f"לא זמין ({count}/{denominator})" if rate is None else f"{rate:.1%} ({count}/{denominator})"


def _pct_count_ci(rate: float | None, count: int, denominator: int, interval: list[float] | None) -> str:
    base = _pct_count(rate, count, denominator)
    if interval is None:
        return base
    return f"{base}<small>רווח סמך 95%: {interval[0]:.1%}–{interval[1]:.1%}</small>"


def _comparison_block(comparison: dict | None) -> str:
    if not comparison:
        return ""
    metric_rows = []
    for metric in comparison["metrics"].values():
        delta = metric["delta"]
        delta_text = (
            "לא בר השוואה" if not comparison["aggregate_comparable"] else
            "לא זמין" if delta is None else f"{delta:+.1%}"
        )
        delta_class = "neutral" if metric["favorable"] is None else ("good" if metric["favorable"] else "bad")
        metric_rows.append(
            f'<tr><td>{_esc(metric["label"])}</td><td>{_pct(metric["previous"])}</td>'
            f'<td>{_pct(metric["current"])}</td><td class="delta {delta_class}">{delta_text}</td></tr>'
        )
    matched = comparison["matched_outcomes"]
    topic_rows = "".join(
        f'<tr><td>{_esc(topic)}</td><td>{data["total"]}</td><td>{data["improved"]}</td>'
        f'<td>{data["regressed"]}</td><td>{data["unchanged_success"]}</td>'
        f'<td>{data["unchanged_failure"]}</td></tr>'
        for topic, data in comparison["by_topic"].items()
    )
    conflict_note = "" if not comparison["id_conflicts"] else (
        f'<div class="note warn">{len(comparison["id_conflicts"])} מזהים הופיעו בשני הדוחות אך לפחות שדה מהותי '
        'בשאלת הכסף השתנה, ולכן הם לא נכללו בהשוואה הישירה.</div>'
    )
    duplicate_count = len(comparison["current_duplicate_ids"]) + len(comparison["previous_duplicate_ids"])
    duplicate_note = "" if not duplicate_count else (
        f'<div class="note warn">נמצאו {duplicate_count} מופעים של מזהי שאלה כפולים בין הקבצים. '
        'המזהים הכפולים הוחרגו מהתאמה כדי למנוע החלפה שקטה של תוצאות.</div>'
    )
    aggregate_note = (
        '<div class="note">מערך הבדיקה זהה בשתי הריצות; שינויי המדדים הכוללים ניתנים להשוואה ישירה.</div>'
        if comparison["aggregate_comparable"] else
        '<div class="note warn">מערכי הבדיקה אינם זהים לחלוטין. הערכים הכוללים מוצגים לתיאור בלבד, '
        'אך שינוייהם הוסתרו כדי שלא להציג שינוי בהרכב המדגם כשינוי בביצועים.</div>'
    )
    no_match_note = "" if comparison["matched_questions"] else (
        '<div class="note warn">לא נמצאו שאלות זהות להשוואה ישירה. ערכי המדדים הכוללים עדיין מוצגים, '
        'אך שינוייהם אינם מוצגים אלא אם מערכי הבדיקה זהים.</div>'
    )
    return f'''<div class="panel comparison"><h2>השוואה לריצה הקודמת</h2>
<div class="note">השוואת שאלות מותאמות דורשת אותו מזהה ואת כל שדות המשימה המהותיים, כולל נוסח, תשובת ייחוס, טענות, ראיות, סוג, צורה, נושא והתנהגות מצופה. הצלחה ישירה פירושה תשובה נכונה, הימנעות נכונה או הבהרה נכונה בהתאם למשימה.</div>
<div class="cards"><div class="card">שאלות מותאמות<b>{comparison["matched_questions"]}</b><small>{comparison["comparable_questions"]} ניתנות להשוואת הצלחה</small></div><div class="card">השתפרו<b>{matched["improved"]}</b></div><div class="card">נסוגו<b>{matched["regressed"]}</b></div><div class="card">שיפור נטו<b>{matched["net_improvement"]:+d}</b><small>השתפרו פחות נסוגו</small></div><div class="card">חדשות / הוסרו<b>{comparison["current_only"]} / {comparison["previous_only"]}</b></div></div>
{conflict_note}{duplicate_note}{no_match_note}{aggregate_note}
<h3>שינוי במדדים הכוללים</h3><table><thead><tr><th>מדד</th><th>קודם</th><th>עכשיו</th><th>שינוי</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table>
{('<h3>שינוי בשאלות מותאמות לפי נושא</h3><table><thead><tr><th>נושא</th><th>מותאמות</th><th>השתפרו</th><th>נסוגו</th><th>הצלחה נשמרה</th><th>כשל נשאר</th></tr></thead><tbody>' + topic_rows + '</tbody></table>') if topic_rows else ''}</div>'''


def _details(record: EvaluationRecord) -> str:
    scores = record.scores
    label, color = OUTCOME_HEBREW[record.outcome], OUTCOME_COLORS[record.outcome]
    score_cells = "" if not scores else "".join((
        f'<span><b>צורת שאלה</b> {_esc(QUESTION_FORM_HEBREW[record.question.question_form])}</span>',
        f'<span><b>התנהגות מצופה</b> {_esc(EXPECTED_BEHAVIOR_HEBREW[record.question.expected_behavior])}</span>',
        f'<span><b>פרטים נדרשים</b> {scores.required_points_total}</span>',
        f'<span><b>פרטים שנענו נכון</b> {scores.answer_points_correct}/{scores.required_points_total}</span>',
        f'<span><b>טענות שגויות</b> {scores.answer_false_claims}</span>',
        f'<span><b>טענות לא מבוססות</b> {scores.answer_unsupported_claims}</span>',
        f'<span><b>טענות עודפות</b> {scores.answer_extraneous_claims}</span>',
        f'<span><b>מידע שנמצא באחזור</b> {scores.retrieval_points_found}/{scores.required_points_total}</span>',
        f'<span><b>מקטעים רלוונטיים</b> {scores.retrieved_chunks_relevant}/{scores.retrieved_chunks_total}</span>',
        f'<span><b>מקטעים סותרים</b> {scores.retrieved_chunks_contradictory}</span>',
    ))
    if scores:
        explanation = scores.explanation
    elif record.result.error:
        explanation = f"שגיאת מערכת בצ׳אטבוט: {record.result.error}"
    else:
        explanation = f"שגיאה בתהליך הבדיקה: {record.judge_error}"
    retrieval_explanation = scores.retrieval_explanation if scores else ""
    missing = scores.missing_or_wrong if scores else ""
    claim_rows = "".join(
        f'<li><b>{_esc(f"C{index:03d}")}</b> {_esc(claim)}</li>'
        for index, claim in enumerate(record.question.reference_claims, 1)
    )
    claim_block = f'<section><h4>טענות ייחוס קבועות</h4><ol>{claim_rows}</ol></section>' if claim_rows else ""
    search = " ".join((record.question.id, record.question.topic, record.question.question, record.result.answer, label)).casefold()
    return f'''<details class="result" data-topic="{_esc(record.question.topic)}" data-outcome="{_esc(record.outcome.value)}" data-search="{_esc(search)}">
<summary><span class="id">{_esc(record.question.id)}</span><span class="topic">{_esc(record.question.topic)}</span><span class="q">{_esc(record.question.question)}</span><span class="badge" style="background:{color}">{_esc(label)}</span></summary>
<div class="detail-grid"><section><h4>המענה המצופה</h4><p>{_esc(record.question.expected_answer)}</p></section><section><h4>תשובת הצ׳אטבוט</h4><p>{_esc(record.result.answer)}</p></section></div>
{claim_block}
<section><h4>מקטעים שאוחזרו</h4><pre>{_esc(record.result.retrieved_context) or 'לא סופקו מקטעים שאוחזרו'}</pre></section><div class="scores">{score_cells}</div>
<div class="detail-grid"><section><h4>הסבר הבדיקה</h4><p>{_esc(explanation)}</p><p class="warn">{_esc(missing)}</p></section><section><h4>אבחון האחזור</h4><p>{_esc(retrieval_explanation)}</p></section></div></details>'''


def _insights_block(insights: EvaluationInsights | None) -> str:
    if not insights:
        return ""
    priorities = {"high": ("עדיפות גבוהה", "#dc2626"), "medium": ("עדיפות בינונית", "#d97706"), "low": ("עדיפות נמוכה", "#64748b")}
    confidences = {"high": "ביטחון גבוה", "medium": "ביטחון בינוני", "low": "ביטחון נמוך"}
    strengths = "".join(f"<li>{_esc(item)}</li>" for item in insights.strengths)
    issues = "".join(
        f'''<article class="insight"><div><span class="badge" style="background:{priorities[issue.priority][1]}">{priorities[issue.priority][0]}</span> <small>{confidences[issue.confidence]} · {issue.evidence_count} מקרים</small></div><h3>{_esc(issue.title)}</h3><p><b>דפוס שנצפה:</b> {_esc(issue.observed_pattern)}</p><p><b>השערה לגורם:</b> {_esc(issue.likely_cause_hypothesis)}</p><p><b>המלצה:</b> {_esc(issue.recommendation)}</p><small>נושאים: {_esc(', '.join(issue.affected_topics) or 'לא סווג')} · שאלות לדוגמה: {_esc(', '.join(issue.example_question_ids) or 'לא צוינו')}</small></article>'''
        for issue in insights.issues
    )
    return f'''<div class="panel insights"><h2>תובנות והמלצות</h2><p class="lead">{_esc(insights.executive_summary)}</p><div class="note"><b>חשוב:</b> זהו ניתוח מסייע שנוצר על ידי מודל שפה. הדפוסים מבוססים על מתאם בין תוצאות ואינם מוכיחים סיבתיות. מומלץ לאמת את ההשערות מול מומחה ובבדיקות ממוקדות.</div><h3>חוזקות שזוהו</h3><ul>{strengths}</ul><div class="insight-grid">{issues}</div><small>{_esc(insights.methodology_note)}</small></div>'''


def write_report(
    records: list[EvaluationRecord],
    output_dir: Path,
    insights: EvaluationInsights | None = None,
    *,
    show_correct_answer_metrics: bool = True,
    previous_records: list[EvaluationRecord] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(records)
    comparison = build_comparison(records, previous_records) if previous_records else None
    if comparison:
        summary["comparison"] = comparison
    json_path = output_dir / "evaluation_summary.json"
    atomic_write_text(json_path, json.dumps(summary, ensure_ascii=False, indent=2))
    outcome_bars = "".join(_bar(OUTCOME_HEBREW[o], summary["outcomes"].get(o.value, 0), summary["total"], OUTCOME_COLORS[o]) for o in Outcome if summary["outcomes"].get(o.value, 0))
    answer_metrics = summary["answer_claim_metrics"]
    retrieval_metrics = summary["retrieval_claim_metrics"]
    answer_scores = "".join((
        _metric_bar("כיסוי הפרטים הנדרשים בתשובה", answer_metrics["required_point_coverage"], f'{answer_metrics["correct_points_total"]}/{answer_metrics["required_points_total"]} פרטים', "#2563eb"),
        _metric_bar("דיוק בפרטים שהתשובה ניסתה לענות עליהם", answer_metrics["addressed_point_accuracy"], f'{answer_metrics["correct_points_total"]}/{answer_metrics["addressed_points_total"]} פרטים', "#1d4ed8"),
    ))
    retrieval_scores = "".join((
        _metric_bar("כיסוי הפרטים באחזור", retrieval_metrics["required_point_coverage"], f'{retrieval_metrics["points_found_total"]}/{retrieval_metrics["required_points_total"]} פרטים', "#0f766e"),
        _metric_bar("שיעור המקטעים הרלוונטיים", retrieval_metrics["relevant_chunk_rate"], f'{retrieval_metrics["relevant_chunks_total"]}/{retrieval_metrics["chunks_total"]} מקטעים', "#0d9488"),
        _metric_bar("שיעור המקטעים ללא סתירה", None if retrieval_metrics["contradictory_chunk_rate"] is None else 1-retrieval_metrics["contradictory_chunk_rate"], f'{retrieval_metrics["contradictory_chunks_total"]} מקטעים סותרים', "#14b8a6"),
    ))
    pipeline_labels = {
        "retrieval_good_answer_good": ("אחזור טוב + תשובה נכונה", "#15803d"),
        "retrieval_good_answer_poor": ("אחזור טוב + כשל ביצירת התשובה", "#dc2626"),
        "retrieval_poor_answer_good": ("אחזור חלש + תשובה נכונה", "#ca8a04"),
        "retrieval_poor_answer_poor": ("אחזור חלש + תשובה לא תקינה", "#ea580c"),
        "retrieval_unavailable": ("אין נתוני אחזור", "#64748b"),
    }
    pipeline_bars = "".join(_bar(label, summary["pipeline"].get(key, 0), summary["pipeline_evaluable"], color) for key, (label, color) in pipeline_labels.items())
    correct_topic_cell = lambda data: (
        f'<td>{_pct_count_ci(data["correct_rate"], data["correct"], data["answer_tasks"], data["correct_interval_95"])}</td>'
        if show_correct_answer_metrics else ""
    )
    sample_warning = lambda data: '<small class="sample-warning">מדגם קטן</small>' if data["small_sample_warning"] else ""
    topic_rows = "".join(
        f'<tr><td>{_esc(topic)}{sample_warning(data)}</td><td>{data["total"]}</td>'
        f'{correct_topic_cell(data)}'
        f'<td>{_pct_count_ci(data["useful_rate"], data["useful"], data["evaluable"], data["useful_interval_95"])}</td>'
        f'<td>{_pct_count_ci(data["risky_rate"], data["risky"], data["evaluable"], data["risky_interval_95"])}</td>'
        f'<td>{_pct_count_ci(data["good_retrieval_rate"], data["good_retrieval"], data["retrieval_evaluated"], data["good_retrieval_interval_95"])}</td></tr>'
        for topic, data in summary["by_topic"].items()
    )
    paired = summary["paired_variants"]
    form_labels = {"natural_user": "ניסוח טבעי", "ambiguous": "ניסוח הדורש הבהרה"}
    paired_rows = "".join(
        f'<tr><td>{form_labels.get(form, _esc(form))}</td><td>{data["pairs"]}</td>'
        f'<td>{_pct(data["variant_success_rate"])}</td><td>{_pct(data["robustness_when_parent_succeeds"])}</td>'
        f'<td>{_pct(data["degradation_rate"])}</td></tr>'
        for form, data in paired["by_form"].items()
    )
    paired_panel = "" if not paired["pairs"] else f'''<div class="panel"><h2>עמידות לניסוחי משתמש</h2><div class="note">המדדים משווים כל וריאציה רק לשאלת המקור שלה. <b>שימור הצלחה</b> הוא שיעור הווריאציות שהצליחו כאשר שאלת המקור הצליחה; <b>הידרדרות</b> היא המקרה ההפוך.</div><div class="cards"><div class="card">זוגות שנבדקו<b>{paired["pairs"]}</b></div><div class="card">הצלחת וריאציות<b>{_pct(paired["variant_success_rate"])}</b></div><div class="card">שימור הצלחה<b>{_pct(paired["robustness_when_parent_succeeds"])}</b></div><div class="card">הידרדרות בניסוח<b>{_pct(paired["degradation_rate"])}</b></div></div><table><thead><tr><th>סוג וריאציה</th><th>זוגות</th><th>הצלחת וריאציה</th><th>שימור הצלחה</th><th>הידרדרות</th></tr></thead><tbody>{paired_rows}</tbody></table></div>'''
    details = "".join(_details(record) for record in records)
    outcome_options = "".join(f'<option value="{o.value}">{_esc(OUTCOME_HEBREW[o])}</option>' for o in Outcome if o.value in summary["outcomes"])
    topic_options = "".join(f'<option value="{_esc(topic)}">{_esc(topic)}</option>' for topic in summary["by_topic"])
    insights_html = _insights_block(insights)
    comparison_html = _comparison_block(comparison)
    correct_answer_card = (
        f'<div class="card">שיעור תשובות נכונות<b>{_pct(summary["correct_answer_rate_on_answerable"])}</b>'
        '<small>מתוך שאלות שניתנות למענה</small></div>'
        if show_correct_answer_metrics else ""
    )
    correct_topic_header = "<th>תשובות נכונות</th>" if show_correct_answer_metrics else ""
    behavior_cards = "".join((
        f'<div class="card">הבהרות מתאימות<b>{_pct(summary["clarification_success_rate"])}</b><small>{summary["clarification_tasks"]} שאלות הבהרה</small></div>' if summary["clarification_tasks"] else "",
        f'<div class="card">הימנעות נכונה<b>{_pct(summary["abstention_success_rate"])}</b><small>{summary["abstention_tasks"]} שאלות ללא מענה נתמך</small></div>' if summary["abstention_tasks"] else "",
    ))
    document = f'''<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>דוח הערכת צ׳אטבוט</title>
<style>:root{{--ink:#172033;--muted:#64748b;--line:#dbe3ef;--panel:#fff;--bg:#f4f7fb}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px Arial,"Noto Sans Hebrew",sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin:0 0 4px}}h2{{margin-top:0}}.subtitle{{color:var(--muted);margin-bottom:24px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0 24px}}.card,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:0 2px 10px #1e293b0a}}.card{{padding:18px}}.card b{{display:block;font-size:25px;margin-top:7px}}.card small,small{{color:var(--muted);display:block;margin-top:3px}}.sample-warning{{color:#b45309;font-weight:bold}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;margin-bottom:16px}}.panel{{padding:20px;overflow:auto;margin-bottom:16px}}.comparison{{border-top:4px solid #2563eb}}.delta{{font-weight:bold;direction:ltr;text-align:right}}.delta.good{{color:#15803d}}.delta.bad{{color:#b91c1c}}.delta.neutral{{color:var(--muted)}}.bar-row,.score-row{{display:grid;grid-template-columns:minmax(145px,1.3fr) 3fr 45px 48px;gap:9px;align-items:center;margin:10px 0}}.score-row{{grid-template-columns:minmax(170px,1.3fr) 3fr 68px minmax(95px,auto)}}.track{{height:12px;background:#e8edf4;border-radius:10px;overflow:hidden}}.track i{{height:100%;display:block;border-radius:10px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}}th{{background:#eef3f8;position:sticky;top:0}}.note{{background:#f8fafc;border-right:4px solid #2563eb;padding:12px 14px;margin:12px 0;border-radius:6px;line-height:1.55}}.insights{{margin-bottom:16px;border-top:4px solid #7c3aed}}.insights .lead{{font-size:17px}}.insight-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin:14px 0}}.insight{{border:1px solid var(--line);border-radius:10px;padding:15px;background:#fcfcff}}.insight h3{{margin:12px 0 6px}}.filters{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}input,select{{border:1px solid #bcc8d8;border-radius:8px;padding:10px;background:white;min-width:190px}}input{{flex:1}}.result{{background:white;border:1px solid var(--line);border-radius:10px;margin:8px 0;overflow:hidden}}summary{{display:grid;grid-template-columns:90px 150px 1fr auto;gap:10px;align-items:center;padding:13px;cursor:pointer}}summary:hover{{background:#f8fafc}}.id,.topic{{color:var(--muted)}}.badge{{color:white;padding:5px 9px;border-radius:999px;font-size:12px;white-space:nowrap}}.detail-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:0 16px}}section{{padding:8px 16px}}h4{{margin:8px 0;color:#334155}}p,pre{{white-space:pre-wrap;line-height:1.55;overflow-wrap:anywhere}}pre{{max-height:260px;overflow:auto;background:#f8fafc;padding:12px;border-radius:8px;font:13px Arial}}.scores{{display:flex;gap:8px;flex-wrap:wrap;padding:10px 16px}}.scores span{{background:#eef4ff;padding:7px;border-radius:7px}}.warn{{color:#b91c1c}}.hidden{{display:none}}@media(max-width:700px){{main{{padding:14px}}.grid{{grid-template-columns:1fr}}summary{{grid-template-columns:1fr}}.detail-grid{{grid-template-columns:1fr}}.score-row{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>דוח הערכת ביצועי הצ׳אטבוט</h1><div class="subtitle">ניתוח איכות התשובות, האחזור והכשלים לאורך צינור ה-RAG</div>
<div class="cards"><div class="card">סה״כ שאלות<b>{summary['total']}</b><small>{summary['evaluable']} ניתנות להערכה</small></div>{correct_answer_card}<div class="card">שיעור תשובות שימושיות<b>{_pct(summary['useful_answer_rate_on_answerable'])}</b><small>מענה נכון, חלקי מועיל או שאלת הבהרה מתאימה</small></div>{behavior_cards}<div class="card">סיכון למידע מטעה<b>{_pct(summary['risky_misinformation_rate'])}</b><small>הזיות, מענה במקום הימנעות או ללא בירור נדרש</small></div><div class="card">אחזור איכותי<b>{_pct(summary['good_retrieval_rate'])}</b><small>{summary['retrieval_evaluated']} שאלות מענה עם נתוני אחזור</small></div><div class="card">כשל יצירה למרות אחזור טוב<b>{summary['generation_failures_despite_good_retrieval']}</b><small>המידע נמצא אך התשובה לא הייתה נכונה</small></div><div class="card">שגיאות תשתית<b>{summary['infrastructure_errors']} ({_pct(summary['infrastructure_error_rate'])})</b><small>מתוך כלל השאלות; לא נכללו במדדי האיכות</small></div></div>
{comparison_html}
<div class="grid"><div class="panel"><h2>התפלגות תוצאות</h2>{outcome_bars}<div class="note"><b>תשובה שימושית</b> היא תשובה נכונה, תשובה חלקית שיש בה מידע נכון או שאלת הבהרה מתאימה כשחסר פרט מהותי. <b>סיכון למידע מטעה</b> כולל הזיה שנשמעת סבירה, מענה לשאלה שהיה נכון להימנע ממנה או תשובה החלטית כשנדרש תחילה בירור.</div></div><div class="panel"><h2>אבחון צינור האחזור והיצירה</h2>{pipeline_bars}<div class="note"><b>אחזור טוב</b> פירושו שכל הפרטים הנדרשים נמצאו, ללא מקטע שסותר את תשובת הייחוס. <b>אחזור חלש</b> פירושו שחסר לפחות פרט נדרש אחד או שנמצא מקטע סותר. אחזור טוב עם תשובה לא תקינה מצביע על כשל בשלב יצירת התשובה.</div></div></div>
<div class="grid"><div class="panel"><h2>מדדי מידע בתשובות</h2>{answer_scores}<div class="note"><b>כיסוי</b> הוא שיעור הפרטים הנדרשים שנענו נכון. <b>דיוק בפרטים שנענו</b> בודק כמה מהפרטים שהצ׳אטבוט ניסה לענות עליהם היו נכונים. בנוסף נמצאו בסך הכול: {answer_metrics['false_claims_total']} טענות שגויות, {answer_metrics['unsupported_claims_total']} טענות לא מבוססות ו-{answer_metrics['extraneous_claims_total']} טענות עודפות.</div></div><div class="panel"><h2>מדדי מידע באחזור</h2>{retrieval_scores}<div class="note"><b>כיסוי המידע</b> הוא מספר הפרטים הנדרשים שנמצאו במקטעים. <b>שיעור מקטעים רלוונטיים</b> מראה כמה מהמקטעים תרמו למענה. נמצאו {retrieval_metrics['irrelevant_chunks_total']} מקטעים לא רלוונטיים ו-{retrieval_metrics['contradictory_chunks_total']} מקטעים סותרים. המדדים מחושבים רק עבור שאלות שבהן סופקו מקטעים.</div></div></div>
{paired_panel}
<div class="panel"><h2>ביצועים לפי נושא</h2><div class="note">כל מדדי הביצוע מוצגים בפורמט <b>אחוז (מונה/מכנה)</b> עם רווח סמך וילסון של 95%. נושאים עם פחות מ-{summary['minimum_topic_sample']} תוצאות מסומנים כמדגם קטן. מדד התשובות הנכונות כולל רק משימות שבהן נדרש מענה עובדתי; הבהרה והימנעות נמדדות בנפרד.</div><table><thead><tr><th>נושא</th><th>שאלות</th>{correct_topic_header}<th>תשובות שימושיות</th><th>סיכון למידע מטעה</th><th>אחזור טוב</th></tr></thead><tbody>{topic_rows}</tbody></table></div>
{insights_html}
<div class="panel" style="margin-top:16px"><h2>פירוט לפי שאלה</h2><div class="filters"><input id="search" placeholder="חיפוש בשאלה, בתשובה או במזהה"><select id="topic"><option value="">כל הנושאים</option>{topic_options}</select><select id="outcome"><option value="">כל התוצאות</option>{outcome_options}</select></div><div id="visibleCount"></div>{details}</div></main>
<script>const rows=[...document.querySelectorAll('.result')],q=document.getElementById('search'),t=document.getElementById('topic'),o=document.getElementById('outcome'),c=document.getElementById('visibleCount');function f(){{const s=q.value.trim().toLocaleLowerCase('he');let n=0;rows.forEach(r=>{{const show=(!s||r.dataset.search.includes(s))&&(!t.value||r.dataset.topic===t.value)&&(!o.value||r.dataset.outcome===o.value);r.classList.toggle('hidden',!show);if(show)n++}});c.textContent='מוצגות '+n+' מתוך '+rows.length+' שאלות'}}[q,t,o].forEach(x=>x.addEventListener('input',f));f();</script></body></html>'''
    html_path = output_dir / "evaluation_report.html"
    atomic_write_text(html_path, document)
    return json_path, html_path
