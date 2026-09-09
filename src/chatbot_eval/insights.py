from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path

from .artifacts import atomic_write_text
from .llm import StructuredLLM
from .models import EvaluationInsights, EvaluationRecord, Outcome
from .report import build_summary

logger = logging.getLogger(__name__)


def insights_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

INSIGHTS_PROMPT = """Analyze a completed internal chatbot evaluation and produce actionable,
evidence-based insights in clear Hebrew. Look for recurring patterns that explain failures across
questions, topics, retrieval, and answer generation—not just a restatement of outcome counts.

Consider patterns such as: personal/sensitive-data questions, multi-condition rules, dates and
numbers, procedural questions, long or ambiguous questions, missing required details, noisy or
contradictory retrieval, correct retrieval ignored by generation, excessive answers, abstention,
and topic-specific weaknesses. Mention a pattern only when the supplied records support it.

Important safeguards:
- Absent context means missing telemetry, not evidence of failed retrieval.
- Without the deployed prompt, do not claim a particular instruction was absent.
- A likely cause is a hypothesis, not a proven causal claim. State uncertainty explicitly.
- Do not infer sensitive attributes or repeat personal values from the data.
- Use question IDs as examples; do not quote identifying personal details.
- example_question_ids must list every record counted as evidence (up to 8), without duplicates.
- evidence_count must equal the number of example_question_ids. affected_topics must be derived only
  from those records. Code will recalculate both fields and discard issues with no valid evidence.
- Prioritize issues by user harm and frequency. Avoid generic advice.
- Recommendations must be concrete and tied to the observed failure stage (data, retrieval,
  prompt/generation, abstention policy, or evaluation set).
- Return at most 6 issues and at most 4 strengths.

AGGREGATE SUMMARY:
{summary}

COMPACT QUESTION RECORDS:
{records}
"""


def _compact_record(record: EvaluationRecord) -> dict:
    scores = record.scores
    return {
        "id": record.question.id,
        "topic": record.question.topic,
        "expected_behavior": record.question.expected_behavior.value,
        "question_form": record.question.question_form.value,
        "context_available": bool(record.result.retrieved_context.strip()),
        "source_evidence_available": bool(record.question.sources or record.question.supporting_quotes),
        "question": record.question.question[:700],
        "expected_answer": record.question.expected_answer[:900],
        "chatbot_answer": record.result.answer[:900],
        "outcome": record.outcome.value,
        "required_details": scores.required_points_total if scores else None,
        "correct_details": scores.answer_points_correct if scores else None,
        "false_claims": scores.answer_false_claims if scores else None,
        "unsupported_claims": scores.answer_unsupported_claims if scores else None,
        "extraneous_claims": scores.answer_extraneous_claims if scores else None,
        "retrieved_details": scores.retrieval_points_found if scores else None,
        "retrieved_chunks": scores.retrieved_chunks_total if scores else None,
        "relevant_chunks": scores.retrieved_chunks_relevant if scores else None,
        "contradictory_chunks": scores.retrieved_chunks_contradictory if scores else None,
        "judge_explanation": scores.explanation[:500] if scores else "",
        "retrieval_explanation": scores.retrieval_explanation[:500] if scores else "",
    }


def generate_insights(
    records: list[EvaluationRecord], llm: StructuredLLM, model: str, *, max_prompt_chars: int = 80_000,
) -> EvaluationInsights:
    analyzable = [record for record in records if record.outcome not in {Outcome.CHATBOT_ERROR, Outcome.JUDGE_ERROR}]
    if not analyzable:
        raise ValueError("No successfully judged records are available for insight generation")
    priority = {
        Outcome.MISLEADING_HALLUCINATION: 0, Outcome.SHOULD_HAVE_ABSTAINED: 0,
        Outcome.MISSING_CLARIFICATION: 0, Outcome.PARTIAL_TOO_LITTLE: 1,
        Outcome.PARTIAL_TOO_MUCH: 1, Outcome.UNRELATED_ANSWER: 1,
        Outcome.INCORRECT_ABSTENTION: 1,
    }
    ordered = sorted(analyzable, key=lambda record: (
        priority.get(record.outcome, 2), record.question.topic, record.question.id,
    ))
    compact: list[dict] = []
    summary = json.dumps(build_summary(records), ensure_ascii=False)
    def render(items):
        coverage = {"analyzable_records": len(analyzable), "included_records": len(items),
                    "omitted_records": len(analyzable) - len(items), "records": items}
        return INSIGHTS_PROMPT.format(summary=summary, records=json.dumps(coverage, ensure_ascii=False))
    for record in ordered:
        item = _compact_record(record)
        if len(render(compact + [item])) <= max_prompt_chars:
            compact.append(item)
    if not compact:
        raise ValueError("max_prompt_chars cannot fit the summary and one complete evidence record")
    prompt = render(compact)
    logger.info(
        "insight_generation_started record_count=%d included_count=%d model=%s",
        len(analyzable), len(compact), model,
    )
    insights = llm.generate(prompt, EvaluationInsights, model)
    included_ids = {item["id"] for item in compact}
    by_id = {record.question.id: record for record in analyzable if record.question.id in included_ids}
    validated_issues = []
    seen_titles: set[str] = set()
    for issue in insights.issues[:6]:
        title_key = " ".join(issue.title.casefold().split())
        if not title_key or title_key in seen_titles:
            continue
        identifiers = list(dict.fromkeys(
            question_id for question_id in issue.example_question_ids if question_id in by_id
        ))[:8]
        if not identifiers:
            logger.warning("insight_issue_discarded title=%r reason=no_valid_evidence", issue.title)
            continue
        issue.example_question_ids = identifiers
        issue.evidence_count = len(identifiers)
        issue.affected_topics = sorted({by_id[identifier].question.topic or "לא סווג" for identifier in identifiers})
        if issue.priority == "high" and issue.evidence_count == 1:
            issue.priority = "medium"
            issue.confidence = "low"
        seen_titles.add(title_key)
        validated_issues.append(issue)
    insights.issues = validated_issues
    insights.strengths = list(dict.fromkeys(item.strip() for item in insights.strengths if item.strip()))[:4]
    logger.info("insight_generation_completed issue_count=%d", len(insights.issues))
    return insights


def write_insights(insights: EvaluationInsights, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "evaluation_insights.json"
    atomic_write_text(path, insights.model_dump_json(indent=2))
    return path
