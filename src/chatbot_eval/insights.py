from __future__ import annotations

import json
import logging
from pathlib import Path

from .llm import StructuredLLM
from .models import EvaluationInsights, EvaluationRecord, Outcome
from .report import build_summary

logger = logging.getLogger(__name__)

INSIGHTS_PROMPT = """Analyze a completed internal chatbot evaluation and produce actionable,
evidence-based insights in clear Hebrew. Look for recurring patterns that explain failures across
questions, topics, retrieval, and answer generation—not just a restatement of outcome counts.

Consider patterns such as: personal/sensitive-data questions, multi-condition rules, dates and
numbers, procedural questions, long or ambiguous questions, missing required details, noisy or
contradictory retrieval, correct retrieval ignored by generation, excessive answers, abstention,
and topic-specific weaknesses. Mention a pattern only when the supplied records support it.

Important safeguards:
- A likely cause is a hypothesis, not a proven causal claim. State uncertainty explicitly.
- Do not infer sensitive attributes or repeat personal values from the data.
- Use question IDs as examples; do not quote identifying personal details.
- evidence_count must reflect supporting records, and example_question_ids must exist below.
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


def generate_insights(records: list[EvaluationRecord], llm: StructuredLLM, model: str) -> EvaluationInsights:
    analyzable = [record for record in records if record.outcome not in {Outcome.CHATBOT_ERROR, Outcome.JUDGE_ERROR}]
    if not analyzable:
        raise ValueError("No successfully judged records are available for insight generation")
    compact = [_compact_record(record) for record in analyzable]
    prompt = INSIGHTS_PROMPT.format(
        summary=json.dumps(build_summary(records), ensure_ascii=False),
        records=json.dumps(compact, ensure_ascii=False),
    )
    logger.info("insight_generation_started record_count=%d model=%s", len(analyzable), model)
    insights = llm.generate(prompt, EvaluationInsights, model)
    known_ids = {record.question.id for record in records}
    for issue in insights.issues:
        issue.example_question_ids = [question_id for question_id in issue.example_question_ids if question_id in known_ids][:8]
    logger.info("insight_generation_completed issue_count=%d", len(insights.issues))
    return insights


def write_insights(insights: EvaluationInsights, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "evaluation_insights.json"
    path.write_text(insights.model_dump_json(indent=2), encoding="utf-8")
    return path
