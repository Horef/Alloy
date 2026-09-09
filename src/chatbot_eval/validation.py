from __future__ import annotations

from collections import Counter
import json
from typing import Any

from .models import ExpectedBehavior, QuestionForm, SilverQuestion


def parse_bool(value: Any, default: bool = True) -> bool:
    """Accept booleans and true/false, 1/0, yes/no, y/n (case insensitive)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError("Invalid answerable value; expected true/false, 1/0, yes/no, or y/n")


def normalize_question_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply explicit legacy defaults identically at each dataset input boundary."""
    data = dict(payload)
    behavior = data.get("expected_behavior")
    default_answerable = behavior != ExpectedBehavior.ABSTAIN
    data["answerable"] = parse_bool(data.get("answerable"), default_answerable)
    data["expected_behavior"] = behavior or ("answer" if data["answerable"] else "abstain")
    data["question_form"] = data.get("question_form") or ("ambiguous" if data["expected_behavior"] == "clarify" else "canonical")
    for name in ("reference_claims", "supporting_quotes", "sources"):
        value = data.get(name)
        if isinstance(value, str):
            try:
                value = json.loads(value) if value.strip() else []
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid {name} JSON") from exc
        data[name] = [] if value is None else value
    for name, default in (("question_type", "basic_knowledge"), ("difficulty", "medium"), ("review_status", "pending")):
        data[name] = data.get(name) or default
    return data


def validate_question_set(questions: list[SilverQuestion]) -> list[SilverQuestion]:
    """Validate and normalize questions at every external input boundary.

    Older silver files did not contain ``reference_claims``. For answer tasks, their
    complete reference answer becomes one stable coarse-grained claim. Newly generated
    files contain reviewable atomic claims.
    """
    if not questions:
        return questions

    duplicate_ids = sorted(identifier for identifier, count in Counter(q.id for q in questions).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"Duplicate question IDs are not allowed: {duplicate_ids}")

    errors: list[str] = []
    for question in questions:
        prefix = f"Question {question.id!r}"
        if not question.id.strip():
            errors.append("Question ID must not be empty")
        if not question.question.strip():
            errors.append(f"{prefix} has empty question text")
        if not question.expected_answer.strip():
            errors.append(f"{prefix} has an empty expected answer")

        if question.expected_behavior == ExpectedBehavior.ANSWER:
            if not question.answerable:
                errors.append(f"{prefix} expects an answer but answerable=false")
            claims = [" ".join(claim.split()) for claim in question.reference_claims if claim.strip()]
            question.reference_claims = list(dict.fromkeys(claims)) or [" ".join(question.expected_answer.split())]
        elif question.expected_behavior == ExpectedBehavior.CLARIFY:
            if not question.answerable or question.question_form != QuestionForm.AMBIGUOUS:
                errors.append(f"{prefix} expects clarification but is not an answerable ambiguous question")
            if question.reference_claims:
                errors.append(f"{prefix} is a clarification task and must not contain factual reference claims")
        elif question.expected_behavior == ExpectedBehavior.ABSTAIN:
            if question.answerable:
                errors.append(f"{prefix} expects abstention but answerable=true")
            if question.reference_claims:
                errors.append(f"{prefix} is an abstention task and must not contain factual reference claims")

    if errors:
        shown = "; ".join(errors[:10])
        remainder = len(errors) - 10
        raise ValueError(shown + (f"; and {remainder} more validation errors" if remainder > 0 else ""))
    return questions
