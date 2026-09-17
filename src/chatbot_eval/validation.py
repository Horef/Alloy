from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any

from .models import ExpectedBehavior, QuestionForm, SilverQuestion

# Sentence terminators used to derive atomic claims when a question supplies none.
# A period is only treated as a terminator when it is NOT between two digits, so numbers
# like "250,000" or "5.5" and rate markers are not split. Hebrew has no distinct sentence
# punctuation, so the same ASCII terminators plus explicit line breaks are used.
_CLAIM_TERMINATORS = re.compile(r"(?<!\d)[.!?;](?!\d)|\n+")
_BULLET_PREFIX = re.compile(r"^\s*(?:[-*•·]|\d+[.)]|[א-ת][.)])\s*")


def derive_reference_claims(expected_answer: str) -> list[str]:
    """Derive deterministic atomic claims from a reference answer.

    Used only when a question provides no explicit ``reference_claims`` (typically premade
    imports). Splitting the reference answer into per-sentence/per-line claims gives the judge
    finer, claim-level scoring instead of a single all-or-nothing claim, so premade sets are
    scored at a granularity closer to generated silver sets. This never changes the reference
    answer itself; it only produces evaluation units. Falls back to the whole answer as one
    claim when it cannot be meaningfully split.
    """
    pieces: list[str] = []
    for raw in _CLAIM_TERMINATORS.split(expected_answer):
        if raw is None:
            continue
        cleaned = " ".join(_BULLET_PREFIX.sub("", raw).split())
        # Ignore fragments with no letters/digits (stray punctuation, empty bullets).
        if cleaned and re.search(r"[^\W_]", cleaned, re.UNICODE):
            pieces.append(cleaned)
    unique = list(dict.fromkeys(pieces))
    return unique or [" ".join(expected_answer.split())]


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

    Older silver files and premade imports may not contain ``reference_claims``. For answer
    tasks that supply none, atomic claims are derived deterministically from the reference
    answer (see ``derive_reference_claims``) so scoring is claim-level rather than a single
    all-or-nothing claim. Newly generated files already contain reviewable atomic claims and
    are left untouched.
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
            question.reference_claims = list(dict.fromkeys(claims)) or derive_reference_claims(question.expected_answer)
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
