from __future__ import annotations

from collections import Counter

from .models import ExpectedBehavior, QuestionForm, SilverQuestion


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
            question.reference_claims = list(dict.fromkeys(claims)) or [question.expected_answer.strip()]
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
