from __future__ import annotations


SUMMARY_GENERATION_FAILURE = "לא הצלחנו ליצור סיכום לשאילתת החיפוש שלך, אבל כן מצאנו כמה תוצאות."


def placeholder_error(answer: str) -> str:
    """Return a categorized error for known non-answer placeholder responses."""
    normalized = " ".join(answer.split())
    if normalized == SUMMARY_GENERATION_FAILURE:
        return "answer_generation_failed: RAG retrieval succeeded but answer generation returned a placeholder"
    return ""
