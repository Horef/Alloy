from __future__ import annotations


SUMMARY_GENERATION_FAILURE = "לא הצלחנו ליצור סיכום לשאילתת החיפוש שלך, אבל כן מצאנו כמה תוצאות."


def placeholder_error(answer: str) -> str:
    """Return a categorized error for known non-answer placeholder responses."""
    normalized = " ".join(answer.split())
    if normalized == SUMMARY_GENERATION_FAILURE:
        return "answer_generation_failed: RAG retrieval succeeded but answer generation returned a placeholder"
    return ""


def bounded_retry_delay(retry_after, attempt: int, base_seconds: float, cap: float) -> float:
    """Bound server-directed and exponential retry waits; invalid hints use fallback."""
    import math
    import random
    from datetime import datetime, timezone
    from email.utils import parsedate_to_datetime

    if retry_after is not None:
        try:
            seconds = float(retry_after)
        except (TypeError, ValueError):
            try:
                date = parsedate_to_datetime(retry_after)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                seconds = (date - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                seconds = float("nan")
        if math.isfinite(seconds) and seconds >= 0:
            return min(cap, seconds)
    base = min(cap, base_seconds * (2 ** min(attempt, 100)))
    return min(cap, base + random.uniform(0, base * 0.25))
