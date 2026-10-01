"""Heuristic sizing of a question set from what a knowledge base actually contains.

Most users cannot say how many evaluation questions a corpus needs or how to split them. The
planner measures the corpus instead: it counts *information units* (distinct factual statements:
sentences, list items, table rows with at least a few words) per chunk, then derives

- per-topic canonical quotas: every chunk is represented, denser chunks get proportionally more
  questions (``coverage_target`` of their statements at ``claims_per_question`` claims each), and
  each topic gets at least enough questions for a per-topic accuracy estimate with the requested
  ``margin_of_error`` -- but never more than its content can support;
- user variations as a share of canonical questions, and boundary questions as a share of the
  total with a per-topic minimum;
- grouping parameters: topic size cap and evidence-window size so one model call has roughly
  ``questions_per_call`` questions of material.

All numbers land in an editable ``generation_plan.json``, so a person can fine-tune them.
"""

from __future__ import annotations

import math
import re

_STATEMENT_SPLIT = re.compile(r"(?<!\d)[.!?;](?!\d)|\n+|\s[-–•▪]\s|\s\d+[.)]\s|\|")
_WORD = re.compile(r"\w+")
_MARKUP = re.compile(r"\*\*|__|`|#{1,6}\s")
MIN_STATEMENT_WORDS = 4


def information_units(text: str) -> int:
    """Distinct factual statements in ``text``; at least 1 for any nonempty chunk."""
    pieces = _STATEMENT_SPLIT.split(_MARKUP.sub(" ", text))
    seen: set[str] = set()
    for piece in pieces:
        words = _WORD.findall(piece.casefold())
        if len(words) >= MIN_STATEMENT_WORDS:
            seen.add(" ".join(words))
    return max(1, len(seen)) if text.strip() else 0
