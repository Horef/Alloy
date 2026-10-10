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

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass

from pydantic import BaseModel, Field

_STATEMENT_SPLIT = re.compile(r"(?<!\d)[.!?;](?!\d)|\n+|\s[-–•▪]\s|\s\d+[.)]\s|\|")
_WORD = re.compile(r"\w+")
_MARKUP = re.compile(r"\*\*|__|`|#{1,6}\s")
MIN_STATEMENT_WORDS = 4
PLAN_VERSION = 1


def information_units(text: str) -> int:
    """Distinct factual statements in ``text``; at least 1 for any nonempty chunk."""
    pieces = _STATEMENT_SPLIT.split(_MARKUP.sub(" ", text))
    seen: set[str] = set()
    for piece in pieces:
        words = _WORD.findall(piece.casefold())
        if len(words) >= MIN_STATEMENT_WORDS:
            seen.add(" ".join(words))
    return max(1, len(seen)) if text.strip() else 0


@dataclass(frozen=True)
class PlanningParameters:
    """Knobs of the sizing heuristic; defaults aim for representative sets without bulk."""

    # Share of a chunk's statements its questions should touch; 1.0 would aim at every statement.
    coverage_target: float = 0.35
    # Average atomic claims per canonical question (measured ~3.2-3.3 on hova/keva).
    claims_per_question: float = 3.2
    # Half-width of a 95% confidence interval for each topic's accuracy; sets a per-topic floor.
    margin_of_error: float = 0.20
    # Floors never ask a topic to cover more than this share of its statements.
    max_topic_coverage: float = 0.6
    # Natural/ambiguous variants per canonical question (0.5 = one natural twin for ~1/3 of canonical
    # questions plus ambiguous ones for ~1/6), and the ambiguous share of variants.
    variation_share: float = 0.5
    ambiguous_share: float = 0.33
    # Boundary (unanswerable) share of the whole set, with a per-topic minimum.
    boundary_share: float = 0.10
    min_boundary_per_topic: int = 2
    questions_per_call: int = 8
    # "user_facing": every canonical and boundary question also gets one natural_user phrasing.
    mode: str = "standard"
    # Corpus + population broad questions when enabled (one per topic is added); -1 = disabled.
    broad_estimate: int = -1


class TopicPlan(BaseModel):
    name: str
    chunks: int
    information_units: int
    canonical: int
    boundary: int
    reason: str = Field(description="Which rule set the canonical quota")


class GenerationPlan(BaseModel):
    """Editable output of the planner; `generate --plan` uses it as the exact budget."""

    version: int = PLAN_VERSION
    corpus: dict
    parameters: dict
    graph: dict = Field(description="Graph settings the topics below were derived with")
    totals: dict
    topics: list[TopicPlan]
    topics_fingerprint: str
    rationale: list[str]
    # Theme vocabulary the topics were built with ({"name", "description"} items). `generate --plan`
    # reuses it instead of deriving a new one, and `plan --themes-from` carries it to a new plan, so
    # document edits re-tag only the documents that changed. Empty in entity mode.
    theme_vocabulary: list[dict] = Field(default_factory=list)


def suggest_graph_parameters(chunk_count: int, questions_per_chunk: float = 2.0, questions_per_call: int = 8) -> dict:
    """Topic granularity and window size that suit a corpus of ``chunk_count`` chunks.

    About sqrt(chunks) topics keeps both small and large corpora legible (hova ~9, keva ~15);
    a theme above twice the average topic size is split. Windows hold roughly one call's worth
    of questions, so each call focuses on a few chunks.
    """
    target_topics = min(30, max(6, round(math.sqrt(max(1, chunk_count)))))
    max_cluster_size = min(40, max(12, math.ceil(2 * chunk_count / target_topics)))
    window = min(12, max(3, round(questions_per_call / max(0.5, questions_per_chunk))))
    return {
        "target_topics": target_topics,
        "max_cluster_size": max_cluster_size,
        "max_graph_topics": min(60, max(10, math.ceil(target_topics * 2.5))),
        "max_cluster_nodes": window,
    }


def topics_fingerprint(topics) -> str:
    """Identity of a topic partition, so a plan is only applied to the graph it was made for."""
    payload = [[topic.name, sorted(topic.node_ids)] for topic in topics]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def _precision_floor(margin: float) -> int:
    return math.ceil(1.96 ** 2 * 0.25 / margin ** 2) if margin > 0 else 0


def _largest_remainder(weights: dict[str, float], total: int, minimum: int) -> dict[str, int]:
    names = list(weights)
    base = {name: minimum for name in names}
    remaining = total - minimum * len(names)
    if remaining <= 0:
        return base
    weight_total = sum(weights.values()) or 1.0
    raw = {name: remaining * weights[name] / weight_total for name in names}
    for name in names:
        base[name] += math.floor(raw[name])
    leftover = total - sum(base.values())
    for name in sorted(names, key=lambda n: (math.floor(raw[n]) - raw[n], n))[:leftover]:
        base[name] += 1
    return base


def build_plan(
    chunks, topics, parameters: PlanningParameters, *, graph: dict, theme_vocabulary: list[dict] | None = None,
) -> GenerationPlan:
    """Size a question set for this corpus and topic partition."""
    units = {chunk.id: information_units(chunk.text) for chunk in chunks}
    files = {chunk.file for chunk in chunks}
    floor = _precision_floor(parameters.margin_of_error)
    topic_plans: list[TopicPlan] = []
    for topic in topics:
        topic_units = [units.get(node_id, 0) for node_id in topic.node_ids]
        content = sum(
            max(1, round(u * parameters.coverage_target / parameters.claims_per_question)) for u in topic_units if u
        )
        ceiling = max(1, math.floor(sum(topic_units) * parameters.max_topic_coverage / parameters.claims_per_question))
        floored = min(floor, ceiling)
        canonical, reason = (floored, "per-topic precision floor") if floored > content else (
            content, "content: every chunk represented, denser chunks covered more")
        topic_plans.append(TopicPlan(
            name=topic.name, chunks=len(topic.node_ids), information_units=sum(topic_units),
            canonical=canonical, boundary=0, reason=reason,
        ))
    canonical_total = sum(plan.canonical for plan in topic_plans)
    variations = round(canonical_total * parameters.variation_share)
    ambiguous = round(variations * parameters.ambiguous_share)
    boundary_minimum = parameters.min_boundary_per_topic * len(topic_plans)
    share = parameters.boundary_share
    boundary = max(boundary_minimum, round(share * (canonical_total + variations) / (1 - share))) if share < 1 else 0
    boundary_quotas = _largest_remainder(
        {plan.name: float(plan.information_units) for plan in topic_plans}, boundary,
        parameters.min_boundary_per_topic if boundary >= boundary_minimum else 0,
    )
    for plan in topic_plans:
        plan.boundary = boundary_quotas.get(plan.name, 0)
    user_facing = parameters.mode == "user_facing"
    natural = canonical_total + boundary if user_facing else variations - ambiguous
    # Broad questions are an upper bound: population questions exist only for populations the documents address.
    broad = parameters.broad_estimate + len(topic_plans) if parameters.broad_estimate >= 0 else 0
    total = canonical_total + boundary + natural + ambiguous + broad
    unit_values = sorted(units.values())
    return GenerationPlan(
        corpus={
            "documents": len(files), "chunks": len(chunks), "information_units": sum(unit_values),
            "units_per_chunk_median": unit_values[len(unit_values) // 2] if unit_values else 0,
            "topics": len(topic_plans),
        },
        parameters=asdict(parameters),
        graph=graph,
        totals={
            "canonical": canonical_total, "variations": variations, "natural_user": natural,
            "ambiguous": ambiguous, "boundary": boundary, "broad_max": broad, "total": total,
            "user_facing": natural + ambiguous + broad if user_facing else total,
            "estimated_model_calls": _estimated_calls(canonical_total, natural + ambiguous, boundary, parameters),
        },
        topics=topic_plans,
        topics_fingerprint=topics_fingerprint(topics),
        theme_vocabulary=list(theme_vocabulary or []),
        rationale=[
            f"Every chunk gets at least one canonical question; a chunk with u statements gets "
            f"round(u x {parameters.coverage_target} / {parameters.claims_per_question}).",
            f"Each topic gets at least {floor} canonical questions (95% CI half-width "
            f"{parameters.margin_of_error:.0%} on its accuracy) unless its content supports fewer "
            f"(at most {parameters.max_topic_coverage:.0%} of its statements).",
            f"Variations are {parameters.variation_share:.0%} of canonical questions "
            f"({parameters.ambiguous_share:.0%} of them ambiguous); boundary questions are "
            f"{parameters.boundary_share:.0%} of the set with at least {parameters.min_boundary_per_topic} per topic.",
        ] + ([
            "user_facing mode: every canonical and boundary question also gets one natural_user phrasing; "
            "the canonical forms become anchors, so the user-facing set is natural_user + ambiguous.",
        ] if user_facing else []),
    )


def _estimated_calls(canonical: int, variations: int, boundary: int, parameters: PlanningParameters) -> int:
    generation = math.ceil(1.3 * canonical / parameters.questions_per_call)
    verification = round(1.3 * canonical * 3)
    return generation + verification + math.ceil(variations / 10) + math.ceil(1.3 * boundary * 1.2)


def plan_quotas(plan: GenerationPlan, topics) -> tuple[tuple[int, int, int], tuple, tuple]:
    """Budgets and per-topic quotas from a plan, after checking it matches the current topics."""
    if plan.version != PLAN_VERSION:
        raise ValueError(f"Unsupported generation plan version {plan.version}")
    if plan.topics_fingerprint != topics_fingerprint(topics):
        raise ValueError(
            "The knowledge graph's topics changed since this plan was made (documents or graph settings "
            "differ); run `plan` again."
        )
    canonical = tuple((topic.name, topic.canonical) for topic in plan.topics)
    boundary = tuple((topic.name, topic.boundary) for topic in plan.topics)
    budgets = (sum(q for _, q in canonical), int(plan.totals["variations"]), sum(q for _, q in boundary))
    return budgets, canonical, boundary
