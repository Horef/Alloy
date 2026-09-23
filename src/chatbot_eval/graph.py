"""Corpus knowledge graph for grounded question generation.

Instead of selecting evidence for a question by lexical overlap with a broad topic label,
Alloy builds a knowledge graph over the corpus: nodes are chunks enriched with extracted
signals (entities, keyphrases, a one-line summary), and edges are deterministic typed
relationships between chunks that share entities, overlap on keyphrases, or sit adjacently in
the same document. Question evidence is then assembled from a connected *cluster* of nodes, so
facts that live in different chunks but belong together (same unit, same benefit, same
procedure) are pulled in together instead of being ranked out.

This module is deliberately dependency-light and fully deterministic apart from the LLM signal
extraction (which lives in ``graph_build``). Edge building and topic derivation use only the
extracted signals, so they are cheap, reproducible, and cache-friendly.

Hebrew note: entity matching is done on a *canonical* form that tolerates the spelling
variations common in Hebrew corpora -- attached one-letter prefixes (ו/ה/ב/ל/כ/מ/ש), the
gershayim/geresh used in acronyms (צה"ל / צה״ל), niqqud, and Hebrew/Latin case/width -- while the
original surface form is preserved for display and provenance.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Literal

from .documents import Chunk

# Relationship edge types. ``same_document`` guarantees no node is evidence-starved even when a
# corpus is entity-sparse; the other two capture semantic relatedness across chunk boundaries.
EdgeType = Literal["shared_entity", "keyphrase_overlap", "same_document"]

# Hebrew attached prefixes that are frequently glued to a following word. Stripping a single
# leading prefix lets "ולצה\"ל" match "צה\"ל". We strip conservatively (one prefix, and only when
# a reasonable stem remains) to avoid over-merging distinct words.
_HEBREW_PREFIXES = ("ו", "ה", "ב", "ל", "כ", "מ", "ש")
_HEBREW_LETTER = re.compile(r"[\u0590-\u05FF]")
_NIQQUD = re.compile(r"[\u0591-\u05C7]")
_GERSHAYIM = re.compile(r"[\"'\u05F3\u05F4\u2018\u2019\u201C\u201D`]")
_NON_WORD = re.compile(r"[^\w\u0590-\u05FF]+", re.UNICODE)


def normalize_entity(surface: str) -> str:
    """Canonicalize an entity/keyphrase surface form losslessly for tolerant matching.

    The canonical form is unicode-normalized (NFKC), niqqud- and gershayim-stripped, lower-cased,
    and whitespace-collapsed. It deliberately does NOT strip Hebrew prefixes, because many base
    words legitimately begin with a prefix letter (e.g. שכר, מעטפת) and blind stripping corrupts
    them or causes false merges. Prefix tolerance is handled separately by :func:`entity_match_keys`,
    which only removes an attached prefix from multi-token phrases or gershayim acronyms where it is
    safe. Returns an empty string when nothing meaningful remains (callers must skip empties).
    """
    text = unicodedata.normalize("NFKC", surface)
    text = _NIQQUD.sub("", text)
    text = _GERSHAYIM.sub("", text)
    text = text.casefold().strip()
    text = _NON_WORD.sub(" ", text).strip()
    return text


def _strip_one_prefix(token: str) -> str | None:
    """Return ``token`` with a single leading Hebrew prefix removed, or None if not safely strippable.

    Only strips when the token starts with a prefix letter, is long enough that a real stem
    remains, and contains a gershayim-style acronym marker OR the two-letter prefix cluster ``ול``
    / ``וה`` / ``ומ`` etc. This is intentionally narrow: it targets the frequent "attached
    conjunction/preposition on a named thing" case (ולצה"ל, ולמעטפת) without touching ordinary
    base words like שכר or מעטפת whose leading letter is part of the word.
    """
    if len(token) < 4 or token[0] not in _HEBREW_PREFIXES:
        return None
    remainder = token[1:]
    # Peel a second stacked prefix (e.g. ו+ל) so "ולצהל" -> "צהל".
    if len(remainder) >= 4 and remainder[0] in _HEBREW_PREFIXES:
        remainder = remainder[1:]
    return remainder if len(remainder) >= 2 else None


def entity_match_keys(surface: str) -> set[str]:
    """Matching keys for an entity: its canonical form plus conservative prefix-stripped variants.

    Two entities are considered the same when their key sets intersect. Because the base canonical
    form is always included, this never *loses* an exact match; the extra variants only *add*
    tolerance for attached Hebrew prefixes on acronyms and multi-token names. Ordinary single base
    words are left untouched, so שכר and מעטפת are never corrupted.
    """
    canonical = normalize_entity(surface)
    if not canonical:
        return set()
    keys = {canonical}
    tokens = canonical.split()
    # Only attempt prefix stripping on the leading token, and only for acronym-like or multi-token
    # entities where an attached prefix is the likely explanation for a mismatch.
    is_acronym_like = bool(_GERSHAYIM.search(surface))
    if is_acronym_like or len(tokens) > 1:
        stripped = _strip_one_prefix(tokens[0])
        if stripped:
            keys.add(" ".join([stripped, *tokens[1:]]))
    return keys


def _match_key_set(values: list[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        result |= entity_match_keys(value)
    return result


def _canonical_set(values: list[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        canonical = normalize_entity(value)
        if canonical:
            result.add(canonical)
    return result


@dataclass
class GraphNode:
    """A chunk enriched with extracted signals. ``content_sha256`` addresses the owning
    document's subgraph so incremental rebuilds can reuse unchanged extractions."""

    chunk_id: str
    file: str
    location: str
    text: str
    entities: list[str] = field(default_factory=list)
    keyphrases: list[str] = field(default_factory=list)
    summary: str = ""
    theme: str = ""

    def to_chunk(self) -> Chunk:
        return Chunk(self.chunk_id, self.file, self.location, self.text)

    @property
    def entity_keys(self) -> set[str]:
        """Match keys for this node's entities (canonical + safe prefix-stripped variants)."""
        return _match_key_set(self.entities)

    @property
    def canonical_keyphrases(self) -> set[str]:
        return _canonical_set(self.keyphrases)


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    type: EdgeType
    weight: float
    shared: tuple[str, ...] = ()


@dataclass
class GraphTopic:
    """A graph-derived topic: a cluster of related nodes with a prevalence-based importance.

    Importance (1-5) is derived from how much of the corpus the cluster covers and how densely
    its nodes are connected, preserving Alloy's prevalence-driven quota philosophy while grounding
    it in graph structure rather than a lexical label match.
    """

    name: str
    description: str
    importance: int
    node_ids: list[str]


@dataclass
class KnowledgeGraph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    # ``GraphBundle`` (below) pairs a graph with its derived topics for caching/return.

    def node_by_id(self) -> dict[str, GraphNode]:
        return {node.chunk_id: node for node in self.nodes}

    def neighbors(self) -> dict[str, list[GraphEdge]]:
        """Adjacency as an id -> incident-edges map (each undirected edge appears on both ends)."""
        adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
        for edge in self.edges:
            adjacency[edge.source_id].append(edge)
            adjacency[edge.target_id].append(edge)
        return adjacency


@dataclass
class GraphBundle:
    """A knowledge graph paired with its derived, labeled topics. This is the unit that the
    generator consumes and the cache stores, so a rebuilt graph and its topics never drift apart."""

    graph: KnowledgeGraph
    topics: list[GraphTopic]


def build_edges(
    nodes: list[GraphNode],
    *,
    keyphrase_overlap_threshold: float = 0.3,
    max_shared_entity_pairs: int | None = None,
) -> list[GraphEdge]:
    """Build deterministic typed edges from node signals only (no LLM, no embeddings).

    - ``shared_entity``: two nodes sharing >=1 canonical entity. Built via an inverted index
      (entity -> nodes) so cost is proportional to co-occurrences, not to N^2. Weight = number of
      shared entities.
    - ``keyphrase_overlap``: Jaccard similarity of canonical keyphrase sets >= threshold. Only
      evaluated for node pairs that already co-occur on some entity or keyphrase, again via an
      inverted index, so we never materialize the full N^2 pair space.
    - ``same_document``: consecutive chunks of the same file (by chunk order), so no node is left
      isolated and document-wide questions always have a spine to walk.
    """
    edges: list[GraphEdge] = []
    entity_keys = {node.chunk_id: node.entity_keys for node in nodes}
    canonical_keyphrases = {node.chunk_id: node.canonical_keyphrases for node in nodes}

    # Inverted index: entity match-key -> node ids that carry it (canonical + prefix variants).
    entity_index: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for key in entity_keys[node.chunk_id]:
            entity_index[key].append(node.chunk_id)

    # Candidate pairs come only from shared entities/keyphrases (sparse), never all N^2.
    shared_entity_counts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entity, members in entity_index.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair = (members[i], members[j]) if members[i] < members[j] else (members[j], members[i])
                shared_entity_counts[pair].add(entity)

    for pair, shared in sorted(shared_entity_counts.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if max_shared_entity_pairs is not None and len(edges) >= max_shared_entity_pairs:
            break
        edges.append(GraphEdge(pair[0], pair[1], "shared_entity", float(len(shared)), tuple(sorted(shared))))

    # Keyphrase overlap only among pairs that already share an entity or a keyphrase token.
    keyphrase_index: dict[str, set[str]] = defaultdict(set)
    for node in nodes:
        for phrase in canonical_keyphrases[node.chunk_id]:
            keyphrase_index[phrase].add(node.chunk_id)
    candidate_pairs: set[tuple[str, str]] = set(shared_entity_counts)
    for members in keyphrase_index.values():
        members_list = sorted(members)
        for i in range(len(members_list)):
            for j in range(i + 1, len(members_list)):
                candidate_pairs.add((members_list[i], members_list[j]))
    for source, target in sorted(candidate_pairs):
        a, b = canonical_keyphrases[source], canonical_keyphrases[target]
        if not a or not b:
            continue
        union = a | b
        jaccard = len(a & b) / len(union) if union else 0.0
        if jaccard >= keyphrase_overlap_threshold:
            edges.append(GraphEdge(source, target, "keyphrase_overlap", round(jaccard, 4), tuple(sorted(a & b))))

    # Same-document spine: consecutive chunks of one file.
    by_file: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        by_file[node.file].append(node.chunk_id)
    for file_ids in by_file.values():
        for previous, current in zip(file_ids, file_ids[1:]):
            edges.append(GraphEdge(previous, current, "same_document", 1.0, ()))

    return edges


def _clustering_edges(edges: list[GraphEdge], min_edge_weight: float) -> list[GraphEdge]:
    """Edges eligible to merge two nodes into one topic cluster.

    ``same_document`` edges are excluded (chaining adjacent chunks would collapse a whole document
    into one topic). Remaining semantic edges must carry at least ``min_edge_weight`` relatedness.
    Requiring, for example, two shared entities (weight >= 2) rather than one prevents a single
    common "hub" term from transitively fusing the whole corpus into one giant cluster -- the
    classic knowledge-graph hairball. Weak single-link edges still exist in the graph for evidence
    assembly; they simply do not drive clustering.
    """
    return [
        edge for edge in edges
        if edge.type != "same_document" and edge.weight >= min_edge_weight
    ]


def _components_from_edges(node_ids: list[str], edges: list[GraphEdge]) -> list[list[str]]:
    parent = {node_id: node_id for node_id in node_ids}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    present = set(node_ids)
    for edge in edges:
        if edge.source_id in present and edge.target_id in present:
            ra, rb = find(edge.source_id), find(edge.target_id)
            if ra != rb:
                parent[ra] = rb
    groups: dict[str, list[str]] = defaultdict(list)
    for node_id in node_ids:
        groups[find(node_id)].append(node_id)
    return [members for _, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))]


def _split_oversized(cluster: list[str], edges: list[GraphEdge], max_size: int) -> list[list[str]]:
    """Split a cluster that exceeds ``max_size`` into smaller, more strongly-connected groups.

    Dense knowledge-graph clusters (e.g. a corpus where nearly every document shares a few hub
    entities) do not disconnect when a single weak edge is removed, so a naive Girvan-Newman step
    stalls. Instead this progressively raises the internal edge-weight threshold -- keeping only the
    strongest links -- until the component fragments, then recurses into any part still too large.
    Because higher shared-entity weight means more entities in common, the surviving groups are the
    most semantically cohesive sub-topics. Deterministic; the small node counts Alloy produces keep
    this cheap even though the graph can be edge-dense.
    """
    if len(cluster) <= max_size:
        return [cluster]
    members = set(cluster)
    internal = [e for e in edges if e.source_id in members and e.target_id in members]
    if not internal:
        return [cluster]
    weights = sorted({e.weight for e in internal})
    # Raise the threshold one weight band at a time; stop at the first threshold that fragments the
    # component (or leaves isolated nodes), so we split at the weakest join rather than shattering.
    for cutoff in weights:
        kept = [e for e in internal if e.weight > cutoff]
        parts = _components_from_edges(cluster, kept)
        if len(parts) > 1:
            result: list[list[str]] = []
            for part in parts:
                result.extend(_split_oversized(part, edges, max_size))
            return result
    # Every edge has the same weight and forms one clique-like block: cannot split by weight.
    return [cluster]


def derive_topic_clusters(
    graph: KnowledgeGraph,
    *,
    min_cluster_nodes: int = 1,
    max_topics: int = 12,
    min_cluster_edge_weight: float = 2.0,
    max_cluster_size: int | None = None,
) -> list[list[str]]:
    """Group nodes into topic clusters by shared meaning, largest (most prevalent) first.

    Clusters are connected components over *strong* semantic edges (weight >=
    ``min_cluster_edge_weight``), which avoids the hairball where one hub entity fuses the whole
    corpus. Oversized clusters are split by raising the internal edge-weight threshold until they
    fragment, so no single topic dominates. When splitting yields more than ``max_topics`` groups,
    the largest are kept as distinct topics and the remaining small ones are bin-packed into a
    bounded number of "other" buckets (each under ``max_cluster_size``), so a dense corpus neither
    collapses into one giant topic nor explodes into many singletons. Prevalence (cluster size)
    drives ordering and, later, importance -- preserving Alloy's quota philosophy.
    """
    node_ids = [node.chunk_id for node in graph.nodes]
    if not node_ids:
        return []
    clustering_edges = _clustering_edges(graph.edges, min_cluster_edge_weight)
    components = _components_from_edges(node_ids, clustering_edges)
    if max_cluster_size is not None:
        split: list[list[str]] = []
        for component in components:
            split.extend(_split_oversized(component, clustering_edges, max_cluster_size))
        components = sorted(split, key=lambda c: (-len(c), c[0]))
    kept = [c for c in components if len(c) >= min_cluster_nodes] or components
    return _cap_topic_count(kept, max_topics=max_topics, max_cluster_size=max_cluster_size)


def _cap_topic_count(
    clusters: list[list[str]], *, max_topics: int, max_cluster_size: int | None,
) -> list[list[str]]:
    """Keep the largest clusters as distinct topics and bin-pack the small-cluster tail.

    When there are more clusters than ``max_topics``, the largest (most prevalent) stay as their own
    topics and the remaining small ones are combined into a bounded number of "other" buckets, each
    under ``max_cluster_size``. Bin-packing the tail (rather than fusing it into one lump) keeps a
    dense corpus from collapsing into one giant "other" topic while avoiding a swarm of singletons.
    """
    clusters = sorted(clusters, key=lambda c: (-len(c), c[0]))
    if len(clusters) <= max_topics:
        return clusters
    head = clusters[: max_topics - 1]
    tail_clusters = clusters[max_topics - 1:]
    if not tail_clusters:
        return head
    cap = max_cluster_size if max_cluster_size is not None else sum(len(c) for c in tail_clusters)
    buckets: list[list[str]] = []
    current: list[str] = []
    for cluster in tail_clusters:  # largest-first, so big residual clusters stay whole
        if current and len(current) + len(cluster) > cap:
            buckets.append(current)
            current = []
        current.extend(cluster)
    if current:
        buckets.append(current)
    return head + buckets


def derive_theme_clusters(
    graph: KnowledgeGraph,
    *,
    max_topics: int = 40,
    max_cluster_size: int | None = None,
    min_cluster_edge_weight: float = 2.0,
) -> list[list[str]]:
    """Group nodes into topics by their assigned *theme* (the high-level layer), largest first.

    This is the theme-first mode: nodes sharing the same controlled-vocabulary theme form one topic,
    so a theme that is spread across many documents (e.g. pay) becomes its own topic even when its
    chunks share few entities. Themes are a separate layer and never create entity edges, so this
    cannot cause the entity hairball. A theme group larger than ``max_cluster_size`` is still split
    by the entity-edge splitter (its internal cohesion), and any nodes without a theme fall back to
    entity-based clustering so nothing is lost. The result is capped/bin-packed like entity mode.
    """
    node_ids = [node.chunk_id for node in graph.nodes]
    if not node_ids:
        return []
    themed: dict[str, list[str]] = defaultdict(list)
    unthemed: list[str] = []
    for node in graph.nodes:
        theme = normalize_entity(node.theme)
        if theme and theme != normalize_entity("אחר"):
            themed[theme].append(node.chunk_id)
        else:
            unthemed.append(node.chunk_id)
    # If themes were never assigned, fall back entirely to entity clustering.
    if not themed:
        return derive_topic_clusters(
            graph, max_topics=max_topics, max_cluster_size=max_cluster_size,
            min_cluster_edge_weight=min_cluster_edge_weight,
        )
    clusters = list(themed.values())
    # Cluster the unthemed / "other" nodes among themselves by entity edges so they are not one lump.
    if unthemed:
        other_edges = _clustering_edges(
            [e for e in graph.edges
             if e.source_id in set(unthemed) and e.target_id in set(unthemed)],
            min_cluster_edge_weight,
        )
        clusters.extend(_components_from_edges(unthemed, other_edges))
    # A very large theme group is split by its internal entity cohesion so it does not dominate.
    if max_cluster_size is not None:
        clustering_edges = _clustering_edges(graph.edges, min_cluster_edge_weight)
        split: list[list[str]] = []
        for cluster in clusters:
            split.extend(_split_oversized(cluster, clustering_edges, max_cluster_size))
        clusters = split
    return _cap_topic_count(clusters, max_topics=max_topics, max_cluster_size=max_cluster_size)


def cluster_importance(cluster_size: int, total_nodes: int) -> int:
    """Map a cluster's corpus coverage to a 1-5 importance, preserving prevalence weighting."""
    if total_nodes <= 0 or cluster_size <= 0:
        return 1
    share = cluster_size / total_nodes
    # Log-spaced thresholds so a handful of dominant clusters get the top marks without a single
    # giant cluster flattening everything to 5.
    if share >= 0.30:
        return 5
    if share >= 0.18:
        return 4
    if share >= 0.09:
        return 3
    if share >= 0.03:
        return 2
    return 1


def cluster_signature(graph: KnowledgeGraph, node_ids: list[str], *, top_k: int = 8) -> dict[str, list[str]]:
    """Summarize a cluster for LLM labeling: its most frequent entities and keyphrases.

    Returns display (surface) forms, ranked by frequency across the cluster's nodes, so the label
    prompt stays small and focused regardless of cluster size.
    """
    by_id = graph.node_by_id()
    entity_counts: Counter = Counter()
    keyphrase_counts: Counter = Counter()
    entity_display: dict[str, str] = {}
    keyphrase_display: dict[str, str] = {}
    for node_id in node_ids:
        node = by_id.get(node_id)
        if not node:
            continue
        for entity in node.entities:
            canonical = normalize_entity(entity)
            if canonical:
                entity_counts[canonical] += 1
                entity_display.setdefault(canonical, entity.strip())
        for phrase in node.keyphrases:
            canonical = normalize_entity(phrase)
            if canonical:
                keyphrase_counts[canonical] += 1
                keyphrase_display.setdefault(canonical, phrase.strip())
    top_entities = [entity_display[c] for c, _ in entity_counts.most_common(top_k)]
    top_keyphrases = [keyphrase_display[c] for c, _ in keyphrase_counts.most_common(top_k)]
    return {"entities": top_entities, "keyphrases": top_keyphrases}


def cluster_evidence_ids(
    graph: KnowledgeGraph,
    seed_ids: list[str],
    *,
    max_nodes: int,
    same_file_only: bool = False,
    cross_file_only: bool = False,
) -> list[str]:
    """Assemble a bounded, connected evidence set for a question, seeded from a cluster.

    Starting from the seed nodes, greedily add the highest-weight neighbors (breadth-first by
    descending edge weight) until ``max_nodes`` is reached. ``same_file_only`` keeps a
    document-wide walk within one file; ``cross_file_only`` prefers neighbors in a different file
    for cross-document questions. This is the graph analogue of the old top-k, but the neighbors
    are chosen by real relatedness rather than topic-label token overlap.
    """
    if max_nodes <= 0:
        return []
    by_id = graph.node_by_id()
    adjacency = graph.neighbors()
    seeds = [sid for sid in seed_ids if sid in by_id]
    if not seeds:
        return []
    selected: list[str] = []
    seen: set[str] = set()
    seed_file = by_id[seeds[0]].file
    frontier: list[tuple[float, str]] = [(math.inf, sid) for sid in seeds]
    while frontier and len(selected) < max_nodes:
        frontier.sort(key=lambda item: item[0], reverse=True)
        _, current = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        node = by_id[current]
        if same_file_only and node.file != seed_file and current not in seeds:
            continue
        selected.append(current)
        for edge in adjacency.get(current, ()):
            other = edge.target_id if edge.source_id == current else edge.source_id
            if other in seen:
                continue
            weight = edge.weight
            if cross_file_only and by_id[other].file != seed_file:
                weight += 1000.0  # prioritize crossing document boundaries
            frontier.append((weight, other))
    return selected
