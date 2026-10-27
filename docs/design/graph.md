# 7. The corpus knowledge graph (`graph_build.py`, `graph.py`)

Status: current (verified 2026-10-27)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md). Citations such as [R1] are listed in [references.md](references.md).

## 7.1 Why a graph at all

Before v1.0 a question's evidence was selected by lexical overlap with a broad topic label. Facts that
belong together but live in different chunks (the eligibility rule in one section, the amount in a
table elsewhere, the procedure in another document) were ranked out, so reference answers were
incomplete. The graph makes "relatedness" a first-class, inspectable structure: evidence for a
question is a *connected neighborhood* of chunks, and topics are *regions* of the graph rather than
keyword matches. This is the same motivation as GraphRAG [R1], but Alloy keeps the graph much
simpler: nodes are chunks (not extracted entities), edges are deterministic, and clustering is
union-find plus weight thresholds instead of Leiden community detection [R14] *(inferred: Leiden
needs an extra dependency and its output is harder to explain to reviewers; the corpora here have
hundreds, not millions, of nodes)*.

## 7.2 Node signals: the only LLM step in graph building

For each chunk, one model call per batch of `graph_extraction_batch_chunks` chunks returns:

- `entities`: named things as written in the text (units, roles, forms, benefits, programs,
  places), without translating or expanding acronyms;
- `keyphrases`: 3-8 short topical noun phrases;
- `summary`: one sentence.

Chunks longer than 4,000 characters are truncated for extraction only. Extraction is **per
document** so its result can be cached by that document's content (section 9). Bad or missing items
degrade to empty signals for that chunk; a failed call raises `IncompleteExtraction`, which carries
the partial signals so the run continues but prevents them from being cached as the document's
permanent signals.

## 7.3 Hebrew-tolerant entity matching

`normalize_entity` canonicalizes a surface form (NFKC, remove niqqud and gershayim/geresh/quotes,
casefold, collapse punctuation to spaces). `entity_match_keys` then adds a prefix-stripped variant,
but **only** for acronym-like entities (the original contained a gershayim mark) or multi-word
entities, and only for the first token: one prefix letter from ו/ה/ב/ל/כ/מ/ש is removed from a token of
four or more letters, plus a second stacked prefix when four letters still remain (so `ולצה"ל`
matches `צה"ל`). Two entities match when their key sets
intersect. Single-word entities keep only their canonical form, because many base words start with a
prefix letter (`שכר`, `מעטפת`) and blind stripping would corrupt them. This is "light stemming", the
approach that works well for IR in Semitic languages without a full analyzer [R25].

## 7.4 Edges

`build_edges` creates three edge types, all deterministic:

| Type                | Rule                                                                                            | Weight                 |
| ------------------- | ----------------------------------------------------------------------------------------------- | ---------------------- |
| `shared_entity`     | The two chunks share at least one entity match key.                                             | Number of shared keys. |
| `keyphrase_overlap` | Jaccard similarity of canonical keyphrase sets is at least `keyphrase_overlap_threshold` (0.3). | The Jaccard value.     |
| `same_document`     | Consecutive chunks of the same file.                                                            | 1.0                    |

Candidate pairs come from inverted indexes (entity → chunks, keyphrase → chunks), so cost grows with
co-occurrences rather than with all N² pairs. The `same_document` spine guarantees that no chunk is
isolated and that document-wide questions have a path to walk.

## 7.5 From graph to topics

Two topic modes exist; `theme` is the default.

**Entity mode** (`derive_topic_clusters`): connected components over *strong* semantic edges only:
`same_document` edges are excluded (they would collapse each document into one topic) and an edge must
have weight ≥ `min_cluster_edge_weight` (default 2, meaning at least two shared entities). The
threshold is the anti-"hairball" measure: one ubiquitous entity (the organization's own name, say)
would otherwise chain the whole corpus into a single topic. Components larger than `max_cluster_size`
are split by raising the internal weight threshold one band at a time until the component fragments
(splitting at the weakest join, in the spirit of Girvan-Newman edge removal [R13], but by weight band
so dense clusters do not stall).

**Theme mode** (`derive_theme_clusters`) adds a separate, high-level layer:

1. One corpus-level call reads every chunk summary and proposes a controlled vocabulary of at most
   `max_theme_vocabulary` broad, user-recognizable, non-overlapping themes (for example "pay and
   payments", "leave and absences").
2. A cheap tagging pass assigns each chunk exactly one theme by name, from its summary, keyphrases,
   and first 600 characters (three extraction batches per call). Returned names are snapped onto
   the vocabulary; anything else becomes unthemed.
3. Chunks with the same theme form one topic. Unthemed chunks are clustered among themselves by
   entity edges. An oversized theme is split by internal entity cohesion and then re-packed into the
   fewest balanced groups so it does not dissolve into singletons.

*Why themes:* entity clustering fragments sparse corpora and buries thin, cross-cutting subjects. The
motivating case was pay (שכר), which is mentioned in many documents but rarely shares two entities
across them. The A/B probe in [theme-layer-ab-results.md](../theme-layer-ab-results.md) found, on one
corpus, 40 entity-mode topics (28 of them single-chunk) against 10 balanced theme-mode topics with pay
as a first-class topic. The two-step "propose a topic list, then assign documents to it" design is the
same pattern TopicGPT validated for LLM topic modeling [R15]. Themes never create edges, so they
cannot reintroduce the hairball.

In both modes, if there are more clusters than `max_graph_topics`, the largest stay distinct and the
small tail is bin-packed into bounded "other" buckets.

The vocabulary is derived from the whole corpus, so without care any document edit would produce a
new vocabulary and re-tag every document. A generation plan therefore stores the vocabulary its
topics were built with: `generate --plan` and `add-broad --plan` reuse it, and
`plan --themes-from <old plan>` carries it into a new plan after documents change, so only edited
documents are re-extracted and re-tagged.

## 7.6 Labels and importance

One call names all clusters from their most frequent entities and keyphrases (and dominant theme),
asking for broad Hebrew names a user would recognize. Duplicate names are disambiguated in code
because topic names key quotas and diagnostics. If labeling fails, names fall back to the top
entities. Each topic also gets a 1-5 `importance` from its share of the corpus (log-spaced
thresholds); it orders topics for the prompt generator but no longer drives quotas (section 11.2).

## 7.7 Evidence walks

`cluster_evidence_ids` grows an evidence set from seed chunks by repeatedly taking the
highest-weight unvisited neighbor (a best-first walk), up to a node budget, optionally staying in
one file or preferring other files. Generation uses it to add up to `context_nodes` strong neighbors
to each focus window.
