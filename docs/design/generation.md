# 11. Question generation (`generator.py`)

Status: current (verified 2026-10-27)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md). Citations such as [R1] are listed in [references.md](references.md).

## 11.1 Budgets

The total is divided into canonical answerable questions, user variants, and boundary questions,
either from ratios (`unanswerable_ratio`, `user_variation_ratio`) or exactly from a plan. Variants are
children of canonical questions and never replace their provenance.

## 11.2 Topic quotas

Without a plan, `allocate_graph_quotas` splits the canonical (answerable) budget over topics in
proportion to how much each topic says.

**A topic's total information units** is the sum of `information_units(chunk.text)` over the chunks
in the topic (section 10). For one chunk, that is the number of distinct statements it makes. The
text is split at sentence ends, line breaks, list bullets, numbered items, and table-cell
separators. A piece counts if it has at least four words, and repeated pieces count once. Any
non-empty chunk counts as at least 1, and so does any topic. For example, this chunk has 3 units:

```
# חופשה
- חייל זכאי לחופשה שנתית של 18 ימים.        <- 1
- את הבקשה מגישים למפקד הישיר.              <- 2
| סוג | ימים |                              <- cells under four words: 0
| חופשה מיוחדת לחייל בודד | 30 |            <- 3 (the "30" cell does not count)
חייל זכאי לחופשה שנתית של 18 ימים.          <- repeat of 1: 0
```

The count needs no model call and gives the same result on every run. It measures how many facts a
topic can support questions about, unlike its chunk count: a short chunk and a dense table of rules
each count as one chunk. Topics do not share chunks (each chunk has one theme or one entity
component), so topic weights add up to the corpus total.

**Allocation.** With these weights:

1. Every topic gets `min_topic_questions`, in descending weight order, but never more than the share
   cap `⌈total × max_topic_share⌉`. If the budget runs out first, the lightest topics get nothing,
   and the diagnostics show their zero quota.
2. The rest is handed out one question at a time. Each goes to the topic with the highest
   `weight / (quota + 1)` that is still under the cap.

This is a highest-averages (D'Hondt-like) apportionment. It tracks proportionality while
guaranteeing coverage and a diversity cap. For example, three topics with 120, 60, and 20 units,
20 questions, a minimum of 2, and a share cap of 0.4 (8 questions) get 8, 8, and 4. A purely
proportional split would give 12, 6, and 2, but the cap stops the largest topic at 8, and its
surplus goes to the others.

*Why information units and not the 1-5 importance bucket:* the coarse bucket oversampled small
topics (the code comment says so).

Boundary quotas (section 11.7) use the same weights, with a minimum of one per topic when the
boundary budget allows it, and no share cap. With `generate --plan`, the plan's per-topic numbers
(section 10, also derived from information units) replace this allocation.

`--topic` gives a single topic the whole quota; if the name is not a discovered topic, its evidence is
seeded from the chunks whose signals and text best match the requested term.

## 11.3 Evidence windows

A topic's chunks are split into contiguous, balanced **focus windows** of at most `max_cluster_nodes`
chunks, with enough windows that each call asks for about `questions_per_call` questions. The topic
quota is spread over windows in proportion to their information units (largest remainder). Each
window is rendered with up to `context_nodes` of its strongest graph neighbors as context, capped at
90,000 characters (a truncated chunk is marked, and quote validation only sees the visible part).

*Why windows:* the alternative, one call per topic with its "top" chunks, never showed the model the
tail of a large topic. Windows guarantee every chunk of a topic is shown, keep each request small
(more reliable for small models, see [R12]), and keep chunks of one document together. Unused quota
in one window carries over to the next.

## 11.4 The generation loop for one window

```
rounds = ceil(goal / questions_per_call) + max_candidate_rounds - 1
for each round while accepted < goal:
    ask for min(missing, questions_per_call) questions, with:
        - the remaining question-type targets,
        - questions already accepted in this topic ("do not repeat"),
        - up to 12 recent rejection reasons ("avoid these")
    for each returned candidate:
        validate_candidate  -> reject with a named reason
        duplicate checks    -> reject
        type over target    -> reject, except on the final round (counted as relaxed)
        optional closed-book check (tag or reject)
        optional completeness verification (keep, correct, or reject)
        accept
```

The feedback in the prompt turns retries into targeted repairs rather than re-rolls. The final-round
relaxation exists so a window that cannot support, say, a `cross_document` question still fills its
quota instead of leaving a shortfall.

## 11.5 `validate_candidate`: the grounding contract

An answerable candidate is accepted only if all of the following hold:

- non-blank question and answer; source IDs exist and are unique;
- at least one supporting quote, and every quote's source is cited;
- every quote occurs **verbatim** in its source after normalizing whitespace, Markdown emphasis and
  heading markers, bidi and zero-width marks, niqqud, and Hebrew/typographic quote variants. Reworded
  quotes are rejected;
- every cited source has at least one quote;
- non-empty, distinct `reference_claims`;
- `document_wide` cites at least two chunks of one file; `cross_document` cites at least two files
  (multi-hop questions in the HotpotQA sense [R32]);
- personal-data types are rejected (no controlled personal-data source exists).

Boundary candidates must have type `unanswerable`, cite nearby sources, and carry no claims or
quotes.

*Why verbatim quotes:* a quote that exists in the cited chunk is a cheap, deterministic proof that the
answer was taken from the document rather than from the model's memory. It is the code-level version
of round-trip consistency filtering [R5] and of attribution checking in RAG evaluation [R2, R4].

*Why atomic claims:* they make judging auditable claim by claim (section 13.3), the approach FActScore
showed to be more reliable than holistic grades for long answers [R3, R16].

## 11.6 Optional checks

| Check                        | What it catches                                                                                                                                                                                                                                                                                                        | Cost                            |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| `verify_answer_completeness` | A reference answer that is incomplete or contradicted once evidence is re-retrieved *for this question* (not for the topic). The model may return a corrected answer, which must pass `validate_candidate` again; otherwise the candidate is dropped. A separate verification pass, as in Chain-of-Verification [R20]. | 1 call per canonical candidate  |
| `closed_book_check`          | Questions a model answers fully with no documents. These test general knowledge, not the knowledge base [R18]. `tag` keeps and marks them (`closed_book_answerable`); `reject` drops them. Tagging is recommended because a legitimate rule can also be public.                                                        | 2 calls per canonical candidate |
| `verify_unanswerable`        | Boundary questions that the corpus actually answers somewhere outside the window that produced them. Re-checks each one against evidence retrieved from the whole corpus.                                                                                                                                              | 1 call per boundary candidate   |

## 11.7 Boundary (unanswerable) questions

Boundary questions test whether the chatbot abstains instead of guessing, the capability SQuAD 2.0
made standard [R6] and that abstention research treats as central to safety [R19]. They are generated
per topic from the same windows, with a mix of three kinds:

- `missing_detail`: in scope, but the specific amount, date, contact, or step is not stated;
- `false_premise`: presupposes a rule or benefit the documents do not establish [R7];
- `out_of_scope`: plausible for this audience but outside the knowledge base.

Without `verify_unanswerable`, "not answerable" is established only against the window's evidence;
the diagnostics say so (`boundary_evidence_scope = cluster_excerpts`).

## 11.8 User variants

Canonical questions are clean and explicit; real users are not. Variants measure the difference.

- **`natural_user`**: what a soldier would actually type: short (usually 4-10 words), first person,
  everyday words, one recognizable anchor term, and often only the main part of a multi-part
  question. The model also returns `kept_claim_ids`, the parent's claims the shorter question still
  asks for, and the variant is graded only on those. A rewrite that is not noticeably shorter than a
  non-trivial parent (above 85% of its length) is rejected as `natural_not_simpler`. Testing the same
  intent under different wording is a CheckList-style invariance test [R21].
- **`ambiguous`**: removes a material discriminator (population, status, timeframe, procedure), never
  adds typos. It comes in two kinds:
  - `must_clarify`: interpretations have different correct answers, so answering one is unsafe and
    only a clarifying question succeeds. This is the clarifying-question setting of Qulac [R8b].
  - `answer_or_clarify`: interpretations are all safe to present together, so a comprehensive answer
    covering all of them is as good as clarifying. This is the AmbigQA view, where the right
    response to an ambiguous question enumerates its readings [R8].

Variants are generated in batches of `variation_batch_size` parents and deduplicated only against
other variants, since paraphrasing the parent is the point.

**`user_facing` mode** gives *every* canonical and boundary question exactly one natural phrasing. The
canonical question becomes its **anchor** (`anchor=true`): it keeps the grounded intent and
provenance, but the user phrasing is what reviewers see and what headline metrics score. Anchors are
still judged, and the report shows the **phrasing gap** (anchor success minus user-phrasing success on
the same pairs), which isolates how much the chatbot loses to realistic wording.

## 11.9 Broad questions

Real users often open with a vague question: "מה הזכויות שלי?" or "אני חייל בודד, מה מגיע לי?". The
right answer is an overview of the relevant areas (or a clarifying question), not a single fact.
Canonical questions cannot test this: each one targets specific claims in a few chunks. Broad
questions test whether the chatbot can map the knowledge base for the user, instead of answering one
detail, refusing, or inventing benefits. They are off by default (`generation.broad_questions`). They
are produced by `generate` (skipped with `--topic`) or added to an existing set with `add-broad`. That
command replaces the set's existing broad questions and keeps all others. Pass `--plan` so the topics
match the ones the set was generated with.

**Three levels** (`broad_level`):

| Level        | Count                                                                                | Asked by                                                      | Map shown to the model                                   |
| ------------ | ------------------------------------------------------------------------------------ | ------------------------------------------------------------- | -------------------------------------------------------- |
| `corpus`     | `broad_corpus_questions` (default 2)                                                 | any user of the knowledge base                                | every document                                           |
| `population` | one per population the documents address, up to `broad_population_limit` (default 8) | a member of that population, such as lone or married soldiers | every document                                           |
| `topic`      | one per topic                                                                        | anyone asking about the whole area                            | the topic's documents, with up to 6 chunk summaries each |

Corpus and population questions come from one call, and each topic adds one call. A population
qualifies only if the documents describe at least `broad_min_key_points` distinct things for it, so
the population count is an upper bound. The plan reports broad questions as `broad_max` for this
reason.

**Step 1: generate from a document map.** A whole corpus does not fit in one prompt, and chunk-level
evidence would push the model toward detail questions. Instead, `_document_map` renders each
document as one line: `D01`, `D02`, and so on, with its front-matter title, subtitle, category, and
topics. The model writes short, casual first-person questions, the way a soldier types into a chat
box. For each question it returns:

- 3 to 8 `key_points`: the main areas a good overview names, such as a right, benefit, procedure, or
  service. Each point lists the map documents that cover it.
- a suggested `min_key_points`;
- an `expected_answer` overview;
- an `acceptable_clarification`, a follow-up question that narrows the request.

**Step 2: ground every key point.** The map shows only titles, so the model could name an area the
documents do not support. `_ground_broad` drops points that cite no valid map document. For each
remaining point, it ranks the corpus by the point's text with BM25 (section 8) and keeps the two
best chunks from the cited documents. If none rank, it falls back to the document's first chunk. One
check call (`BROAD_CHECK_PROMPT`) then reads those chunks and decides, point by point, whether the
evidence states the point. A point is kept only if it is marked supported and cites a chunk that was
actually shown. A question with fewer than `broad_min_key_points` supported points is rejected.

The supported points become the question's `reference_claims`, and their chunks become its
`sources`. The question type is `cross_document` when those chunks span several files and
`document_wide` otherwise. Unlike canonical questions, broad questions carry no verbatim quotes. A
key point names an area, such as "החזרי נסיעה", that is rarely one sentence of text. The model check
is therefore the grounding guarantee: deterministic validation still applies, but quote provenance
does not. This makes broad questions weaker silver than canonical ones, and they need review *(the
reasoning is inferred from the design)*.

**Step 3: decide how many points an answer must name.** `min_key_points` is the model's suggestion,
capped at half of the supported points (rounded up) and raised to at least 2. With six supported
points, a suggestion of 4 becomes 3, and a suggestion of 1 becomes 2. An overview that names every
area is not required. A good answer names enough of them to orient the user.

**Filtering and IDs.** Each call may return only its own levels, up to their counts. Extra questions
are rejected as `broad_level_over_budget`. Questions that duplicate each other, the set, or the
exclusion list are dropped as `broad_duplicate`. Broad questions get `B` IDs (section 11.10) and the
topic "שאלות כלליות", except topic-level ones, which keep their topic. They are generated after
variants, so they never get variants of their own. They are already in user phrasing, and
`user_facing` totals count them as such.

The remaining rejection reasons are `broad_point_without_document`, `broad_too_few_points`,
`broad_point_unsupported`, and `broad_too_few_supported_points`. They appear in the generation
diagnostics, or in `broad_diagnostics` in the `add-broad` run manifest, together with counts per
level.

**Judging** (section 13.3). The judge prompt gets a broad-question task note. The reference claims
are areas, and naming `min_key_points` of them correctly is enough. Extra true areas are not
counted as unsupported, and a long, structured overview is not marked `too_much`. Since
`clarification_acceptable` is set, a clarifying question that narrows the request is a
`correct_clarification`. False claims still fail the answer, as they do for any question.

## 11.10 IDs, merging, concurrency, and diagnostics

- **Stable IDs** are `prefix-sha256(content)[:16]`, with prefixes `Q` (canonical), `U` (boundary),
  `V` (variant), `B` (broad). Re-generating identical content yields the same ID, which keeps
  reviewer decisions and run comparisons attached to the right question. `--sequential-ids` restores
  run-local `Q0001` IDs.
- **`--merge-into`** appends new questions to an existing set, keeps existing rows verbatim, drops
  near-duplicates and orphaned variants, and never reuses an existing ID.
- **Concurrency:** topic workers keep independent state and are merged in topic order, so output does
  not depend on scheduling. Cross-topic duplicates are removed at merge time.
- **Resume:** generation calls are journaled with a signature covering chunks, options, topics, model,
  route, and implementation; `--resume` replays them in order and continues after the last one.
- **Diagnostics:** `generation_diagnostics.json` records planned versus accepted totals, every
  rejection reason, per-topic quotas and shortfalls, the exact source IDs rendered per topic, graph
  statistics, and outcomes of every optional check. A shortfall is expected behavior when evidence
  cannot support the request, and the diagnostics say which topics fell short and why.
