# Theme-aware topic layer — live A/B results

Goal of the theme layer: get topic coverage to self-organize so a chatbot requester only has to
*create and read* — every major theme (the team's example was **שכר / pay**) should surface as its
own topic automatically, without anyone hand-writing a topic list.

The probe (`docs/theme_ab_probe.py`) builds the corpus knowledge graph twice from the **same** cached
node signals — once in the default **entity** clustering mode and once in the new **theme-first**
mode — and prints both topic maps. Theme mode re-extracts node signals once (the theme signature
changes the per-document cache key); after that both graphs rebuild cheaply.

Run against the live gateway (`gemini-3.7-flash`, `--max-concurrency 6`).

## hova

| | entity mode (default) | theme mode (opt-in) |
|---|---|---|
| topics | 40 | 10 |
| single-chunk topics | 28 of 40 | 0 |
| largest topic | 29 chunks | 15 chunks |
| שכר as its own topic | only a thin 1-chunk topic | yes — "רמות פעילות ותוספות שכר" (grouped) |

Entity mode fragments hova into a swarm of singletons plus one 29-chunk lump; pay is barely visible.
Theme mode organizes the corpus into 10 balanced, human-legible themes with no giant topic, and pay
is a first-class topic.

## keva

| | entity mode (default) | theme mode (opt-in) |
|---|---|---|
| topics | 42 | 40 |
| giant (~40-chunk) topics | 4 | 0 (largest 38) |
| שכר coverage | smeared across 4 topics | consolidated into "שכר, דירוגים וגמול השתלמות" (35 chunks) |

Entity mode spreads pay across four different topics and produces four ~40-chunk lumps. Theme mode
gives pay a single dedicated topic and no worse a largest-topic size. Theme mode still emits a tail
of single-chunk topics for a dense parenting-rights document (the unthemed/"other" fallback), which
is expected and stays within the `max_graph_topics` cap.

## Conclusion

On both corpora, theme mode makes the major themes — especially שכר — surface as their own
consolidated topics with no manager authoring, and does not create an entity hairball. This is the
"create and read" behavior we were after.

Theme mode remains **opt-in** (`topic_mode = "theme"`); entity mode stays the default so existing
runs are unchanged. Enable theme mode on corpora where coverage of every theme matters more than
tight entity cohesion.

## Full hova question-generation run (theme mode) vs the previous entity run

Beyond the topic-map probe, a complete `generate` was run on hova in theme mode
(`hova/outputs/questions-kb-theme`, 300 questions) and compared to the previous entity-mode run
(`hova/outputs/questions-kb`, 295 questions). Both used the same 85-node graph and a 12-topic budget.

**Topic balance (from generation diagnostics):**

| | entity run | theme run |
|---|---|---|
| questions | 295 | 300 |
| distinct topics used | 12 | 12 |
| largest topic (graph chunks) | 33 | 14 |
| smallest topic | 1 | 3 |
| singleton topics | 2 | 0 |
| questions touching שכר | 12 | 14 |

**Question distribution:** the entity run concentrated 94 of 295 questions (32%) in one topic
("בקשות תנאי שירות...") and another 45 in "זכויות פרט, תשלומים והיתרי עבודה" — over half the set in
two broad buckets, with pay folded inside them. The theme run spread questions far more evenly: the
largest topic held 42 of 300 (14%) and the twelve topics are human-legible, non-overlapping subjects
(economic aid, lone soldiers, attendance/command authority, leave, service length, legal counsel,
welfare, travel reimbursement, discharge, dietary provision, housing, mental health).

**Takeaway:** on a real generation, theme mode produced a more balanced, better-organized question
set — no dominant catch-all topic, no singleton topics, and slightly more pay coverage — matching
what the topic-map probe predicted. This is why theme mode is now the default.

## Cache pruning (validated on the same run)

The theme run was also the first live exercise of default cache pruning. On completion it swept
`chunks=2 graph=4 node_signals=136 theme_vocabulary=1` superseded entries, leaving node_signals at
68 (= 34 documents × 2 legitimate key-sets: the base extraction plus the theme-tagged extraction).
Both live key-sets from the two in-run `load_node_signals` calls were correctly retained; only the
truly orphaned entries were removed.
