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
