# 8. Retrieval (`retrieval.py`)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md). Citations such as [R1] are listed in [references.md](references.md).

Retrieval is used wherever evidence must be found *for a specific question* rather than for a topic:
the completeness check, corpus-wide boundary checks, reground, broad-question support checks, and
reviewer source hints.

- **Lexical:** Okapi BM25 [R9] (k1 = 1.5, b = 0.75) over tokens produced by the same Hebrew
  normalization as the graph. Attached prefixes are tolerated only against an attested bare word: a
  document indexes each prefixed token's candidate stems as marked terms, and a query adds its own
  stems plus a marked copy of each token. So `לחייל` matches `חייל` in either direction, while `הורה`
  and `מורה`, which only share the remainder `ורה`, do not match. IDF keeps frequent function words
  from dominating.
- **Dense (optional):** when `embedding_model` is set, chunks and queries are embedded and ranked by
  cosine similarity [R26]. Vectors default to 768 dimensions; Gemini embedding models are trained so
  truncated vectors keep most of their quality (Matryoshka representations [R11]), which keeps the
  cache and similarity checks light.
- **Fusion:** reciprocal rank fusion, `score = Σ 1 / (60 + rank)`, with k = 60 as in the original
  paper [R10]. RRF needs no score calibration between BM25 and cosine, which is why it is the standard
  way to combine them.
- **Failure mode:** any embedding failure logs a warning and permanently falls back to lexical
  ranking for that run; it never fails generation.

Embeddings are cached in `<cache>/embeddings/<identity>.jsonl`, keyed by the SHA-256 of the text, in
a file whose name encodes model, dimensions, transport, and endpoint.

The same normalization powers near-duplicate detection: `token_similarity` equates a prefixed token
with a bare token present in either question (`השכר` with `שכר`), and two questions are duplicates
when that token Jaccard is ≥ 0.78 (lexical) or, with
embeddings, cosine ≥ `semantic_duplicate_threshold` (0.92). Jaccard over token sets is the classic
resemblance measure [R27]; removing near-duplicates matters because a benchmark that repeats one
question in five wordings over-weights that fact [R17].
