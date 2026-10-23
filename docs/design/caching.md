# 9. Caching, identity, and invalidation (`cache.py`, `contracts.py`)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md).

## 9.1 What is cached, and by what key

| Kind                                            | Granularity      | Key includes                                                                                                                                                                                                      |
| ----------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chunks`                                        | corpus           | Every supported file's relative path and SHA-256, chunk size and overlap, code fingerprint of `documents.py` and `pdf_markdown.py` (PDFs are read through the converter), cache schema.                           |
| `node_signals`                                  | **per document** | That document's chunk IDs and text, signals fingerprint (prompt + schema + truncation bound), model, model route, batch size, theme signature (tagging pass), hash of the document's base signals (tagging pass). |
| `theme_vocabulary`                              | corpus           | All chunk IDs and text, vocabulary fingerprint, model, route, ceiling. Empty vocabularies (the failure fallback) are not cached.                                                                                  |
| `graph`                                         | corpus           | Chunks, every chunk's signals, edge and clustering parameters, topic mode, model, route, code fingerprint of `graph.py` + `graph_build.py`.                                                                       |
| `question_topics`, `insights`, `prompt_package` | per workflow     | Their inputs, model, route, and the implementing module's code fingerprint.                                                                                                                                       |
| `llm_calls`                                     | per call         | Model, schema, prompt, required fields; file name carries route, seed, call contract.                                                                                                                             |
| `embeddings`                                    | per text         | Text hash; file name carries model, dimensions, route.                                                                                                                                                            |

The model route (`contracts.cache_identity`) is transport, endpoint hash, structured-call contract
version, and seed.

## 9.2 Why the keys look like this

- **Content hashes, not timestamps.** File modification times are not trusted; only content and
  settings determine reuse.
- **Fingerprint prompts and schemas, not whole modules,** for the expensive stage (signal extraction).
  Editing a log line in `llm.py` must not force re-extracting every document. An earlier version
  hashed whole modules and did exactly that; commit `0c23fd0` narrowed it.
- **Code fingerprints ignore comments and docstrings.** Where deterministic code shapes the output
  (chunking, graph building, judging, insights, prompt packages), keys use
  `contracts.code_fingerprint`: a hash of each module's syntax tree without docstrings or positions.
  Behavior and prompt text change the key; comments, docstrings, and formatting do not. The syntax
  tree format belongs to the Python minor version, so these entries are not shared across Python
  versions.
- **Per-document signals make rebuilds incremental.** Adding or editing one document re-extracts
  only that document. The graph itself is cheap to rebuild from cached signals (deterministic edges
  plus one labeling call).
- **Theme tagging is keyed on the base signals,** so a re-extracted document is re-tagged.

## 9.3 Pruning

Content-addressed entries are never overwritten, so changed inputs leave orphans. After each command
the cache is swept: for every kind the command *used*, entries whose keys were not looked up in this
run are deleted. Kinds the command never touched are left alone, because the run cannot know their
live set. `--keep-stale-cache` disables the sweep. The call and embedding stores are append-only JSONL
files and are not pruned.

## 9.4 Safety rules

The cache directory must be outside `--documents` (otherwise cache JSON would be read as corpus input),
and it contains extracted document text, so it must be protected like the corpus. Corrupt or
schema-mismatched entries are logged and recomputed.
