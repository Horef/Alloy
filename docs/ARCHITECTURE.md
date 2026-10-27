# Architecture

Status: current (verified 2026-10-27). Only the boundaries and invariants that every change must keep, and an index of
the design reference. How each component works and why is in [design/](design/overview.md), one
file per component; the file-level map is generated in [MAP.md](MAP.md). The
[README](../README.md) is the user manual.

## Overview

Alloy has two halves that share data contracts but no runtime state. **Generation** turns a
document corpus into a *silver* Hebrew question set (chunks → knowledge graph → topics → validated
candidates → variants), which people review. **Evaluation** sends approved questions to a chatbot
or reads saved answers, has an LLM judge score each answer claim by claim, classifies the scores
with deterministic code, and reports metrics. A system-prompt generator and optional insights reuse
the same pieces. The system map and design principles are in
[design/overview.md](design/overview.md).

## Boundaries and invariants

Enforced by `python tools/docmap.py check` (`[[tool.docmap.forbid]]` in `pyproject.toml`):

- Only `llm.py` imports the Gemini SDK (`google`). Every other module depends on the
  `StructuredLLM` protocol, so the whole suite runs offline with fakes.
- No module imports `cli.py`. It is the only module that wires components together.

Enforced by tests:

- **Model output is a proposal; code decides.** Questions are accepted only if their quotes occur
  verbatim in the cited chunks (`generator.validate_candidate`); judge totals are recomputed from
  per-claim assessments (`evaluator.apply_fixed_claims`) and outcomes come from `classify`; prompt
  packages pass `validate_prompt_package`. Every rejection is counted under a named reason
  (`tests/test_generator.py`, `test_evaluator.py`, `test_prompt_generator.py`).
- **Errors are not quality.** Chatbot and judge failures become `chatbot_error` / `judge_error`
  and are excluded from every quality denominator (`tests/test_report.py`).
- **Cache keys change whenever outputs would change**, and only then: keys fingerprint prompts,
  schemas and deterministic code with `contracts.code_fingerprint`, which ignores comments and
  docstrings (`tests/test_cache.py`, `test_contracts.py`). The cache directory must lie outside the
  corpus (`tests/test_cli.py`).
- **Reproducible output.** Seeded calls, content-hash IDs and a fixed merge order; concurrency does
  not change the result (`tests/test_generator.py`).
- **Optional checks that cost calls are off by default** and report their outcome in the diagnostics
  or manifest (`tests/test_config.py`, `test_generator.py`).
- **Secrets never reach artifacts.** API keys come only from the environment and are removed from
  run manifests (`tests/test_artifacts.py`); logs carry counts and IDs, not content
  (`tests/test_llm.py` for quota logging).
- **Docs follow the code.** Every setting, CLI option and module is documented
  (`tests/test_docs_consistency.py`).

By convention (reviewed, not tested):

- Every prompt treats documents, questions and answers as untrusted data and asks for Hebrew output;
  the model is never given tools (automatic function calling is disabled).
- Corpora, outputs, caches, `.env` and `config.toml` are never committed (`.gitignore`); caches
  contain document text and are protected like the corpus.

## Design reference

1. [What Alloy does, in one page](design/overview.md#1-what-alloy-does-in-one-page)
2. [System map](design/overview.md#2-system-map)
3. [Design principles that cut across every component](design/overview.md#3-design-principles-that-cut-across-every-component)
4. [Configuration and entry points](design/configuration.md)
5. [The model layer (`llm.py`)](design/model-layer.md)
6. [Ingestion and chunking (`documents.py`, `pdf_markdown.py`)](design/ingestion.md)
7. [The corpus knowledge graph (`graph_build.py`, `graph.py`)](design/graph.md)
8. [Retrieval (`retrieval.py`)](design/retrieval.md)
9. [Caching, identity, and invalidation (`cache.py`, `contracts.py`)](design/caching.md)
10. [Automatic sizing (`planning.py`)](design/planning.md)
11. [Question generation (`generator.py`)](design/generation.md)
12. [Human review and set maintenance (`review.py`)](design/review.md)
13. [Evaluation (`adapters.py`, `results_io.py`, `evaluator.py`)](design/evaluation.md)
14. [Metrics and reporting (`report.py`)](design/reporting.md)
15. [Insights and topic inference (`insights.py`, `topics.py`)](design/insights-and-topics.md)
16. [System-prompt generation (`prompt_generator.py`, `prompt_policy.py`)](design/prompt-generation.md)
17. [Artifacts, reliability, and observability (`artifacts.py`, `io.py`)](design/artifacts.md)
18. [Security and data handling](design/security.md)
19. [Known limitations](design/limitations.md)
22. [Personalization (`profiles.py`, `personas.py`)](design/personalization.md)

Section 22 was added after the split. Section 21, [References](design/references.md), lists the works cited as [R1], [R2], and so on.
Sections keep their original numbers, so "section 9" still means caching. The original section 20,
the fixes made while this reference was written, is a dated record in
[archive/architecture-review-fixes.md](archive/architecture-review-fixes.md).
