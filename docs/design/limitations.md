# 19. Known limitations

Status: current (verified 2026-10-27)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md). Citations such as [R1] are listed in [references.md](references.md).

These are inherent tradeoffs worth knowing when interpreting results, with the settings that
mitigate them.

1. **The judge is not calibrated yet.** LLM judges have length, style, and model-family biases
   [R28], and even strong judges diverge from humans [R31]. Before using a metric as a release gate,
   double-review a sample by hand and measure agreement (Cohen's kappa rather than percent agreement,
   as [R31] recommends).
2. **Boundary questions are only as unanswerable as the evidence checked.** Without
   `verify_unanswerable`, absence is established against one window; with it, against a retrieval
   sample from the whole corpus. Neither is exhaustive, which is one more reason boundary questions
   are reviewed by a person.
3. **Retrieval can miss.** Completeness verification and reground depend on BM25 (plus optional
   embeddings) surfacing the right chunk. Enabling `embedding_model` helps with paraphrased
   evidence; a reviewer's `source_hint` helps reground directly.
4. **Topics follow the corpus.** A pinned theme vocabulary (section 7.5) keeps theme names stable, but
   editing a document still relabels topics and can move its chunks to another theme, so a plan made
   before the edit has to be re-made (with `plan --themes-from`).
5. **Code fingerprints are per Python version.** Caches built under one Python minor version are
   rebuilt once under another.
6. **PDF conversion is heuristic.** The local backend recognizes only ruled tables (borderless
   tables come out as text lines), decides direction per line, and infers headings from font size
   and weight. OCR needs Tesseract installed. Use the `gemini` backend for complex layouts and scans;
   its pages are verified against the text layer, but pages without one cannot be verified.
7. **Generated system prompts are starting points** that a manager must review; they are not a
   security control.
