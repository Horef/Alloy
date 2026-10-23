# 6. Ingestion and chunking (`documents.py`)

Part of the Alloy design reference; index and invariants in [../ARCHITECTURE.md](../ARCHITECTURE.md). Citations such as [R1] are listed in [references.md](references.md).

Supported inputs are `.txt/.md/.rst`, `.csv` (rows become ` | `-joined lines), `.json/.jsonl`,
`.pdf` (per page, through the local converter of section 6.1, no OCR), and `.docx` (paragraphs and
tables in their XML order, so a table stays next to the text that introduces it). Markdown with
`<!-- page: N -->` markers is split into page sections, so converted PDFs keep page locations.
Hidden files and folders (`.alloy/`, `.git/`, editor settings) are never read as corpus.

Pages or sections that yield no text, and files that yield none at all, are recorded in an
extraction summary that is stored with the chunk cache, logged as a warning on every run, and listed
under `cache.extraction` in the run manifest. A scanned PDF therefore never disappears from the
corpus silently; it shows up as an OCR candidate.

Chunking is **structure-aware, then size-bounded**:

1. Text is split into blocks at blank lines and at headings (a line starting with `#`, or a short
   line ending in `:`).
2. A block longer than `chunk_chars` is cut into overlapping windows.
3. Consecutive blocks are packed into a chunk up to `chunk_chars`; the next chunk starts with as
   many trailing blocks as fit in `chunk_overlap_chars`.

Chunk IDs are `relative/path#chunk-N` with a location such as `page 3, blocks 4-7`, so every citation
in a generated question points to a human-findable place.

*Why these choices:* paragraph and heading boundaries keep a rule and its conditions together; the
overlap protects facts that straddle a boundary; and a short chunk is never dropped, because a
two-line section can hold the only operative rule in a document (a comment in the code says exactly
this). The shipped configs use ~2,000 characters (~500 tokens): small enough that each document
becomes several graph nodes (which `document_wide` questions need) and that a small model can extract
signals reliably, large enough to keep a typical rule intact. Keeping evidence small and focused also
avoids the well-documented drop in model accuracy for facts buried in the middle of long contexts
[R12].

## 6.1 PDF to Markdown (`pdf_markdown.py`)

PDFs are the weakest input format: they store glyphs at positions, not text in reading order.
Markdown is the target because headings, lists, and tables survive as plain text that models read
well, chunking can follow that structure, and per-page markers keep provenance. `convert-pdf`
produces it with two backends that share the output format and the report.

**Local backend.** Built on `pdfplumber` (MIT), which exposes every character with its position,
size, and font.

- *Lines and bidi.* Characters are grouped into lines, split into segments at wide gaps (columns,
  form fields), and spaced by gap width. Extractors order characters left to right on the page, so a
  Hebrew line comes out reversed. Each segment's base direction is decided by whether it has at
  least as many Hebrew words as Latin words. A right-to-left segment is then reversed, and every
  embedded left-to-right run (Latin words, numbers, dates, URLs, `5%`) is reversed back. This is the
  visual-to-logical inverse of the Unicode Bidirectional Algorithm's reordering [R33]. Separate
  numbers are not joined into one run, because the algorithm lays them out right to left.
- *Tables.* Ruled tables (`find_tables`) become Markdown tables. Cell text goes through the same line
  logic, and right-to-left tables are mirrored so their first column comes first. A one-cell frame
  is a boxed paragraph, not a table.
- *Structure.* The most common font size is body text. Larger sizes become `#`, `##`, `###` by
  rank, and short all-bold lines become the next level. Bullet glyphs (including Word's private-use
  bullet) become list items, and consecutive lines join into paragraphs when they are close and
  share direction. Lines in the top or bottom 8% of the page that repeat on at least half the pages
  (digits ignored) are running headers or footers and are dropped, as are page-number lines.
- *Columns.* Full-width lines split a page into bands. A band whose two halves each hold at least
  three lines, two of them wide, is read column by column, right column first on right-to-left
  pages. The width condition keeps label/value forms row by row.
- *No text layer.* A page with almost no text, or whose text is mostly unmapped glyphs (private-use
  code points, `(cid:N)`), is rendered at 300 dpi and OCR'd with Tesseract [R34] when it is
  installed, or reported under `pages_needing_ocr`.

**Gemini backend.** Vision-language models now transcribe complex layouts and scans better than
rule-based extraction, but they can silently drop a paragraph or "fix" a number. The backend
therefore treats the model as a proposal, like every other model call in Alloy:

1. Pages go in small batches (`pages_per_call`, default 4) as native PDF parts (each page is about
   258 tokens), together with the local text layer of the same pages to pin exact spelling.
2. Each returned page is compared with the text layer by word recall (does it contain what the page
   says?) and precision (does it say anything the page does not?). A word also matches its reversal,
   so a reversed text layer does not penalize a correct transcription.
3. A page below `min_text_recall` or `min_text_precision` is retried alone, since smaller requests
   omit less, and is otherwise replaced by the local result and flagged `local_fallback`. Pages
   without a text layer are kept but reported as unverified.

Calls carry the PDF's SHA-256 in the prompt and the attachment's hash in the call-cache key, so
re-running is free and two documents never share a cached answer.

*Alternatives considered.* Docling [R35] (MIT) is the strongest open-source layout pipeline, but it
needs PyTorch models and has no right-to-left handling of its own. Marker (GPL-3.0, with
model-weight license restrictions) and PyMuPDF4LLM (AGPL-3.0) have licenses that are hard to clear
for internal enterprise use. A rule-based converter that understands Hebrew, combined with a
verified model pass for hard pages, covers this corpus with a light, permissive dependency set.
