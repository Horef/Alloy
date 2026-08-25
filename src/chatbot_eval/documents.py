from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from .progress import track


SUPPORTED_SUFFIXES = {".txt", ".md", ".rst", ".csv", ".json", ".jsonl", ".pdf", ".docx"}


@dataclass(frozen=True)
class Chunk:
    id: str
    file: str
    location: str
    text: str


def _read_sections(path: Path) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".rst"}:
        return [("document", path.read_text(encoding="utf-8", errors="replace"))]
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            return [("document", "\n\n".join(" | ".join(row) for row in csv.reader(handle)))]
    if suffix == ".json":
        return [("document", json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))]
    if suffix == ".jsonl":
        return [("document", path.read_text(encoding="utf-8", errors="replace"))]
    if suffix == ".pdf":
        return [(f"page {number}", page.extract_text() or "") for number, page in enumerate(PdfReader(path).pages, 1)]
    if suffix == ".docx":
        document = Document(path)
        sections = [("paragraphs", "\n\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()))]
        for number, table in enumerate(document.tables, 1):
            rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
            sections.append((f"table {number}", "\n\n".join(rows)))
        return sections
    raise ValueError(f"Unsupported file type: {path}")


def _blocks(text: str) -> list[str]:
    normalized_lines = [" ".join(line.split()) for line in text.splitlines()]
    blocks: list[str] = []
    current: list[str] = []
    for line in normalized_lines:
        heading = line.startswith("#") or (line.endswith(":") and len(line) < 160)
        if not line or heading:
            if current:
                blocks.append(" ".join(current))
                current = []
            if heading:
                blocks.append(line)
            continue
        current.append(line)
    if current:
        blocks.append(" ".join(current))
    return [block for block in blocks if block]


def _structured_chunks(text: str, chunk_chars: int, overlap_chars: int) -> list[tuple[str, str]]:
    blocks = _blocks(text)
    if not blocks:
        return []
    expanded: list[str] = []
    for block in blocks:
        if len(block) <= chunk_chars:
            expanded.append(block)
            continue
        step = chunk_chars - overlap_chars
        expanded.extend(block[start : start + chunk_chars] for start in range(0, len(block), step))

    chunks: list[tuple[str, str]] = []
    start = 0
    while start < len(expanded):
        selected: list[str] = []
        used = 0
        end = start
        while end < len(expanded):
            addition = len(expanded[end]) + (2 if selected else 0)
            if selected and used + addition > chunk_chars:
                break
            selected.append(expanded[end])
            used += addition
            end += 1
        content = "\n\n".join(selected)
        if len(content.strip()) >= 80:
            chunks.append((f"blocks {start + 1}-{end}", content))
        if end >= len(expanded):
            break
        overlap_start = end
        overlap_used = 0
        while overlap_start > start and overlap_used + len(expanded[overlap_start - 1]) <= overlap_chars:
            overlap_start -= 1
            overlap_used += len(expanded[overlap_start]) + 2
        start = overlap_start if overlap_start > start else end
    return chunks


def load_chunks(root: Path, chunk_chars: int, overlap_chars: int, progress_enabled: bool = False) -> list[Chunk]:
    if overlap_chars >= chunk_chars:
        raise ValueError("chunk overlap must be smaller than chunk size")
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    if not files:
        raise ValueError(f"No supported documents found under {root}")
    chunks: list[Chunk] = []
    for path in track(files, enabled=progress_enabled, description="Reading documents", total=len(files)):
        relative = str(path.relative_to(root))
        chunk_number = 0
        for section_location, raw_text in _read_sections(path):
            for block_location, part in _structured_chunks(raw_text, chunk_chars, overlap_chars):
                chunk_number += 1
                location = f"{section_location}, {block_location}"
                chunks.append(Chunk(f"{relative}#chunk-{chunk_number}", relative, location, part))
    if not chunks:
        raise ValueError("Documents contained no extractable text")
    return chunks
