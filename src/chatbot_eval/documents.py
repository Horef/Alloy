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
            return [("document", "\n".join(" | ".join(row) for row in csv.reader(handle)))]
    if suffix == ".json":
        return [("document", json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))]
    if suffix == ".jsonl":
        return [("document", path.read_text(encoding="utf-8", errors="replace"))]
    if suffix == ".pdf":
        return [(f"page {number}", page.extract_text() or "") for number, page in enumerate(PdfReader(path).pages, 1)]
    if suffix == ".docx":
        return [("document", "\n".join(paragraph.text for paragraph in Document(path).paragraphs))]
    raise ValueError(f"Unsupported file type: {path}")


def load_chunks(root: Path, chunk_chars: int, overlap_chars: int, progress_enabled: bool = False) -> list[Chunk]:
    if overlap_chars >= chunk_chars:
        raise ValueError("chunk overlap must be smaller than chunk size")
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    if not files:
        raise ValueError(f"No supported documents found under {root}")
    chunks: list[Chunk] = []
    for path in track(files, enabled=progress_enabled, description="Reading documents", total=len(files)):
        relative = str(path.relative_to(root))
        step = chunk_chars - overlap_chars
        chunk_number = 0
        for section_location, raw_text in _read_sections(path):
            text = " ".join(raw_text.split())
            for start in range(0, len(text), step):
                part = text[start : start + chunk_chars]
                if len(part.strip()) < 80:
                    continue
                chunk_number += 1
                location = f"{section_location}, chars {start}-{start + len(part)}"
                chunks.append(Chunk(f"{relative}#chunk-{chunk_number}", relative, location, part))
    if not chunks:
        raise ValueError("Documents contained no extractable text")
    return chunks
