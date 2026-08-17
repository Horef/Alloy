from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".txt", ".md", ".rst", ".csv", ".json", ".jsonl", ".pdf", ".docx"}


@dataclass(frozen=True)
class Chunk:
    id: str
    file: str
    location: str
    text: str


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".rst"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            return "\n".join(" | ".join(row) for row in csv.reader(handle))
    if suffix == ".json":
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
    if suffix == ".jsonl":
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        pages = []
        for number, page in enumerate(PdfReader(path).pages, 1):
            pages.append(f"[Page {number}]\n{page.extract_text() or ''}")
        return "\n\n".join(pages)
    if suffix == ".docx":
        return "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    raise ValueError(f"Unsupported file type: {path}")


def load_chunks(root: Path, chunk_chars: int, overlap_chars: int) -> list[Chunk]:
    if overlap_chars >= chunk_chars:
        raise ValueError("chunk overlap must be smaller than chunk size")
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    if not files:
        raise ValueError(f"No supported documents found under {root}")
    chunks: list[Chunk] = []
    for path in files:
        text = " ".join(_read_file(path).split())
        relative = str(path.relative_to(root))
        step = chunk_chars - overlap_chars
        for index, start in enumerate(range(0, len(text), step), 1):
            part = text[start : start + chunk_chars]
            if len(part.strip()) < 80:
                continue
            chunks.append(Chunk(f"{relative}#chunk-{index}", relative, f"chars {start}-{start + len(part)}", part))
    if not chunks:
        raise ValueError("Documents contained no extractable text")
    return chunks

