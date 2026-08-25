from chatbot_eval import documents


def test_chunks_preserve_section_location(tmp_path, monkeypatch):
    root = tmp_path / "docs"
    root.mkdir()
    pdf = root / "sample.pdf"
    pdf.touch()
    monkeypatch.setattr(documents, "_read_sections", lambda path: [("page 2", "useful policy text " * 10)])

    chunks = documents.load_chunks(root, chunk_chars=100, overlap_chars=10)

    assert chunks
    assert all(chunk.location.startswith("page 2, blocks") for chunk in chunks)


def test_structured_chunks_preserve_heading_and_paragraph_boundaries():
    text = "# Eligibility\n\nFirst paragraph contains enough policy detail for evaluation.\n\nSecond paragraph contains a separate procedure and deadline."

    chunks = documents._structured_chunks(text, chunk_chars=180, overlap_chars=20)

    assert len(chunks) == 1
    assert chunks[0][1].startswith("# Eligibility\n\nFirst paragraph")
    assert "\n\nSecond paragraph" in chunks[0][1]
