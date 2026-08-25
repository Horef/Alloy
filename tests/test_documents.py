from chatbot_eval import documents


def test_chunks_preserve_section_location(tmp_path, monkeypatch):
    root = tmp_path / "docs"
    root.mkdir()
    pdf = root / "sample.pdf"
    pdf.touch()
    monkeypatch.setattr(documents, "_read_sections", lambda path: [("page 2", "useful policy text " * 10)])

    chunks = documents.load_chunks(root, chunk_chars=100, overlap_chars=10)

    assert chunks
    assert all(chunk.location.startswith("page 2, chars") for chunk in chunks)
