import hashlib
import json

import pytest

from chatbot_eval import pdf_markdown
from chatbot_eval.llm import CachedStructuredLLM
from chatbot_eval.pdf_markdown import (
    PdfPagesMarkdown, TextLine, _column_order, _page_markdown, _RawPage, _segments, convert_gemini,
    convert_local, text_agreement, visual_to_logical,
)

# Visual-order lines as pdfplumber returns them from a Word-made Hebrew PDF (synthetic text).
VISUAL_TO_LOGICAL = [
    (
        'Personnel( א"כא לש 3.2.1 להונל םאתהב ,הנשב השפוח ימי 18-ל יאכז רידס תורישב לייח',
        'חייל בשירות סדיר זכאי ל-18 ימי חופשה בשנה, בהתאם לנוהל 3.2.1 של אכ"א (Personnel',
    ),
    (".שארמ םימי 7 תוחפל 1234 ספוט תועצמאב תשגומ השקבה .)Directorate", "Directorate). הבקשה מוגשת באמצעות טופס 1234 לפחות 7 ימים מראש."),
    ("15/11/2026-ה דע םלושי רזחהה .מ\"ק 10 לע הלוע", "עולה על 10 ק\"מ. ההחזר ישולם עד ה-15/11/2026"),
    (".רודמה שאר ידי לע ,דבלב 50% לש רזחה", "החזר של 50% בלבד, על ידי ראש המדור."),
    ("ישדוח קנעמ 1,250.50 דדוב לייח", "חייל בודד 1,250.50 מענק חודשי"),
    (".תכרעמב 401 ספוט תשגה .1", "1. הגשת טופס 401 במערכת."),
    ("www.idf.il רתאל וא", "או לאתר www.idf.il"),
    # A left-to-right line keeps its order; only the embedded Hebrew run is reversed.
    ("The term רושיק ןיצק is used.", "The term קצין קישור is used."),
]


@pytest.mark.parametrize(("visual", "logical"), VISUAL_TO_LOGICAL)
def test_visual_lines_become_logical_reading_order(visual, logical):
    assert visual_to_logical(visual) == logical


def _chars(visual, *, x0=100.0, top=100.0, size=10.0, width=5.0, font="Arial"):
    chars, x = [], x0
    for character in visual:
        chars.append({"text": character, "x0": x, "x1": x + width, "top": top, "bottom": top + size,
                      "size": size, "fontname": font})
        x += width
    return chars


def test_segments_restore_hebrew_and_split_columns_at_wide_gaps():
    right = _chars("ימי 18-ל יאכז", x0=400)
    left = _chars("1234 ספוט", x0=100)
    doubled = _chars("א", x0=400, top=200) + _chars("א", x0=400.4, top=200) + _chars("ב", x0=405, top=200)
    segments = _segments(left + right)
    assert [segment.text for segment in segments] == ["טופס 1234", "זכאי ל-18 ימי"]
    assert all(segment.rtl for segment in segments)
    # A glyph drawn twice to fake bold is kept once; two equal neighboring letters are both kept.
    assert _segments(doubled)[0].text == "בא"
    assert _segments(_chars("לייח", x0=100))[0].text == "חייל"


def _line(text, top, *, x0=300.0, x1=520.0, size=11.0, bold=False):
    return TextLine(text, x0, x1, top, top + size, size, bold, pdf_markdown.is_rtl(text))


def test_two_column_rtl_band_reads_the_right_column_first():
    wide_right = [_line(f"ימין שורה ארוכה מספר {i}", 200 + 15 * i, x0=320, x1=520) for i in range(3)]
    wide_left = [_line(f"שמאל שורה ארוכה מספר {i}", 200 + 15 * i, x0=90, x1=290) for i in range(3)]
    heading = _line("כותרת על כל הרוחב", 150, x0=90, x1=520)
    ordered = _column_order(wide_left + wide_right + [heading], 612)
    assert [line.text for line in ordered] == [heading.text] + [l.text for l in wide_right] + [l.text for l in wide_left]
    # Narrow label/value pairs (a form) are not columns: they are read row by row.
    form = [_line("שם:", 300 + 15 * i, x0=480, x1=520) for i in range(4)] + [
        _line(f"ערך {i}", 300 + 15 * i, x0=90, x1=130) for i in range(4)]
    assert [line.top for line in _column_order(form, 612)] == sorted(line.top for line in form)


def test_page_markdown_builds_headings_lists_tables_and_drops_running_lines():
    lines = [
        _line("נוהל לדוגמה", 20, size=9),  # running header in the top margin
        _line("כותרת המסמך", 100, size=16, bold=True),
        _line("פסקה ראשונה שנמשכת", 130),
        _line("אל השורה הבאה.", 143),
        _line("• פריט ראשון", 180),
        _line("• פריט שני", 195),
        _line("3", 760, size=9),  # page number in the bottom margin
    ]
    table = pdf_markdown.TableBlock(rows=[["עמודה", "ערך"], ["א", "1"]], top=300, bottom=340)
    page = _RawPage(1, 612, 792, lines, [table], 0, False)
    markdown = _page_markdown(page, {"נוהל לדוגמה"}, 11.0, [16.0])
    assert markdown == (
        "# כותרת המסמך\n\nפסקה ראשונה שנמשכת אל השורה הבאה.\n\n- פריט ראשון\n- פריט שני\n\n"
        "| עמודה | ערך |\n| --- | --- |\n| א | 1 |"
    )


# --- A tiny PDF built at test time (Helvetica, so no font files are needed) ---------------------

def _text(x, top, size, text, bold=False):
    return f"BT /{'F2' if bold else 'F1'} {size} Tf {x} {792 - top - size} Td ({text}) Tj ET"


def _rule(x0, top0, x1, top1):
    return f"{x0} {792 - top0} m {x1} {792 - top1} l S"


def _pdf(pages):
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
    }
    kids, number = [], 5
    for operations in pages:
        stream = "\n".join(operations).encode("latin-1")
        objects[number] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        objects[number + 1] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {number} 0 R >>"
        ).encode()
        kids.append(number + 1)
        number += 2
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode()
    out, offsets = b"%PDF-1.4\n", {}
    for key in sorted(objects):
        offsets[key] = len(out)
        out += f"{key} 0 obj\n".encode() + objects[key] + b"\nendobj\n"
    xref, size = len(out), max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offsets[i]:010d} 00000 n \n".encode() for i in range(1, size))
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


BODY = "Leave requests are submitted through the personnel portal at least seven days in advance"


@pytest.fixture
def policy_pdf(tmp_path):
    grid = [_rule(72, 300, 372, 300), _rule(72, 320, 372, 320), _rule(72, 340, 372, 340),
            _rule(72, 300, 72, 340), _rule(222, 300, 222, 340), _rule(372, 300, 372, 340)]
    page_one = [
        _text(72, 30, 9, "Leave Policy Draft"),
        _text(72, 80, 18, "Leave Policy", bold=True),
        _text(72, 120, 11, BODY),
        _text(72, 134, 11, "and approved by the direct commander."),
        _text(72, 170, 14, "Eligibility", bold=True),
        _text(72, 200, 11, "- Reservists receive two extra days."),
        _text(72, 214, 11, "- Parents receive one extra day per child."),
        *grid,
        _text(80, 305, 11, "Group"), _text(230, 305, 11, "Days"),
        _text(80, 325, 11, "Reservists"), _text(230, 325, 11, "2"),
        _text(300, 760, 9, "1"),
    ]
    page_two = [
        _text(72, 30, 9, "Leave Policy Draft"),
        _text(72, 80, 14, "Appendix", bold=True),
        _text(72, 120, 11, BODY + " again."),
        _text(300, 760, 9, "2"),
    ]
    path = tmp_path / "policy.pdf"
    path.write_bytes(_pdf([page_one, page_two]))
    return path


def test_local_conversion_of_a_real_pdf(policy_pdf):
    result = convert_local(policy_pdf, ocr="off")
    markdown = result.markdown
    assert markdown.startswith('---\nid: "policy"\ntitle: "Leave Policy"\nsource: "policy.pdf"\npages: 2\n')
    assert "<!-- page: 1 -->\n\n# Leave Policy\n\n" + BODY + " and approved by the direct commander." in markdown
    assert "## Eligibility\n\n- Reservists receive two extra days.\n- Parents receive one extra day per child." in markdown
    assert "| Group | Days |\n| --- | --- |\n| Reservists | 2 |" in markdown
    assert "<!-- page: 2 -->\n\n## Appendix" in markdown
    # Running header and page numbers are gone.
    assert "Draft" not in markdown and "\n1\n" not in markdown and "\n2\n" not in markdown
    assert result.report()["methods"] == {"text": 2}


def test_page_without_text_is_reported_for_ocr(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(_pdf([[_rule(72, 100, 300, 100)]]))
    result = convert_local(path, ocr="off")
    assert result.pages[0].method == "empty" and result.report()["pages_needing_ocr"] == [1]


def test_text_agreement_tolerates_reversed_reference_words():
    reference = " ".join(f"מילה{index}" for index in range(20))
    assert text_agreement(reference, reference) == (1.0, 1.0)
    reversed_reference = " ".join(word[::-1] for word in reference.split())
    assert text_agreement(reversed_reference, reference) == (1.0, 1.0)
    assert text_agreement("קצר מדי", "משהו") is None
    recall, precision = text_agreement(reference, " ".join(reference.split()[:10]))
    assert recall == 0.5 and precision == 1.0


class FakeGemini:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def generate(self, prompt, schema, model, *, media=None):
        assert schema is PdfPagesMarkdown
        self.calls.append((prompt, media))
        pages = [int(number) for number in prompt.split("holds pages ", 1)[1].split(" of ", 1)[0].split(", ")]
        return PdfPagesMarkdown(pages=[{"page": page, "markdown": self.answer(page)} for page in pages])


def test_gemini_pages_are_verified_against_the_text_layer(policy_pdf):
    local = {page.number: page.markdown for page in convert_local(policy_pdf, ocr="off").pages}
    llm = FakeGemini(lambda page: local[page].replace("Eligibility", "Eligibility rules"))
    result = convert_gemini(policy_pdf, llm, "flash-lite", pages_per_call=2)
    assert [page.method for page in result.pages] == ["gemini", "gemini"]
    assert result.pages[0].verified is True and result.pages[0].recall == 1.0
    prompt, media = llm.calls[0]
    assert media[0][0] == "application/pdf" and media[0][1].startswith(b"%PDF")
    assert hashlib.sha256(policy_pdf.read_bytes()).hexdigest() in prompt and "Reservists" in prompt
    assert "\n# Leave Policy" in result.markdown and "(gemini)" in result.markdown


def test_gemini_page_that_drops_text_is_retried_then_replaced_by_the_local_page(policy_pdf):
    llm = FakeGemini(lambda page: "# Leave Policy\n\nShort summary only." if page == 1 else "")
    result = convert_gemini(policy_pdf, llm, "flash-lite", pages_per_call=2)
    first = result.pages[0]
    assert first.method == "local_fallback" and first.verified is False and first.recall < 0.5
    assert "disagreed with the text layer" in first.warnings[-1]
    assert BODY in first.markdown
    assert len(llm.calls) == 3  # one batch, then each failing page alone
    assert result.pages[1].method == "local_fallback" and "model output missing" in result.pages[1].warnings[-1]


def test_call_cache_keys_include_attachments(tmp_path):
    class Echo:
        calls = 0

        def generate(self, prompt, schema, model, **kwargs):
            Echo.calls += 1
            return PdfPagesMarkdown(pages=[{"page": 1, "markdown": str(Echo.calls)}])

    cache = CachedStructuredLLM(Echo(), tmp_path / "calls.jsonl")
    first = cache.generate("p", PdfPagesMarkdown, "m", media=[("application/pdf", b"one")])
    again = cache.generate("p", PdfPagesMarkdown, "m", media=[("application/pdf", b"one")])
    other = cache.generate("p", PdfPagesMarkdown, "m", media=[("application/pdf", b"two")])
    assert first == again and other != first and Echo.calls == 2


def test_convert_pdf_command_writes_markdown_and_hidden_reports_without_an_api_key(tmp_path, policy_pdf, monkeypatch):
    from chatbot_eval import cli
    from chatbot_eval.documents import load_chunks

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = tmp_path / "config.toml"
    config.write_text("[runtime]\nprogress_enabled = false\n", encoding="utf-8")
    corpus, output = tmp_path / "pdfs", tmp_path / "markdown"
    (corpus / "nested").mkdir(parents=True)
    policy_pdf.rename(corpus / "nested" / "policy.pdf")
    assert cli.main(["--config", str(config), "convert-pdf", "--input", str(corpus), "--output", str(output), "--ocr", "off"]) == 0
    report = json.loads((output / ".alloy" / "conversion_report.json").read_text(encoding="utf-8"))
    assert report[0]["source"] == "nested/policy.pdf" and report[0]["methods"] == {"text": 2}
    chunks = load_chunks(output, 2000, 200)
    # The Markdown is a ready corpus: page provenance survives and the hidden reports are not read.
    assert {chunk.file for chunk in chunks} == {"nested/policy.md"}
    assert [chunk.location.split(",")[0] for chunk in chunks] == ["page 1", "page 2"]
