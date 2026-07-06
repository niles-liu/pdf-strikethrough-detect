"""v0.7.0 "beyond PDFs" surface: image-file input, cloud-OCR adapters (Textract / Document AI),
DOCX detection, and RAG provenance text."""
import io
import json
import zipfile

import fitz
from PIL import Image

import pdf_strikethrough as st
from pdf_strikethrough.ocr import Word


# ------------------------------------------------------------------------- fixtures

def _synthetic_native_pdf():
    """Born-digital PDF: one line with a vector strike through the consecutive 'deleted text'."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "keep this deleted text here", fontsize=12)
    words = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}
    r1, r2 = words["deleted"], words["text"]
    ymid = (r1.y0 + r1.y1) / 2
    page.draw_line(fitz.Point(r1.x0 - 1, ymid), fitz.Point(r2.x1 + 1, ymid), width=1.0)
    return doc


def _mixed_native_scanned_pdf():
    """Page 0 native (struck); page 1 an image-only render of it (classified scanned)."""
    doc = _synthetic_native_pdf()
    scan = doc.new_page(width=595, height=842)
    scan.insert_image(scan.rect, pixmap=doc[0].get_pixmap(dpi=72))
    return doc


def _redline_image_png(dpi=200):
    """PNG bytes of a rendered page with a vector strike through 'struck', + its Word boxes."""
    doc = fitz.open()
    page = doc.new_page(width=300, height=120)
    page.insert_text((20, 60), "keep struck", fontsize=20)
    b = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}
    r = b["struck"]
    ymid = (r.y0 + r.y1) / 2
    page.draw_line(fitz.Point(r.x0, ymid), fitz.Point(r.x1, ymid), width=1.5)
    png = page.get_pixmap(dpi=dpi).tobytes("png")
    W, H = page.rect.width, page.rect.height
    words = [Word(w[4], (w[0] / W, w[1] / H, w[2] / W, w[3] / H)) for w in page.get_text("words")]
    doc.close()
    return png, words


def _docx_bytes(body_xml):
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xml = (f'<?xml version="1.0"?><w:document xmlns:w="{ns}"><w:body>'
           f'{body_xml}</w:body></w:document>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


# ------------------------------------------------------------------------- R-cloud adapters

def test_words_from_textract_pages_and_confidence():
    resp = {"Blocks": [
        {"BlockType": "PAGE", "Page": 1},
        {"BlockType": "WORD", "Text": "hello", "Confidence": 99.0, "Page": 1,
         "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.2, "Width": 0.1, "Height": 0.03}}},
        {"BlockType": "WORD", "Text": "world", "Confidence": 50.0, "Page": 2,
         "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.2, "Width": 0.1, "Height": 0.03}}},
        {"BlockType": "WORD", "Text": "   ", "Page": 1,
         "Geometry": {"BoundingBox": {"Left": 0.0, "Top": 0.0, "Width": 0.1, "Height": 0.1}}},
    ]}
    wbp = st.words_from_textract(resp)
    assert set(wbp) == {0, 1}                             # 0-based pages
    assert [w.text for w in wbp[0]] == ["hello"]          # blank word skipped
    assert abs(wbp[0][0].confidence - 0.99) < 1e-9        # 0..100 scaled to 0..1
    assert wbp[0][0].bbox == (0.1, 0.2, 0.2, 0.23)        # Left/Top + Width/Height -> x0y0x1y1


def test_words_from_docai_camel_and_snake_case():
    doc = {"text": "hello world", "pages": [{
        "dimension": {"width": 800, "height": 1000},
        "tokens": [
            {"layout": {"textAnchor": {"textSegments": [{"startIndex": "0", "endIndex": "5"}]},
                        "boundingPoly": {"normalizedVertices": [
                            {"x": .1, "y": .2}, {"x": .2, "y": .2}, {"x": .2, "y": .23}]},
                        "confidence": 0.97}},
            {"layout": {"text_anchor": {"text_segments": [{"start_index": "6", "end_index": "11"}]},
                        "bounding_poly": {"vertices": [{"x": 80, "y": 200}, {"x": 160, "y": 230}]},
                        "confidence": 0.9}},
        ]}]}
    wbp = st.words_from_docai(doc)
    assert [w.text for w in wbp[0]] == ["hello", "world"]
    assert wbp[0][0].bbox == (0.1, 0.2, 0.2, 0.23)        # normalized vertices used directly
    assert wbp[0][1].bbox == (0.1, 0.2, 0.2, 0.23)        # pixel vertices normalized by dimension


def test_detect_pdf_words_by_page_routes_and_defaults_confidence_free(monkeypatch):
    """words_by_page supplies a scanned page's words in place of an ocr backend, and defaults
    scan_config to confidence-free (cloud confidences aren't calibrated to the classifier)."""
    from pdf_strikethrough import detect as D
    doc = _mixed_native_scanned_pdf()
    pw, ph = doc[1].rect.width, doc[1].rect.height
    x0, y0, x1, y1 = {w[4]: w[:4] for w in doc[0].get_text("words")}["deleted"]
    supplied = [Word("deleted", (x0 / pw, y0 / ph, x1 / pw, y1 / ph), 0.5)]
    seen = {}

    def fake_scanned(gray, words, config=None, meta=None, dpi=200, crop_sink=None):
        seen["words"], seen["gating"] = words, config.confidence_gating
        return []
    monkeypatch.setattr(D, "detect_scanned_image", fake_scanned)
    st.detect_pdf(doc, words_by_page={1: supplied}, pages=[1])
    assert seen["words"] is supplied              # routed the supplied words (no ocr backend)
    assert seen["gating"] is False                # confidence-free default
    doc.close()


# ------------------------------------------------------------------------- R-img image input

def test_detect_image_file_flags_struck_word():
    png, words = _redline_image_png()
    res = st.detect_image_file(png, words=words, dpi=200)
    assert res["page_sources"] == ["scanned"] and res["page_count"] == 1
    assert [w["chars"] for w in res["words"] if w["final"]] == ["struck"]
    assert res["clean_text"] == "keep"


def test_detect_image_file_multipage_words_requires_ocr():
    png, words = _redline_image_png()
    im = Image.open(io.BytesIO(png)).convert("L")
    buf = io.BytesIO()
    im.save(buf, format="TIFF", save_all=True, append_images=[im])   # 2-frame TIFF
    try:
        st.detect_image_file(buf.getvalue(), words=words)
        assert False, "expected ValueError for a multi-frame image + single-frame words="
    except ValueError:
        pass


def test_detect_image_file_ocr_defaults_confidence_free(monkeypatch):
    """An image has no Azure-DI calibration source, so an ocr backend must default to
    confidence-free (RapidOCR/Tesseract confidences aren't calibrated to the classifier)."""
    from pdf_strikethrough import detect as D
    _png, words = _redline_image_png()
    seen = {}

    def fake_scanned(gray, w, config=None, meta=None, dpi=200, crop_sink=None):
        seen["gating"] = config.confidence_gating
        return []
    monkeypatch.setattr(D, "detect_scanned_image", fake_scanned)
    st.detect_image_file(_png, ocr=lambda img: words)
    assert seen["gating"] is False


def test_image_frames_reads_and_omits_dpi_metadata():
    from pdf_strikethrough import detect as D
    buf = io.BytesIO()
    Image.new("L", (100, 50), 255).save(buf, format="PNG", dpi=(150, 150))
    frames = D._image_frames(buf.getvalue())
    assert len(frames) == 1 and frames[0][1] == 150
    buf2 = io.BytesIO()
    Image.new("L", (10, 10)).save(buf2, format="PPM")                # PPM carries no dpi
    assert D._image_frames(buf2.getvalue())[0][1] is None


# ------------------------------------------------------------------------- R-docx

def test_docx_strike_and_tracked_deletion():
    body = ('<w:p><w:r><w:t>keep </w:t></w:r>'
            '<w:r><w:rPr><w:strike/></w:rPr><w:t>gone</w:t></w:r>'
            '<w:r><w:rPr><w:strike w:val="false"/></w:rPr><w:t>stay</w:t></w:r></w:p>'
            '<w:p><w:del w:author="Ed" w:date="2026-07-05T00:00:00Z" w:id="3">'
            '<w:r><w:delText>removed</w:delText></w:r></w:del></w:p>')
    recs = st.strikethroughs_in_docx(_docx_bytes(body))
    assert [(r["chars"], r["docx_change"]) for r in recs] == [
        ("gone", "format"), ("removed", "deletion")]      # 'stay' (w:val=false) excluded
    assert [r["para"] for r in recs] == [0, 1]
    assert recs[1]["docx_author"] == "Ed" and recs[1]["docx_date"].startswith("2026")
    assert all(r["tier"] == "docx" and r["final"] for r in recs)


def test_docx_double_strike_flagged():
    recs = st.strikethroughs_in_docx(_docx_bytes(
        '<w:p><w:r><w:rPr><w:dstrike/></w:rPr><w:t>doubled</w:t></w:r></w:p>'))
    assert recs[0]["docx_double"] is True and recs[0]["chars"] == "doubled"


def test_docx_bad_input_raises():
    try:
        st.strikethroughs_in_docx(b"not a zip file")
        assert False, "expected ValueError for a non-docx input"
    except ValueError:
        pass


# ------------------------------------------------------------------------- R-rag provenance

def test_provenance_text_marks_and_merges_passages():
    doc = _synthetic_native_pdf()                         # strikes consecutive 'deleted text'
    res = st.detect_pdf(doc, method="vector")
    prov = st.provenance_text(res)
    assert "[deleted: deleted text]" in prov              # a passage merges to one marker
    assert "keep this" in prov and "here" in prov
    assert st.provenance_text(res, template="<{}>").count("<deleted text>") == 1
    doc.close()


def test_mark_provenance_merge_and_newline_boundary():
    from pdf_strikethrough import markdown as M
    assert M.mark_provenance("a ~~b~~ c") == "a [deleted: b] c"
    assert M.mark_provenance("~~x~~ ~~y~~") == "[deleted: x y]"          # space gap -> merged
    assert M.mark_provenance("~~x~~\n~~y~~") == "[deleted: x]\n[deleted: y]"   # newline -> kept


# ------------------------------------------------------------------------- CLI routing

def test_cli_detects_docx_and_json(tmp_path):
    from pdf_strikethrough import __main__ as cli
    p = tmp_path / "d.docx"
    p.write_bytes(_docx_bytes(
        '<w:p><w:r><w:rPr><w:strike/></w:rPr><w:t>gone</w:t></w:r></w:p>'
        '<w:p><w:del w:author="Z"><w:r><w:delText>x</w:delText></w:r></w:del></w:p>'))
    assert cli.main(["detect", str(p), "--fail-if-found"]) == 3      # struck runs -> exit 3
    out = tmp_path / "o.json"
    assert cli.main(["detect", str(p), "--json", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["kind"] == "docx"
    assert payload["words"][1]["docx_author"] == "Z"


def test_cli_image_with_textract_result(tmp_path):
    from pdf_strikethrough import __main__ as cli
    png, words = _redline_image_png()
    img = tmp_path / "s.png"
    Image.open(io.BytesIO(png)).save(img, dpi=(200, 200))
    blocks = [{"BlockType": "WORD", "Text": w.text, "Confidence": 99.0, "Page": 1,
               "Geometry": {"BoundingBox": {"Left": w.bbox[0], "Top": w.bbox[1],
                            "Width": w.bbox[2] - w.bbox[0], "Height": w.bbox[3] - w.bbox[1]}}}
              for w in words]
    tj = tmp_path / "t.json"
    tj.write_text(json.dumps({"Blocks": blocks}), encoding="utf-8")
    out = tmp_path / "o.json"
    assert cli.main(["detect", str(img), "--textract-result", str(tj), "--json", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["page_sources"] == ["scanned"]
    assert any(w["chars"] == "struck" for w in payload["words"])


def test_cli_provenance_output(tmp_path):
    from pdf_strikethrough import __main__ as cli
    pdf = tmp_path / "r.pdf"
    doc = _synthetic_native_pdf()
    pdf.write_bytes(doc.tobytes())
    doc.close()
    out = tmp_path / "prov.txt"
    assert cli.main(["detect", str(pdf), "--provenance", str(out)]) == 0
    assert "[deleted:" in out.read_text(encoding="utf-8")
