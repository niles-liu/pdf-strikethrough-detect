"""The cloud OCR adapters clamp a word box the service places past the edge of the page, as
`rapidocr_backend()` does, so one such box no longer makes `Word` refuse it and, with it, the whole
document. A page where most boxes lie past 1.5, as with boxes in the wrong units, still raises.
A page with no usable words, as a blank scanned page has, comes back with an empty word list.
The payloads are synthetic, in each service's result shape; nothing calls a service."""
import pymupdf
import pytest

import pdf_strikethrough as st
from pdf_strikethrough.ocr import Word, words_from_azure_di, words_from_docai, words_from_textract

# (text, page-fraction box): four on the page, then past the right edge, past the bottom, wholly
# past the right edge, and a little past the left and top edges (a box Word accepted before,
# clamped now). Three of the eight lie past 1.5, a minority, so the page is clamped.
_ON_PAGE = [("kept", (0.1, 0.2, 0.3, 0.4)), ("one", (0.1, 0.5, 0.2, 0.55)),
            ("two", (0.3, 0.5, 0.4, 0.55)), ("three", (0.5, 0.5, 0.6, 0.55))]
_BOXES = _ON_PAGE + [("x=1", (0.778, 0.2, 1.569, 0.4)),
                     ("low", (0.4, 0.9, 0.5, 1.6)),
                     ("edge", (1.6, 0.2, 1.9, 0.4)),
                     ("near", (-0.02, -0.01, 0.1, 0.05))]
_CLAMPED = [b for _, b in _ON_PAGE] + [(0.778, 0.2, 1.0, 0.4), (0.4, 0.9, 0.5, 1.0),
                                       (1.0, 0.2, 1.0, 0.4), (0.0, 0.0, 0.1, 0.05)]
_NAN = float("nan")


def _assert_clamped(words, texts=None, boxes=None):
    assert [w.text for w in words] == (texts or [t for t, _ in _BOXES])
    for w, box in zip(words, boxes or _CLAMPED):
        assert w.bbox == pytest.approx(box)


# ----------------------------------------------------------------------- the service payloads

def _di_page(boxes, width=8.5, height=11.0, page=1):
    """An Azure DI page (REST JSON, flat polygons in inches) holding the given words."""
    return {"pageNumber": page, "width": width, "height": height, "unit": "inch",
            "words": [{"content": t, "confidence": 0.99,
                       "polygon": [x0 * width, y0 * height, x1 * width, y0 * height,
                                   x1 * width, y1 * height, x0 * width, y1 * height]}
                      for t, (x0, y0, x1, y1) in boxes]}


def _textract(*pages):
    """A Textract result: a PAGE block for each page, then the page's WORD blocks."""
    blocks = []
    for p, boxes in enumerate(pages, start=1):
        blocks += [{"BlockType": "PAGE", "Page": p}] + [
            {"BlockType": "WORD", "Text": t, "Confidence": 99.0, "Page": p,
             "Geometry": {"BoundingBox": {"Left": x0, "Top": y0, "Width": x1 - x0,
                                          "Height": y1 - y0}}}
            for t, (x0, y0, x1, y1) in boxes]
    return {"Blocks": blocks}


def _docai(*pages, pixel=False, width=1700, height=2200):
    """A Document AI document: pages whose tokens carry normalized vertices, or pixel vertices
    and the page's dimension."""
    text, out = "", []
    for p, boxes in enumerate(pages, start=1):
        tokens = []
        for t, (x0, y0, x1, y1) in boxes:
            sx, sy = (width, height) if pixel else (1, 1)
            verts = [{"x": x0 * sx, "y": y0 * sy}, {"x": x1 * sx, "y": y0 * sy},
                     {"x": x1 * sx, "y": y1 * sy}, {"x": x0 * sx, "y": y1 * sy}]
            tokens.append({"layout": {
                "textAnchor": {"textSegments": [{"startIndex": len(text),
                                                 "endIndex": len(text) + len(t)}]},
                "boundingPoly": {"vertices" if pixel else "normalizedVertices": verts}}})
            text += t + " "
        page = {"pageNumber": p, "tokens": tokens}
        if pixel:
            page["dimension"] = {"width": width, "height": height}
        out.append(page)
    return {"text": text, "pages": out}


# ----------------------------------------------------------------------- the boxes handed on

def test_azure_di_clamps_a_box_past_the_page():
    _assert_clamped(words_from_azure_di(_di_page(_BOXES)))


def test_textract_clamps_a_box_past_the_page():
    _assert_clamped(words_from_textract(_textract(_BOXES))[0])


@pytest.mark.parametrize("pixel", [False, True])
def test_docai_clamps_a_box_past_the_page(pixel):
    _assert_clamped(words_from_docai(_docai(_BOXES, pixel=pixel))[0])


def test_a_box_that_is_not_finite_is_dropped():
    boxes = [("kept", (0.1, 0.2, 0.3, 0.4)), ("nan", (0.1, _NAN, 0.3, _NAN))]
    one = (["kept"], [(0.1, 0.2, 0.3, 0.4)])
    _assert_clamped(words_from_azure_di(_di_page(boxes)), *one)
    _assert_clamped(words_from_textract(_textract(boxes))[0], *one)
    _assert_clamped(words_from_docai(_docai(boxes))[0], *one)


def test_azure_di_page_with_only_a_box_that_is_not_finite_raises():
    with pytest.raises(ValueError, match="polygon"):
        words_from_azure_di(_di_page([("nan", (0.1, _NAN, 0.3, _NAN))]))


_NAN_ONLY = [("nan", (0.1, _NAN, 0.3, _NAN))]


@pytest.mark.parametrize("middle", [[], _NAN_ONLY], ids=["blank", "not_finite"])
@pytest.mark.parametrize("adapter, payload", [(words_from_textract, _textract),
                                              (words_from_docai, _docai)], ids=["textract", "docai"])
def test_a_page_with_no_usable_words_is_kept_with_none(adapter, payload, middle):
    """A blank scanned page (for Textract, a PAGE block and no WORD blocks) is a key with an empty
    list, so detect_pdf does not take it for a page the OCR left out."""
    wbp = adapter(payload(_ON_PAGE[:1], middle, _ON_PAGE[1:2]))
    assert {p: [w.text for w in ws] for p, ws in wbp.items()} == {0: ["kept"], 1: [], 2: ["one"]}


def test_a_blank_azure_di_page_has_no_words():
    """Only a page with words and no usable box raises; a page with no words is empty."""
    assert words_from_azure_di(_di_page([])) == []


def test_a_word_built_directly_still_refuses_a_box_past_1_5():
    with pytest.raises(ValueError, match="normalized page fractions"):
        Word("x=1", (0.778, 0.2, 1.569, 0.4))


def _in_pixels(boxes, width=1700, height=2200):
    """The boxes in pixels of a 200 dpi letter page, where page fractions belong."""
    return [(t, (x0 * width, y0 * height, x1 * width, y1 * height))
            for t, (x0, y0, x1, y1) in boxes]


def _di_page_px(boxes, width=8.5, height=11.0):
    """An Azure DI page whose polygons are in 200 dpi pixels but whose size is in inches."""
    return dict(_di_page(boxes, width * 200, height * 200), width=width, height=height)


@pytest.mark.parametrize("adapter", [
    lambda: words_from_azure_di(_di_page_px(_ON_PAGE)),
    lambda: words_from_textract(_textract(_in_pixels(_ON_PAGE))),
    lambda: words_from_docai(_docai(_in_pixels(_ON_PAGE))),
], ids=["azure_di", "textract", "docai"])
def test_boxes_in_the_wrong_units_still_raise(adapter):
    with pytest.raises(ValueError, match="pixel coordinates"):
        adapter()


def test_a_page_is_clamped_unless_most_of_its_boxes_lie_past_1_5():
    far = ("far", (1.6, 0.2, 1.9, 0.4))
    half = words_from_textract(_textract([_ON_PAGE[0], far]))[0]
    assert [w.bbox for w in half] == [(0.1, 0.2, 0.3, 0.4), (1.0, 0.2, 1.0, 0.4)]
    with pytest.raises(ValueError, match="pixel coordinates"):
        words_from_textract(_textract([_ON_PAGE[0], far, far]))


# ----------------------------------------------------------------------- the whole document

def _scanned_struck_line():
    """An image-only PDF of 'keep struck' with 'struck' struck through, and its words plus a third,
    'tail', whose box runs from x 0.778 to 1.569, past the 1.5 that Word refuses."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=120)
    page.insert_text((20, 60), "keep struck", fontsize=20)
    W, H = page.rect.width, page.rect.height
    words = [(w[4], (w[0] / W, w[1] / H, w[2] / W, w[3] / H)) for w in page.get_text("words")]
    _, (x0, y0, x1, y1) = words[1]
    page.draw_line((x0 * W, (y0 + y1) / 2 * H), (x1 * W, (y0 + y1) / 2 * H), width=1.5)
    png = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY).tobytes("png")
    scan = pymupdf.open()
    scan.new_page(width=W, height=H).insert_image(pymupdf.Rect(0, 0, W, H), stream=png)
    return scan.tobytes(), words + [("tail", (0.778, y0, 1.569, y1))]


def _assert_read(res):
    assert res["page_sources"] == ["scanned"]
    assert [w["chars"] for w in res["words"] if w["final"]] == ["struck"]
    assert res["clean_text"].split() == ["keep", "tail"]


def test_detect_pdf_reads_past_an_off_page_azure_di_box():
    pdf, words = _scanned_struck_line()
    _assert_read(st.detect_pdf(pdf, di_result={"pages": [_di_page(words, 300 / 72, 120 / 72)]},
                               scan_config=st.ScanConfig.confidence_free()))


def test_detect_pdf_reads_past_an_off_page_textract_box():
    pdf, words = _scanned_struck_line()
    _assert_read(st.detect_pdf(pdf, words_by_page=words_from_textract(_textract(words))))


def test_detect_pdf_reads_past_an_off_page_docai_box():
    pdf, words = _scanned_struck_line()
    _assert_read(st.detect_pdf(pdf, words_by_page=words_from_docai(_docai(words))))


def _scan_with_a_blank_page():
    """The struck-line scan, a blank scanned page, then the struck line again, and each page's
    words (none on the blank page)."""
    one, words = _scanned_struck_line()
    src, scan = pymupdf.open(stream=one), pymupdf.open()
    scan.insert_pdf(src)
    blank = scan.new_page(width=src[0].rect.width, height=src[0].rect.height)
    white = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 60, 24), False)
    white.set_rect(white.irect, (255,))
    blank.insert_image(blank.rect, pixmap=white)
    scan.insert_pdf(src)
    return scan.tobytes(), [words[:2], [], words[:2]]


@pytest.mark.parametrize("provider", ["textract", "docai", "azure_di"])
def test_detect_pdf_reads_a_scan_with_a_blank_page(provider):
    pdf, pages = _scan_with_a_blank_page()
    if provider == "azure_di":
        kw = {"di_result": {"pages": [_di_page(ws, 300 / 72, 120 / 72, page=p)
                                      for p, ws in enumerate(pages, start=1)]},
              "scan_config": st.ScanConfig.confidence_free()}
    else:
        adapter, payload = {"textract": (words_from_textract, _textract),
                            "docai": (words_from_docai, _docai)}[provider]
        kw = {"words_by_page": adapter(payload(*pages))}
    res = st.detect_pdf(pdf, **kw)
    assert res["page_sources"] == ["scanned"] * 3
    assert [(w["page"], w["chars"]) for w in res["words"] if w["final"]] == [(0, "struck"),
                                                                             (2, "struck")]


def test_detect_pdf_still_raises_on_azure_di_polygons_in_the_wrong_units():
    """Before the clamp this raised; clamped, every box would land on the corner and the struck
    word would come back as clean text, with no warning."""
    pdf, words = _scanned_struck_line()
    page = _di_page_px(words[:2], 300 / 72, 120 / 72)
    with pytest.raises(ValueError, match="pixel coordinates"):
        st.detect_pdf(pdf, di_result={"pages": [page]}, scan_config=st.ScanConfig.confidence_free())
