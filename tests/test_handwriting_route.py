"""The opt-in handwriting route on the scanned path (ScanConfig.handwriting).

The line finder still runs on every word; afterwards each handwritten word's records are replaced
by one record for the whole word, marked struck when StrikeNet scores it at or above p_hi. The
record carries the line finder's geometry and line from a record the line path scores, so the
DI-confidence veto and line support read it as they do there. "styled" picks the words the OCR
engine marks handwritten (``Word.handwritten``: Azure DI's styles through
``words_from_azure_di(..., styles=...)``, Textract's TextType), "all" every word. Off by default,
where the output is unchanged.
"""
import dataclasses
import json

import numpy as np
import pymupdf
import pytest

import pdf_strikethrough as st
from pdf_strikethrough import cnn, detect
from pdf_strikethrough.ocr import Word, words_from_azure_di
from pdf_strikethrough.scanned import ScanConfig

CF = ScanConfig.confidence_free()


# ------------------------------------------------------------------------- Word and the adapter

def test_word_handwritten_defaults_to_unknown():
    w = Word("a", (0.1, 0.1, 0.2, 0.2), 0.9)
    assert w.handwritten is None
    assert Word("a", (0.1, 0.1, 0.2, 0.2), 0.9, 1).handwritten is True
    assert Word("a", (0.1, 0.1, 0.2, 0.2), handwritten=0).handwritten is False


def _di_page():
    """A DI page (inches) with three words: 'Os' and 'livros' in a handwritten style, 'Pag.'
    printed, and one word whose span is missing."""
    def word(text, x, off):
        return {"content": text, "polygon": [x, 1.0, x + 0.5, 1.0, x + 0.5, 1.2, x, 1.2],
                "confidence": 0.9, "span": {"offset": off, "length": len(text)}}
    words = [word("Os", 1.0, 0), word("livros", 2.0, 3), word("Pag.", 4.0, 10)]
    words.append({"content": "x", "polygon": [5.0, 1.0, 5.2, 1.0, 5.2, 1.2, 5.0, 1.2]})
    return {"width": 8.5, "height": 11.0, "unit": "inch", "words": words}


STYLES = [{"isHandwritten": True, "confidence": 1, "spans": [{"offset": 0, "length": 9}]},
          {"isHandwritten": False, "spans": [{"offset": 10, "length": 4}]}]


def test_styles_mark_words_handwritten():
    ws = words_from_azure_di(_di_page(), styles=STYLES)
    assert [(w.text, w.handwritten) for w in ws] == [
        ("Os", True), ("livros", True), ("Pag.", False), ("x", False)]


def test_without_styles_handwriting_is_unknown():
    assert {w.handwritten for w in words_from_azure_di(_di_page())} == {None}


def test_sdk_style_objects_are_read():
    class Obj(dict):
        def as_dict(self):
            return dict(self)
    ws = words_from_azure_di(_di_page(), styles=[Obj(STYLES[0])])
    assert [w.handwritten for w in ws][:2] == [True, True]


def test_handwritten_is_left_out_of_equality_and_the_hash():
    """Words are used as dict keys and compared downstream; a Word built before the field existed
    still equals one an adapter marks."""
    a, b = Word("a", (0.1, 0.1, 0.2, 0.2), 0.9), Word("a", (0.1, 0.1, 0.2, 0.2), 0.9, True)
    assert a == b and hash(a) == hash(b) and {a: 1}[b] == 1
    assert Word("a", (0.1, 0.1, 0.2, 0.2), 0.8, True) != b                # the rest still counts


def test_textract_text_type_marks_words():
    from pdf_strikethrough.ocr import words_from_textract
    box = {"Left": 0.1, "Top": 0.1, "Width": 0.1, "Height": 0.05}
    blocks = [{"BlockType": "PAGE"}] + [
        {"BlockType": "WORD", "Text": t, "Confidence": 99.0, "Geometry": {"BoundingBox": box},
         **({"TextType": k} if k else {})}
        for t, k in (("pen", "HANDWRITING"), ("print", "PRINTED"), ("old", None))]
    got = words_from_textract({"Blocks": blocks})[0]
    assert [(w.text, w.handwritten) for w in got] == [("pen", True), ("print", False),
                                                      ("old", None)]


def test_word_mostly_outside_a_style_is_not_handwritten():
    styles = [{"isHandwritten": True, "spans": [{"offset": 3, "length": 2}]}]   # 2 of 6 chars
    assert words_from_azure_di(_di_page(), styles=styles)[1].handwritten is False


# --------------------------------------------------------------------------------- ScanConfig

def test_config_default_and_validation():
    assert ScanConfig().handwriting == "off" == ScanConfig.confidence_free().handwriting
    with pytest.raises(ValueError, match="handwriting"):
        ScanConfig(handwriting="yes")


# --------------------------------------------------------------------------------- the route

def _page():
    """One line of text, 'the' struck by a 1 pt line, rendered at 200 dpi: (gray, words)."""
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=120)
    page.insert_text((24, 64), "keep the deposit safe", fontname="helv", fontsize=12)
    ws = page.get_text("words")
    y = 64 - 12 * pymupdf.Font("helv").glyph_bbox(ord("x")).y1 / 2
    page.draw_line((ws[1][0], y), (ws[1][2], y), width=1.0)
    words = [Word(w[4], (w[0] / 420, w[1] / 120, w[2] / 420, w[3] / 120), None) for w in ws]
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    return gray, words


def _scores(monkeypatch, by_text):
    """StrikeNet replaced by a table keyed on the word each crop belongs to, in call order."""
    seen = []

    def score(crops, **kw):
        out = np.array([by_text[seen.pop(0)] for _ in crops])
        return out

    real = cnn.word_crop_px

    def crop(gray, bbox, *a, **k):
        c = real(gray, bbox, *a, **k)
        if c is not None:
            seen.append(_TEXT[bbox])
        return c

    monkeypatch.setattr(cnn, "score_crops", score)
    monkeypatch.setattr(cnn, "word_crop_px", crop)


_TEXT = {}


def test_off_is_the_line_path():
    gray, words = _page()
    a = st.detect_scanned_image(gray, words, config=CF, dpi=200)
    b = st.detect_scanned_image(gray, words, config=ScanConfig(confidence_gating=False,
                                                               handwriting="off"), dpi=200)
    assert a == b and a


def test_all_sends_every_word_to_the_model(monkeypatch):
    gray, words = _page()
    _TEXT.clear()
    _TEXT.update({w.bbox: w.text for w in words})
    _scores(monkeypatch, {"keep": 0.10, "the": 0.20, "deposit": 0.97, "safe": 0.50})
    cfg = ScanConfig(confidence_gating=False, handwriting="all", cnn_p_hi=0.9, cnn_p_lo=0.15)
    recs = st.detect_scanned_image(gray, words, config=cfg, dpi=200)
    assert [r["text"] for r in recs] == ["keep", "the", "deposit", "safe"]
    assert all(r["handwriting"] and r["tier"] == "review" and not r["partial"] for r in recs)
    assert all(isinstance(r["score"], float) and r["score"] == 0.0 for r in recs)   # no geometry
    # the line through "the" is not used: the model alone decides, at p_hi
    assert [r["text"] for r in recs if r["final"]] == ["deposit"]
    assert [r["char_span"] for r in recs] == [(0, 4), (0, 3), (0, 7), (0, 4)]


def test_styled_routes_only_the_marked_words(monkeypatch):
    gray, words = _page()
    words = [Word(w.text, w.bbox, w.confidence, w.text in ("deposit", "safe")) for w in words]
    _TEXT.clear()
    _TEXT.update({w.bbox: w.text for w in words})
    _scores(monkeypatch, {"keep": 0.0, "the": 0.99, "deposit": 0.95, "safe": 0.10})
    cfg = ScanConfig(confidence_gating=False, handwriting="styled", cnn_p_hi=0.9, cnn_p_lo=0.15)
    recs = st.detect_scanned_image(gray, words, config=cfg, dpi=200)
    by = {r["text"]: r for r in recs}
    assert by["the"].get("handwriting") is None            # printed: found on its line
    assert by["deposit"]["handwriting"] and by["deposit"]["final"]
    assert by["safe"]["handwriting"] and not by["safe"]["final"]
    assert all(by[t].get("handwriting") is None for t in by if t in ("keep", "the"))


def test_styled_with_no_handwritten_words_is_the_line_path():
    gray, words = _page()
    cfg = ScanConfig(confidence_gating=False, handwriting="styled")
    assert (st.detect_scanned_image(gray, words, config=cfg, dpi=200)
            == st.detect_scanned_image(gray, words, config=CF, dpi=200))


def test_detect_pdf_reads_the_styles_from_the_di_result(monkeypatch):
    """detect_pdf passes the result's styles to the adapter, so "styled" routes DI's handwritten
    words; with StrikeNet confirming everything, every handwritten word is marked."""
    gray, words = _page()
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=120)
    pix = pymupdf.Pixmap(pymupdf.csGRAY, gray.shape[1], gray.shape[0], gray.tobytes(), False)
    page.insert_image(page.rect, pixmap=pix)
    di_words, off = [], 0
    for w in words:
        x0, y0, x1, y1 = (w.bbox[0] * 4.2, w.bbox[1] * 1.2, w.bbox[2] * 4.2, w.bbox[3] * 1.2)
        di_words.append({"content": w.text, "polygon": [x0, y0, x1, y0, x1, y1, x0, y1],
                         "confidence": 0.99, "span": {"offset": off, "length": len(w.text)}})
        off += len(w.text) + 1
    di = {"analyzeResult": {
        "pages": [{"pageNumber": 1, "width": 4.2, "height": 1.2, "unit": "inch",
                   "words": di_words}],
        "styles": [{"isHandwritten": True, "spans": [{"offset": 0, "length": off}]}]}}
    monkeypatch.setattr(cnn, "score_crops", lambda crops, **kw: np.ones(len(crops)))
    res = st.detect_pdf(doc.tobytes(), di_result=di,
                        scan_config=ScanConfig(handwriting="styled", confidence_gating=False))
    marked = [w for w in res["words"] if w.get("final")]
    assert sorted(w["text"] for w in marked) == ["deposit", "keep", "safe", "the"]
    assert all(w.get("handwriting") for w in marked)


DI = ScanConfig.azure_di()


def _struck_page(confs):
    """'keep the deposit minimum' at 16 pt and 200 dpi, 'deposit' struck by a straight 1.2 pt line;
    every word handwritten, read at `confs`: (gray, words)."""
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=120)
    page.insert_text((24, 64), "keep the deposit minimum", fontname="helv", fontsize=16)
    ws = page.get_text("words")
    y = 64 - 16 * pymupdf.Font("helv").glyph_bbox(ord("x")).y1 / 2
    page.draw_line((ws[2][0], y), (ws[2][2], y), width=1.2)
    words = [Word(w[4], (w[0] / 420, w[1] / 120, w[2] / 420, w[3] / 120), c, True)
             for w, c in zip(ws, confs)]
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    return gray, words


def _as_pdf(gray):
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=120)
    pix = pymupdf.Pixmap(pymupdf.csGRAY, gray.shape[1], gray.shape[0], gray.tobytes(), False)
    page.insert_image(page.rect, pixmap=pix)
    return doc.tobytes()


def _rec(text, box, tier, geom, line_idx):
    """A layer-1 record as consolidate_struck makes one."""
    return {"text": text, "chars": text, "char_span": (0, len(text)), "partial": False,
            "bbox_frac": box, "conf": 0.99, "off": 0.0, "wcov": 1.0, "line_idx": line_idx,
            "twin": False, "geom_corroborated": geom, "score": 0.3, "tier": tier}


def _di_page_with_a_spanless_word():
    def word(text, x, off):
        return {"content": text, "polygon": [x, 1.0, x + 0.5, 1.0, x + 0.5, 1.2, x, 1.2],
                "confidence": 0.9, "span": {"offset": off, "length": len(text)}}
    words = [word("Os", 1.0, 0), word("livros", 2.0, 3), word("Pag.", 4.0, 10)]
    words.append({"content": "x", "polygon": [5.0, 1.0, 5.2, 1.0, 5.2, 1.2, 5.0, 1.2]})
    return {"width": 8.5, "height": 11.0, "unit": "inch", "words": words}


# ------------------------------------------------------------------- the veto and line support

def test_the_veto_reads_the_line_finders_geometry_on_a_routed_word(monkeypatch):
    """With confidence gating, a routed word read above max_clean_conf is kept when the line
    finder found strike geometry on it, as the line path keeps it, and vetoed when it found none."""
    gray, words = _struck_page([0.8, 0.8, 0.99, 0.99])
    monkeypatch.setattr(cnn, "score_crops", lambda crops, **kw: np.full(len(crops), 0.99))
    off = {r["text"]: r for r in st.detect_scanned_image(gray, words, config=DI, dpi=200)}
    assert off["deposit"]["final"] and off["deposit"]["geom_corroborated"]
    on = {r["text"]: r for r in st.detect_scanned_image(
        gray, words, config=dataclasses.replace(DI, handwriting="styled"), dpi=200)}
    assert on["deposit"]["handwriting"] and on["deposit"]["final"]
    assert on["deposit"]["geom_corroborated"] and "conf_veto" not in on["deposit"]
    assert on["minimum"]["handwriting"] and not on["minimum"]["final"] and on["minimum"]["conf_veto"]
    assert on["keep"]["final"] and on["the"]["final"]           # read at 0.8: never vetoed


def test_a_routed_word_takes_evidence_only_from_a_record_the_line_path_scores():
    """A 'weak' line record never reaches the CNN on the line path, so it lends the route record
    no geometry; an auto/review one lends its geom_corroborated and line_idx."""
    a, b, c = (0.1, 0.1, 0.2, 0.15), (0.3, 0.1, 0.4, 0.15), (0.5, 0.1, 0.6, 0.15)
    words = [Word("weak", a, 0.99, True), Word("seen", b, 0.99, True),
             Word("none", c, 0.99, True)]
    struck = [_rec("weak", a, "weak", True, [0]), _rec("seen", b, "review", True, [1])]
    out = {h["text"]: h for h in detect._handwriting_route(struck, words, "styled")}
    assert [out[t]["geom_corroborated"] for t in ("weak", "seen", "none")] == [False, True, False]
    assert [out[t]["line_idx"] for t in ("weak", "seen", "none")] == [[], [1], []]
    assert all(h["handwriting"] and h["tier"] == "review" and h["score"] == 0.0
               for h in out.values())


def test_line_support_keeps_a_short_routed_word_as_on_the_line_path(monkeypatch):
    """'deleted of words' under one stroke, StrikeNet confirming the long words only: line support
    keeps "of" with the route on, as with it off."""
    H, W, Y0, Y1 = 400, 900, 180, 220
    spec = [("deleted", 100, 300), ("of", 312, 340), ("words", 352, 552)]
    gray = np.full((H, W), 255, np.uint8)
    for _, x0, x1 in spec:
        gray[Y0 + 6:Y0 + 10, x0:x1] = 20
        gray[Y1 - 10:Y1 - 6, x0:x1] = 20
        for x in range(x0, x1, 8):
            gray[Y0 + 6:Y1 - 6, x:x + 2] = 20
    gray[199:202, 100:552] = 0
    words = [Word(t, (a / W, Y0 / H, b / W, Y1 / H), None, True) for t, a, b in spec]
    probs = {w.bbox: p for w, p in zip(words, (0.99, 0.30, 0.99))}
    monkeypatch.setattr(cnn, "word_crop_px", lambda gray, bbox, **kw: bbox)
    monkeypatch.setattr(cnn, "std_crop", lambda crop: crop)
    monkeypatch.setattr(cnn, "score_crops",
                        lambda crops, **kw: np.array([probs[c] for c in crops]))
    for mode in ("off", "all"):
        cfg = dataclasses.replace(ScanConfig.confidence_free(), handwriting=mode)
        by = {r["text"]: r for r in st.detect_scanned_image(gray, words, config=cfg, dpi=200)}
        assert by["of"]["final"] and by["of"].get("line_support"), mode


# ------------------------------------------------------------- order, blank words, duplicates

def test_route_records_are_one_per_word_in_reading_order_and_none_for_a_blank_word():
    left, right, blank = (0.1, 0.1, 0.2, 0.15), (0.5, 0.1, 0.6, 0.15), (0.3, 0.1, 0.35, 0.15)
    words = [Word("pen", left, 0.9, True), Word("print", right, 0.9, False),
             Word("pen", left, 0.9, True), Word(" ", blank, 0.9, True)]
    out = detect._handwriting_route([_rec("print", right, "auto", True, [0])], words, "styled")
    assert [(h["text"], bool(h.get("handwriting"))) for h in out] == [("pen", True),
                                                                       ("print", False)]


def test_snake_case_styles_and_an_empty_styles_list():
    """azure-ai-formrecognizer's to_dict() spells it is_handwritten; a result whose styles list is
    empty marks every word printed, as the docs say (with styles: False)."""
    styles = [{"is_handwritten": True, "spans": [{"offset": 0, "length": 9}]}]
    assert [w.handwritten for w in words_from_azure_di(_di_page_with_a_spanless_word(), styles=styles)] == [
        True, True, False, False]
    assert {w.handwritten for w in words_from_azure_di(_di_page_with_a_spanless_word(), styles=[])} == {False}


# ------------------------------------------------------------------- the record contract

def test_telemetry_counts_layer_one_tiers_with_the_route_on(monkeypatch):
    gray, words = _struck_page([0.8, 0.8, 0.99, 0.99])
    monkeypatch.setattr(cnn, "score_crops", lambda crops, **kw: np.full(len(crops), 0.99))
    pdf = _as_pdf(gray)
    tiers = [st.detect_pdf(pdf, words_by_page={0: words}, telemetry=True,
                           scan_config=dataclasses.replace(DI, handwriting=m))
             ["telemetry"]["tiers"] for m in ("off", "all")]
    assert tiers[0] and tiers[0] == tiers[1]


def test_dump_crops_manifest_marks_route_records(tmp_path, monkeypatch):
    gray, words = _struck_page([0.8, 0.8, 0.99, 0.99])
    monkeypatch.setattr(cnn, "score_crops", lambda crops, **kw: np.full(len(crops), 0.99))
    st.dump_crops(_as_pdf(gray), str(tmp_path), words_by_page={0: words},
                  scan_config=dataclasses.replace(DI, handwriting="all"))
    rows = [json.loads(s) for s in
            (tmp_path / "crops.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4 and all(r.get("handwriting") is True for r in rows)


def test_handwriting_is_a_documented_record_key():
    assert "handwriting" in st.StruckWord.__annotations__


def test_styles_given_as_one_dict_is_refused():
    style = {"isHandwritten": True, "spans": [{"offset": 0, "length": 9}]}
    with pytest.raises(TypeError, match="styles"):
        words_from_azure_di(_di_page_with_a_spanless_word(), styles=style)
