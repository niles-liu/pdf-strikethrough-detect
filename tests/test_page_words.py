"""page_words=True lists every word each page's markdown and clean_text are built from, with the
index of its struck record, and render_text rebuilds the text from them under a reviewer's calls."""
import copy
import io
import json

import pymupdf
import pytest
from PIL import Image

import pdf_strikethrough as st
from _synth import _redline_image_png, _synthetic_native_pdf
from test_garbled_text import _broken_pdf
from test_native_overlay import S, _ov, _page

TEXT_KEYS = ("markdown", "clean_text", "passages")


def _text(res):
    return {k: res[k] for k in TEXT_KEYS}


def _mixed():
    """Page 0 born-digital with 'deleted text' struck, page 1 blank, page 2 a scan of 'keep struck'
    with 'struck' struck, its words supplied."""
    png, words = _redline_image_png()
    doc = _synthetic_native_pdf()
    doc.new_page(width=300, height=120)
    page = doc.new_page(width=300, height=120)
    page.insert_image(page.rect, stream=png)
    return doc.tobytes(), {2: words}


def test_page_words_is_off_by_default():
    data, wbp = _mixed()
    assert "page_words" not in st.detect_pdf(data, words_by_page=wbp)


def test_every_word_of_every_page_with_its_record():
    data, wbp = _mixed()
    res = st.detect_pdf(data, words_by_page=wbp, page_words=True)
    assert res["page_sources"] == ["native", "blank", "scanned"]
    assert [p["page"] for p in res["page_words"]] == [0, 1, 2]
    assert res["page_words"][1]["words"] == []
    for pg in res["page_words"]:
        for w in pg["words"]:
            if w["record"] is not None:
                rec = res["words"][w["record"]]
                assert (rec["page"], rec["text"], rec["bbox_frac"]) == (
                    pg["page"], w["text"], w["bbox_frac"])
    native = [(w["text"], w["record"] is not None) for w in res["page_words"][0]["words"]]
    assert native == [("keep", False), ("this", False), ("deleted", True), ("text", True),
                      ("here", False)]
    assert [w["text"] for w in res["page_words"][2]["words"]] == ["keep", "struck"]


def test_render_text_gives_back_the_result_text():
    data, wbp = _mixed()
    res = st.detect_pdf(data, words_by_page=wbp, page_words=True)
    assert "~~deleted~~" in res["markdown"]
    assert st.render_text(res) == _text(res)
    stored = json.loads(json.dumps(res))                  # as a consumer saves and reloads it
    assert st.render_text(stored) == json.loads(json.dumps(_text(res)))


def test_render_text_applies_a_reviewers_calls():
    data, wbp = _mixed()
    res = st.detect_pdf(data, words_by_page=wbp, page_words=True)
    words = copy.deepcopy(res["words"])
    for w in words:
        w["final"] = False                                # every call overturned
    out = st.render_text(res, words)
    assert "~~" not in out["markdown"] and out["passages"] == []
    assert out["clean_text"].startswith("keep this deleted text here")
    assert st.render_text(res) == _text(res)             # the result itself is untouched


def test_render_text_checks_its_inputs():
    data, wbp = _mixed()
    with pytest.raises(ValueError, match="page_words=True"):
        st.render_text(st.detect_pdf(data, words_by_page=wbp))
    res = st.detect_pdf(data, words_by_page=wbp, page_words=True)
    with pytest.raises(ValueError, match="words"):
        st.render_text(res, res["words"][:-1])


def test_page_words_without_markdown():
    data, wbp = _mixed()
    full = st.detect_pdf(data, words_by_page=wbp, page_words=True)
    bare = st.detect_pdf(data, words_by_page=wbp, page_words=True, include_markdown=False)
    assert "markdown" not in bare and bare["page_words"] == full["page_words"]
    assert st.render_text(bare) == _text(full)


def test_a_page_subset_lists_only_its_pages():
    data, wbp = _mixed()
    res = st.detect_pdf(data, words_by_page=wbp, page_words=True, pages=[2, 0])
    assert [p["page"] for p in res["page_words"]] == [0, 2]
    assert st.render_text(res) == _text(res)


def test_overlaid_characters_leave_the_rebuilt_clean_text():
    # 'vector' and 'both' drop U+0336-overlaid characters a record's span leaves out of
    # clean_text; 'flag' keeps them. Each page carries which it did.
    noto = pymupdf.Font(script=0)
    if not noto.has_glyph(ord(S)):
        pytest.skip("this PyMuPDF build's Noto Serif has no U+0336")
    doc = _page(noto, f"keep {_ov('a')}b{_ov('cd')}", ("e", "drawn"), " x")
    for method, dropped in (("vector", True), ("both", True), ("flag", False)):
        res = st.detect_pdf(doc, method=method, page_words=True)
        assert res["page_words"][0]["drop_overlaid"] is dropped
        assert st.render_text(res) == _text(res)


def test_garbled_words_left_out_of_the_text_are_left_out_of_page_words():
    with pytest.warns(st.GarbledTextWarning):
        res = st.detect_pdf(_broken_pdf(), on_garbled="drop", page_words=True)
    listed = {w["text"] for w in res["page_words"][0]["words"]}
    assert res["garbled_words"] and not listed & {g["text"] for g in res["garbled_words"]}
    assert st.render_text(res) == _text(res)


def test_image_frames():
    png, words = _redline_image_png()
    res = st.detect_image_file(png, words=words, page_words=True)
    assert [w["text"] for w in res["page_words"][0]["words"]] == ["keep", "struck"]
    assert res["page_words"][0]["drop_overlaid"] is False
    assert st.render_text(res) == _text(res)
    assert "page_words" not in st.detect_image_file(png, words=words)


def test_a_valid_document_text_is_unchanged_by_the_option():
    data, wbp = _mixed()
    off = st.detect_pdf(data, words_by_page=wbp)
    on = st.detect_pdf(data, words_by_page=wbp, page_words=True)
    del on["page_words"]
    assert on == off and off["page_count"] == 3


def _tiff(png, n):
    frame = Image.open(io.BytesIO(png))
    frame.load()
    buf = io.BytesIO()
    frame.save(buf, "TIFF", save_all=True, append_images=[frame.copy() for _ in range(n - 1)],
               dpi=(200, 200))
    return buf.getvalue()


def _check_records(res):
    for pg in res["page_words"]:
        for w in pg["words"]:
            if w["record"] is not None:
                rec = res["words"][w["record"]]
                assert (rec["page"], rec["text"], rec["bbox_frac"]) == (
                    pg["page"], w["text"], w["bbox_frac"])


def test_frames_of_a_multi_page_image():
    png, words = _redline_image_png()
    tiff, wbp = _tiff(png, 2), {0: words, 1: words}
    full = st.detect_image_file(tiff, words_by_page=wbp, page_words=True)
    assert [p["page"] for p in full["page_words"]] == [0, 1]
    assert any(w["record"] is not None for w in full["page_words"][1]["words"])
    _check_records(full)
    bare = st.detect_image_file(tiff, words_by_page=wbp, page_words=True, include_markdown=False)
    assert "markdown" not in bare and bare["page_words"] == full["page_words"]
    assert st.render_text(bare) == _text(full)


def test_a_scan_read_through_its_text_layer():
    # a scanned page with no OCR words for it but a text layer is read through the text layer
    png, _ = _redline_image_png()
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=120)
    page.insert_image(page.rect, stream=png)
    page.insert_text((20, 60), "keep struck", fontsize=20, render_mode=3)
    data = doc.tobytes()
    with pytest.warns(UserWarning, match="extractable text"):
        full = st.detect_pdf(data, page_words=True)
    with pytest.warns(UserWarning, match="extractable text"):
        bare = st.detect_pdf(data, page_words=True, include_markdown=False)
    assert full["page_sources"] == ["scanned"]
    assert [w["text"] for w in full["page_words"][0]["words"]] == ["keep", "struck"]
    assert bare["page_words"] == full["page_words"]
    assert st.render_text(bare) == _text(full)


def test_a_frame_rapidocr_cannot_read_is_listed_without_words():
    # a frame skipped for its shape keeps its (empty) place in the text, as in the result
    class ResizeImgError(Exception):
        """Named as RapidOCR names it: detection knows it by its name."""

    png, words = _redline_image_png()
    tiff = _tiff(png, 2)

    def ocr(rgb):
        if not seen:
            seen.append(rgb.shape)
            raise ResizeImgError()
        return list(words)

    for markdown in (True, False):
        seen = []
        with pytest.warns(UserWarning, match="frame 0"):
            res = st.detect_image_file(tiff, ocr=ocr, page_words=True, include_markdown=markdown)
        assert res["page_words"][0] == {"page": 0, "drop_overlaid": False, "words": []}
        assert res["page_words"][1]["words"]
        if markdown:
            assert res["markdown"].startswith("\n\n")
            assert st.render_text(res) == _text(res)
