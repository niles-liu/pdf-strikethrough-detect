"""A page RapidOCR cannot read because it is too long for its width (its ResizeImgError) is skipped
with a warning, and the rest of the document is still read. RapidOCR scales a long side over its
max_side_len (2000 px) down to it and rounds the short side to a multiple of 32; it raises when
that rounds to 0, so 4800 x 40 px fails and 2000 x 10 px reads."""
import io
import types

import numpy as np
import pymupdf
import pytest
from PIL import Image

import pdf_strikethrough as st
from pdf_strikethrough.ocr import rapidocr_backend


class ResizeImgError(Exception):
    """Named as RapidOCR names it: the backend knows it by its name."""


class _Engine:
    """A RapidOCR stand-in that refuses an image more than 100 times as long as it is wide and
    reads nothing on any other."""

    def __call__(self, image, return_word_box):
        h, w = image.shape[:2]
        if max(h, w) > 100 * min(h, w):
            raise ResizeImgError()
        return types.SimpleNamespace(word_results=())


def _png(size):
    buf = io.BytesIO()
    Image.new("L", size, 255).save(buf, "PNG")
    return buf.getvalue()


STRIP = (4800, 40)


def test_an_image_rapidocr_cannot_read_is_skipped_with_a_warning():
    with pytest.warns(UserWarning, match="frame 0: RapidOCR cannot read"):
        res = st.detect_image_file(_png(STRIP), ocr=rapidocr_backend(engine=_Engine()),
                                   telemetry=True)
    assert res["page_count"] == 1 and res["words"] == []
    assert res["markdown"] == "" and res["clean_text"] == ""
    (note,) = res["warnings"]
    assert "4800 x 40 px" in note and "skipped" in note
    assert res["telemetry"]["pages"][0]["skipped"] is True


def test_the_other_frames_are_still_read():
    buf = io.BytesIO()
    Image.new("L", STRIP, 255).save(buf, "TIFF", save_all=True,
                                    append_images=[Image.new("L", (400, 200), 255)])
    calls = []
    engine = _Engine()

    def counting(image, return_word_box):
        calls.append(image.shape[:2])
        return engine(image, return_word_box)
    with pytest.warns(UserWarning, match="frame 0"):
        res = st.detect_image_file(buf.getvalue(), ocr=rapidocr_backend(engine=counting), dpi=200)
    assert res["page_count"] == 2 and len(res["warnings"]) == 1
    assert calls == [(40, 4800), (200, 400)]


def test_a_scanned_pdf_page_rapidocr_cannot_read_is_skipped():
    doc = pymupdf.open()
    for w, h in (STRIP, (400, 200)):
        page = doc.new_page(width=w * 72 / 200, height=h * 72 / 200)   # 200 dpi, as rendered
        page.insert_image(page.rect, stream=_png((w, h)))
    with pytest.warns(UserWarning, match="page 0: RapidOCR cannot read"):
        res = st.detect_pdf(doc.tobytes(), ocr=rapidocr_backend(engine=_Engine()), telemetry=True)
    assert res["page_sources"] == ["scanned", "scanned"] and len(res["warnings"]) == 1
    assert [p.get("skipped", False) for p in res["telemetry"]["pages"]] == [True, False]


def test_the_backend_alone_still_raises_rapidocrs_error():
    # only detection skips the page; a caller of the backend gets RapidOCR's own error
    with pytest.raises(ResizeImgError):
        rapidocr_backend(engine=_Engine())(np.zeros((40, 4800, 3), np.uint8))


def test_any_other_rapidocr_error_still_raises():
    def broken(image, return_word_box):
        raise RuntimeError("model missing")
    with pytest.raises(RuntimeError, match="model missing"):
        st.detect_image_file(_png(STRIP), ocr=rapidocr_backend(engine=broken))


def test_real_rapidocr_skips_a_strip_it_cannot_read():
    pytest.importorskip("rapidocr")
    with pytest.warns(UserWarning, match="RapidOCR cannot read"):
        res = st.detect_image_file(_png(STRIP), ocr=rapidocr_backend())
    assert res["words"] == [] and len(res["warnings"]) == 1
