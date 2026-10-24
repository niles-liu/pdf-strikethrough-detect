"""The RapidOCR adapter: the engine it builds (orientation classifier off unless the caller turns it
on, a caller's engine used as is) and the word boxes it hands on (one placed off the page is
clipped to it, not refused along with the whole document). Fake engines stand in for RapidOCR,
so none of this needs the extra installed."""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pymupdf
import pytest

import pdf_strikethrough as st
from pdf_strikethrough.ocr import rapidocr_backend

_BLANK = np.zeros((100, 200, 3), np.uint8)


def _fake_rapidocr_module(monkeypatch):
    """Install a stand-in ``rapidocr`` module; returns the kwargs of every engine it builds."""
    built = []

    class RapidOCR:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def __call__(self, image, return_word_box):
            return SimpleNamespace(word_results=())

    mod = types.ModuleType("rapidocr")
    mod.__version__ = "3.10.0"
    mod.RapidOCR = RapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr", mod)
    return built


class _Engine:
    """A stand-in RapidOCR engine reading one line of words. Boxes are given in page fractions and
    reported as pixel corner points of the image it is called on, as RapidOCR reports them."""

    def __init__(self, words):
        self.words, self.calls = words, []

    def __call__(self, image, return_word_box):
        self.calls.append((image.shape, return_word_box))
        h, w = image.shape[:2]
        line = tuple((text, 0.99, [[x0 * w, y0 * h], [x1 * w, y0 * h], [x1 * w, y1 * h],
                                   [x0 * w, y1 * h]])
                     for text, (x0, y0, x1, y1) in self.words)
        return SimpleNamespace(word_results=(line,))


# ----------------------------------------------------------------------- the engine it builds

def test_the_built_engine_has_the_orientation_classifier_off(monkeypatch):
    built = _fake_rapidocr_module(monkeypatch)
    backend = rapidocr_backend()
    assert built == []                                  # built lazily, on the first page
    assert backend(_BLANK) == []
    assert built == [{"params": {"Global.use_cls": False}}]


@pytest.mark.parametrize("params, expected", [
    ({"Global.use_cls": True}, {"Global.use_cls": True}),                       # the opt-in
    ({"Global.text_score": 0.3}, {"Global.use_cls": False, "Global.text_score": 0.3}),
    (None, {"Global.use_cls": False}),
])
def test_the_callers_params_win(monkeypatch, params, expected):
    built = _fake_rapidocr_module(monkeypatch)
    passed = None if params is None else dict(params)
    rapidocr_backend(config_path="ocr.yaml", params=passed)(_BLANK)
    assert built == [{"config_path": "ocr.yaml", "params": expected}]
    assert passed == params                             # the caller's dict is left as it was


def test_a_prebuilt_engine_is_used_as_is(monkeypatch):
    built = _fake_rapidocr_module(monkeypatch)
    engine = _Engine([("word", (0.1, 0.2, 0.3, 0.4))])
    words = rapidocr_backend(engine=engine)(_BLANK)
    assert built == []                                  # no engine of its own
    assert engine.calls == [(_BLANK.shape, True)] and [w.text for w in words] == ["word"]


# ----------------------------------------------------------------------- the boxes it hands on

def test_a_word_box_off_the_page_is_clamped_to_it():
    engine = _Engine([("kept", (0.1, 0.2, 0.3, 0.4)),
                      ("x=1", (0.778, 0.2, 1.569, 0.4)),       # runs off the right edge
                      ("low", (0.4, 0.9, 0.5, 1.6)),           # and off the bottom
                      ("edge", (1.6, 0.2, 1.9, 0.4)),          # wholly off: text kept, zero width
                      ("nan", (float("nan"), 0.2, 0.5, 0.4))])
    words = rapidocr_backend(engine=engine)(_BLANK)
    assert [w.text for w in words] == ["kept", "x=1", "low", "edge"]
    expected = [(0.1, 0.2, 0.3, 0.4), (0.778, 0.2, 1.0, 0.4), (0.4, 0.9, 0.5, 1.0),
                (1.0, 0.2, 1.0, 0.4)]
    for w, box in zip(words, expected):
        assert w.bbox == pytest.approx(box)


def _struck_line():
    """A 200-dpi PNG of 'keep struck' with 'struck' struck through, and an engine reading it plus a
    third word whose box runs from x 0.778 to 1.569, past the 1.5 that Word refuses."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=120)
    page.insert_text((20, 60), "keep struck", fontsize=20)
    W, H = page.rect.width, page.rect.height
    words = [(w[4], (w[0] / W, w[1] / H, w[2] / W, w[3] / H)) for w in page.get_text("words")]
    _, (x0, y0, x1, y1) = words[1]
    page.draw_line((x0 * W, (y0 + y1) / 2 * H), (x1 * W, (y0 + y1) / 2 * H), width=1.5)
    png = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY).tobytes("png")
    return png, _Engine(words + [("tail", (0.778, y0, 1.569, y1))])


def _struck(res):
    return [w["chars"] for w in res["words"] if w["final"]]


def test_detect_image_file_reads_past_an_off_page_box():
    png, engine = _struck_line()
    res = st.detect_image_file(png, ocr=rapidocr_backend(engine=engine))
    assert _struck(res) == ["struck"] and res["clean_text"].split() == ["keep", "tail"]


def test_detect_pdf_reads_past_an_off_page_box():
    png, engine = _struck_line()
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=120)
    page.insert_image(page.rect, stream=png)                # an image-only, scanned page
    res = st.detect_pdf(doc.tobytes(), ocr=rapidocr_backend(engine=engine))
    assert res["page_sources"] == ["scanned"]
    assert _struck(res) == ["struck"] and res["clean_text"].split() == ["keep", "tail"]
