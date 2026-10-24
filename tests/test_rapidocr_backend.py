"""The RapidOCR adapter: the engine it builds (orientation classifier off unless the caller turns it
on, a caller's engine used as is), the word boxes it hands on (one placed off the page is
clipped to it, not refused along with the whole document) and the lone CJK characters it removes
from a page without CJK text. Fake engines stand in for RapidOCR, so all but the last test run
without the extra installed."""
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
    """A stand-in RapidOCR engine reading one line of words, or one line per argument. Boxes are
    given in page fractions and reported as pixel corner points of the image it is called on, as
    RapidOCR reports them."""

    def __init__(self, words, *more_lines):
        self.lines, self.calls = (words, *more_lines), []

    def __call__(self, image, return_word_box):
        self.calls.append((image.shape, return_word_box))
        h, w = image.shape[:2]
        return SimpleNamespace(word_results=tuple(
            tuple((text, 0.99, [[x0 * w, y0 * h], [x1 * w, y0 * h], [x1 * w, y1 * h],
                                [x0 * w, y1 * h]])
                  for text, (x0, y0, x1, y1) in line)
            for line in self.lines))


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


# ------------------------------------------------------------------ lone CJK characters it removes

def _row(*texts):
    """One line of words, each 0.04 wide, left to right."""
    return [(t, (0.1 + 0.05 * i, 0.2, 0.14 + 0.05 * i, 0.23)) for i, t in enumerate(texts)]


def _texts(words):
    return [w.text for w in words]


_TABLE_ROW = _row("67.8", "(", "斗", ")", "17.38")     # a footnote marker (a) read as a Han character


def test_a_lone_cjk_character_on_an_english_page_is_removed():
    words = rapidocr_backend(engine=_Engine(_TABLE_ROW))(_BLANK)
    assert _texts(words) == ["67.8", "(", ")", "17.38"]


def test_drop_lone_cjk_false_keeps_it(monkeypatch):
    words = rapidocr_backend(engine=_Engine(_TABLE_ROW), drop_lone_cjk=False)(_BLANK)
    assert _texts(words) == ["67.8", "(", "斗", ")", "17.38"]
    built = _fake_rapidocr_module(monkeypatch)
    rapidocr_backend(drop_lone_cjk=False)(_BLANK)
    assert built == [{"params": {"Global.use_cls": False}}]   # the switch never reaches RapidOCR


@pytest.mark.parametrize("texts", [
    ("第", "三", "条", "2024", "斗"),             # RapidOCR returns each Han character as a word
    ("Mr.", "王", "小", "明", "(Wang)", "斗"),     # a name in an English line
    ("Ver.", "データ", "2", "二"),                 # a kana word
    ("the", "人々", "of", "二"),                   # the iteration mark counts as a character
])
def test_a_page_with_cjk_text_is_left_as_read(texts):
    words = rapidocr_backend(engine=_Engine(_row(*texts)))(_BLANK)
    assert _texts(words) == list(texts)            # the lone character beside the run stays too


def test_two_characters_ending_and_starting_two_lines_are_cjk_text():
    first = [("Signed", (0.1, 0.2, 0.2, 0.23)), ("王", (0.21, 0.2, 0.23, 0.23))]
    second = [("明", (0.1, 0.3, 0.12, 0.33)), ("Wang", (0.13, 0.3, 0.2, 0.33)),
              ("斗", (0.21, 0.3, 0.23, 0.33))]
    words = rapidocr_backend(engine=_Engine(first, second))(_BLANK)
    assert _texts(words) == ["Signed", "王", "明", "Wang", "斗"]


def test_a_cjk_character_inside_a_word_becomes_a_space():
    row = _row("xニ1", "2019ー2020", "ニ1", "aニ", "Total", "二", "。", "（a）", "A・B")
    words = rapidocr_backend(engine=_Engine(row))(_BLANK)
    # inside a word a space, at either end cut; a lone Han character dropped; CJK punctuation,
    # fullwidth forms and the katakana middle dot kept
    assert _texts(words) == ["x 1", "2019 2020", "1", "a", "Total", "。", "（a）", "A・B"]
    for w, (_, box) in zip(words[:4], row[:4]):
        assert w.bbox == pytest.approx(box)                 # the box of the word it came from


def _han_page():
    """A 200-dpi RGB render of a Chinese and a Japanese line, set in PyMuPDF's built-in CJK fonts."""
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=150)
    page.insert_text((30, 60), "合同双方应当按照约定全面履行义务。", fontsize=16, fontname="china-s")
    page.insert_text((30, 100), "第五条 秘密保持義務は存続する。", fontsize=16, fontname="japan")
    pix = page.get_pixmap(dpi=200)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]


def test_real_rapidocr_reads_a_cjk_page_unchanged():
    pytest.importorskip("rapidocr")
    from rapidocr import RapidOCR

    from pdf_strikethrough.ocr import _CJK
    engine, image = RapidOCR(params={"Global.use_cls": False}), _han_page()
    kept = rapidocr_backend(engine=engine)(image)
    read = rapidocr_backend(engine=engine, drop_lone_cjk=False)(image)
    assert _texts(kept) == _texts(read)
    assert sum(len(_CJK.findall(w.text)) for w in kept) >= 20
