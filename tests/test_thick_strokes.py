"""Thick strikes on the scanned path: a 2-3 pt line through text is kept, while the bars of a bold
heading, about as thick as its stems, are not."""
import numpy as np
import pymupdf
import pytest
from scipy import ndimage

import pdf_strikethrough as st
from pdf_strikethrough import lines
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import ScanConfig

CF = ScanConfig.confidence_free()


def _page(text, font, size, strike=None, width=2.0, dpi=200, blur=0.0):
    """One line of text rendered at `dpi`: (gray, words). `strike` = (first, last) word index,
    struck by a `width` pt line through the middle of the x-height. `blur` is a Gaussian sigma in
    px, a mild stand-in for a scanner's softening."""
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=120)
    page.insert_text((24, 64), text, fontname=font, fontsize=size)
    ws = page.get_text("words")
    if strike:
        y = 64 - size * pymupdf.Font(font).glyph_bbox(ord("x")).y1 / 2
        page.draw_line((ws[strike[0]][0], y), (ws[strike[1]][2], y), width=width)
    W, H = page.rect.width, page.rect.height
    words = [Word(w[4], (w[0] / W, w[1] / H, w[2] / W, w[3] / H), None) for w in ws]
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    if blur:
        gray = np.clip(ndimage.gaussian_filter(gray.astype(float), blur), 0, 255).astype(np.uint8)
    return gray, words


def _thick(found, dpi):
    return [ln for ln in found if ln["run_px"] > lines.THIN_STROKE_RUN_PX * dpi / lines.RENDER_DPI]


def _struck(gray, words, dpi=200):
    recs = st.detect_scanned_image(gray, words, config=CF, dpi=dpi)
    return sorted(r["text"] for r in recs if r["final"]), recs


@pytest.mark.parametrize("dpi", [150, 200, 300])
@pytest.mark.parametrize("width", [2.0, 3.0])
def test_thick_strike_through_regular_text_is_kept(width, dpi):
    """A 2 or 3 pt strike is 5.6-8.3 px at 200 dpi, past the old 4 px cap; it is clearly thicker
    than the stems of 11 pt Times, so it is kept and the word is marked."""
    gray, words = _page("hold the deposit in a segregated account", "tiro", 11, (2, 2), width, dpi)
    assert _thick(st.strike_lines(gray, dpi=dpi), dpi)
    assert _struck(gray, words, dpi)[0] == ["deposit"]


def test_thick_strike_was_dropped_by_the_old_cap(monkeypatch):
    """The regression the stem test replaces: at 0.12.1's 4 px cap the same strike is lost."""
    gray, words = _page("hold the deposit in a segregated account", "tiro", 11, (2, 2), 2.0)
    monkeypatch.setattr(lines, "MAX_STROKE_RUN_PX", 4)
    assert _struck(gray, words)[0] == []


def test_3pt_strike_at_150_dpi_uses_the_slack():
    """At 150 dpi the scaled cap is 6.75 px and a 3 pt strike measures 7: the 1 px of slack on
    the cap keeps it."""
    gray, words = _page("a summer was over now, we saw a nurse", "helv", 11, (1, 2), 3.0, 150)
    assert _struck(gray, words, 150)[0] == ["summer", "was"]


def test_blurred_3pt_strike_is_kept():
    """A softened 3 pt strike, which 0.12.1 lost. Blur can still merge a 3 pt stroke with the
    glyphs it crosses into a piece thicker than the stitching takes (11-13 pt Times), and then it
    is lost as before."""
    gray, words = _page("a summer was over now, we saw a nurse", "tiro", 9, (1, 2), 3.0, blur=0.6)
    assert _struck(gray, words)[0] == ["summer", "was"]


def test_thick_line_with_no_text_beside_it_is_kept():
    """With no glyph ink beside it there are no stems to compare with: the line is kept."""
    gray = np.full((200, 600), 255, np.uint8)
    gray[96:102, 100:500] = 0                                   # 6 px, 2 in long
    assert any(ln["len_in"] >= 1.9 and ln["run_px"] == 6 for ln in st.strike_lines(gray, dpi=200))


@pytest.mark.parametrize("font,size", [("hebo", 18), ("hebo", 24), ("tibo", 12)])
def test_bold_heading_gives_no_strike_candidates(font, size, monkeypatch):
    """A bold glyph's bars, and chains of them stitched along a heading, are about as thick as its
    stems: no thick line comes out and no word reaches the CNN. Without the stem test (ratio 0)
    the same heading does produce thick lines."""
    gray, words = _page("TERMS OF THE AGREEMENT Effective", font, size)
    assert _thick(st.strike_lines(gray, dpi=200), 200) == []
    assert _struck(gray, words)[1] == []
    monkeypatch.setattr(lines, "STEM_RATIO", 0)
    assert _thick(st.strike_lines(gray, dpi=200), 200)


@pytest.mark.parametrize("font", ["hebo", "tibo"])
def test_thick_strike_through_small_bold_text_is_kept(font):
    """At 10 pt a bold font's stems are about 4 px, so a 2 pt strike (5.6 px) is still clearly
    thicker. From about 11 pt the stems are too close to the strike's thickness for the test to
    tell the two apart: such a strike is dropped, as it was under the old cap."""
    gray, words = _page("TERMS OF THE AGREEMENT", font, 10, (1, 2), 2.0)
    assert _struck(gray, words)[0] == ["OF", "THE"]
