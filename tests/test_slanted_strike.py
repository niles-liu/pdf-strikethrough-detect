"""A slanted strike's ink test, on the scanned path.

The two-sided ink test read the glyph ink above and below a line level with the line's height at
the middle of the word. A strike leaning 3 degrees or more moves several px across a word, so
that level band took the glyph ink above the stroke at one end and below it at the other, and the
struck word read as ink on one side only. A word the tests drop under a line whose height changes
by at least ``INK_ALONG_SLOPE_PX`` (1 px) across it, and in the strike band, is now tried again
with the ink read along the line, column by column. Words the tests pass keep their verdict.
"""
import math

import numpy as np
import pymupdf
import pytest

import pdf_strikethrough as st
from pdf_strikethrough import scanned
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import ScanConfig

CF = ScanConfig.confidence_free()


# ------------------------------------------------------------------------- the reading itself

def _slope_ink(rise=20):
    """Glyph ink in two strips that follow a line from (100, 150) to (300, 150 + rise): rows 4 to
    6 px above it and 4 to 6 px below it, at every column."""
    ink = np.zeros((300, 400), bool)
    for x in range(100, 301):
        y = 150 + rise * (x - 100) / 200
        ink[int(y) - 6:int(y) - 3, x] = True
        ink[int(y) + 4:int(y) + 7, x] = True
    return ink, ((100.0, 150.0), (300.0, 150.0 + rise))


def test_reading_along_follows_the_line():
    """Along the line, every column has ink above and below it; level with its middle, the ink
    above the line at one end falls in the band left out or below it."""
    ink, ends = _slope_ink()
    box = (100, 110, 300, 210)
    level = scanned._ink_above_below(ink, ends, 2.0, box)
    along = scanned._ink_above_below(ink, ends, 2.0, box, along=True)
    assert along == (1.0, 1.0)
    assert min(level) < 0.75


@pytest.mark.parametrize("seed", range(5))
def test_level_line_reads_the_same_both_ways(seed):
    """On a level line the two readings are the same, whatever the ink: sparse enough that one
    row more or less above or below changes the share of columns with ink."""
    rng = np.random.default_rng(seed)
    ink = rng.random((200, 300)) < 0.02
    ends = ((40.0, 100.4), (260.0, 100.4))
    for gap in (0, 2):
        assert (scanned._ink_above_below(ink, ends, 3.0, (50, 70, 250, 130), gap)
                == scanned._ink_above_below(ink, ends, 3.0, (50, 70, 250, 130), gap, along=True))


def test_reading_along_a_line_off_the_box_is_empty():
    ink, ends = _slope_ink()
    assert scanned._ink_above_below(ink, ends, 2.0, (320, 110, 380, 210), along=True) == (0.0, 0.0)


# ------------------------------------------------------------------------ real text, end to end

def _page(text, font, size, strike, width, sign):
    """One line of text with a strike leaning 3 degrees or more and rising at least 1.8 pt, as the
    frontier benchmark's generator draws its slanted strikes; `sign` sets which way it leans."""
    doc = pymupdf.open()
    page = doc.new_page(width=460, height=120)
    page.insert_text((24, 64), text, fontname=font, fontsize=size)
    ws = page.get_text("words")
    x0, x1 = ws[strike[0]][0], ws[strike[1]][2]
    y = 64 - size * pymupdf.Font(font).glyph_bbox(ord("x")).y1 / 2
    rise = (max(1.8, (x1 - x0) * math.tan(math.radians(3))) + 0.25) / 2 * sign
    page.draw_line((x0, y + rise), (x1, y - rise), width=width)
    words = [Word(w[4], (w[0] / 460, w[1] / 120, w[2] / 460, w[3] / 120), None) for w in ws]
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    return gray, words


def _found(gray, words):
    recs = st.detect_scanned_image(gray, words, config=CF, dpi=200)
    return sorted(r["text"] for r in recs if r["final"])


CASES = [("a summer was over now", (1, 3), "tiro", 11, 0.6, -1, ["was"], ["summer", "was"]),
         ("the lessee may cure it", (1, 3), "cour", 9.5, 0.8, 1, [], ["may"])]


@pytest.mark.parametrize("text,strike,font,size,width,sign,before,after", CASES)
def test_slanted_strike_on_real_text(text, strike, font, size, width, sign, before, after,
                                     monkeypatch):
    """Found read along the line; at INK_ALONG_SLOPE_PX = 0, the release's reading, not."""
    gray, words = _page(text, font, size, strike, width, sign)
    assert _found(gray, words) == after
    monkeypatch.setattr(scanned, "INK_ALONG_SLOPE_PX", 0)
    assert _found(gray, words) == before


@pytest.mark.parametrize("text,strike,font,size,width,sign,before,after", CASES)
def test_a_line_rising_less_than_the_threshold_is_not_retried(text, strike, font, size, width,
                                                               sign, before, after, monkeypatch):
    """The same strikes with the threshold above their rise across a word: not retried."""
    gray, words = _page(text, font, size, strike, width, sign)
    monkeypatch.setattr(scanned, "INK_ALONG_SLOPE_PX", 40)
    assert _found(gray, words) == before


def test_the_rise_is_measured_across_the_word(monkeypatch):
    """The strike over "summer was over" rises about 12 px end to end but about 6 px across
    "summer": with the threshold at 8 px, "summer" is not retried."""
    text, strike, font, size, width, sign = CASES[0][:6]
    gray, words = _page(text, font, size, strike, width, sign)
    monkeypatch.setattr(scanned, "INK_ALONG_SLOPE_PX", 8)
    assert _found(gray, words) == CASES[0][6]


def test_sloped_underline_is_not_retried():
    """A strike-free line of Courier Oblique over an underline sloping down across it, at 300 dpi:
    its high end runs into the bottom of the first word, out of the strike band. Read along it,
    there is ink above and below; the retry is only for words in the band."""
    doc = pymupdf.open()
    page = doc.new_page(width=460, height=120)
    page.insert_text((24, 64), "wavy kitty sky jury", fontname="coit", fontsize=11)
    ws = page.get_text("words")
    xa, xb = ws[0][0], ws[-1][2]
    under, rise = 64 + 11 * 0.12, 0.02 * (ws[-1][2] - ws[0][0])
    page.draw_line((xa, under - rise), (xb, under + rise), width=0.6)
    words = [Word(w[4], (w[0] / 460, w[1] / 120, w[2] / 460, w[3] / 120), None) for w in ws]
    pix = page.get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    recs = st.detect_scanned_image(gray, words, config=CF, dpi=300)
    assert [r["text"] for r in recs if r["final"]] == []


def test_default():
    assert scanned.INK_ALONG_SLOPE_PX == 1


# ------------------------------------------------------------- what the retry reads and leaves

def _stems(n, top, bot, y0, y1, fill=0.8, run=2.0):
    """`n` letters of three stems over rows `top` to `bot`, and a 2 px stroke from row `y0` at the
    left to `y1` at the right, with the record strike_lines would give for it."""
    X0, ADV = 200, 18
    x1 = X0 + (n - 1) * ADV + 14
    gray = np.full((300, 900), 255, np.uint8)
    for i in range(n):
        for s in (0, 5, 11):
            gray[top:bot, X0 + i * ADV + s:X0 + i * ADV + s + 3] = 20
    for x in range(X0 - 2, x1 + 2):
        y = y0 + (y1 - y0) * (x - X0 + 2) / (x1 + 4)
        gray[int(round(y)):int(round(y)) + 2, x] = 20
    line = {"bbox_px": (X0 - 2, min(y0, y1), x1 + 2, max(y0, y1) + 1),
            "ends_px": ((X0 - 2.0, y0 + 0.5), (x1 + 2.0, y1 + 0.5)),
            "len_in": (x1 - X0 + 4) / 200, "angle_deg": 3.0, "fill": fill, "run_px": run,
            "straightness": 2.0}
    word = Word("x" * n, (X0 / 900, 116 / 300, x1 / 900, 176 / 300), None)
    return gray, line, word


def _records(gray, line, word, slope, monkeypatch):
    monkeypatch.setattr(scanned, "INK_ALONG_SLOPE_PX", slope)
    return scanned.classify_lines([line], [word], gray, config=CF)[1]


def test_short_slanted_strike_needs_ink_on_both_sides_along_it(monkeypatch):
    """Three letters, rows 141 to 151, a stroke falling 8 px across them from row 140: a short
    line, so it needs glyph ink on both sides. Level with its middle the band covers the letters'
    top rows, and there is none above; along the line, where it has fallen, there is."""
    gray, line, word = _stems(3, 141, 152, 140, 148)
    assert line["len_in"] < scanned.INK_SHORT_LEN_IN
    assert _records(gray, line, word, 0, monkeypatch) == []
    assert len(_records(gray, line, word, 1, monkeypatch)) == 1


def test_a_word_the_tests_pass_is_not_read_again(monkeypatch):
    """Letters filling the box's middle, a stroke falling across them: the level reading passes,
    so the reading along the line is not taken, even one that would say nothing beside it."""
    gray, line, word = _stems(12, 130, 170, 145, 155)
    before = _records(gray, line, word, 0, monkeypatch)
    assert before
    monkeypatch.setattr(scanned, "_ink_along", lambda *a, **k: (0.0, 0.0))
    assert _records(gray, line, word, 1, monkeypatch) == before


def test_off_never_reads_along(monkeypatch):
    """With INK_ALONG_SLOPE_PX = 0 the reading along a line is never taken, the all-ink fallback
    of a long solid sloped line with no glyph ink beside it included."""
    def along(*a, **k):
        raise AssertionError("read along the line")
    monkeypatch.setattr(scanned, "_ink_along", along)
    gray, line, word = _stems(12, 141, 152, 140, 148)
    _records(gray, line, word, 0, monkeypatch)
    # a long solid rule falling 4 px across the word, nothing beside it: its steps are over 50 px,
    # so glyph ink leaves them out and the table-rule test reads all ink
    bare = np.full((300, 900), 255, np.uint8)
    for x in range(198, 416):
        y = 140 + 4 * (x - 198) / 218
        bare[int(round(y)):int(round(y)) + 2, x] = 20
    ends = ((198.0, 140.5), (416.0, 144.5))
    _records(bare, {**line, "fill": 0.97, "ends_px": ends}, word, 0, monkeypatch)


def test_reading_along_off_the_image_is_empty():
    """_ink_along on a box whose columns or rows lie off the image reads nothing."""
    ink = np.ones((50, 50), bool)
    ends = ((0.0, 25.0), (49.0, 30.0))
    assert scanned._ink_along(ink, ends, 3.0, (60, 0, 80, 40)) == (0.0, 0.0)
    assert scanned._ink_along(ink, ends, 3.0, (0, 60, 40, 80)) == (0.0, 0.0)
