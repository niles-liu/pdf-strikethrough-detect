"""A detected line a few px off its stroke, on the scanned path.

Stitching can run a line a few px off a thin or light stroke, along glyph pieces. Measured from
there, the ink on one side of the line is under the glyphs, so two gates dropped the struck word:
a line out of the strike band (or short) needs ink on both sides, and a long solid line with ink
on one side only is taken for a table rule. A word either gate drops is now tried again at the
stroke's own row: the band within ``STROKE_SNAP_PX`` (4 px at 200 dpi) whose unbroken ink run
covers at least ``STROKE_SNAP_MIN_RUN`` of the columns the word shares with the line. Words the
gates pass are judged as before.

Synthetic rasters: each letter is three ink stems, so the only long horizontal ink is a stroke,
an underline or a rule; and real text with real strikes, its lines moved off the stroke.
"""
import numpy as np
import pymupdf
import pytest

from pdf_strikethrough import detect, scanned
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import ScanConfig, classify_lines

CF = ScanConfig.confidence_free()
H, W = 300, 900
TOP, BOT = 130, 170             # the letters' x-height band, no ascenders or descenders
BOX = (100, 190)                # the OCR box: the line's full height
X0, ADV, N = 200, 18, 12        # where the word starts, the letter advance, letters


def _page(k=1):
    """The word: N letters of three stems each, at scale `k` (2 for 400 dpi)."""
    gray = np.full((H * k, W * k), 255, np.uint8)
    for i in range(N):
        for s in (0, 5, 11):
            x = (X0 + i * ADV + s) * k
            gray[TOP * k:BOT * k, x:x + 3 * k] = 20
    return gray


def _x1():
    return X0 + (N - 1) * ADV + 14


def _word(k=1):
    return Word("abcdefghijkl", (X0 * k / (W * k), BOX[0] / H, _x1() * k / (W * k), BOX[1] / H),
                None)


def _stroke(gray, y, k=1, xa=None, xb=None):
    xa, xb = (X0 - 2 if xa is None else xa), (_x1() + 2 if xb is None else xb)
    gray[(y - 1) * k:(y + 1) * k, xa * k:xb * k] = 20


def _line(y, fill=0.8, k=1, xa=None, xb=None):
    """The record strike_lines would give for a line at row `y` across the word, or from `xa` to
    `xb` (px at 200 dpi)."""
    xa, xb = (X0 - 2 if xa is None else xa) * k, (_x1() + 2 if xb is None else xb) * k
    return {"bbox_px": (xa, y * k - 1, xb, y * k + 1),
            "ends_px": ((float(xa), float(y * k)), (float(xb), float(y * k))),
            "len_in": (xb - xa) / (200 * k), "angle_deg": 0.0, "fill": fill, "run_px": 2.0 * k,
            "straightness": 3.0}


def _struck(gray, line, k=1, snap=None, monkeypatch=None):
    """The records for one line through the word, the reach STROKE_SNAP_PX = `snap`."""
    if snap is not None:
        monkeypatch.setattr(scanned, "STROKE_SNAP_PX", snap)
    _tagged, struck = classify_lines([line], [_word(k)], gray, config=CF, dpi=200 * k)
    return struck


# ------------------------------------------------------------------- the words the gates dropped

def test_line_below_a_low_stroke_out_of_the_band(monkeypatch):
    """The stroke low in the letters, the line 4 px under it: out of the strike band, and nothing
    under it but the last rows of the stems, so no ink below."""
    gray = _page()
    _stroke(gray, 163)
    line = _line(167)
    assert _struck(gray, line, snap=0, monkeypatch=monkeypatch) == []
    rec, = _struck(gray, line, snap=4, monkeypatch=monkeypatch)
    assert rec["char_span"] == (0, N) and not rec["partial"]
    assert rec["off"] > scanned.STRIKE_TOL                 # still out of the band, now two-sided


def test_long_solid_line_above_a_high_stroke_is_not_a_table_rule(monkeypatch):
    """The stroke high in the letters, a long solid line 4 px above it: in the band, with no ink
    above it, so it read as a table rule."""
    gray = _page()
    _stroke(gray, 136)
    line = _line(132, fill=0.95)
    assert _struck(gray, line, snap=0, monkeypatch=monkeypatch) == []
    rec, = _struck(gray, line, snap=4, monkeypatch=monkeypatch)
    assert rec["char_span"] == (0, N) and rec["geom_corroborated"]


def test_line_out_of_the_band_under_a_stroke_in_it(monkeypatch):
    """Sparse letters (a stem every 12 px), too little ink for the both-sides test even at the
    stroke. The stroke is in the band, its line 4 px lower is not: judged at the stroke's row the
    word is in the band, where a long line needs no ink test."""
    gray = np.full((H, W), 255, np.uint8)
    for x in range(X0, _x1(), 12):
        gray[TOP:BOT, x:x + 3] = 20
    _stroke(gray, 156)
    line = _line(160)
    assert _struck(gray, line, snap=0, monkeypatch=monkeypatch) == []
    rec, = _struck(gray, line, snap=4, monkeypatch=monkeypatch)
    assert abs(rec["off"]) <= scanned.STRIKE_TOL and not rec["geom_corroborated"]


def test_spans_are_read_at_the_stroke_row(monkeypatch):
    """A loose box, a stroke from 12 px before the first letter to 4 px into the seventh. Read
    from the line 4 px under the stroke, the stroke's own rows would count as glyph ink, move the
    word's start back to where the stroke starts, and widen every character enough to strike the
    seventh."""
    gray = _page()
    xb = X0 + 6 * ADV + 4
    _stroke(gray, 163, xa=X0 - 12, xb=xb)
    word = Word("abcdefghijkl", ((X0 - 16) / W, BOX[0] / H, (_x1() + 4) / W, BOX[1] / H), None)
    line = {**_line(167), "bbox_px": (X0 - 12, 166, xb, 168),
            "ends_px": ((X0 - 12.0, 167.0), (float(xb), 167.0)), "len_in": (xb - X0 + 12) / 200}
    monkeypatch.setattr(scanned, "STROKE_SNAP_PX", 4)
    rec, = classify_lines([line], [word], gray, config=CF, dpi=200)[1]
    assert rec["char_span"] == (0, 6) and rec["partial"]


def _sparse(x1=None, step=12):
    """Letters of one stem every `step` px: too little ink for the both-sides test anywhere."""
    gray = np.full((H, W), 255, np.uint8)
    for x in range(X0, _x1() if x1 is None else x1, step):
        gray[TOP:BOT, x:x + 3] = 20
    return gray


# ---------------------------------------------- a retried word is judged as at the stroke's row

def test_the_offset_moves_by_the_retry_in_box_heights(monkeypatch):
    """Line 4 px under a stroke in the band: the word's offset is read at the stroke's row,
    3 px up (the nearest band holding it), in units of the box's height."""
    gray = _sparse()
    _stroke(gray, 157)
    rec, = _struck(gray, _line(161), snap=4, monkeypatch=monkeypatch)
    assert rec["off"] == round((161 - 3 - BOX[0]) / (BOX[1] - BOX[0]) - 0.5, 2)


def test_a_line_the_tests_pass_is_not_moved(monkeypatch):
    """A long line in the band, 3 px above its stroke, passes as it is: its offset is the line's."""
    gray = _page()
    _stroke(gray, 153)
    rec, = _struck(gray, _line(150), snap=4, monkeypatch=monkeypatch)
    assert rec["off"] == round((150 - BOX[0]) / (BOX[1] - BOX[0]) - 0.5, 2)


def test_a_partial_strike_is_read_at_the_stroke_row():
    """A loose box, a stroke from 12 px before the word over four letters, its line 4 px lower
    and outside the band: the same four letters as with the line on the stroke."""
    gray = _page()
    xa, xb = X0 - 12, X0 + 4 * ADV + 4
    _stroke(gray, 163, xa=xa, xb=xb)
    word = Word("abcdefghijkl", ((X0 - 16) / W, BOX[0] / H, (_x1() + 4) / W, BOX[1] / H), None)
    for y in (167, 163):
        rec, = classify_lines([_line(y, xa=xa, xb=xb)], [word], gray, config=CF, dpi=200)[1]
        assert rec["char_span"] == (0, 4) and rec["partial"], y


def test_a_word_the_line_barely_enters_is_read_at_the_stroke_row():
    """A struck word, a word space, then a word whose box starts 2 px before its ink; the stroke
    runs 1.3 letters into it. Its line 4 px lower, the second word's span is read at the stroke's
    row: one letter, as with the line on the stroke."""
    gray = np.full((H, W), 255, np.uint8)
    xs1 = [X0 + i * ADV for i in range(6)]
    start = xs1[-1] + 14 + 24                              # a 24 px word space
    xs2 = [start + i * ADV for i in range(8)]
    for x in xs1 + xs2:
        for s in (0, 5, 11):
            gray[TOP:BOT, x + s:x + s + 3] = 20
    first = Word("abcdef", (X0 / W, BOX[0] / H, (xs1[-1] + 14) / W, BOX[1] / H), None)
    second = Word("ghijklmn", ((start - 2) / W, BOX[0] / H, (xs2[-1] + 14) / W, BOX[1] / H), None)
    xa, xb = X0 - 2, int(start + 1.3 * ADV)
    _stroke(gray, 163, xa=xa, xb=xb)
    for y in (167, 163):
        recs = classify_lines([_line(y, xa=xa, xb=xb)], [first, second], gray, config=CF,
                              dpi=200)[1]
        assert [(r["text"], r["char_span"]) for r in recs] == [
            ("abcdef", (0, 6)), ("ghijklmn", (0, 1))], y


def test_an_isolated_partial_strike_moved_into_the_band_is_judged_in_it():
    """A stroke over the middle of a long word, nothing else struck on its line: a partial strike
    there must sit within PARTIAL_ISO_MAX_OFF of the middle when its line is in the band. Moved
    into the band from 4 px lower, it is dropped, as with the line drawn on the stroke."""
    gray = _sparse(x1=X0 + 360)
    xa, xb = X0 + 100, X0 + 220
    _stroke(gray, 156, xa=xa, xb=xb)
    word = Word("a" * 30, (X0 / W, BOX[0] / H, (X0 + 360) / W, BOX[1] / H), None)
    for y in (160, 156):
        assert classify_lines([_line(y, xa=xa, xb=xb)], [word], gray, config=CF,
                              dpi=200)[1] == [], y


def test_the_confidence_rescue_applies_at_the_stroke_row(monkeypatch):
    """Azure DI words: a short line out of the band, under a stroke in it, through sparse letters
    read at 0.5. At the stroke's row the line is in the band, where low OCR confidence rescues a
    short line without ink on both sides."""
    gray = _sparse(x1=X0 + 60)
    _stroke(gray, 156, xa=X0 - 2, xb=X0 + 62)
    word = Word("abcd", (X0 / W, BOX[0] / H, (X0 + 60) / W, BOX[1] / H), 0.5)
    di = ScanConfig.azure_di()
    for y in (160, 156):
        rec, = classify_lines([_line(y, xa=X0 - 2, xb=X0 + 62)], [word], gray, config=di,
                              dpi=200)[1]
        assert rec["char_span"] == (0, 4), y
    monkeypatch.setattr(scanned, "STROKE_SNAP_PX", 0)
    assert classify_lines([_line(160, xa=X0 - 2, xb=X0 + 62)], [word], gray, config=di,
                          dpi=200)[1] == []


def test_reach_is_four_px_at_200_dpi(monkeypatch):
    gray = _page()
    _stroke(gray, 163)
    assert _struck(gray, _line(168), snap=4, monkeypatch=monkeypatch)       # within reach
    assert not _struck(gray, _line(170), snap=4, monkeypatch=monkeypatch)   # past it


@pytest.mark.parametrize("dpi, reach", [(200, 4), (400, 8), (100, 2), (300, 6)])
def test_reach_scales_with_dpi(monkeypatch, dpi, reach):
    seen = []
    real = scanned._stroke_row_dy
    monkeypatch.setattr(scanned, "_stroke_row_dy",
                        lambda *a: seen.append(a[-1]) or real(*a))
    gray = _page()
    _stroke(gray, 163)
    classify_lines([_line(167)], [_word()], gray, config=CF, dpi=dpi)
    assert seen == [reach]


def test_at_400_dpi_the_reach_is_8_px(monkeypatch):
    gray = _page(k=2)
    _stroke(gray, 163, k=2)                                 # rows 324-327
    line = _line(168, k=2)                                  # row 336: 9 px under the stroke
    assert _struck(gray, line, k=2, snap=0, monkeypatch=monkeypatch) == []
    rec, = _struck(gray, line, k=2, snap=4, monkeypatch=monkeypatch)
    assert rec["char_span"] == (0, N)


# ------------------------------------------------------------------- what must stay dropped

@pytest.mark.parametrize("line_y", [173, 176, 170])
def test_an_underline_stays_an_underline(monkeypatch, line_y):
    """An underline under the letters, the line on it or a few px off it: the band with the long
    run is the underline itself, with no ink under it."""
    gray = _page()
    _stroke(gray, 173)
    assert _struck(gray, _line(line_y), snap=4, monkeypatch=monkeypatch) == []


def test_a_table_rule_under_the_letters_stays_a_rule(monkeypatch):
    gray = _page()
    gray[172:174, 50:850] = 20                              # a rule across the page
    for y in (169, 173, 176):
        assert _struck(gray, _line(y, fill=0.98), snap=4, monkeypatch=monkeypatch) == []


def test_no_stroke_near_the_line_no_retry(monkeypatch):
    """A line under the letters with no stroke within reach: glyph rows break at every letter."""
    gray = _page()
    for y in (166, 167, 172):
        assert _struck(gray, _line(y), snap=4, monkeypatch=monkeypatch) == []


def test_a_stroke_over_part_of_the_word_is_not_retried(monkeypatch):
    """The retry needs an unbroken run over STROKE_SNAP_MIN_RUN of the columns: a stroke over 60%
    of the word, its line out of the band below it, stays dropped."""
    gray = _page()
    _stroke(gray, 163, xa=X0 - 2, xb=X0 + int(0.6 * (_x1() - X0)))
    assert _struck(gray, _line(167), snap=4, monkeypatch=monkeypatch) == []


def test_words_the_gates_pass_are_judged_as_before(monkeypatch):
    """A line on its stroke, through the middle and low in the letters: the same records with and
    without the retry."""
    for y in (150, 160, 140):
        gray = _page()
        _stroke(gray, y)
        on = _struck(gray, _line(y), snap=4, monkeypatch=monkeypatch)
        off = _struck(gray, _line(y), snap=0, monkeypatch=monkeypatch)
        assert on and on == off


# ------------------------------------------------------------------- _stroke_row_dy

def test_stroke_row_dy_moves_to_the_nearest_band_on_the_stroke():
    ink = np.zeros((60, 200), bool)
    ink[30:32, 10:190] = True                               # a 2 px stroke on rows 30-31
    ends = ((10.0, 37.0), (190.0, 37.0))                    # bands are rows y-1 to y+1
    assert scanned._stroke_row_dy(ink, ends, 2.0, 10, 190, 6) == -5   # band 31-33, the nearest
    assert scanned._stroke_row_dy(ink, ends, 2.0, 10, 190, 4) == 0    # out of reach
    on = ((10.0, 31.0), (190.0, 31.0))
    assert scanned._stroke_row_dy(ink, on, 2.0, 10, 190, 4) == 0      # already on it
    assert scanned._stroke_row_dy(ink, ends, 2.0, 10, 190, 0) == 0
    assert scanned._stroke_row_dy(ink, ends, 2.0, 195, 199, 6) == 0   # past the line's end
    up = ((10.0, 25.0), (190.0, 25.0))
    assert scanned._stroke_row_dy(ink, up, 2.0, 10, 190, 6) == 4      # and down: band 28-30


def test_stroke_row_dy_follows_a_sloped_line():
    ink = np.zeros((80, 300), bool)
    for x in range(20, 281):
        y = 30 + (x - 20) * 0.05                            # a stroke falling 13 px over 260
        ink[int(y):int(y) + 2, x] = True
    ends = ((20.0, 26.0), (280.0, 39.0))                    # the same slope, 4 px above it
    assert scanned._stroke_row_dy(ink, ends, 2.0, 20, 280, 4) == 3
    flat = ((20.0, 26.0), (280.0, 26.0))                    # level: it meets the stroke at one end
    assert scanned._stroke_row_dy(ink, flat, 2.0, 20, 280, 4) == 0


def test_stroke_row_dy_needs_the_run_share(monkeypatch):
    ink = np.zeros((60, 200), bool)
    ink[30:32, 10:150] = True                               # a stroke over 140 of 181 columns
    ends = ((10.0, 34.0), (190.0, 34.0))
    monkeypatch.setattr(scanned, "STROKE_SNAP_MIN_RUN", 0.80)
    assert scanned._stroke_row_dy(ink, ends, 2.0, 10, 190, 4) == 0
    monkeypatch.setattr(scanned, "STROKE_SNAP_MIN_RUN", 0.75)
    assert scanned._stroke_row_dy(ink, ends, 2.0, 10, 190, 4) == -2
    halves = np.zeros((60, 200), bool)                     # two half strokes on different rows:
    halves[30:32, 10:100] = halves[36:38, 100:191] = True  # together across, apart neither is
    monkeypatch.setattr(scanned, "STROKE_SNAP_MIN_RUN", 0.80)
    assert scanned._stroke_row_dy(halves, ends, 2.0, 10, 190, 4) == 0


def test_longest_run():
    assert scanned._longest_run(np.array([], bool)) == 0
    assert scanned._longest_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == 3
    assert scanned._longest_run(np.ones(5, bool)) == 5


# ------------------------------------------------------------------- end to end, real StrikeNet

def _text_page(font, size, frac):
    """Real text, a 0.7 pt strike through "remove this clause" at `frac` of the x-height."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=120)
    page.insert_text((30, 60), "keep these words and remove this clause now", fontname=font,
                     fontsize=size)
    words = page.get_text("words")
    struck = [w for w in words if w[4] in ("remove", "this", "clause")]
    y = 60.0 - frac * {"helv": 0.523, "tiro": 0.45}[font] * size
    page.draw_line((struck[0][0] - 1, y), (struck[-1][2] + 1, y), width=0.7)
    gray = detect._render_gray(page, 200)
    h, w = gray.shape
    s = 200 / 72
    return gray, [Word(t[4], (t[0] * s / w, t[1] * s / h, t[2] * s / w, t[3] * s / h), None)
                  for t in words]


@pytest.mark.parametrize("font, size, frac, dy", [("tiro", 10, 0.35, 4), ("helv", 12, 0.5, 4)])
def test_detect_scanned_image_keeps_the_word_its_line_misses(monkeypatch, font, size, frac, dy):
    gray, words = _text_page(font, size, frac)
    real = scanned.strike_lines

    def shifted(g, dpi=200, ink=None):                     # the lines `dy` px under the strike
        return [{**ln, "ends_px": tuple((x, y + dy) for x, y in ln["ends_px"]),
                 "bbox_px": (ln["bbox_px"][0], ln["bbox_px"][1] + dy, ln["bbox_px"][2],
                             ln["bbox_px"][3] + dy)} for ln in real(g, dpi=dpi, ink=ink)]

    def final(snap):
        monkeypatch.setattr(scanned, "STROKE_SNAP_PX", snap)
        return sorted(r["text"] for r in detect.detect_scanned_image(gray, words, dpi=200)
                      if r["final"])

    assert final(4) == ["clause", "remove", "this"]        # the lines where strike_lines puts them
    monkeypatch.setattr(scanned, "strike_lines", shifted)
    assert final(0) != ["clause", "remove", "this"]        # off the stroke: a word lost
    assert final(4) == ["clause", "remove", "this"]        # tried again at the stroke's row
