"""A double strike on the scanned path: two thin lines through the x-height.

Each line of a double strike sits off the middle of the x-height, and the line finder gives one of
them (or one stitched along the glyph pieces beside it, which the retry at the stroke's row moves
onto it). The ink tests read glyph ink from 2 px
past that line, which skips the one to three rows of the x-height on its outer side, so the struck
word read as a table rule or as ink on one side only. A word the ink tests drop now passes them
when a second unbroken stroke runs beside its line within ``PAIR_REACH`` of the box height, the
rows between the two hold glyph ink, both strokes run at least ``STROKE_RUN_MIN_CHARS`` character
widths and about as long as each other (``PAIR_LEN_RATIO``), and their middle, halfway to the
nearest band on the second stroke that puts it in the strike band (its edge nearest the line, or
further onto it), is in that band. Only under a line at
most ``PAIR_MAX_LEN_IN`` long and ``PAIR_MAX_RUN_PX`` thick, and straighter than
``PAIR_MAX_WOBBLE``. A double rule has nothing between its lines and a double underline's middle
is below the band; one glyph's bars are short; a line across the page is the ruling of lined
paper; a thick or wobbly line is a pen stroke, and beside a pen stroke the other is a ruling or a
stroke of another length.

Synthetic rasters (letters of three ink stems) and real text with the real StrikeNet.
"""
import dataclasses

import numpy as np
import pymupdf
import pytest

import pdf_strikethrough as st
from pdf_strikethrough import scanned
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import ScanConfig, classify_lines

CF = ScanConfig.confidence_free()
H, W = 300, 900
X0, ADV, N = 200, 18, 12        # where the word starts, the letter advance, letters
TOP, BOT = 140, 158             # the letters' x-height band


def _x1(n=N):
    return X0 + (n - 1) * ADV + 14


def _page(top=TOP, bot=BOT, n=N, strokes=(), xa=None, xb=None):
    """N letters of three stems over rows `top` to `bot`, and 2 px strokes at the rows in
    `strokes`, from `xa` to `xb`."""
    gray = np.full((H, W), 255, np.uint8)
    for i in range(n):
        for s in (0, 5, 11):
            x = X0 + i * ADV + s
            gray[top:bot, x:x + 3] = 20
    xa, xb = (X0 - 2 if xa is None else xa), (_x1(n) + 2 if xb is None else xb)
    for y in strokes:
        gray[y:y + 2, xa:xb] = 20
    return gray


def _word(box=(119, 179), n=N):
    return Word("abcdefghijkl"[:n], (X0 / W, box[0] / H, _x1(n) / W, box[1] / H), None)


def _line(y, n=N, fill=0.95, xa=None, xb=None):
    """The record strike_lines would give for a 2 px line at row `y`."""
    xa, xb = (X0 - 2 if xa is None else xa), (_x1(n) + 2 if xb is None else xb)
    return {"bbox_px": (xa, y, xb, y + 1), "ends_px": ((float(xa), y + 0.5), (float(xb), y + 0.5)),
            "len_in": (xb - xa) / 200, "angle_deg": 0.0, "fill": fill, "run_px": 2.0,
            "straightness": 1.0}


def _struck(gray, line, word=None, reach=None, monkeypatch=None, dpi=200):
    if reach is not None:
        monkeypatch.setattr(scanned, "PAIR_REACH", reach)
    return classify_lines([line], [word or _word()], gray, config=CF, dpi=dpi)[1]


# ------------------------------------------------------------------- the words the tests dropped

def test_double_strike_found_from_its_upper_line(monkeypatch):
    """Strokes at rows 142 and 152 through letters on rows 140-157: the line on the upper one
    has two rows of letters above it, which the 2 px gap takes, so the long solid line read as a
    table rule. Its partner below, with the letters' middles between, makes it a double strike."""
    gray = _page(strokes=(142, 152))
    line = _line(142)
    assert _struck(gray, line, reach=0, monkeypatch=monkeypatch) == []
    rec, = _struck(gray, line, reach=0.30, monkeypatch=monkeypatch)
    assert rec["char_span"] == (0, N) and rec["geom_corroborated"]
    assert abs(rec["off"]) <= scanned.STRIKE_TOL


def test_double_strike_found_from_its_lower_line(monkeypatch):
    """The line on the lower stroke, low in the box, the letters ending two rows below it: out of
    the strike band, with no ink below it past the gap. The pair's middle, halfway to the band where
    the upper stroke is found, 8 px up at its edge nearest the line, is in the band: the word
    counts as strike geometry, its `off` moved there."""
    gray = _page(bot=155, strokes=(142, 152))
    line = _line(152)
    word = _word(box=(110, 170))
    assert _struck(gray, line, word, reach=0, monkeypatch=monkeypatch) == []
    rec, = _struck(gray, line, word, reach=0.30, monkeypatch=monkeypatch)
    assert rec["geom_corroborated"]
    assert rec["off"] == pytest.approx((148.5 - 110) / 60 - 0.5, abs=0.005)


def test_default_is_on():
    assert scanned.PAIR_REACH == 0.30
    assert len(_struck(_page(strokes=(142, 152)), _line(142))) == 1


# ------------------------------------------------------------------------ what stays dropped

def test_double_rule_beside_the_letters_is_not_a_strike(monkeypatch):
    """Two rules 10 px apart under the letters, nothing between them: no glyph ink between."""
    gray = _page(top=120, bot=140, strokes=(144, 154))
    assert _struck(gray, _line(144), _word(box=(110, 170)), monkeypatch=monkeypatch) == []


def test_double_underline_is_out_of_the_band(monkeypatch):
    """Two lines low on the letters, the upper through their last four rows: glyph ink between
    the two, and the row midway clear, but the pair's middle is below the band."""
    gray = _page(top=120, bot=146, strokes=(142, 150))
    word = _word(box=(110, 160))
    assert scanned.STRIKE_TOL < (146 - 110) / 50 - 0.5 < 0.25
    assert _struck(gray, _line(142), word, monkeypatch=monkeypatch) == []
    monkeypatch.setattr(scanned, "STRIKE_TOL", 0.25)
    assert len(_struck(gray, _line(142), word)) == 1


def test_sparse_ink_between_is_not_a_pair(monkeypatch):
    """Two lines with only every other letter's descender crossing the gap between them, in a box
    tall enough that their middle is in the band: glyph ink in too few of the columns between
    the two (INK_MIN_FRAC)."""
    gray = _page(top=120, bot=140, strokes=(142, 150))
    for i in range(0, N, 2):                                # descenders down to row 156
        gray[140:156, X0 + i * ADV:X0 + i * ADV + 3] = 20
    word = _word(box=(100, 190))
    assert abs((146 - 100) / 90 - 0.5) <= scanned.STRIKE_TOL
    assert _struck(gray, _line(142), word, monkeypatch=monkeypatch) == []


def test_bars_of_one_glyph_are_too_short(monkeypatch):
    """A one-letter word whose two bars (a P's bowl, an =) run across its box with its stem
    between them: a pair of strokes by every other measure. Each bar is shorter than
    STROKE_RUN_MIN_CHARS characters, so it is not taken for a double strike."""
    gray = np.full((H, W), 255, np.uint8)
    gray[140:158, X0:X0 + 3] = 20                          # the stem
    gray[142:144, X0:X0 + 14] = 20                         # the bowl's top and bottom
    gray[152:154, X0:X0 + 14] = 20
    word = Word("P", (X0 / W, 119 / H, (X0 + 14) / W, 179 / H), None)
    line = _line(142, xa=X0, xb=X0 + 60)
    assert _struck(gray, line, word, monkeypatch=monkeypatch) == []
    monkeypatch.setattr(scanned, "STROKE_RUN_MIN_CHARS", 0)
    assert len(_struck(gray, line, word)) == 1


def test_line_not_on_a_stroke_is_not_half_of_a_pair(monkeypatch):
    """A line through the letters where nothing runs unbroken (a glyph chain), with a stroke 8 px
    above it and the letters between: the line's own band must be a stroke too. With the length
    match off, which would also reject it: followed along the chain, the line stops at the gaps
    between letters, 31 px against the stroke's 215."""
    monkeypatch.setattr(scanned, "PAIR_LEN_RATIO", 0)
    gray = _page(bot=154, strokes=(142,))
    assert _struck(gray, _line(150), _word(box=(110, 170)), monkeypatch=monkeypatch) == []


def test_pair_middle_out_of_the_band(monkeypatch):
    """A double strike in a box sitting high above the letters: the line on the lower stroke and
    the pair's middle are both out of the strike band, so it is not taken for a double strike."""
    gray = _page(bot=155, strokes=(142, 152))
    word = _word(box=(100, 160))
    assert (147 - 100) / 60 - 0.5 > scanned.STRIKE_TOL
    assert _struck(gray, _line(152), word, monkeypatch=monkeypatch) == []


def test_pair_middle_judged_at_the_second_strokes_near_edge():
    """Strokes on rows 142 and 152, the line on the upper one, in 40 px boxes: the band check takes
    the second stroke at the band where it is found, 9 px down at its edge nearest the line, so the
    middle is row 147 (147.5 at the stroke's center). In the box from row 121 that is in the band,
    and the pair is found; one row higher it is out, and there is no pair."""
    gray = _page(strokes=(142, 152))
    assert (147 - 120) / 40 - 0.5 > scanned.STRIKE_TOL >= (147 - 121) / 40 - 0.5
    assert _struck(gray, _line(142), _word(box=(120, 160))) == []
    assert len(_struck(gray, _line(142), _word(box=(121, 161)))) == 1


def test_pair_middle_may_be_taken_at_the_second_strokes_far_edge():
    """The line on the lower stroke, below the band, in a 60 px box from row 107.5: with the upper
    stroke taken at its edge nearest the line (8 px up), or at its center (10 px), the pair's
    middle is out of the band; the band 11 px up, at its far edge, still holds the stroke and puts
    the middle in, so the pair is found there. From row 107 no band on the stroke puts it in."""
    gray = _page(bot=155, strokes=(142, 152))
    assert (147.5 - 107.5) / 60 - 0.5 > scanned.STRIKE_TOL >= (147 - 107.5) / 60 - 0.5
    rec, = _struck(gray, _line(152), _word(box=(107.5, 167.5)))
    assert rec["off"] == pytest.approx((147 - 107.5) / 60 - 0.5, abs=0.005)
    assert _struck(gray, _line(152), _word(box=(107, 167))) == []


def test_double_overline_is_out_of_the_band(monkeypatch):
    """Two lines high on the letters, glyph ink between them, the pair's middle above the band:
    not a pair. With a wider band it is one."""
    gray = _page(top=130, bot=146, strokes=(132, 142))
    word = _word(box=(126, 166))
    assert (138.5 - 126) / 40 - 0.5 < -scanned.STRIKE_TOL
    assert _struck(gray, _line(142), word) == []
    monkeypatch.setattr(scanned, "STRIKE_TOL", 0.25)
    assert len(_struck(gray, _line(142), word)) == 1


def test_one_thick_stroke_is_not_a_pair(monkeypatch):
    """A short 8 px stroke over a two-letter word, its line claiming 2 px: bands 4 px apart both
    lie on it, and its own rows between them are glyph ink (too short for a rule). With no gap
    between them, they are one stroke."""
    gray = _page(n=2, strokes=())
    gray[142:150, X0 - 2:_x1(2) + 2] = 20
    word = _word(n=2)
    line = _line(142, n=2, fill=1.0)
    assert line["len_in"] < scanned.INK_SHORT_LEN_IN
    assert _struck(gray, line, word, monkeypatch=monkeypatch) == []


def test_pair_over_part_of_a_word(monkeypatch):
    """A double strike over the first four of twelve letters: the line covers less than
    MIN_WORD_XOVER of the word, so it is not tried as a pair."""
    xb = X0 + 3 * ADV + 16
    gray = _page(strokes=(142, 152), xb=xb)
    assert _struck(gray, _line(142, xb=xb), monkeypatch=monkeypatch) == []


def test_partner_out_of_reach(monkeypatch):
    """Strokes 18 px apart, more than PAIR_REACH of a 40 px box: not a pair."""
    gray = _page(strokes=(140, 158))
    word = _word(box=(129, 169))
    assert 18 > scanned.PAIR_REACH * 40
    assert _struck(gray, _line(140), word, monkeypatch=monkeypatch) == []


# ------------------------------------------------------------------ where the pair test is tried

def test_line_across_the_page_is_not_tried_for_a_pair(monkeypatch):
    """The same strokes across the page, 4.4 in: past PAIR_MAX_LEN_IN a line is the ruling of
    lined paper, where a pen stroke above it with handwriting between would make a pair, so the
    pair test is not tried and the word stays dropped. With the cap off, or raised past the line,
    it is found."""
    gray, line = _page(strokes=(142, 152), xa=10, xb=890), _line(142, xa=10, xb=890)
    assert scanned.PAIR_MAX_LEN_IN == 4.0 < line["len_in"]
    assert _struck(gray, line) == []
    for cap in (0, 4.5):
        monkeypatch.setattr(scanned, "PAIR_MAX_LEN_IN", cap)
        rec, = _struck(gray, line)
        assert rec["char_span"] == (0, N)


def test_line_exactly_at_the_cap_is_tried():
    """A line exactly PAIR_MAX_LEN_IN long (4 in) is still tried for a pair."""
    gray, line = _page(strokes=(142, 152), xa=50, xb=850), _line(142, xa=50, xb=850)
    assert line["len_in"] == scanned.PAIR_MAX_LEN_IN
    assert len(_struck(gray, line)) == 1


def test_thick_line_is_not_tried_for_a_pair(monkeypatch):
    """Strokes 5 px thick at rows 140 and 152, the line on the upper one (run_px 5): thicker than
    PAIR_MAX_RUN_PX, as a pen stroke is, so the pair test is not tried. With the limit off the
    pair is found. The limit is the line finder's own thin-stroke limit, past which the thick
    retry is tried instead."""
    gray = _page(strokes=())
    for y in (140, 152):
        gray[y:y + 5, X0 - 2:_x1() + 2] = 20
    line = {**_line(140), "bbox_px": (X0 - 2, 140, _x1() + 2, 145),
            "ends_px": ((float(X0 - 2), 142.0), (float(_x1() + 2), 142.0)), "run_px": 5.0}
    assert scanned.PAIR_MAX_RUN_PX == scanned.THIN_STROKE_RUN_PX == 4.0
    assert _struck(gray, line) == []
    monkeypatch.setattr(scanned, "PAIR_MAX_RUN_PX", 0)
    assert len(_struck(gray, line)) == 1


def test_line_exactly_at_the_thickness_limit_is_tried(monkeypatch):
    """Strokes 4 px thick, the line exactly PAIR_MAX_RUN_PX thick: still tried for a pair, and
    found by it."""
    gray = _page(strokes=())
    for y in (140, 152):
        gray[y:y + 4, X0 - 2:_x1() + 2] = 20
    line = {**_line(140), "bbox_px": (X0 - 2, 140, _x1() + 2, 143),
            "ends_px": ((float(X0 - 2), 141.5), (float(_x1() + 2), 141.5)), "run_px": 4.0}
    assert line["run_px"] == scanned.PAIR_MAX_RUN_PX
    assert len(_struck(gray, line)) == 1
    assert _struck(gray, line, reach=0, monkeypatch=monkeypatch) == []


def test_thickness_limit_scales_with_the_dpi(monkeypatch):
    """A double strike of 3 px strokes drawn at twice the size and read at 400 dpi, where the
    limit is 8 px: its 6 px line is tried and the pair found. The letters start at the upper
    stroke's top, so the ink tests' 2 px gap, which does not scale, leaves no ink above it."""
    gray = _page(top=141, strokes=())
    for y in (141, 152):
        gray[y:y + 3, X0 - 2:_x1() + 2] = 20
    gray = gray.repeat(2, axis=0).repeat(2, axis=1)
    xa, xb = 2 * (X0 - 2), 2 * (_x1() + 2)
    line = {**_line(141), "bbox_px": (xa, 282, xb, 287),
            "ends_px": ((float(xa), 284.5), (float(xb), 284.5)), "run_px": 6.0}
    assert line["run_px"] > scanned.PAIR_MAX_RUN_PX
    assert len(_struck(gray, line, dpi=400)) == 1
    assert _struck(gray, line, reach=0, monkeypatch=monkeypatch, dpi=400) == []


def test_thickness_limit_is_at_least_2px(monkeypatch):
    """Read at 75 dpi the limit scaled with the dpi is 1.5 px, held at 2 px as the line finder's
    own thin-stroke limit is: a 2 px double strike is still tried, and found by it."""
    gray, line = _page(strokes=(142, 152)), _line(142)
    assert scanned.PAIR_MAX_RUN_PX * 75 / scanned.RENDER_DPI < line["run_px"] == 2.0
    assert len(_struck(gray, line, dpi=75)) == 1
    assert _struck(gray, line, reach=0, monkeypatch=monkeypatch, dpi=75) == []


XA, XB = X0 - 2, _x1() + 2          # the strokes over the word: columns 198 to 413


def test_partner_a_ruling_is_not_a_pair(monkeypatch):
    """The line on a stroke over the word, 215 px, and the stroke below it a ruling across the page,
    879 px: a pen stroke above a ruling, not a double strike, whose lines run as long as each
    other. With the length match off the pair is found."""
    gray = _page(strokes=(142,))
    gray[152:154, 10:890] = 20
    assert scanned.PAIR_LEN_RATIO == 0.5
    assert _struck(gray, _line(142)) == []
    monkeypatch.setattr(scanned, "PAIR_LEN_RATIO", 0)
    assert len(_struck(gray, _line(142))) == 1


def test_only_the_nearest_stroke_is_measured():
    """A double strike found from its lower line, on lined paper whose ruling runs 7 px below it,
    nearer than the upper line: the ruling is the nearest stroke that passes the rest, too long
    for a partner, and only the nearest is measured, so the word stays dropped. Without the
    ruling the pair is found."""
    gray = _page(top=128, bot=146, strokes=(132, 142))
    word = _word(box=(110, 180))
    assert len(_struck(gray, _line(142), word)) == 1
    gray[149:151, 10:890] = 20
    assert _struck(gray, _line(142), word) == []


def test_stroke_out_of_the_band_is_passed_over():
    """A double-struck word that is underlined too, the underline nearer the found line than the
    upper line is: the underline's middle with the line is below the band, so the search goes on
    to the upper line, and the pair is found."""
    gray = _page(top=128, bot=146, strokes=(132, 142))
    gray[149:151, XA:XB] = 20
    word = _word(box=(118, 158))
    assert (148.5 + 142.5) / 2 > 118 + (0.5 + scanned.STRIKE_TOL) * 40
    rec, = _struck(gray, _line(142), word)
    assert rec["off"] == pytest.approx((138.5 - 118) / 40 - 0.5, abs=0.01)


def test_only_the_stroke_out_of_the_band_is_passed_over():
    """Three strokes, the line on the lowest, below the band: the stroke 6 px above it makes a
    middle still below the band and is passed over, and the one 16 px above, found 14 px up at its
    edge nearest the line, makes the pair. Only that stroke is passed over, not the rest of its
    side."""
    gray = _page(top=128, bot=155, strokes=(136, 146, 152))
    word = _word(box=(109, 169))
    assert (149 - 109) / 60 - 0.5 > scanned.STRIKE_TOL
    rec, = _struck(gray, _line(152), word)
    assert rec["off"] == pytest.approx((145.5 - 109) / 60 - 0.5, abs=0.005)


def test_underline_past_the_partner_is_not_part_of_it():
    """A word-long underline two rows under the second stroke: the stroke is taken at the band
    where it is found, at its edge nearest the line, so ink past it does not move the pair's
    middle, which stays in the band, and the pair is found."""
    gray = _page(bot=155, strokes=(142, 152))
    gray[156:158, XA:XB] = 20
    assert len(_struck(gray, _line(142), _word(box=(122, 162)))) == 1


def test_partner_exactly_twice_as_long_is_a_pair():
    """Each stroke is followed along its band: the line's 215 px, and a partner of 430 px, exactly
    twice as long, is still a pair; 1 px longer it is not."""
    for end, found in ((XA + 431, 1), (XA + 432, 0)):
        gray = _page(strokes=(142,))
        gray[152:154, XA:end] = 20
        ink = gray < 128
        assert scanned._stroke_length(ink, _line(142)["ends_px"], 2.0, 9, 306.0) == end - 1 - XA
        assert len(_struck(gray, _line(142))) == found


def test_lengths_followed_from_the_word_middle():
    """The stroke below starts a letter (20 px) into the word: each stroke is followed both ways
    from the middle of the word, so the two still run about as long as each other."""
    gray = _page(strokes=(142,))
    gray[152:154, XA + 20:XB] = 20
    assert len(_struck(gray, _line(142))) == 1


def test_breaks_of_up_to_3px_are_bridged():
    """The stroke below runs on past the word after a break: across a 3 px break it is followed
    (a ruling, too long for a partner), across a 4 px one it ends with the word."""
    for brk, found in ((3, 0), (4, 1)):
        gray = _page(strokes=(142,))
        gray[152:154, XA:XB] = 20
        gray[152:154, XB + brk:890] = 20
        assert len(_struck(gray, _line(142))) == found


def test_stroke_followed_from_the_band_where_it_is_found():
    """A ruling that runs on past the word up to 5 px nearer the line or 2 px further from it, as a
    slightly tilted one does: the second stroke is read within the line's half thickness plus 2 px
    of the band where it is found, at its edge nearest the line, so it is followed, and too long
    for a partner. 6 px nearer or 3 px further, it ends with the word."""
    for step, found in ((-5, 0), (2, 0), (-6, 1), (3, 1)):
        gray = _page(strokes=(142,))
        gray[152:154, XA:XB] = 20
        gray[152 + step:154 + step, XB:890] = 20
        assert len(_struck(gray, _line(142))) == found


def test_length_measure_scales_with_the_dpi():
    """At 400 dpi the bridged break and the rows read beside each stroke double: the 3 px double
    strike drawn at twice the size (6 px strokes), its lower stroke running on past the word after
    a 6 px break, or 12 px nearer the line or 4 px further from it, is followed, a ruling; after an
    8 px break, or 5 px further, it ends with the word. (From 13 px nearer, the line's own rows
    reach it too, and both strokes run on.)"""
    def page(brk=0, step=0):
        gray = _page(top=141, strokes=())
        gray[141:144, XA:XB] = 20
        gray[152:155, XA:XB] = 20
        gray = gray.repeat(2, axis=0).repeat(2, axis=1)
        gray[304 + step:310 + step, 2 * XB + brk:1780] = 20
        return gray
    line = {**_line(141), "bbox_px": (2 * XA, 282, 2 * XB, 287),
            "ends_px": ((float(2 * XA), 284.5), (float(2 * XB), 284.5)), "run_px": 6.0}
    for brk, step, found in ((6, 0, 0), (0, -12, 0), (0, 4, 0), (8, 0, 1), (0, 5, 1)):
        assert len(_struck(page(brk, step), line, dpi=400)) == found


def test_strokes_followed_along_the_line_slope(monkeypatch):
    """The ruling test on a page skewed 1.2 degrees, everything on it turned with it: the ruling is
    still too long for a partner. With the length match off the pair is found."""
    s = np.tan(np.radians(1.2))
    gray = _page(strokes=(142,))
    gray[152:154, 10:890] = 20
    for x in range(W):                                      # each column down by its skew
        gray[:, x] = np.roll(gray[:, x], int(round(s * (x - XA))))
    line = {**_line(142), "bbox_px": (XA, 142, XB, 147), "angle_deg": 1.2,
            "ends_px": ((float(XA), 142.5), (float(XB), 142.5 + s * (XB - XA)))}
    assert _struck(gray, line) == []
    monkeypatch.setattr(scanned, "PAIR_LEN_RATIO", 0)
    assert len(_struck(gray, line)) == 1


def test_length_measured_along_the_line_slope():
    """A 600 px stroke at 2 degrees, followed from its middle along the line's slope, measures
    599 px: read along a level row it would leave the rows read within 4 px of drift."""
    s = np.tan(np.radians(2.0))
    ink = np.zeros((H, W), bool)
    for x in range(100, 700):
        ink[int(round(150 + s * (x - 100))) + np.arange(2), x] = True
    ends = ((100.0, 150.5), (699.0, 150.5 + s * 599))
    assert scanned._stroke_length(ink, ends, 2.0, 0, 400.0) == 599


def test_partner_offset_is_its_nearest_band():
    """The offset found is the nearest band searched that lies on the second stroke and passes the
    test for glyph ink between: 9 px for a stroke on rows 152-153 under a line on rows 142-143,
    with letters through both; 10 px with glyph ink only in the two rows next to the stroke (the
    band at 9 px reaches it, but the rows between hold no glyph ink); 4 px for a stroke two rows
    below the line. With the offsets searched ending at 8 px, short of the stroke, there is none."""
    dys = [d for d in range(-18, 19) if abs(d) >= 4]
    for strokes, top, bot, off in (((142, 152), TOP, BOT, 9), ((142, 152), 150, 152, 10),
                                   ((142, 146), 138, 152, 4)):
        ink = _page(top=top, bot=bot, strokes=strokes) < 128
        assert scanned._partner_stroke(ink, scanned._glyph_ink(ink), _line(142)["ends_px"], 2.0,
                                       XA, XB, dys) == off
    ink = _page(strokes=(142, 152)) < 128
    assert scanned._partner_stroke(ink, scanned._glyph_ink(ink), _line(142)["ends_px"], 2.0, XA, XB,
                                   [d for d in range(-8, 9) if abs(d) >= 4]) == 0


def test_wobbly_line_is_not_tried_for_a_pair(monkeypatch):
    """A line whose wobble is PAIR_MAX_WOBBLE or more, as a pen stroke's is, or unmeasured, which
    counts as not straight: the pair test is not tried. Just under it, or with the limit off, the
    pair is found."""
    gray = _page(strokes=(142, 152))
    assert scanned.PAIR_MAX_WOBBLE == 1.3
    for wobble, found in ((1.29, 1), (1.3, 0), (None, 0)):
        assert len(_struck(gray, {**_line(142), "straightness": wobble})) == found
    monkeypatch.setattr(scanned, "PAIR_MAX_WOBBLE", 0)
    for wobble in (1.3, None):
        assert len(_struck(gray, {**_line(142), "straightness": wobble})) == 1


def test_bar_between_two_rules_is_not_glyph_ink():
    """Two rules 12 px apart over an empty word box, a 3 px bar between them off the row midway: the
    rows between hold the bar, a rule too, so no glyph ink, and there is no pair."""
    gray = _page(n=0, strokes=(142, 154), xa=XA, xb=XB)
    gray[145:148, XA:XB] = 20
    assert _struck(gray, _line(142)) == []


def test_no_pair_test_without_glyph_ink():
    """With ScanConfig(glyph_ink=False) the ink tests read all ink, where a thick bar between two
    rules has its own rows between them: the pair test is for glyph ink only, so the double strike
    found from its upper line above stays dropped."""
    gray, line = _page(strokes=(142, 152)), _line(142)
    assert len(_struck(gray, line)) == 1
    cfg = dataclasses.replace(CF, glyph_ink=False)
    assert classify_lines([line], [_word()], gray, config=cfg, dpi=200)[1] == []


# ------------------------------------------------------------------------- real text, end to end

def _real(text, font, size, strike, width):
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=120)
    page.insert_text((24, 64), text, fontname=font, fontsize=size)
    ws = page.get_text("words")
    y = 64 - size * pymupdf.Font(font).glyph_bbox(ord("x")).y1 / 2
    for dy in (-0.09 * size, 0.09 * size):                  # two lines, as Word draws one
        page.draw_line((ws[strike[0]][0], y + dy), (ws[strike[1]][2], y + dy), width=width)
    words = [Word(w[4], (w[0] / 420, w[1] / 120, w[2] / 420, w[3] / 120), None) for w in ws]
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    return gray, words


@pytest.mark.parametrize("text,strike,font,size,width,before,after", [
    ("no more than one", (0, 2), "tiro", 11, 0.4, ["than"], ["more", "no", "than"]),
    ("in respects to the wire transfer", (4, 5), "tiro", 13, 0.7, [], ["transfer", "wire"])])
def test_double_strike_on_real_text(text, strike, font, size, width, before, after, monkeypatch):
    gray, words = _real(text, font, size, strike, width)

    def found():
        recs = st.detect_scanned_image(gray, words, config=CF, dpi=200)
        return sorted(r["text"] for r in recs if r["final"])

    assert found() == after
    monkeypatch.setattr(scanned, "PAIR_REACH", 0)
    assert found() == before
