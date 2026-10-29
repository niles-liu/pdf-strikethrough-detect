"""A thick strike's ink test, on the scanned path.

A 2-3 pt bar through the middle of the x-height covers most of it and leaves two to four rows of
glyph ink above and below. The ink tests read the ink from ``INK_GAP_PX`` (2 px) past the stroke's
edges, which takes those rows, so the struck word read as ink on one side only (a long solid line:
a table rule) or on neither (a short line: no strike). A word either test drops under a thick line
(``run_px`` above ``THIN_STROKE_RUN_PX``, scaled with dpi) is now tried again with the ink read from
``INK_GAP_THICK_PX`` (0 px) past it, when the line is in the strike band and covers the word.
Words the tests pass keep their verdict.

Synthetic rasters: each letter is three ink stems, so the only long horizontal ink is the bar.
"""
import dataclasses

import numpy as np
import pytest

from pdf_strikethrough import scanned
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import ScanConfig, classify_lines

CF = ScanConfig.confidence_free()
H, W = 300, 900
X0, ADV, N = 200, 18, 12        # where the word starts, the letter advance, letters
TOP, BOT = 140, 153             # the letters' x-height band, no ascenders or descenders
BAR = (143, 149)                # the bar's rows: 6 px, three rows of letters above, four below
LY = 145.6                      # the line's height, as strike_lines measures it off the bar


def _x1(n=N):
    return X0 + (n - 1) * ADV + 14


def _page(k=1, top=TOP, bot=BOT, bar=BAR):
    """The word, N letters of three stems each over rows `top` to `bot`, struck by a bar over rows
    `bar`, at scale `k`."""
    gray = np.full((H * k, W * k), 255, np.uint8)
    for i in range(N):
        for s in (0, 5, 11):
            x = (X0 + i * ADV + s) * k
            gray[top * k:bot * k, x:x + 3 * k] = 20
    gray[bar[0] * k:bar[1] * k, (X0 - 2) * k:(_x1() + 2) * k] = 20
    return gray


def _word(box=(116, 175), x1=None):
    """The OCR box, rows `box` (centered on the bar: in the strike band), out to `x1`."""
    x1 = _x1() if x1 is None else x1
    return Word("abcdefghijkl", (X0 / W, box[0] / H, x1 / W, box[1] / H), None)


def _line(k=1, run=6.0, y=LY, fill=0.95, xa=None, xb=None):
    """The record strike_lines would give for the bar, from `xa` to `xb` (px at 200 dpi)."""
    xa, xb = (X0 - 2 if xa is None else xa) * k, (_x1() + 2 if xb is None else xb) * k
    return {"bbox_px": (xa, int(y * k) - 3 * k, xb, int(y * k) + 3 * k),
            "ends_px": ((float(xa), y * k), (float(xb), y * k)),
            "len_in": (xb - xa) / (200 * k), "angle_deg": 0.0, "fill": fill, "run_px": run * k,
            "straightness": 1.0}


def _struck(gray, line, word=None, k=1, gap=None, monkeypatch=None):
    """The records for one line through the word, the retry's gap INK_GAP_THICK_PX = `gap`."""
    if gap is not None:
        monkeypatch.setattr(scanned, "INK_GAP_THICK_PX", gap)
    _tagged, struck = classify_lines([line], [word or _word()], gray, config=CF, dpi=200 * k)
    return struck


# ------------------------------------------------------------------- the words the tests dropped

def test_long_thick_bar_is_not_a_table_rule(monkeypatch):
    """A long solid bar: read from 2 px past it, the three rows of letters above it are taken and
    two of the four below, so it read as ink on one side, a table rule. From its edges, both."""
    gray, line = _page(), _line()
    assert line["len_in"] >= scanned.TABLE_RULE_MIN_LEN_IN
    assert _struck(gray, line, gap=2, monkeypatch=monkeypatch) == []
    rec, = _struck(gray, line, gap=0, monkeypatch=monkeypatch)
    assert rec["char_span"] == (0, N) and not rec["partial"]
    assert rec["geom_corroborated"]                        # in the band, ink on both sides


def test_short_thick_bar_has_ink_on_both_sides(monkeypatch):
    """A short bar over a short word needs ink on both sides at any height. Read from 2 px past it
    there is none above; from its edges there is."""
    gray = _page()
    xb = X0 + 2 * ADV + 16                                  # three letters, 0.27 in
    line = _line(xb=xb)
    word = _word(x1=xb - 2)
    assert line["len_in"] < scanned.INK_SHORT_LEN_IN
    assert _struck(gray, line, word, gap=2, monkeypatch=monkeypatch) == []
    rec, = _struck(gray, line, word, gap=0, monkeypatch=monkeypatch)
    assert not rec["partial"]


def test_default_reads_from_the_edges():
    """The shipped setting is the retry from the bar's edges."""
    assert scanned.INK_GAP_THICK_PX == 0 < scanned.INK_GAP_PX == 2
    assert len(_struck(_page(), _line())) == 1


# ------------------------------------------------------------------------ what the retry leaves

def test_a_word_the_tests_pass_is_judged_as_before(monkeypatch):
    """The bar over the tops of the letters: read from 2 px past it there is no ink on either
    side, and a long solid line with none is kept. From its edges the letters' feet show below
    and none above, a table rule's reading. The retry comes only after a drop, so the word stays
    struck, as in the release."""
    gray = _page(top=143, bot=152, bar=(143, 149))
    line = _line(y=146.0)
    rec2, = _struck(gray, line, gap=2, monkeypatch=monkeypatch)
    rec0, = _struck(gray, line, gap=0, monkeypatch=monkeypatch)
    assert rec0 == rec2
    above, below = scanned._ink_above_below(scanned._glyph_ink(scanned.ink_mask(gray)),
                                            line["ends_px"], 6.0,
                                            (X0, 116, _x1(), 175), gap=0)
    assert scanned._one_sided(above, below)                # the reading the retry would give


def test_bar_between_two_rules_stays_a_rule(monkeypatch):
    """A thick bar with a thin rule two rows past each edge and no letters: a triple border. Its
    rules are not glyph ink, so there is no glyph ink beside it from its edges either, and it is
    judged on all ink as in the release, from 2 px past it, where only the lower rule shows: one
    side, a rule. Read on all ink from its edges, both rules would show and it would pass."""
    gray = np.full((H, W), 255, np.uint8)
    xa, xb = X0 - 2, _x1() + 2
    gray[BAR[0]:BAR[1], xa:xb] = 20
    gray[140:142, xa:xb] = 20                              # a rule above, and one below
    gray[151:153, xa:xb] = 20
    line = _line()
    assert _struck(gray, line, gap=0, monkeypatch=monkeypatch) == []
    ink = scanned.ink_mask(gray)
    box = (X0, 116, _x1(), 175)
    assert not scanned._one_sided(*scanned._ink_above_below(ink, line["ends_px"], 6.0, box, 0))


def test_bar_out_of_the_band_is_not_retried(monkeypatch):
    """The same bar under an OCR box sitting higher, so the bar is out of the strike band, as an
    underline is: it needs ink on both sides, and it is not tried from its edges."""
    gray, line = _page(), _line()
    word = _word(box=(95, 156))
    rel = (LY - 95) / (156 - 95)
    assert scanned.STRIKE_TOL < rel - 0.5 < scanned.INK_MAX_OFF
    assert _struck(gray, line, word, gap=0, monkeypatch=monkeypatch) == []


def test_bar_over_part_of_a_word_is_not_retried(monkeypatch):
    """A bar over the first four of twelve letters, lying wholly inside the word, a partial hit by
    its own measure. It covers less than MIN_WORD_XOVER of the word, as a glyph chain in bold text
    does, so it is not tried from its edges."""
    gray = _page()
    xb = X0 + 3 * ADV + 16
    gray[BAR[0]:BAR[1], xb:_x1() + 2] = 255                 # the bar stops after four letters
    line = _line(xb=xb)
    assert (xb - X0) / (_x1() - X0) < scanned.MIN_WORD_XOVER            # wcov, as the guard reads
    assert _struck(gray, line, gap=0, monkeypatch=monkeypatch) == []


def test_thin_line_is_not_retried(monkeypatch):
    """A 4 px line (THIN_STROKE_RUN_PX) with two rows of letters above it: read from 2 px past it
    there is no ink above, and it is not tried from its edges."""
    gray = _page(top=TOP, bar=(142, 146))
    line = _line(run=4.0, y=143.6)
    assert _struck(gray, line, gap=0, monkeypatch=monkeypatch) == []


def test_thick_is_scaled_with_the_dpi(monkeypatch):
    """The same pixels at 400 dpi: a 6 px line there is a thin stroke (THIN_STROKE_RUN_PX scales
    to 8 px), so it is not tried from its edges. At 200 dpi it is thick and the word is found."""
    gray, line = _page(), _line()
    assert len(_struck(gray, line, gap=0, monkeypatch=monkeypatch)) == 1
    word = Word("abcdefghijkl", (X0 / W, 116 / H, _x1() / W, 175 / H), None)
    _tagged, struck = classify_lines([line], [word], gray, config=CF, dpi=400)
    assert struck == []


def test_the_retry_scales_with_the_dpi(monkeypatch):
    """At 400 dpi, letters twice the size and a 12 px bar, two rows of letters above it (the 2 px
    gap is not scaled): dropped at 2 px, found from the bar's edges."""
    gray, line = _page(k=2, top=142), _line(k=2)
    assert _struck(gray, line, k=2, gap=2, monkeypatch=monkeypatch) == []
    assert len(_struck(gray, line, k=2, gap=0, monkeypatch=monkeypatch)) == 1


@pytest.mark.parametrize("gap", [2, 3])
def test_gap_at_or_past_the_thin_one_turns_it_off(gap, monkeypatch):
    """INK_GAP_THICK_PX = INK_GAP_PX (2) gives the release's output; a larger one does too."""
    assert _struck(_page(), _line(), gap=gap, monkeypatch=monkeypatch) == []


def test_gap_past_the_thin_one_is_off_where_reading_further_out_would_pass(monkeypatch):
    """Five of twelve letters with ascenders, the rest topping out 2 px past the bar: from 2 px
    past it the ink above covers half the columns and none below, a table rule's reading; from 3
    px past it only the ascenders show, under INK_MIN_FRAC, so it is not one-sided and would pass.
    INK_GAP_THICK_PX = 3 must still give the release's output, so it is no retry at all."""
    gray = np.full((H, W), 255, np.uint8)
    for i in range(N):
        t = 120 if i in (0, 2, 5, 8, 10) else 139
        for s in (0, 5, 11):
            gray[t:151, X0 + i * ADV + s:X0 + i * ADV + s + 3] = 20
    gray[BAR[0]:BAR[1], X0 - 2:_x1() + 2] = 20
    gi = scanned._glyph_ink(scanned.ink_mask(gray))
    line, box = _line(), (X0, 116, _x1(), 175)
    assert scanned._one_sided(*scanned._ink_above_below(gi, line["ends_px"], 6.0, box, 2))
    assert not scanned._one_sided(*scanned._ink_above_below(gi, line["ends_px"], 6.0, box, 3))
    assert _struck(gray, line, gap=2, monkeypatch=monkeypatch) == []
    assert _struck(gray, line, gap=3, monkeypatch=monkeypatch) == []


def test_thick_retry_reads_at_the_stroke_row(monkeypatch):
    """A thick line 7 px below its 6 px bar, so its own band misses the bar: the stroke-row retry
    moves it 1 px up, onto the bar's last row, and the retry from the edges reads there, finding
    the letters on both sides. Read at the line, there is no ink below and it would be dropped."""
    gray = _page(top=140, bot=156)
    y = 152.6
    line = _line(y=y)
    assert scanned._stroke_row_dy(scanned.ink_mask(gray), line["ends_px"], 6.0, X0, _x1(), 4) == -1
    assert _struck(gray, line, gap=2, monkeypatch=monkeypatch) == []
    rec, = _struck(gray, line, gap=0, monkeypatch=monkeypatch)
    assert rec["off"] == round((y - 1 - 116) / 59 - 0.5, 2)       # judged at the row it moved to


def test_triple_border_stays_a_rule_with_glyph_ink_off(monkeypatch):
    """With ScanConfig(glyph_ink=False) the ink tests read all ink, where a rule's partner is ink
    beside the bar. The retry reads glyph ink only, so it is off: the triple border stays a rule."""
    gray = np.full((H, W), 255, np.uint8)
    xa, xb = X0 - 2, _x1() + 2
    gray[BAR[0]:BAR[1], xa:xb] = 20
    for x in range(xa, xb, 20):                            # a dashed rule above, a solid one below
        gray[140:142, x:x + 12] = 20
    gray[151:153, xa:xb] = 20
    monkeypatch.setattr(scanned, "INK_GAP_THICK_PX", 0)
    cfg = dataclasses.replace(CF, glyph_ink=False)
    assert classify_lines([_line()], [_word()], gray, config=cfg)[1] == []


def test_two_px_line_is_thin_at_low_dpi(monkeypatch):
    """At 72 dpi THIN_STROKE_RUN_PX scales to 1.44 px, but a stroke of 2 px or less is thin there,
    as in lines.strike_lines: it is not retried. Letters two rows above and below a 2 px bar."""
    gray = np.full((H, W), 255, np.uint8)
    for i in range(N):
        for s in (0, 5, 11):
            gray[143:151, X0 + i * ADV + s:X0 + i * ADV + s + 3] = 20
    gray[146:148, X0 - 2:_x1() + 2] = 20
    line = {**_line(y=146.5), "run_px": 2.0}
    monkeypatch.setattr(scanned, "INK_GAP_THICK_PX", 0)
    assert classify_lines([line], [_word()], gray, config=CF, dpi=72)[1] == []


def test_line_with_unmeasured_run_px_and_no_word_under_it():
    """A caller-built line whose run_px is None and that crosses no word is tagged as before: the
    thickness is compared only for a word that is retried."""
    line = {**_line(), "run_px": None}
    far = Word("far", (0.05, 0.85, 0.10, 0.90), None)
    tagged, struck = classify_lines([line], [far], _page(), config=CF)
    assert struck == [] and tagged[0]["label"] == "rule"


def test_ink_gap_px_is_read_when_called(monkeypatch):
    """INK_GAP_PX is read on each call, so patching it moves every reading. Five of twelve
    letters with ascenders, the rest topping out 2 px past the bar: the readings from 2 and 3 px
    past it differ."""
    gray = np.full((H, W), 255, np.uint8)
    for i in range(N):
        t = 120 if i in (0, 2, 5, 8, 10) else 139
        for s in (0, 5, 11):
            gray[t:151, X0 + i * ADV + s:X0 + i * ADV + s + 3] = 20
    gray[BAR[0]:BAR[1], X0 - 2:_x1() + 2] = 20
    ink = scanned._glyph_ink(scanned.ink_mask(gray))
    box, ends = (X0, 116, _x1(), 175), _line()["ends_px"]
    at2 = scanned._ink_above_below(ink, ends, 6.0, box, 2)
    at3 = scanned._ink_above_below(ink, ends, 6.0, box, 3)
    assert at2 != at3 and scanned._ink_above_below(ink, ends, 6.0, box) == at2
    monkeypatch.setattr(scanned, "INK_GAP_PX", 3)
    assert scanned._ink_above_below(ink, ends, 6.0, box) == at3


def test_bar_does_not_read_its_own_last_row(monkeypatch):
    """A short 6 px bar under two letters, nothing below it, its line measured 0.9 px above the
    bar's middle (strike_lines measures a little high). Read from its edges, the bar's last row
    falls below the band and, the bar being too short to be left out of glyph ink, reads as glyph
    ink below. Rows just past the band inked across the word are the stroke's own and are
    skipped, so the word is dropped, as in the release."""
    gray = np.full((H, W), 255, np.uint8)
    for i in range(2):
        for s in (0, 5, 11):
            gray[130:143, X0 + i * ADV + s:X0 + i * ADV + s + 3] = 20
    gray[143:149, X0 - 2:_x1(2) + 2] = 20
    line = {"bbox_px": (X0 - 2, 141, _x1(2) + 2, 148),
            "ends_px": ((X0 - 2.0, 144.6), (_x1(2) + 2.0, 144.6)),
            "len_in": (_x1(2) - X0 + 4) / 200, "angle_deg": 0.0, "fill": 1.0, "run_px": 6.0,
            "straightness": 1.0}
    word = Word("ab", (X0 / W, 116 / H, _x1(2) / W, 175 / H), None)
    gi = scanned._glyph_ink(scanned.ink_mask(gray))
    box = (X0, 116, _x1(2), 175)
    assert scanned._ink_above_below(gi, line["ends_px"], 6.0, box, 0)[1] == 1.0   # its own row
    assert scanned._ink_above_below(gi, line["ends_px"], 6.0, box, 0, skip_full=True)[1] == 0.0
    monkeypatch.setattr(scanned, "INK_GAP_THICK_PX", 0)
    assert classify_lines([line], [word], gray, config=CF)[1] == []


def test_line_longer_than_the_cap_is_not_retried(monkeypatch):
    """A thick bar across the page is the ruling of lined paper: the retry is only for a line up
    to INK_GAP_THICK_MAX_LEN_IN long. The test bar is 1.1 in; with the cap at 1 in it is not
    retried."""
    gray, line = _page(), _line()
    monkeypatch.setattr(scanned, "INK_GAP_THICK_PX", 0)
    assert len(classify_lines([line], [_word()], gray, config=CF)[1]) == 1
    monkeypatch.setattr(scanned, "INK_GAP_THICK_MAX_LEN_IN", 1.0)
    assert classify_lines([line], [_word()], gray, config=CF)[1] == []
