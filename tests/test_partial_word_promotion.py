"""A scanned word is promoted to a whole-word strike only when the stroke reaches both ends of its
glyph ink. Before, any stroke over 70% of a word's characters, as estimated from its OCR
box, struck the whole word, so the live rest of a partly struck long word was dropped too.

Synthetic rasters: each letter is three ink stems, so the only long horizontal ink at the stroke's
height is the stroke itself.
"""
import numpy as np
import pytest

from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import (ScanConfig, analyze_scanned_page, classify_lines,
                                       consolidate_struck)

H, W = 300, 900
TOP, BOT = 130, 170             # the letters' x-height band
MID = (TOP + BOT) // 2          # the stroke runs through its middle
ADV, LETTER = 18, 14            # letter advance and ink width, px
X0 = 200                        # where the word's ink starts
TEXT = "abcdefghijkl"           # 12 letters, 212 px of ink


def _page(text=TEXT, x0=X0):
    gray = np.full((H, W), 255, np.uint8)
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        lx = x0 + i * ADV
        for s in (0, 5, 11):
            gray[TOP:BOT, lx + s:lx + s + 3] = 20
    return gray


def _ink_end(n, x0=X0):
    """x where the n-th letter's ink ends (n letters from x0)."""
    return x0 + (n - 1) * ADV + LETTER


def _strike(gray, xa, xb, y=MID):
    gray[y - 1:y + 1, int(xa):int(xb)] = 20


def _line(xa, xb, y=MID):
    """The line record strike_lines would give for a stroke from xa to xb."""
    return {"bbox_px": (int(xa), y - 1, int(xb), y + 1),
            "ends_px": ((float(xa), float(y)), (float(xb), float(y))),
            "len_in": (xb - xa) / 200, "angle_deg": 0.0, "fill": 0.8, "run_px": 2.0,
            "straightness": 3.0}


def _word(text=TEXT, x0=X0, margin=0):
    x1 = x0 + (len(text) - 1) * ADV + LETTER
    return Word(text, ((x0 - margin) / W, (TOP - 12) / H, (x1 + margin) / W, (BOT + 8) / H),
                None)


def _record(gray, line, word):
    _tagged, struck = classify_lines([line], [word], gray, config=ScanConfig.confidence_free())
    assert len(struck) == 1, struck
    return struck[0]


@pytest.mark.parametrize("xb, span, partial", [
    (_ink_end(7), (0, 7), True),            # 58%: a partial strike, as before
    (_ink_end(10), (0, 10), True),          # 83%: over the old 70% bar, two letters stay live
    (_ink_end(12) - 6, (0, 12), False),     # 97%: a third of a letter short is the whole word
    (_ink_end(12), (0, 12), False),         # 100%
], ids=["58%", "83%", "97%", "100%"])
def test_long_word_struck_over_its_first_letters(xb, span, partial):
    gray = _page()
    _strike(gray, X0, xb)
    rec = _record(gray, _line(X0, xb), _word())
    assert (rec["char_span"], rec["partial"]) == (span, partial)
    assert rec["chars"] == TEXT[span[0]:span[1]]


def test_struck_suffix_keeps_its_partial_span():
    gray = _page()
    xa = X0 + 3 * ADV                                   # the last 9 of 12 letters: 75%
    _strike(gray, xa, _ink_end(12))
    rec = _record(gray, _line(xa, _ink_end(12)), _word())
    assert (rec["char_span"], rec["partial"]) == ((3, 12), True)


def test_stroke_overshooting_both_edges_is_the_whole_word():
    gray = _page()
    xa, xb = X0 - 25, _ink_end(12) + 25
    _strike(gray, xa, xb)
    rec = _record(gray, _line(xa, xb), _word())
    assert (rec["char_span"], rec["partial"]) == ((0, 12), False)


@pytest.mark.parametrize("letters, span, partial", [(10, (0, 10), True), (12, (0, 12), False)],
                         ids=["83%", "100%"])
def test_loose_box_measures_the_glyphs_not_the_margin(letters, span, partial):
    """An OCR box with 40 px of whitespace either side: the stroke is measured against the ink."""
    gray = _page()
    xb = _ink_end(letters)
    _strike(gray, X0, xb)
    rec = _record(gray, _line(X0, xb), _word(margin=40))
    assert (rec["char_span"], rec["partial"]) == (span, partial)


def test_speck_in_a_loose_box_margin_does_not_widen_the_word():
    gray = _page()
    _strike(gray, X0, _ink_end(12))
    gray[TOP + 5:TOP + 7, _ink_end(12) + 20:_ink_end(12) + 22] = 60     # a 2x2 speck
    rec = _record(gray, _line(X0, _ink_end(12)), _word(margin=30))
    assert (rec["char_span"], rec["partial"]) == ((0, 12), False)


def test_box_reaching_into_the_next_word_does_not_widen_the_word():
    gray = _page()
    nx = _ink_end(12) + 16                                              # the next word's ink
    for s in (0, 5, 11):
        gray[TOP:BOT, nx + s:nx + s + 3] = 20
    _strike(gray, X0, _ink_end(12))
    word = Word(TEXT, ((X0 - 3) / W, (TOP - 12) / H, (nx + 4) / W, (BOT + 8) / H), None)
    rec = _record(gray, _line(X0, _ink_end(12)), word)
    assert (rec["char_span"], rec["partial"]) == ((0, 12), False)


def test_skewed_phrase_underline_in_a_loose_box_does_not_widen_the_word():
    """An underline running past the word, half a degree off level, inside a loose box."""
    gray = _page()
    _strike(gray, X0, _ink_end(12))
    xu0, xu1 = X0 - 40, _ink_end(12) + 40
    for x in range(xu0, xu1):
        y = int(round(BOT + 3 + 0.0087 * (x - xu0)))
        gray[y:y + 2, x] = 20
    rec = _record(gray, _line(X0, _ink_end(12)), _word(margin=25))
    assert (rec["char_span"], rec["partial"]) == ((0, 12), False)


def test_dashed_stroke_is_one_stroke():
    """Solid over four letters, then a dash over each letter with a 2 px break between."""
    gray = _page()
    _strike(gray, X0, X0 + 4 * ADV)
    for k in range(4, 12):
        _strike(gray, X0 + k * ADV - 1, X0 + k * ADV + LETTER + 1)
    rec = _record(gray, _line(X0, _ink_end(12)), _word())
    assert (rec["char_span"], rec["partial"]) == ((0, 12), False)


def test_pen_break_before_the_last_letter_keeps_the_whole_word():
    gray = _page()
    _strike(gray, X0, _ink_end(11) + 1)
    _strike(gray, X0 + 11 * ADV - 1, _ink_end(12))                     # a 2 px break
    rec = _record(gray, _line(X0, _ink_end(12)), _word())
    assert (rec["char_span"], rec["partial"]) == ((0, 12), False)


def test_line_stitched_past_the_stroke_keeps_the_stroke_span():
    """The detector can stitch a stroke to a collinear glyph piece beyond its end. The span follows
    the stroke's own ink, not the stitched line's end."""
    gray = _page()
    _strike(gray, X0, _ink_end(7))
    gray[MID - 1:MID + 1, X0 + 9 * ADV + 3:X0 + 9 * ADV + 11] = 20      # a crossbar in letter 10
    rec = _record(gray, _line(X0, X0 + 9 * ADV + 11), _word())
    assert (rec["char_span"], rec["partial"]) == ((0, 7), True)


@pytest.mark.parametrize("letters, span, partial", [(10, (0, 10), True), (12, (0, 12), False)],
                         ids=["83%", "100%"])
def test_leaning_stroke_off_its_fitted_line(letters, span, partial):
    """A pen stroke rising 6 px over the word, while the line it was attributed through runs level
    at the word's middle: the stroke is up to 6 px off the line and is still measured in full."""
    gray = _page()
    xb = _ink_end(letters)
    for x in range(X0, xb):
        y = int(round(MID + 3 - 6 * (x - X0) / (_ink_end(12) - X0)))
        gray[y - 1:y + 1, x] = 20
    rec = _record(gray, _line(X0, xb), _word())
    assert (rec["char_span"], rec["partial"]) == (span, partial)


def test_phrase_box_with_one_word_struck():
    """A phrase-grained engine boxes two words as one: the struck word alone is the span."""
    text = "alpha beta"
    gray = _page(text)
    xa, xb = X0 + 6 * ADV - 2, _ink_end(10)
    _strike(gray, xa, xb)
    rec = _record(gray, _line(xa, xb), _word(text))
    assert (rec["char_span"], rec["partial"], rec["chars"]) == ((6, 10), True, "beta")


def test_phrase_box_with_each_word_struck_on_its_own():
    """Each word of a phrase box struck by its own stroke, with a break at the space: the stroke
    over the one-letter word is short, but it is the same stroke, so the whole box is struck."""
    text = "a cdefghijkl"
    gray = _page(text)
    _strike(gray, X0 - 3, X0 + LETTER + 2)
    _strike(gray, X0 + 2 * ADV - 2, _ink_end(12))
    rec = _record(gray, _line(X0 - 3, _ink_end(12)), _word(text, margin=4))
    assert (rec["char_span"], rec["partial"]) == ((0, 12), False)


def test_end_to_end_detected_stroke_over_most_of_a_word():
    """Through the stroke detector: 10 of 12 letters struck stays partial; all 12 is whole."""
    for letters, span, partial in ((10, (0, 10), True), (12, (0, 12), False)):
        gray = _page()
        _strike(gray, X0, _ink_end(letters))
        _tagged, struck = analyze_scanned_page(gray, [_word()], config=ScanConfig.confidence_free())
        assert [(r["char_span"], r["partial"]) for r in struck] == [(span, partial)]


def test_hits_without_glyph_measure_keep_the_coverage_rule():
    """consolidate_struck on hits built elsewhere (no glyph-ink measure): 70% coverage is still
    the whole word."""
    hit = {"text": "abcdefghij", "char_span": (0, 8), "strong": True, "bbox_frac": (0, 0, 1, 1),
           "off": 0.0, "wcov": 0.8, "line_idx": 0, "conf": None}
    tagged = [{"len_in": 1.0, "fill": 0.9, "struck": [hit]}]
    (rec,) = consolidate_struck([hit], tagged, use_conf=False)
    assert (rec["char_span"], rec["partial"]) == ((0, 10), False)
