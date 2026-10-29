"""The thick retry's stem test: on a line that wobbles at least ``INK_GAP_THICK_STEM_WOBBLE``, or
whose wobble is not measured, a word is tried again from the line's edges only when the line is at
least ``INK_GAP_THICK_STEM_RATIO`` times as thick as the median ink run along it in the rows the
retry adds. Under a printed 2-3 pt bar those rows hold the narrow stems of the struck letters.
Beside a pen stroke in handwriting they hold the stroke's own uneven edge and strokes of the same
pen, about as wide as the stroke, which the retry reads as glyph ink on both sides. A straighter
line, as a printed bar often is, is retried without the test.

Synthetic rasters as in test_thick_strike_ink_gap.py: letters of three stems under a 6 px bar.
"""
import numpy as np

from pdf_strikethrough import scanned
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import ScanConfig, classify_lines

CF = ScanConfig.confidence_free()
H, W = 300, 900
X0, ADV, N = 200, 18, 12
TOP, BOT = 140, 153
BAR = (143, 149)
LY = 145.6
WOBBLY = 2.0                                              # a pen stroke's straightness


def _x1(n=N):
    return X0 + (n - 1) * ADV + 14


def _page(pen=False, k=1):
    """N letters over rows TOP to BOT struck by a bar over rows BAR, at scale `k`: three 3 px
    stems a letter (print), or with `pen` two 6 px stems, as wide as the 6 px line (strokes of
    the same pen)."""
    gray = np.full((H * k, W * k), 255, np.uint8)
    stems, width = ((0, 10), 6) if pen else ((0, 5, 11), 3)
    for i in range(N):
        for s in stems:
            x = (X0 + i * ADV + s) * k
            gray[TOP * k:BOT * k, x:x + width * k] = 20
    gray[BAR[0] * k:BAR[1] * k, (X0 - 2) * k:(_x1() + 2) * k] = 20
    return gray


def _word():
    return Word("abcdefghijkl", (X0 / W, 116 / H, _x1() / W, 175 / H), None)


def _line(k=1, run=6.0, wobble=WOBBLY):
    xa, xb = (X0 - 2) * k, (_x1() + 2) * k
    return {"bbox_px": (xa, int(LY * k) - 3 * k, xb, int(LY * k) + 3 * k),
            "ends_px": ((float(xa), LY * k), (float(xb), LY * k)),
            "len_in": (xb - xa) / (200 * k), "angle_deg": 0.0, "fill": 0.95, "run_px": run * k,
            "straightness": wobble}


def _struck(gray, line=None, word=None):
    return classify_lines([line or _line()], [word or _word()], gray, config=CF, dpi=200)[1]


def _short(rows):
    """A 0.27 in bar over three letters' worth of columns, with ink beside it only in `rows`: for
    each row, (step, runs), the runs as (x offset, width) repeated every `step` px from the bar's
    start. Returns the page, the line, the word and its box in px."""
    xb = X0 + 2 * ADV + 16
    gray = np.full((H, W), 255, np.uint8)
    gray[BAR[0]:BAR[1], X0 - 2:xb] = 20
    for y, (step, runs) in rows.items():
        for x in range(X0 - 2, xb, step):
            for dx, w in runs:
                gray[y, x + dx:min(xb, x + dx + w)] = 20
    line = {**_line(), "bbox_px": (X0 - 2, int(LY) - 3, xb, int(LY) + 3),
            "ends_px": ((float(X0 - 2), LY), (float(xb), LY)), "len_in": (xb - X0 + 2) / 200}
    word = Word("abc", (X0 / W, 116 / H, (xb - 2) / W, 175 / H), None)
    return gray, line, word, (X0, 116, xb - 2, 175)


def _glyph_ink(gray):
    return scanned._glyph_ink(scanned.ink_mask(gray))


def test_bar_over_thin_stems_is_retried():
    """A 6 px bar over 3 px stems: twice their width, so the retry runs and finds the word."""
    assert scanned.INK_GAP_THICK_STEM_RATIO == 1.5 and scanned.INK_GAP_THICK_STEM_WOBBLE == 1.2
    rec, = _struck(_page())
    assert rec["char_span"] == (0, N)


def test_stroke_as_wide_as_the_strokes_beside_it_is_not_retried(monkeypatch):
    """The same 6 px line through 6 px stems (strokes of the same pen): the retry would find ink
    on both sides, but the line is no thicker than the runs it reads there, so the word stays
    dropped. With the stem test off it is retried."""
    gray = _page(pen=True)
    assert scanned._edge_runs(_glyph_ink(gray), _line()["ends_px"], 6.0,
                              (X0, 116, _x1(), 175), 0, 2) == 6.0
    assert _struck(gray) == []
    monkeypatch.setattr(scanned, "INK_GAP_THICK_STEM_RATIO", 0)
    assert len(_struck(gray)) == 1


def test_a_straight_line_is_retried_without_the_test(monkeypatch):
    """The pen-width case on a line that wobbles less than INK_GAP_THICK_STEM_WOBBLE, as a printed
    bar often does, skips the stem test and is retried. At the threshold, with the wobble not
    measured, or with the threshold at 0, the test applies and the word stays dropped."""
    gray = _page(pen=True)
    gate = scanned.INK_GAP_THICK_STEM_WOBBLE
    assert len(_struck(gray, _line(wobble=gate - 0.01))) == 1
    assert _struck(gray, _line(wobble=gate)) == []
    assert _struck(gray, _line(wobble=None)) == []
    monkeypatch.setattr(scanned, "INK_GAP_THICK_STEM_WOBBLE", 0)
    assert _struck(gray, _line(wobble=gate - 0.01)) == []


def test_too_few_runs_to_measure_is_not_retried(monkeypatch):
    """A short 6 px stroke over a three-letter box with no letters, each edge one long run of ink
    in the row just past it (a pen stroke's uneven edge): read from 2 px past it there is no ink,
    which a short line needs on both sides; from its edges each edge reads as glyph ink. Two runs
    are too few to measure stems by, so the word is not retried."""
    gray, line, word, box = _short({141: (900, ((8, 32),)), 149: (900, ((8, 32),))})
    assert line["len_in"] < scanned.INK_SHORT_LEN_IN
    assert scanned._edge_runs(_glyph_ink(gray), line["ends_px"], 6.0, box, 0, 2) is None
    assert _struck(gray, line, word) == []
    monkeypatch.setattr(scanned, "INK_GAP_THICK_STEM_RATIO", 0)
    assert len(_struck(gray, line, word)) == 1


def test_a_ragged_edge_row_over_thin_stems_is_still_a_printed_bar():
    """A scanned bar's first row past each edge ragged (6 px dashes over the stems), the next row
    the struck letters' 3 px stems: the median over both rows is a stem's width, so the word is
    retried."""
    gray = _page()
    for y in (141, 149):
        for x in range(X0 - 2, _x1() + 2, 12):
            gray[y, x:x + 6] = 20
    rec, = _struck(gray)
    assert rec["char_span"] == (0, N)


def test_the_stems_are_read_over_the_word_only(monkeypatch):
    """One 6 px line through two words: printed letters (3 px stems) and, after a gap, letters
    of 6 px stems. Each word is judged by the runs over its own columns: the printed word is
    retried, the other is not."""
    n, gap = 6, 30
    xa0 = X0 + n * ADV + gap                              # where the second word starts
    gray = np.full((H, W), 255, np.uint8)
    for i in range(n):
        for s in (0, 5, 11):
            gray[TOP:BOT, X0 + i * ADV + s:X0 + i * ADV + s + 3] = 20
        for s in (0, 10):
            gray[TOP:BOT, xa0 + i * ADV + s:xa0 + i * ADV + s + 6] = 20
    xb = xa0 + (n - 1) * ADV + 16
    gray[BAR[0]:BAR[1], X0 - 2:xb + 2] = 20
    line = {**_line(), "bbox_px": (X0 - 2, int(LY) - 3, xb + 2, int(LY) + 3),
            "ends_px": ((float(X0 - 2), LY), (float(xb + 2), LY)), "len_in": (xb + 4 - X0) / 200}
    words = [Word("abcdef", (X0 / W, 116 / H, (X0 + (n - 1) * ADV + 14) / W, 175 / H), None),
             Word("ghijkl", (xa0 / W, 116 / H, xb / W, 175 / H), None)]
    struck = classify_lines([line], words, gray, config=CF, dpi=200)[1]
    assert [r["text"] for r in struck] == ["abcdef"]
    monkeypatch.setattr(scanned, "INK_GAP_THICK_STEM_RATIO", 0)
    struck = classify_lines([line], words, gray, config=CF, dpi=200)[1]
    assert [r["text"] for r in struck] == ["abcdef", "ghijkl"]


def test_both_rows_the_retry_adds_are_read():
    """A short bar whose first row past each edge is ragged (8 px dashes) and whose second holds
    three letters' 3 px stems: over both rows the stems outnumber the dashes, so the word is
    retried; the first row alone would read as a pen stroke's edge."""
    stems = (ADV, ((2, 3), (7, 3), (13, 3)))
    gray, line, word, box = _short({141: (16, ((0, 8),)), 149: (16, ((0, 8),)),
                                    140: stems, 150: stems})
    gi = _glyph_ink(gray)
    assert 6.0 / scanned._edge_runs(gi, line["ends_px"], 6.0, box, 0, 1) < 1.5    # dashes only
    assert 6.0 / scanned._edge_runs(gi, line["ends_px"], 6.0, box, 0, 2) >= 1.5   # stems outnumber
    assert len(_struck(gray, line, word)) == 1


def test_the_rows_read_follow_the_retry_gap(monkeypatch):
    """The stem test reads the rows the retry adds, from INK_GAP_THICK_PX to INK_GAP_PX past the
    band: from 1 px past it, only the second row, which here holds 3 px stems while the first holds
    8 px dashes."""
    stems = (ADV, ((2, 3), (7, 3), (13, 3)))
    gray, line, word, box = _short({141: (10, ((0, 8),)), 149: (10, ((0, 8),)),
                                    140: stems, 150: stems})
    real = scanned._edge_runs
    gi = _glyph_ink(gray)
    assert real(gi, line["ends_px"], 6.0, box, 0, 1) == 8.0
    assert real(gi, line["ends_px"], 6.0, box, 1, 2) == 3.0
    seen = []
    monkeypatch.setattr(scanned, "_edge_runs", lambda *a: seen.append(a[4:]) or real(*a))
    for gap in (0, 1):
        monkeypatch.setattr(scanned, "INK_GAP_THICK_PX", gap)
        _struck(gray, line, word)
    assert seen == [(0, scanned.INK_GAP_PX), (1, scanned.INK_GAP_PX)]


def test_a_line_exactly_at_the_ratio_is_retried(monkeypatch):
    """The bar is twice its letters' stems: retried at a ratio of 2, not at one just over."""
    monkeypatch.setattr(scanned, "INK_GAP_THICK_STEM_RATIO", 2.0)
    assert len(_struck(_page())) == 1
    monkeypatch.setattr(scanned, "INK_GAP_THICK_STEM_RATIO", 2.01)
    assert _struck(_page()) == []


def test_ratio_is_the_same_at_400_dpi():
    """At 400 dpi stems and bar double together, so the ratio of the line's thickness to the runs
    beside it is the same: 2 for print, 1 for the pen."""
    for pen, want in ((False, 2.0), (True, 1.0)):
        for k in (1, 2):
            gray = _page(pen=pen, k=k)
            ink = scanned._glyph_ink(scanned.ink_mask(gray), 200 * k)
            runs = scanned._edge_runs(ink, _line(k)["ends_px"], 6.0 * k,
                                      (X0 * k, 116 * k, _x1() * k, 175 * k), 0, 2)
            assert 6.0 * k / runs == want


def test_edge_runs_reads_the_rows_between_the_gaps():
    """From 0 to 2 px past the band: the first two rows on each side, here stems of 2 and 3 px
    that give a median of 2; rows further out are not read, and from 2 px to 2 px none is."""
    ink = np.zeros((40, 60), bool)
    ink[18:23, 5:55] = True                       # the line's band: 5 px around y = 20
    ink[14:18, 10:12] = ink[14:18, 20:22] = True  # 2 px stems above, rows 14-17
    ink[23:27, 30:32] = True                      # below, 2 px and 3 px
    ink[23:27, 40:43] = True
    ink[10:12, 5:55] = True                       # a rule 6 rows out: not read
    assert scanned._edge_runs(ink, ((5, 20.0), (55, 20.0)), 5.0, (5, 10, 55, 30), 0, 2) == 2.0
    assert scanned._edge_runs(ink, ((5, 20.0), (55, 20.0)), 5.0, (5, 10, 55, 30), 2, 2) is None


def test_edge_runs_follows_a_sloped_line():
    """A 6 px bar rising 1 px in 10, a 3 px stem every 30 px in the two rows above and below it:
    the rows read follow the bar column by column, so they hold the stems. Read level from the
    middle, they would cross the bar."""
    ink = np.zeros((120, 260), bool)
    for x in range(20, 240):
        yc = 90 - (x - 20) / 10
        top, bot = int(np.floor(yc - 3)), int(np.floor(yc + 3))
        ink[top:bot + 1, x] = True
        if (x - 20) % 30 < 3:
            ink[top - 2:top, x] = ink[bot + 1:bot + 3, x] = True
    ends = ((20.0, 90.0), (240.0, 68.0))
    assert scanned._edge_runs(ink, ends, 6.0, (20, 0, 239, 120), 0, 2) == 3.0


def test_runs_end_at_the_word_and_at_each_row():
    """Ink in the last column the word shares with the line is a run, and a run ending one row
    and one starting the next are two: three runs here, enough to measure."""
    ink = np.zeros((40, 60), bool)
    ink[18:23, 5:55] = True                       # the band, as above: rows read 16, 15, 23, 24
    ink[16, 40] = True                            # first row above: the last column, 40
    ink[15, 5:9] = True                           # second row above: from the first, 5
    ink[23, 20:24] = True
    assert scanned._edge_runs(ink, ((5, 20.0), (55, 20.0)), 5.0, (5, 10, 40, 30), 0, 2) == 4.0


def test_off_is_the_retry_as_before(monkeypatch):
    """At 0 the stem test is off: the pen-width case is retried, and the printed case is
    unchanged."""
    monkeypatch.setattr(scanned, "INK_GAP_THICK_STEM_RATIO", 0)
    assert len(_struck(_page(pen=True))) == 1
    assert len(_struck(_page())) == 1
