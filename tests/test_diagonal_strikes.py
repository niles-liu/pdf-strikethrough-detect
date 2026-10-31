"""ScanConfig(max_angle_deg=...): the line finder can return strokes leaning up to 60 degrees, so
a diagonal strike through a short token can be found, with guards against a glyph's own diagonal.
At the default, 25, the output is the release's."""
import dataclasses
import math

import numpy as np
import pymupdf
import pytest

from pdf_strikethrough import cnn, detect, lines as L, scanned
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import ScanConfig, classify_lines

CF = ScanConfig.confidence_free()


def _render(draw, text=(("74", 60, 60),), size=14, w=300, h=120):
    """A page with `text` [(token, x, baseline)] and `draw(page)`; (gray, words, page size)."""
    doc = pymupdf.open()
    page = doc.new_page(width=w, height=h)
    for t, x, y in text:
        page.insert_text((x, y), t, fontname="helv", fontsize=size)
    draw(page)
    ws = page.get_text("words")
    words = [Word(x[4], (x[0] / w, x[1] / h, x[2] / w, x[3] / h), None) for x in ws]
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    return gray, words, ws


def _slash(rising=True, width=1.0, over=1.0):
    """A straight line from one corner of the first token's box to the other."""
    def draw(page):
        x0, y0, x1, y1 = page.get_text("words")[0][:4]
        top, bot = y0 + 2, y1 - 2
        a, b = ((x0 - over, bot), (x1 + over, top)) if rising else ((x0 - over, top), (x1 + over, bot))
        page.draw_line(a, b, width=width)
    return draw


@pytest.fixture
def struck_by_strikenet(monkeypatch):
    monkeypatch.setattr(cnn, "score_crops", lambda crops: np.full(len(crops), 0.99))


def test_config_default_and_validation():
    assert ScanConfig().max_angle_deg == 25.0 == L.MAX_ANGLE_DEG
    assert ScanConfig(max_angle_deg=60).max_angle_deg == 60.0
    for bad in (61, -1, 0, 24.9, True, "45", None):
        with pytest.raises(ValueError, match="max_angle_deg"):
            ScanConfig(max_angle_deg=bad)


def test_at_25_the_line_finder_is_unchanged():
    gray, _, _ = _render(_slash())
    assert L.strike_lines(gray, 200) == L.strike_lines(gray, 200, max_angle_deg=25)


@pytest.mark.parametrize("rising", [True, False])
def test_a_diagonal_is_found_only_above_25_and_measured_across(rising):
    gray, _, _ = _render(_slash(rising, width=1.5), text=(("10", 60, 60),), size=20)
    steep = lambda lines: [ln for ln in lines if ln["angle_deg"] > 25]  # noqa: E731
    assert steep(L.strike_lines(gray, 200)) == []
    # the slash: the steep line running across the whole token (the glyphs' curves give short
    # steep lines of their own, which the word-level guards below are for)
    got = [ln for ln in steep(L.strike_lines(gray, 200, max_angle_deg=60))
           if ln["bbox_px"][2] - ln["bbox_px"][0] > 40]
    assert len(got) == 1
    ln = got[0]
    assert 35 < ln["angle_deg"] < 60
    # 1.5 pt is about 4 px across the stroke; its vertical runs read 1 / cos(angle) as long
    assert ln["run_px"] <= 5.0


def test_steep_fragments_join_along_their_axis_and_an_x_does_not():
    a = ((100.0, 200.0), (130.0, 170.0))                 # 45 degrees, rising
    b = ((138.0, 162.0), (168.0, 132.0))                 # the same line, after an 8 px gap
    cross = ((100.0, 170.0), (130.0, 200.0))             # the other diagonal of an X
    groups = L._stitch_fragments([a, b, cross], steep=True)
    sizes = sorted(len(m) for _, m in groups)
    assert sizes == [1, 2]
    joined = [m for _, m in groups if len(m) == 2][0]
    assert a in joined and b in joined
    # judged by rise, as for near-horizontal lines, the gap's 8 px of rise would not join them
    assert sorted(len(m) for _, m in L._stitch_fragments([a, b], dy=6.0)) == [1, 1]
    # 2 px off the first one's axis, across it: still one stroke
    off = ((139.4, 163.4), (169.4, 133.4))
    assert sorted(len(m) for _, m in L._stitch_fragments([a, off], steep=True)) == [2]


def test_a_chevron_does_not_join():
    """Two strokes meeting at an apex (a V, a check mark) lean opposite ways."""
    up = ((100.0, 200.0), (130.0, 170.0))
    down = ((131.0, 170.0), (161.0, 200.0))
    assert sorted(len(m) for _, m in L._stitch_fragments([up, down], steep=True)) == [1, 1]


def test_a_level_line_and_a_diagonal_from_its_end_stay_two_lines():
    def draw(page):
        page.draw_line((40, 60), (120, 60), width=1.0)
        page.draw_line((120, 60), (150, 30), width=1.0)
    gray, _, _ = _render(draw, text=())
    got = sorted(round(ln["angle_deg"]) for ln in L.strike_lines(gray, 200, max_angle_deg=60))
    assert len(got) == 2 and got[0] < 5 and 40 < got[1] < 50


def _line(x0, y0, x1, y1, run=2.0):
    a = float(np.degrees(np.arctan2(abs(y1 - y0), x1 - x0)))
    return {"bbox_px": (x0, min(y0, y1), x1, max(y0, y1)), "ends_px": ((x0, y0), (x1, y1)),
            "len_in": float(np.hypot(x1 - x0, y1 - y0)) / 200, "angle_deg": round(a, 1),
            "fill": 0.95, "run_px": run, "straightness": 0.5}


def _token(glyph_rows):
    """A 40 x 40 px token at (100, 100): vertical stems on `glyph_rows`; its box (fractions)."""
    g = np.full((300, 300), 255, np.uint8)
    for r0, r1 in glyph_rows:
        for x in range(102, 138, 5):                     # stems on 24 of the 40 columns
            g[r0:r1, x:x + 3] = 20
    return g, (100 / 300, 95 / 300, 140 / 300, 145 / 300)


# Azure DI's defaults with diagonals on: the guards below apply only when asked for
DI = dataclasses.replace(ScanConfig.azure_di(), max_angle_deg=60)


@pytest.fixture
def scored(monkeypatch):
    """Every record scores 0.9 (auto), so these tests read the geometric guards, not the score
    of a toy token."""
    monkeypatch.setattr(scanned, "score_struck", lambda rec, tagged, use_conf=True: 0.9)


def test_a_steep_line_needs_glyph_ink_on_both_sides(scored):
    line = _line(98.0, 142.0, 142.0, 98.0)               # 45 degrees through the box's middle
    g, box = _token([(100, 140)])                        # stems above and below the line
    _, recs = classify_lines([line], [Word("74", box, 0.9)], g, config=DI, dpi=200)
    assert len(recs) == 1 and recs[0]["tier"] == "review"
    g, box = _token([(100, 112)])                        # ink above the middle only, like a 7
    _, recs = classify_lines([line], [Word("74", box, 0.9)], g, config=DI, dpi=200)
    assert recs == []


def test_a_steep_line_off_the_middle_band_is_dropped(scored):
    g, box = _token([(100, 140)])
    high = _line(98.0, 130.0, 142.0, 90.0)               # at the word's middle, 0.2 of the box high
    _, recs = classify_lines([high], [Word("74", box, 0.9)], g, config=DI, dpi=200)
    assert recs == []


def test_a_steep_line_gets_no_confidence_rescue(scored):
    line = _line(98.0, 142.0, 142.0, 98.0)
    g, box = _token([(100, 112)])
    _, recs = classify_lines([line], [Word("74", box, 0.9)], g, config=DI, dpi=200)
    assert recs == []
    flat = _line(98.0, 120.0, 142.0, 120.0)              # the same word, a short level line: rescued
    _, recs = classify_lines([flat], [Word("74", box, 0.9)], g, config=DI, dpi=200)
    assert len(recs) == 1


def test_a_steep_line_must_run_across_the_word(scored):
    g, box = _token([(100, 140)])
    short = _line(112.0, 130.0, 128.0, 110.0)            # inside the letters: a glyph's diagonal
    _, recs = classify_lines([short], [Word("74", box, 0.9)], g, config=DI, dpi=200)
    assert recs == []
    full = _line(98.0, 142.0, 142.0, 98.0)
    assert classify_lines([full], [Word("74", box, 0.9)], g, config=DI, dpi=200)[1]


def test_a_word_found_only_on_diagonals_is_never_auto(scored):
    line = _line(98.0, 142.0, 142.0, 98.0)
    level = _line(98.0, 120.0, 142.0, 120.0)
    g, box = _token([(100, 140)])
    w = [Word("74", box, 0.5)]
    _, recs = classify_lines([line], w, g, config=DI, dpi=200)
    assert recs and all(r["tier"] == "review" for r in recs)
    _, both = classify_lines([line, level], w, g, config=DI, dpi=200)
    _, only_level = classify_lines([level], w, g, config=DI, dpi=200)
    assert only_level[0]["tier"] == "auto"
    assert both[0]["tier"] == only_level[0]["tier"]      # a level line still decides the tier


def test_no_horizontal_retry_runs_under_a_steep_line(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a horizontal-line retry ran under a diagonal")

    real = scanned._ink_above_below

    def no_thick_reading(ink, ends, run, wpx, gap=None, skip_full=False):
        if gap is not None:                              # only the thick-edge retry passes a gap
            boom()
        return real(ink, ends, run, wpx, gap, skip_full)

    monkeypatch.setattr(scanned, "_stroke_row_dy", boom)
    monkeypatch.setattr(scanned, "_partner_stroke", boom)
    monkeypatch.setattr(scanned, "_edge_runs", boom)
    monkeypatch.setattr(scanned, "_ink_above_below", no_thick_reading)
    g, box = _token([(100, 112)])                        # fails the ink test: retries would run
    for run in (2.0, 6.0):                               # a thin line, and a thick one
        line = _line(98.0, 142.0, 142.0, 98.0, run=run)
        classify_lines([line], [Word("74", box, None)], g,
                       config=dataclasses.replace(CF, max_angle_deg=60), dpi=200)


def test_end_to_end_a_slashed_number(struck_by_strikenet):
    gray, words, _ = _render(_slash(width=1.2), text=(("74", 60, 60), ("12", 140, 60)), size=20)
    off = detect.detect_scanned_image(gray, words, config=CF, dpi=200)
    on = detect.detect_scanned_image(gray, words, config=ScanConfig(confidence_gating=False,
                                                                   max_angle_deg=60), dpi=200)
    assert not [r for r in off if r["text"] == "74"]
    hit = [r for r in on if r["text"] == "74"]
    assert len(hit) == 1 and hit[0]["final"] and hit[0]["tier"] == "review"
    assert not [r for r in on if r["text"] == "12"]


def _draw(g, p0, p1, w=3):
    (x0, y0), (x1, y1) = p0, p1
    for t in np.linspace(0, 1, int(max(abs(x1 - x0), abs(y1 - y0)) * 4) + 2):
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        g[int(round(y)) - w // 2:int(round(y)) + w // 2 + 1,
          int(round(x)) - w // 2:int(round(x)) + w // 2 + 1] = 0


# the line's own pixels are not glyph ink
@pytest.mark.parametrize("rows", [[(100, 112)], []], ids=["ink above only", "no glyph ink"])
def test_a_steep_lines_own_pixels_are_not_glyph_ink(scored, rows):
    g, box = _token(rows)
    _draw(g, (98, 142), (142, 98))                      # the line is IN the raster
    line = _line(98.0, 142.0, 142.0, 98.0, run=3.0)
    _, recs = classify_lines([line], [Word("74", box, 0.9)], g, config=DI, dpi=200)
    assert recs == []


def test_a_steep_line_through_glyphs_still_passes(scored):
    g, box = _token([(100, 140)])
    _draw(g, (98, 142), (142, 98))
    line = _line(98.0, 142.0, 142.0, 98.0, run=3.0)
    _, recs = classify_lines([line], [Word("74", box, 0.9)], g, config=DI, dpi=200)
    assert len(recs) == 1


# the near-horizontal lines at 60 are those at 25
def _text_page(dpi=150):
    words = ("the of and to in a is that for it as was with be by on not 74 47 1/2 AVOW WAX "
             "kayak Section Amendment").split()
    rng = np.random.default_rng(3)
    doc = pymupdf.open()
    pg = doc.new_page(width=420, height=200)
    y = 30
    while y < 180:
        x = 20
        while x < 360:
            w = words[rng.integers(len(words))]
            pg.insert_text((x, y), w, fontname="hebo" if rng.random() < 0.3 else "helv", fontsize=11)
            wl = pymupdf.get_text_length(w, fontname="helv", fontsize=11)
            if rng.random() < 0.2:
                pg.draw_line((x - 1, y - 3.6), (x + wl + 1, y - 3.6 + rng.uniform(-1, 1)),
                             width=float(rng.choice((0.6, 1.0, 2.0, 3.0))))
            x += wl + 6
        y += 18
    pix = pg.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def test_the_near_horizontal_lines_at_60_are_those_at_25():
    gray = _text_page()
    key = lambda ln: (ln["ends_px"], ln["run_px"], ln["fill"], ln["straightness"])  # noqa: E731
    at25 = sorted(map(key, L.strike_lines(gray, 150)))
    at60 = sorted(key(ln) for ln in L.strike_lines(gray, 150, max_angle_deg=60)
                  if ln["angle_deg"] <= 25)
    assert at60 == at25


# the steep stitch's gap is read along the axis
def _stroke(g, x0, y0, ang, length, w=3):
    a = math.radians(ang)
    for t in np.linspace(0, length, int(length * 4)):
        x, y = x0 + t * math.cos(a), y0 - t * math.sin(a)
        g[int(round(y)) - w // 2:int(round(y)) + w // 2 + 1,
          int(round(x)) - w // 2:int(round(x)) + w // 2 + 1] = 0


@pytest.mark.parametrize("gap", [44, 54])
def test_two_steep_strokes_a_level_gap_apart_stay_two_lines(gap):
    """At 0 degrees two 40 px strokes 30 px or more apart stay two lines (STITCH_GAP_PX is 26);
    along a 60 degree stroke the same gap must not join them, nor lose both."""
    g = np.full((420, 420), 255, np.uint8)
    a = math.radians(60)
    _stroke(g, 60, 360, 60, 40)
    _stroke(g, 60 + (40 + gap) * math.cos(a), 360 - (40 + gap) * math.sin(a), 60, 40)
    got = [ln for ln in L.strike_lines(g, 200, max_angle_deg=60) if ln["angle_deg"] > 25]
    assert len(got) == 2 and all(ln["len_in"] * 200 < 50 for ln in got)


# the limit is validated where it is taken
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 0, 24.9, 61, 90, True, "45"])
def test_strike_lines_rejects_a_limit_outside_25_to_60(bad):
    with pytest.raises(ValueError, match="max_angle_deg"):
        L.strike_lines(np.full((60, 60), 255, np.uint8), 200, max_angle_deg=bad)


@pytest.mark.parametrize("bad", [0, 10, 24.9])
def test_scanconfig_rejects_a_limit_under_25(bad):
    with pytest.raises(ValueError, match="max_angle_deg"):
        ScanConfig(max_angle_deg=bad)


# at the default, a caller's steep line is read as the release read it
def test_the_default_config_reads_a_callers_steep_line_as_the_release_did():
    g, box = _token([(100, 112)])
    line = _line(98.0, 142.0, 142.0, 98.0)
    w = [Word("74", box, 0.5)]
    _, recs = classify_lines([line], w, g, config=ScanConfig.azure_di(), dpi=200)
    assert [(r["text"], r["tier"]) for r in recs] == [("74", "auto")]     # the release's record
    _, recs = classify_lines([line], w, g, config=DI, dpi=200)
    assert recs == []                                                       # the guards, when asked


def _bold_slashed_page(size=34, width=3.4):
    doc = pymupdf.open()
    pg = doc.new_page(width=420, height=140)
    for i, s in enumerate(("WAX", "74", "kayak")):
        pg.insert_text((20 + 120 * i, 80), s, fontname="hebo", fontsize=size)
    for x0, y0, x1, y1, *_ in pg.get_text("words"):
        pg.draw_line((x0, y1 - 4), (x1, y0 + 4), width=width)
    pix = pg.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


@pytest.mark.parametrize("size,width", [(34, 3.4), (24, 2.0), (18, 2.4)])
def test_steep_peeled_bands_stay_out_of_the_level_lines(size, width):
    """A thick slash through bold text peels into steep bands; none may join the level second pass."""
    gray = _bold_slashed_page(size, width)
    key = lambda ln: (ln["ends_px"], ln["run_px"], ln["fill"], ln["straightness"])  # noqa: E731
    at25 = sorted(map(key, L.strike_lines(gray, 200)))
    at60 = sorted(key(ln) for ln in L.strike_lines(gray, 200, max_angle_deg=60)
                  if ln["angle_deg"] <= 25)
    assert at60 == at25
