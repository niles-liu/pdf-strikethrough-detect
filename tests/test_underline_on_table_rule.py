"""Underlined text sitting on a table rule, on the scanned path.

A line below a word's middle band is kept as a strike only when it has ink on both sides. Under an
underline that sits on a table rule, the "ink below" was the rule, so underlined numbers in ruled
tables were marked struck. The two-sided test now reads glyph ink: the ink without its horizontal
runs of ``GLYPH_INK_RULE_LEN_IN`` or more, found by ``lines.rule_mask``, so ruled text is judged
like unruled text. Synthetic rasters.
"""
import numpy as np
import pymupdf
import pytest
from scipy import ndimage

from pdf_strikethrough import scanned
from pdf_strikethrough.lines import ink_mask, line_kernel, rule_mask
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import ScanConfig, analyze_scanned_page, classify_lines

CF = ScanConfig.confidence_free()                      # the RapidOCR path, where this showed up
OLD = ScanConfig(confidence_gating=False, glyph_ink=False)


# ------------------------------------------------------------------------------------ rule_mask

def test_rule_mask_is_a_true_horizontal_opening():
    rng = np.random.default_rng(0)
    for _ in range(300):
        shape = (int(rng.integers(1, 25)), int(rng.integers(1, 80)))
        ink = rng.random(shape) < rng.uniform(0.4, 0.95)
        n = int(rng.integers(1, 90))                   # odd and even, and longer than the width
        want = ndimage.binary_opening(ink, structure=line_kernel(n, 0))
        assert np.array_equal(rule_mask(ink, n), want), (ink.shape, n)


def test_rule_mask_threshold_and_borders():
    ink = np.zeros((3, 100), bool)
    ink[0, :20] = ink[0, 80:] = True                   # exactly 20 long, touching each edge
    ink[1, 40:59] = True                               # 19 long
    ink[2, :] = True                                   # the full width
    got = rule_mask(ink, 20)
    assert np.array_equal(got[0], ink[0]) and not got[1].any() and got[2].all()


def test_rule_mask_tolerance_follows_a_stepped_rule():
    ink = np.zeros((40, 320), bool)
    for i in range(4):                                 # a thin rule stepping down a row every 30 px
        ink[10 + i, 30 * i:30 * (i + 1)] = True
    stair = ink.copy()
    ink[20, 150:300] = True                            # a long rule ...
    ink[0:40, 200:203] = True                          # ... crossed by a stem
    assert not rule_mask(stair, 50).any()              # no step is 50 long on its own
    assert np.array_equal(rule_mask(stair, 50, tolerance_px=1), stair)
    stem = slice(200, 203)
    lost0 = np.flatnonzero(rule_mask(ink, 50)[:, stem].any(axis=1))
    lost1 = np.flatnonzero(rule_mask(ink, 50, tolerance_px=1)[:, stem].any(axis=1))
    assert lost0.tolist() == [20] and lost1.tolist() == [19, 20, 21]
    assert not (rule_mask(ink, 50, tolerance_px=2) & ~ink).any()      # only ever ink pixels


def test_rule_mask_degenerate_inputs():
    ink = np.zeros((5, 30), np.uint8)
    ink[2, 3:25] = 255                                 # a 0/255 mask is read as bool
    assert not rule_mask(ink, 0).any() and not rule_mask(ink, -3).any()
    assert np.array_equal(rule_mask(ink, 1), ink > 0)
    assert not rule_mask(ink, 31).any()
    assert rule_mask(np.zeros((0, 7), bool), 3).shape == (0, 7)


# ------------------------------------------------------------------------ the scanned classifier

def _ruled_row(k=1.0, strike=False):
    """Three numbers in a ruled table row, each underlined (or struck), the row's rule 4 px under
    the underlines and inside the word boxes. `k` scales the geometry (1.5 for 300 dpi)."""
    s = lambda v: int(round(v * k))                    # noqa: E731
    gray = np.full((s(400), s(900)), 255, np.uint8)
    H, W = gray.shape
    words = []
    for i, x0 in enumerate((150, 350, 550)):
        x1 = x0 + 60
        for x in range(x0, x1, 6):
            gray[s(150):s(186), s(x):s(x) + s(3)] = 20  # digit stems
        if strike:
            gray[s(166):s(169), s(x0 - 2):s(x1 + 2)] = 0
        else:
            gray[s(189):s(191), s(x0 - 2):s(x1 + 2)] = 0
        words.append(Word(f"53{i}", (s(x0) / W, s(150) / H, s(x1) / W, s(198) / H), 0.99))
    gray[s(195):s(197), s(100):s(800)] = 0             # the table rule
    return gray, words


def test_underline_on_a_table_rule_is_not_a_strike():
    gray, words = _ruled_row()
    _tagged, struck = analyze_scanned_page(gray, words, config=CF, dpi=200)
    assert struck == []
    # measured on all ink, as before the fix (or given that as the glyph-ink mask), the rule is
    # the underline's "ink below" and every number is a candidate
    _tagged, old = analyze_scanned_page(gray, words, config=OLD, dpi=200)
    assert sorted(r["text"] for r in old) == ["530", "531", "532"]
    lines = scanned.strike_lines(gray, dpi=200)
    _tagged, given = classify_lines(lines, words, gray, config=CF, glyph_ink=ink_mask(gray))
    assert len(given) == 3


def test_strike_through_text_above_a_table_rule_is_kept():
    gray, words = _ruled_row(strike=True)
    _tagged, struck = analyze_scanned_page(gray, words, config=CF, dpi=200)
    assert sorted(r["text"] for r in struck) == ["530", "531", "532"]
    assert all(r["geom_corroborated"] for r in struck)  # still ink on both sides


def _lone_digit(digit, frac, ruled):
    """A short strike at `frac` of the box height through one Helvetica 12 pt digit, with or
    without a cell rule under it inside the word box."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=100)
    page.insert_text((100, 55), digit, fontname="helv", fontsize=12)
    (x0, y0, x1, y1, text, *_), = page.get_text("words")
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    H, W = gray.shape
    s = 200 / 72
    if ruled:
        gray[int(55 * s) + 3:int(55 * s) + 5, 20:W - 20] = 0
    yc = int(round((y0 + frac * (y1 - y0)) * s))
    gray[yc - 1:yc + 1, int(x0 * s) - 2:int(x1 * s) + 2] = 10
    return gray, [Word(text, (x0 * s / W, y0 * s / H, x1 * s / W, y1 * s / H), 0.99)]


@pytest.mark.parametrize("digit, frac", [("7", 0.5), ("7", 0.6), ("1", 0.5)])
def test_a_rule_under_a_lone_digit_does_not_change_the_verdict(digit, frac):
    """Ruled text is judged like unruled text. Measured on all ink, the rule was the ink below a
    short strike through a lone digit and decided the verdict."""
    def verdict(config, ruled):
        _tagged, struck = analyze_scanned_page(*_lone_digit(digit, frac, ruled), config=config,
                                               dpi=200)
        return bool(struck)
    assert verdict(CF, ruled=True) == verdict(CF, ruled=False)
    assert verdict(OLD, ruled=True) != verdict(OLD, ruled=False)


@pytest.mark.parametrize("config", [CF, ScanConfig.azure_di()], ids=["confidence_free", "azure_di"])
def test_double_rule_is_still_a_rule(config):
    """Two rules 6 px apart under one OCR token centred on the upper rule. Glyph ink sees nothing
    on either side of the upper rule, so it is judged on all ink, where the lower rule is the ink
    on its one side: a rule, as before glyph ink."""
    gray = np.full((400, 900), 255, np.uint8)
    gray[200:202, 100:800] = gray[206:208, 100:800] = 0
    H, W = gray.shape
    word = Word("====", (100 / W, 193 / H, 800 / W, 211 / H), 0.99)
    _tagged, struck = analyze_scanned_page(gray, [word], config=config, dpi=200)
    assert struck == []


def test_glyph_ink_is_on_in_every_preset():
    presets = (ScanConfig(), ScanConfig.azure_di(), ScanConfig.confidence_free(),
               ScanConfig.ruled_forms(), ScanConfig.recall_first(), ScanConfig.precision_first())
    assert all(p.glyph_ink for p in presets)


# ---------------------------------------------------------------------------------- dpi scaling

@pytest.mark.parametrize("dpi, n", [(200, 50), (300, 75), (100, 25)])
def test_glyph_ink_opening_scales_with_dpi(monkeypatch, dpi, n):
    seen = []
    monkeypatch.setattr(scanned, "rule_mask",
                        lambda ink, m, *a, **k: seen.append(m) or rule_mask(ink, m, *a, **k))
    gray, words = _ruled_row(k=dpi / 200)
    _tagged, struck = analyze_scanned_page(gray, words, config=CF, dpi=dpi)
    assert seen == [n] and struck == []
    ink = np.zeros((3, 200), bool)
    ink[0, :n] = ink[1, :n - 1] = True
    glyph = scanned._glyph_ink(ink, dpi)
    assert not glyph[0].any() and glyph[1].sum() == n - 1
