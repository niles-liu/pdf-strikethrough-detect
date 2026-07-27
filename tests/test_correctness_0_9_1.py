"""Regression tests for the v0.9.1 correctness patch (GitHub issue #4).

Scanned, tightly-ruled tables and forms (via Azure DI) over-flagged clean text: a solid full-width
table rule rode high spine-fill straight to tier 'auto', and StrikeNet saturates (prob -> 1.0) on
faint scans so it confirmed nearly everything. Two fixes:

  Fix B (geometry) — an in-band, long, solid line with ink on only ONE side is a table rule /
    underline, not a strike; reject it instead of accepting on fill alone.
  Fix A (verdict)  — on the calibrated-confidence (DI) path, downgrade a struck candidate that
    OCRs at/above ``max_clean_conf`` AND lacks corroborating strike geometry.

The originating corpus is private; these use synthetic rasters that reproduce the same geometry.
See CHANGELOG 0.9.1.
"""
import numpy as np
from pdf_strikethrough import cnn, detect
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import FILL_STRONG, ScanConfig, analyze_scanned_page

DPI = 200


def _blank(h=500, w=900):
    return np.full((h, w), 255, np.uint8)


def _glyphs(gray, x0, x1, y0, y1, step=8, wide=3):
    """Fill a band with dense vertical 'glyph' ticks (ink)."""
    for x in range(x0, x1, step):
        gray[y0:y1, x:x + wide] = 20


def test_fix_b_rejects_in_band_one_sided_table_rule():
    """A solid full-width rule sitting at the vertical centre of a tall (table-cell) word box,
    with glyph ink only ABOVE it, must NOT be classified as a strike. Before 0.9.1 the in-band
    long high-fill line bypassed the ink test and rode fill to a strong struck hit."""
    gray = _blank()
    # tall cell box y=150..250; text in the upper half; a solid rule across the middle (in-band).
    _glyphs(gray, 100, 700, 150, 194)          # ink ABOVE the rule only
    gray[198:201, 100:700] = 0                  # solid full-width rule at the box centre (~y=200)
    H, W = gray.shape
    word = Word("BERTH", (100 / W, 150 / H, 700 / W, 250 / H), confidence=0.99)
    _tagged, struck = analyze_scanned_page(gray, [word], config=ScanConfig.azure_di(), dpi=DPI)
    strong = [s for s in struck if not s.get("partial") and s.get("wcov", 0) >= 0.45]
    assert not strong, f"one-sided table rule was flagged as a strike: {struck}"


def test_fix_b_keeps_two_sided_through_strike():
    """Positive control: a strike through the middle of a full-height glyph block (ink on BOTH
    sides) is still detected — Fix B only rejects one-sided lines."""
    gray = _blank()
    _glyphs(gray, 100, 700, 160, 240)          # glyphs fill the whole box height
    gray[198:201, 100:700] = 0                  # strike through the x-height centre
    H, W = gray.shape
    word = Word("deleted", (100 / W, 160 / H, 700 / W, 240 / H), confidence=0.6)
    _tagged, struck = analyze_scanned_page(gray, [word], config=ScanConfig.azure_di(), dpi=DPI)
    assert any(not s.get("partial") for s in struck), f"through-strike lost: {struck}"


def _saturated_cnn(monkeypatch):
    """Reproduce the observed StrikeNet saturation: it confirms every candidate at prob 1.0."""
    monkeypatch.setattr(cnn, "score_crops", lambda crops, **kw: np.ones(len(crops)))


def test_fix_a_confidence_veto_drops_high_conf_uncorroborated(monkeypatch):
    """DI path + saturated CNN: a high-OCR-confidence word crossed only by a SOLID (non-shattered)
    bar has no corroborating strike geometry, so the confidence veto drops it; a genuinely damaged
    (low-confidence) word on the same kind of bar is kept."""
    _saturated_cnn(monkeypatch)
    gray = _blank()
    _glyphs(gray, 100, 700, 150, 250)          # row 1 glyphs (full height)
    gray[198:201, 100:700] = 0                  # solid bar -> high fill -> NOT corroborating
    _glyphs(gray, 100, 700, 350, 450)          # row 2 glyphs
    gray[398:401, 100:700] = 0                  # solid bar
    H, W = gray.shape
    kept = Word("kept", (100 / W, 150 / H, 700 / W, 250 / H), confidence=0.99)   # clean text
    gone = Word("gone", (100 / W, 350 / H, 700 / W, 450 / H), confidence=0.60)   # struck (damaged)
    recs = detect.detect_scanned_image(gray, [kept, gone], config=ScanConfig.azure_di(), dpi=DPI)
    by = {r["text"]: r for r in recs}
    assert "kept" in by and by["kept"]["final"] is False and by["kept"].get("conf_veto")
    assert "gone" in by and by["gone"]["final"] is True


def test_fix_a_veto_off_on_confidence_free_path(monkeypatch):
    """The same high-confidence word is NOT vetoed under ``confidence_free`` — weak-confidence
    engines (RapidOCR etc.) must not regress; detection there rests on geometry + the CNN."""
    _saturated_cnn(monkeypatch)
    gray = _blank()
    _glyphs(gray, 100, 700, 150, 250)
    gray[198:201, 100:700] = 0
    H, W = gray.shape
    kept = Word("kept", (100 / W, 150 / H, 700 / W, 250 / H), confidence=0.99)
    recs = detect.detect_scanned_image(gray, [kept], config=ScanConfig.confidence_free(), dpi=DPI)
    assert any(r["final"] for r in recs), f"confidence-free path regressed: {recs}"
    assert not any(r.get("conf_veto") for r in recs)


def test_fix_a_corroborated_high_conf_word_survives_veto(monkeypatch):
    """The veto is conf AND missing-geometry, never confidence alone: a high-confidence word that
    DOES carry corroborating strike geometry (a shattered through-strike, fill < FILL_STRONG, ink
    both sides) is spared, so a real strike on a clean OCR is never silently dropped."""
    _saturated_cnn(monkeypatch)
    gray = _blank()
    _glyphs(gray, 100, 700, 160, 240)
    # shattered strike: dashes leave gaps so spine fill lands below the solid-rule threshold
    for x in range(100, 700, 12):
        gray[198:201, x:x + 9] = 0
    H, W = gray.shape
    word = Word("struck", (100 / W, 160 / H, 700 / W, 240 / H), confidence=0.99)
    # a second, OCR-damaged word makes the page read as pen-edited (edit_prior), so the marginal-
    # fill strike survives the glyph-chain gate — matching a real page that carries edits.
    decoy = Word("edited", (100 / W, 350 / H, 300 / W, 400 / H), confidence=0.55)
    words = [word, decoy]
    _tagged, struck = analyze_scanned_page(gray, words, config=ScanConfig.azure_di(), dpi=DPI)
    assert struck, "shattered strike not detected at all"
    assert any(s["text"] == "struck" and s.get("geom_corroborated") for s in struck), (
        f"shattered strike should corroborate (fill must be < {FILL_STRONG}): "
        f"{[(s['text'], s.get('geom_corroborated')) for s in struck]}")
    recs = detect.detect_scanned_image(gray, words, config=ScanConfig.azure_di(), dpi=DPI)
    by = {r["text"]: r for r in recs}
    assert by.get("struck", {}).get("final"), f"corroborated high-conf strike was wrongly vetoed: {recs}"
