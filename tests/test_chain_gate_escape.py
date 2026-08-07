"""Regression tests for ``ScanConfig.rescue_clean_chains`` (issue #7 follow-up).

The chain gate in :func:`pdf_strikethrough.scanned.classify_lines` rejects a marginal-fill line
whose every word OCRs CLEANLY as a glyph chain — a real pen strike corrupts what it crosses, so
undamaged OCR on every hit is the glyph-chain signature. The gate has an escape: on a page that
looks pen-edited it spares the line anyway and lets the CNN decide.

On degraded ruled forms that escape is the dominant residual false-positive source, because
``edit_prior`` measures scan quality rather than edits and the CNN saturates. ``ruled_forms()``
therefore turns it off. These tests pin BOTH directions, so neither the default rescue nor the
ruled-forms suppression can be lost silently.

Synthetic rasters (the originating corpus is private) — same construction as
``test_correctness_0_9_1.py``.
"""
import numpy as np
from pdf_strikethrough import cnn, detect
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import ScanConfig, analyze_scanned_page

DPI = 200


def _blank(h=500, w=900):
    return np.full((h, w), 255, np.uint8)


def _glyphs(gray, x0, x1, y0, y1, step=8, wide=3):
    for x in range(x0, x1, step):
        gray[y0:y1, x:x + wide] = 20


def _saturated_cnn(monkeypatch):
    """StrikeNet pinned at prob 1.0 — the faint-scan failure mode the escape hands words to.
    Same helper as ``test_correctness_0_9_1``."""
    monkeypatch.setattr(cnn, "score_crops", lambda crops, **kw: np.ones(len(crops)))


def _clean_chain_page():
    """A marginal-fill line across a CLEAN-OCR word, on a page that reads as pen-edited.

    Dashes leave gaps so spine fill lands below FILL_STRONG (the chain gate's entry condition); the
    word OCRs at 0.99 (above max_clean_conf) so the gate calls it a chain; a damaged decoy elsewhere
    pushes edit_prior over page_edited_min so the escape is reachable.
    """
    gray = _blank()
    _glyphs(gray, 100, 700, 160, 240)
    for x in range(100, 700, 12):
        gray[198:201, x:x + 9] = 0
    H, W = gray.shape
    word = Word("caption", (100 / W, 160 / H, 700 / W, 240 / H), confidence=0.99)
    decoy = Word("edited", (100 / W, 350 / H, 300 / W, 400 / H), confidence=0.55)
    return gray, [word, decoy]


def test_escape_on_by_default_marks_the_clean_chain_twin():
    """Default config: the escape fires, so the clean-OCR line is stamped twin and survives layer 1
    for the CNN to judge. This is the recall safeguard — a real strike that happens to leave OCR
    undamaged must not be dropped on confidence alone."""
    gray, words = _clean_chain_page()
    _tagged, struck = analyze_scanned_page(gray, words, config=ScanConfig.azure_di(), dpi=DPI)
    caption = [s for s in struck if s["text"] == "caption"]
    assert caption, f"escape did not fire under the default config: {struck}"
    assert any(s.get("twin") for s in caption), (
        f"escape fired but did not stamp twin: {[(s['text'], s.get('twin')) for s in struck]}")


def test_ruled_forms_suppresses_the_escape(monkeypatch):
    """``ruled_forms()`` turns the escape off, so the same clean-OCR glyph chain is rejected at
    layer 1 and never reaches the saturated CNN. Guards the measured 42 -> 28 FP drop."""
    _saturated_cnn(monkeypatch)
    gray, words = _clean_chain_page()
    _tagged, struck = analyze_scanned_page(gray, words, config=ScanConfig.ruled_forms(), dpi=DPI)
    assert not [s for s in struck if s["text"] == "caption"], (
        f"clean-OCR glyph chain survived under ruled_forms(): {struck}")
    recs = detect.detect_scanned_image(gray, words, config=ScanConfig.ruled_forms(), dpi=DPI)
    assert not any(r["text"] == "caption" and r.get("final") for r in recs), (
        f"chain reached the CNN and was confirmed: {recs}")


def test_escape_is_independently_controllable():
    """The switch is its own field, NOT bundled into ``ruled_forms()``.

    Deliberate: the printed-rule veto and this escape are separate code paths that each cost recall
    on their own terms, and a caller who has validated one must not silently acquire the other by
    upgrading. The combination is opt-in per switch.
    """
    assert ScanConfig().rescue_clean_chains is True
    assert ScanConfig.ruled_forms().rescue_clean_chains is True     # preset does NOT bundle it
    assert ScanConfig.ruled_forms().veto_printed_rules is True
    assert ScanConfig(rescue_clean_chains=False).veto_printed_rules is False   # and not vice versa
    both = ScanConfig.ruled_forms(rescue_clean_chains=False)
    assert both.veto_printed_rules is True and both.rescue_clean_chains is False


def test_switch_is_inert_without_confidence_gating():
    """Known limitation, pinned so it stays documented: the whole chain gate lives behind
    ``confidence_gating``, so this switch does nothing on a confidence-free engine (RapidOCR) or
    when no word carries a confidence. Unlike the printed-rule veto, which is geometry-only and
    covers that path. Measured on the corpus: identical output either way."""
    gray, words = _clean_chain_page()
    blind = [Word(w.text, w.bbox, confidence=None) for w in words]
    for ws in (words, blind):
        cfg = ScanConfig(confidence_gating=False)
        a = analyze_scanned_page(gray, ws, config=cfg, dpi=DPI)[1]
        b = analyze_scanned_page(gray, ws, config=ScanConfig(confidence_gating=False,
                                                            rescue_clean_chains=False), dpi=DPI)[1]
        assert [s["text"] for s in a] == [s["text"] for s in b], (
            f"the switch had an effect with confidence gating off: {a} vs {b}")


def test_damaged_ocr_strike_is_untouched_by_the_switch():
    """The escape is only ever reached when EVERY hit OCRs clean. A real strike damages its word's
    OCR (the corpus strikes read 0.71-0.94), so it is spared by the gate's primary condition and
    must be detected identically with the escape on or off."""
    gray = _blank()
    _glyphs(gray, 100, 700, 160, 240)
    for x in range(100, 700, 12):
        gray[198:201, x:x + 9] = 0
    H, W = gray.shape
    words = [Word("struck", (100 / W, 160 / H, 700 / W, 240 / H), confidence=0.71),
             Word("edited", (100 / W, 350 / H, 300 / W, 400 / H), confidence=0.55)]
    on = analyze_scanned_page(gray, words, config=ScanConfig.azure_di(), dpi=DPI)[1]
    off = analyze_scanned_page(gray, words, config=ScanConfig(rescue_clean_chains=False),
                               dpi=DPI)[1]
    assert [s["text"] for s in on if s["text"] == "struck"], f"damaged-OCR strike lost: {on}"
    assert ([s["text"] for s in on if s["text"] == "struck"]
            == [s["text"] for s in off if s["text"] == "struck"]), (
        f"the switch changed a damaged-OCR strike: on={on} off={off}")
