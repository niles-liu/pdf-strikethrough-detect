"""Edge cases where a fallback or a default disagreed with the rest of the package: scoring with
no OCR confidences, the low-level scanned functions' default calibration, a caller-built line
without a `fill`, annotation records passed through the CNN verdict, a page `words_by_page` does
not cover, the tight re-stitch at low dpi, and `markdown.strip_struck`. Synthetic rasters only."""
import inspect
import warnings

import numpy as np
import pytest

from pdf_strikethrough import cnn, detect, lines, markdown, native
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import ScanConfig, analyze_scanned_page, classify_lines
from _synth import _glyphs, _scanned_pdf

DPI = 200


def _page():
    return np.full((500, 900), 255, np.uint8)


def _struck(x1=700, solid=True):
    """Glyph ticks across a word box (y 160-240) with a strike through the middle."""
    gray = _page()
    _glyphs(gray, 100, x1, 160, 240)
    if solid:
        gray[198:201, 100:x1] = 0
    else:
        for x in range(100, x1, 10):
            gray[198:201, x:x + 8] = 0
    return gray


def _box(gray, x1=700, y0=160, y1=240):
    H, W = gray.shape
    return (100 / W, y0 / H, x1 / W, y1 / H)


def _summary(struck):
    return [(s["text"], s["score"], s["tier"]) for s in struck]


# ------------------------------------------------------------- scoring without OCR confidences

def test_words_without_confidences_score_as_confidence_free():
    """Under ScanConfig() a page whose words carry no confidence skips the chain gate, and its
    scores drop the confidence term too: before, that term read every word as clean, which cut
    each score to 60% (0.47 'review' where confidence_free() gives 0.78 'auto')."""
    gray = _struck()
    w = Word("deleted", _box(gray), confidence=None)
    _, di = analyze_scanned_page(gray, [w], config=ScanConfig(), dpi=DPI)
    _, free = analyze_scanned_page(gray, [w], config=ScanConfig.confidence_free(), dpi=DPI)
    assert di and _summary(di) == _summary(free)
    assert di[0]["tier"] == "auto"


def test_words_with_confidences_keep_the_confidence_term():
    gray = _struck()
    w = Word("deleted", _box(gray), confidence=0.6)
    _, di = analyze_scanned_page(gray, [w], config=ScanConfig(), dpi=DPI)
    _, free = analyze_scanned_page(gray, [w], config=ScanConfig.confidence_free(), dpi=DPI)
    assert di[0]["score"] > free[0]["score"]          # the damaged read adds evidence


# --------------------------------------------------------- the low-level functions' default config

@pytest.mark.parametrize("fn", [classify_lines, analyze_scanned_page, detect.detect_scanned_image])
def test_low_level_scanned_functions_default_to_confidence_free(fn):
    assert inspect.signature(fn).parameters["config"].default == ScanConfig.confidence_free()


def test_default_does_not_read_uncalibrated_confidences():
    """A struck word an engine reads at 0.99 (RapidOCR's confidences sit near 1.0 on struck and
    clean words alike) scores as on the high-level paths, not as Azure DI clean text."""
    gray = _struck(x1=260, solid=False)
    w = Word("deleted", _box(gray, x1=260), confidence=0.99)
    _, default = analyze_scanned_page(gray, [w], dpi=DPI)
    _, free = analyze_scanned_page(gray, [w], config=ScanConfig.confidence_free(), dpi=DPI)
    _, di = analyze_scanned_page(gray, [w], config=ScanConfig.azure_di(), dpi=DPI)
    assert _summary(default) == _summary(free)
    assert default[0]["tier"] == "auto" and di[0]["tier"] == "review"


# ------------------------------------------------------- a caller-built line without a `fill`

def _one_sided():
    """A long in-band bar with glyph ink above it only: a table rule if solid, else a candidate."""
    gray = _page()
    _glyphs(gray, 100, 700, 150, 194)
    gray[198:201, 100:700] = 0
    line = {"bbox_px": (100, 198, 700, 201), "ends_px": ((100.0, 199.5), (700.0, 199.5)),
            "len_in": 3.0, "run_px": 3.0}
    return gray, line, Word("TOTAL", _box(gray, y0=150, y1=250))


def test_line_without_fill_is_not_taken_for_a_solid_rule():
    """The table-rule test drops a long, solid, one-sided line. A missing `fill` read as 1.0
    there, so a caller-built line lost its words; it now passes as unmeasured, like a missing
    `fill` in the printed-rule veto, and scores with FILL_UNMEASURED."""
    gray, line, word = _one_sided()
    cfg = ScanConfig.confidence_free()
    _, missing = classify_lines([line], [word], gray, config=cfg, dpi=DPI)
    _, none = classify_lines([{**line, "fill": None}], [word], gray, config=cfg, dpi=DPI)
    _, typical = classify_lines([{**line, "fill": 0.80}], [word], gray, config=cfg, dpi=DPI)
    _, solid = classify_lines([{**line, "fill": 1.0}], [word], gray, config=cfg, dpi=DPI)
    assert missing and _summary(missing) == _summary(none) == _summary(typical)
    assert not solid                                   # a measured solid line is still a rule


def test_line_without_fill_skips_the_chain_gate():
    """The chain gate rejects a marginal-fill line whose words all read cleanly; a line with no
    `fill` was never put through it and still is not."""
    gray = _struck()
    line = {"bbox_px": (100, 198, 700, 201), "ends_px": ((100.0, 199.5), (700.0, 199.5)),
            "len_in": 3.0, "run_px": 3.0}
    words = [Word("clean", _box(gray), confidence=0.99)]
    tagged, struck = classify_lines([line], words, gray, config=ScanConfig.azure_di(), dpi=DPI)
    assert struck and tagged[0]["label"] == "strike"
    tagged, struck = classify_lines([{**line, "fill": 0.80}], words, gray,
                                    config=ScanConfig.azure_di(), dpi=DPI)
    assert not struck and tagged[0]["label"] == "chain"


# ------------------------------------------------------------ native records and the CNN verdict

@pytest.mark.parametrize("tier", ["vector", "flag", "annot"])
def test_cnn_verdict_never_drops_a_native_record(monkeypatch, tier):
    """apply_cnn_verdict exempted 'vector' and 'flag' but not 'annot', so an annotation record
    passed through it was dropped whenever StrikeNet read the crop as clean."""
    monkeypatch.setattr(cnn, "score_crops", lambda crops, **kw: np.zeros(len(crops)))
    gray = _struck()
    rec = {"text": "deleted", "chars": "deleted", "char_span": (0, 7), "partial": False,
           "bbox_frac": _box(gray), "tier": tier}
    (out,) = detect.apply_cnn_verdict([rec], gray, {"p_hi": 0.85, "p_lo": 0.15},
                                      config=ScanConfig.azure_di())
    assert out["final"] is True and out["verdict"] == "struck"


# ------------------------------------------------------------- a page words_by_page misses

def test_page_missing_from_words_by_page_names_words_by_page():
    """It still raises (as a partial di_result does), but no longer says no OCR backend was
    provided when the caller supplied words for other pages."""
    with pytest.raises(detect.OcrRequiredError, match=r"words_by_page covers pages \[0\]") as e:
        detect.detect_pdf(_scanned_pdf(), words_by_page={0: []})
    assert "page 1 " in str(e.value) and "no OCR backend" not in str(e.value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = detect.detect_pdf(_scanned_pdf(), words_by_page={0: []}, on_missing_ocr="skip")
    assert res["warnings"] and "words_by_page" in res["warnings"][0]


def test_page_with_no_word_source_at_all_keeps_its_message():
    with pytest.raises(detect.OcrRequiredError, match="no OCR backend was provided"):
        detect.detect_pdf(_scanned_pdf())


# ------------------------------------------------------------------ the tight re-stitch

def test_tight_restitch_stays_tighter_than_the_first_stitch_at_low_dpi():
    """A stroke in three pieces stitched to a dashed rule 1.5-2 px below its end fails as one
    line and is re-stitched tighter, which splits the stroke off. The re-stitch tolerance is not
    floored like the first stitch's: at 100 dpi a 2 px floor rejoins the two and the stroke is
    lost."""
    gray = np.full((220, 500), 255, np.uint8)
    gray[100:102, 50:80] = 0
    gray[101:104, 86:116] = 0
    gray[103:105, 122:152] = 0
    for x in range(160, 160 + 14 * 12, 14):
        gray[105:108, x:x + 6] = 0
    found = lines.strike_lines(gray, dpi=100)
    assert [ln["bbox_px"][0::2] for ln in found] == [(49, 151)]


# ------------------------------------------------------------------- markdown.strip_struck

def test_strip_struck_drops_overlaid_characters_like_strip_struck_markdown():
    """markdown.strip_struck was a copy of native.strip_struck_markdown from before U+0336
    overlays were read, so it kept the characters they strike."""
    s = "\N{COMBINING LONG STROKE OVERLAY}"
    md = f" keep ~~gone~~ un l{s}a{s}n\n\n\n\nnext "
    assert markdown.strip_struck(md) == native.strip_struck_markdown(md).strip()
    assert s not in markdown.strip_struck(md)
