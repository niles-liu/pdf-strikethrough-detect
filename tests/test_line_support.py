"""ScanConfig.line_support: a short word StrikeNet leaves unconfirmed is kept when one stroke runs
through it and strikes confirmed words on both sides of it.

A word of three characters or fewer is mostly padding at the CNN's input width, so a struck "of"
scores in the unsure band even where the words around it are confirmed. The rule must not keep a
live short word that a line only passes over: two words struck one by one can be stitched into a
single line across the live word between them, whose glyphs give it ink on both sides of the line
and, in letters such as "a" or "&", ink at the line's own height.

Two kinds of raster: hand-drawn glyph blocks with StrikeNet replaced by a lookup on each word's
box (the rule alone), and Helvetica rows rendered by PyMuPDF with the shipped StrikeNet.
"""
import json

import numpy as np
import pymupdf
import pytest
from pdf_strikethrough import __main__ as cli
from pdf_strikethrough import cnn, detect
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import ScanConfig

DPI = 200
H, W = 400, 900
Y0, Y1 = 180, 220            # every word box
STROKE = slice(199, 202)     # rows of a strike through the words' middle
CF = ScanConfig.confidence_free()


def _page(spec, confs=None):
    """A raster with glyph-like ink for each (text, x0, x1): bowls above and below the middle
    across the word and stems through it, as letters have. Returns (gray, words)."""
    gray = np.full((H, W), 255, np.uint8)
    for _, x0, x1 in spec:
        gray[Y0 + 6:Y0 + 10, x0:x1] = 20
        gray[Y1 - 10:Y1 - 6, x0:x1] = 20
        for x in range(x0, x1, 8):
            gray[Y0 + 6:Y1 - 6, x:x + 2] = 20
    confs = confs or [None] * len(spec)
    return gray, [Word(t, (a / W, Y0 / H, b / W, Y1 / H), c) for (t, a, b), c in zip(spec, confs)]


def _helvetica_row(texts, struck, size=10):
    """Helvetica words on one line at 200 dpi, each struck word through its own stroke at its
    x-height middle (as many documents strike: word by word, the spaces left clear)."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=80)
    page.insert_text((20, 45), " ".join(texts), fontsize=size, fontname="helv")
    found = page.get_text("words")
    assert [w[4] for w in found] == list(texts)
    for w, s in zip(found, struck):
        if s:
            y = w[1] + 0.62 * (w[3] - w[1])
            page.draw_line((w[0], y), (w[2], y), width=0.8)
    pix = page.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    pw, ph = page.rect.width, page.rect.height
    words = [Word(w[4], (w[0] / pw, w[1] / ph, w[2] / pw, w[3] / ph), None) for w in found]
    return gray, words


@pytest.fixture
def cnn_says(monkeypatch):
    """Replace StrikeNet with fixed probabilities, one per word in order."""
    def install(words, probs):
        by_box = {w.bbox: p for w, p in zip(words, probs)}
        monkeypatch.setattr(cnn, "word_crop_px", lambda gray, bbox, **kw: bbox)
        monkeypatch.setattr(cnn, "std_crop", lambda crop: crop)
        monkeypatch.setattr(cnn, "score_crops",
                            lambda crops, **kw: np.array([by_box[c] for c in crops]))
    return install


def _by_text(recs):
    return {r["text"]: r for r in recs}


def _struck_run(middle="of", x1=340, confs=None):
    spec = [("deleted", 100, 300), (middle, 312, x1), ("words", x1 + 12, x1 + 212)]
    gray, words = _page(spec, confs)
    gray[STROKE, 100:x1 + 212] = 0                  # one stroke through all three words
    return gray, words


def test_short_word_inside_confirmed_struck_run_is_kept(cnn_says):
    gray, words = _struck_run()
    cnn_says(words, [0.99, 0.30, 0.99])
    by = _by_text(detect.detect_scanned_image(gray, words, config=CF, dpi=DPI))
    assert by["of"]["final"] and by["of"]["verdict"] == "struck" and by["of"]["line_support"]
    assert by["of"]["cnn_prob"] == 0.3 and by["of"]["cnn_agrees"] is False
    assert by["deleted"]["final"] and by["words"]["final"]
    assert "line_support" not in by["deleted"]


def test_off_leaves_the_short_word_unconfirmed(cnn_says):
    gray, words = _struck_run()
    cnn_says(words, [0.99, 0.30, 0.99])
    off = ScanConfig(confidence_gating=False, line_support=False)
    by = _by_text(detect.detect_scanned_image(gray, words, config=off, dpi=DPI))
    assert by["of"]["final"] is False and by["of"]["verdict"] == "unsure"
    assert "line_support" not in by["of"]


def test_confirmed_neighbour_on_one_side_only_is_not_enough(cnn_says):
    gray, words = _struck_run()
    cnn_says(words, [0.99, 0.30, 0.20])             # "words" is left unconfirmed too
    by = _by_text(detect.detect_scanned_image(gray, words, config=CF, dpi=DPI))
    assert by["of"]["final"] is False and "line_support" not in by["of"]
    assert by["words"]["final"] is False             # five characters: never supported


def test_longer_word_is_left_to_the_cnn(cnn_says):
    gray, words = _struck_run(middle="within", x1=380)
    cnn_says(words, [0.99, 0.30, 0.99])
    by = _by_text(detect.detect_scanned_image(gray, words, config=CF, dpi=DPI))
    assert by["within"]["final"] is False


def test_live_word_under_a_stitched_line_is_not_kept(cnn_says, monkeypatch):
    """Two words struck one by one, a live "the" between them: its crossbar at the strokes' height
    lets the detector stitch one line across all three, and its glyphs give it ink on both sides,
    but the line's band holds no unbroken stretch of ink across it."""
    spec = [("deleted", 100, 300), ("the", 310, 350), ("words", 360, 560)]
    gray, words = _page(spec)
    gray[STROKE, 100:302] = 0
    gray[STROKE, 358:560] = 0
    gray[STROKE, 322:336] = 0                       # the crossbar of its "e"
    cnn_says(words, [0.99, 0.002, 0.99])
    by = _by_text(detect.detect_scanned_image(gray, words, config=CF, dpi=DPI))
    the = by["the"]
    assert the["geom_corroborated"] and the["line_idx"] == by["deleted"]["line_idx"]
    assert by["deleted"]["final"] and by["words"]["final"]
    assert the["final"] is False and "line_support" not in the
    # the stroke test is what turns it away
    monkeypatch.setattr(detect, "LINE_SUPPORT_MIN_RUN", 0.0)
    by = _by_text(detect.detect_scanned_image(gray, words, config=CF, dpi=DPI))
    assert by["the"]["final"] and by["the"]["line_support"]


@pytest.mark.parametrize("short", ["of", "a"])
def test_each_word_struck_on_its_own_keeps_the_short_one(short):
    """The shipped StrikeNet leaves a struck "of" or "a" between struck words unconfirmed
    (padding fills most of its crop); its own stroke and the confirmed neighbours keep it."""
    gray, words = _helvetica_row(["deleted", short, "words", "here"], [True, True, True, False])
    by = _by_text(detect.detect_scanned_image(gray, words, config=CF, dpi=DPI))
    assert by["deleted"]["final"] and by["words"]["final"]
    assert by[short]["final"] and by[short]["line_support"]
    assert not by.get("here", {}).get("final")


@pytest.mark.parametrize("short", ["a", "&"])
def test_live_short_word_between_words_struck_on_their_own_stays_live(short, monkeypatch):
    """A live "a" or "&" between two struck words: the per-word strokes are stitched into one line
    across it, and its glyph puts ink at the line's height, but no stroke of its own."""
    gray, words = _helvetica_row(["deleted", short, "words", "here"], [True, False, True, False])
    by = _by_text(detect.detect_scanned_image(gray, words, config=CF, dpi=DPI))
    assert by["deleted"]["final"] and by["words"]["final"]
    assert not by[short]["final"] and "line_support" not in by[short]
    # without the own-stroke test the line rule would keep it
    monkeypatch.setattr(detect, "_own_stroke", lambda *a, **kw: (1.0, 1.0))
    by = _by_text(detect.detect_scanned_image(gray, words, config=CF, dpi=DPI))
    assert by[short]["final"] and by[short]["line_support"]


def test_clean_ocr_on_the_calibrated_path_is_never_supported(cnn_says):
    """On the Azure DI calibration, clean text reads above max_clean_conf: a short word read at
    0.99 is live whatever the line says; the same word read as damaged is kept."""
    for conf, kept in ((0.99, False), (0.80, True)):
        gray, words = _struck_run(confs=[0.70, conf, 0.72])
        cnn_says(words, [0.99, 0.30, 0.99])
        by = _by_text(detect.detect_scanned_image(gray, words, config=ScanConfig.azure_di(),
                                                  dpi=DPI))
        assert by["deleted"]["final"] and by["words"]["final"]
        assert bool(by["of"]["final"]) is kept and bool(by["of"].get("line_support")) is kept


@pytest.mark.parametrize("line_y", [141.0, 145.5])
def test_the_own_stroke_is_read_at_the_row_its_geometry_came_from(monkeypatch, line_y):
    """A finder line 4 px above the stroke (141.0), or on it (145.5): the stroke-row retry gives
    "of" its geometry at the stroke's row, so its own stroke is read there too."""
    from pdf_strikethrough.lines import ink_mask
    from pdf_strikethrough.scanned import classify_lines
    gray, words, x = np.full((300, 900), 255, np.uint8), [], 200
    for txt in ("aaaa", "of", "bbbb"):
        for i in range(len(txt)):
            for s in (0, 5, 11):
                gray[140:158, x + i * 18 + s:x + i * 18 + s + 3] = 20
        words.append(Word(txt, (x / 900, 119 / 300, (x + (len(txt) - 1) * 18 + 14) / 900,
                                179 / 300), None))
        x += len(txt) * 18 + 18
    xb = int(words[-1].bbox[2] * 900) + 2
    gray[145:147, 198:xb] = 20
    ink = ink_mask(gray)
    line = {"bbox_px": (198, int(line_y) - 1, xb, int(line_y) + 1),
            "ends_px": ((198.0, line_y), (float(xb), line_y)), "len_in": (xb - 198) / 200,
            "angle_deg": 0.0, "fill": 0.95, "run_px": 2.0, "straightness": 1.0}
    tagged, struck = classify_lines([line], words, gray, ink=ink, config=CF, dpi=DPI)
    probs = iter({"aaaa": 0.99, "of": 0.5, "bbbb": 0.99}[h["text"]] for h in struck
                 if h["tier"] != "weak")
    monkeypatch.setattr(cnn, "score_crops", lambda crops: np.array([next(probs) for _ in crops]))
    recs = detect.apply_cnn_verdict(struck, gray, config=CF)
    detect._line_support(recs, tagged, ink, CF, DPI)
    by = _by_text(recs)
    assert by["of"]["geom_corroborated"] and by["of"]["final"] and by["of"]["line_support"]


def test_line_support_reaches_cli_json(tmp_path, monkeypatch):
    gray, words = _helvetica_row(["deleted", "of", "words", "here"], [True, True, True, False])
    from PIL import Image
    img = tmp_path / "row.png"
    Image.fromarray(gray).save(img, dpi=(DPI, DPI))
    monkeypatch.setattr(cli, "_check_ocr_available", lambda name: None)
    monkeypatch.setattr(cli, "_build_ocr", lambda name: lambda image: list(words))
    out = tmp_path / "out.json"
    assert cli.main(["detect", str(img), "--ocr", "rapidocr", "--json", str(out)]) == 0
    by = {w["text"]: w for w in json.loads(out.read_text(encoding="utf-8"))["words"]}
    assert by["of"]["line_support"] is True and "line_support" not in by["deleted"]


def test_presets():
    assert ScanConfig().line_support and ScanConfig.confidence_free().line_support
    assert ScanConfig.azure_di().line_support and ScanConfig.recall_first().line_support
    assert ScanConfig.ruled_forms().line_support
    assert not ScanConfig.precision_first().line_support   # reports CNN-confirmed strikes only
    assert ScanConfig.precision_first(line_support=True).line_support
