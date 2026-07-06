"""Smoke tests: imports, model load, OCR-free geometry, and the scanned word-level path."""
import numpy as np
import pdf_strikethrough as st
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import ScanConfig


def test_model_loads():
    meta = st.get_model_meta()
    assert 0.0 < meta["p_lo"] < meta["p_hi"] < 1.0
    assert meta["runtime"] in ("onnx", "torch")


def _synthetic_struck_page(struck=True):
    """White page, a black word-height bar of 'glyphs', optionally a strike through the middle."""
    H, W = 400, 800
    gray = np.full((H, W), 255, np.uint8)
    for x in range(100, 500, 12):          # fake glyphs: vertical ticks, y=182..210, x=100..500
        gray[182:210, x:x + 4] = 20
    if struck:
        gray[194:197, 100:500] = 0         # thin horizontal strike through the middle band
    return gray


def test_strike_lines_finds_a_strike():
    found = st.strike_lines(_synthetic_struck_page(True), dpi=200)
    assert any(ln["len_in"] >= 1.0 and ln["angle_deg"] <= 5 for ln in found), found


def test_strike_lines_empty_on_clean_text():
    found = st.strike_lines(_synthetic_struck_page(False), dpi=200)
    assert not any(ln["len_in"] >= 1.0 for ln in found), found


def test_cnn_scores_struck_higher_than_clean():
    struck = st.std_crop(_synthetic_struck_page(True)[170:220, 90:510].astype("float32"))
    clean = st.std_crop(_synthetic_struck_page(False)[170:220, 90:510].astype("float32"))
    ps, pc = st.score_crops([struck, clean])
    assert ps > pc


def test_scanned_word_path_flags_struck_word():
    gray = _synthetic_struck_page(True)
    H, W = gray.shape
    # one Word covering the struck bar (x 100..500, y 182..210), in page fractions
    words = [Word("deleted", (100 / W, 182 / H, 500 / W, 210 / H), confidence=0.6)]
    recs = st.detect_scanned_image(gray, words, config=ScanConfig.confidence_free())
    assert recs and recs[0]["final"], recs


def test_scanned_word_path_clean_word_not_flagged():
    gray = _synthetic_struck_page(False)
    H, W = gray.shape
    words = [Word("kept", (100 / W, 182 / H, 500 / W, 210 / H), confidence=0.99)]
    recs = st.detect_scanned_image(gray, words, config=ScanConfig.confidence_free())
    assert not any(r.get("final") for r in recs), recs


def test_markdown_assembly_and_clean_text():
    from pdf_strikethrough import markdown as md
    # two words on one row; the second is fully struck
    items = [("keep", (0.1, 0.5, 0.2, 0.52), None),
             ("gone", (0.25, 0.5, 0.35, 0.52),
              {"final": True, "partial": False, "char_span": (0, 4)})]
    out = md.page_markdown(items)
    assert "~~gone~~" in out and "keep" in out
    assert md.strip_struck(out).strip() == "keep"
    passages = md.group_passages(items)
    assert len(passages) == 1 and passages[0]["text"] == "gone"


def test_partial_strike_wraps_only_struck_chars():
    from pdf_strikethrough import markdown as md
    rec = {"final": True, "partial": True, "char_span": (0, 5)}   # 'semi-' of 'semi-monthly'
    assert md.mark_word("semi-monthly", rec) == "~~semi-~~monthly"


def _synthetic_native_pdf():
    """Born-digital PDF: one line of text with a vector strike through 'deleted text'."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "keep this deleted text here", fontsize=12)
    words = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}
    r1, r2 = words["deleted"], words["text"]
    ymid = (r1.y0 + r1.y1) / 2
    page.draw_line(fitz.Point(r1.x0 - 1, ymid), fitz.Point(r2.x1 + 1, ymid), width=1.0)
    return doc


def test_native_vector_flag_and_both_on_synthetic_pdf():
    doc = _synthetic_native_pdf()
    assert sorted(r["text"] for r in st.native_page_strikes(doc[0], 0)) == ["deleted", "text"]
    assert sorted(r["text"] for r in st.native_flag_strikes(doc[0], 0)) == ["deleted", "text"]
    both = st.page_strikes(doc[0], 0, "both")
    assert sorted(r["text"] for r in both) == ["deleted", "text"]   # union dedups, no doubles
    doc.close()


def test_native_words_arg_equivalent_to_extracting():
    """F7: passing words= yields records identical to letting the detector extract them itself,
    for vector, flag, and both."""
    doc = _synthetic_native_pdf()
    page = doc[0]
    words = page.get_text("words")
    assert st.native_page_strikes(page, 0) == st.native_page_strikes(page, 0, words=words)
    assert st.native_flag_strikes(page, 0) == st.native_flag_strikes(page, 0, words=words)
    for method in ("vector", "flag", "both"):
        assert st.page_strikes(page, 0, method) == st.page_strikes(page, 0, method, words=words)
    doc.close()


def test_detect_pdf_extracts_words_once_per_native_page(monkeypatch):
    """F7: under native_method='both', a native page extracts get_text('words') once and threads
    it through the vector + flag detectors and the markdown match, instead of re-extracting per
    call (was up to four extractions/page). classify's own light-image-branch extraction is the
    only other one, so the page totals two — not four."""
    import pymupdf
    doc = _synthetic_native_pdf()
    calls = {"words": 0}
    orig = pymupdf.Page.get_text

    def counting(self, *a, **k):
        if (a and a[0] == "words") or k.get("option") == "words":
            calls["words"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(pymupdf.Page, "get_text", counting)
    res = st.detect_pdf(doc, method="both")
    assert res["n_struck_final"] == 2
    assert calls["words"] == 2      # classify (1) + one threaded per-page extraction (1)
    doc.close()


def test_detect_pdf_native_end_to_end():
    doc = _synthetic_native_pdf()
    res = st.detect_pdf(doc, method="both")
    assert res["page_sources"] == ["native"]
    assert res["n_struck_final"] == 2
    assert res["markdown"] == "keep this ~~deleted~~ ~~text~~ here"
    assert res["clean_text"] == "keep this here"
    assert [(p["text"], p["n_words"]) for p in res["passages"]] == [("deleted text", 2)]
    doc.close()


def test_strikethroughs_in_pdf_accepts_bytes_and_methods():
    doc = _synthetic_native_pdf()
    pdf_bytes = doc.tobytes()
    doc.close()
    for method in ("vector", "flag", "both"):
        recs = st.strikethroughs_in_pdf(pdf_bytes, method=method)
        assert sorted(r["text"] for r in recs) == ["deleted", "text"], method


def test_word_bbox_coerced_to_tuple():
    w = Word("x", [0.1, 0.2, 0.3, 0.4], 0.9)          # list bbox must not break dict keys
    assert isinstance(w.bbox, tuple)
    hash((w.bbox, w.text))                             # used as a dict key downstream


def test_version_matches_distribution():
    from importlib.metadata import version
    assert version("pdf-strikethrough-detect") == st.__version__


# --------------------------------------------------------------------- review-fix regressions

def test_float01_image_scores_sanely():
    """[0,1]-float images used to truncate to all-zeros and score everything 'struck'."""
    struck01 = _synthetic_struck_page(True).astype(np.float64) / 255.0
    clean01 = _synthetic_struck_page(False).astype(np.float64) / 255.0
    box = (100 / 800, 170 / 400, 500 / 800, 220 / 400)
    ps, pc = st.score_word(struck01, box), st.score_word(clean01, box)
    assert ps > 0.5 > pc, (ps, pc)


def test_pixel_coordinate_bbox_raises():
    gray = _synthetic_struck_page(True)
    try:
        st.score_word(gray, (100, 200, 500, 240))
        assert False, "expected ValueError for pixel-coordinate bbox"
    except ValueError:
        pass


def test_rgb_and_float_images_accepted():
    gray = _synthetic_struck_page(True)
    H, W = gray.shape
    words = [Word("gone", (100 / W, 182 / H, 500 / W, 210 / H), 0.6)]
    rgb = np.stack([gray] * 3, axis=-1)
    assert any(r["final"] for r in
               st.detect_scanned_image(rgb, words, config=ScanConfig.confidence_free()))
    assert any(r["final"] for r in
               st.detect_scanned_image(gray.astype(np.float64), words,
                                       config=ScanConfig.confidence_free()))


def test_strike_lines_json_serializable_and_positive_area():
    import json
    found = st.strike_lines(_synthetic_struck_page(True), dpi=200)
    json.dumps(found)                                   # np.float32 used to break this
    assert found and all((b["bbox_px"][2] - b["bbox_px"][0]) > 0
                         and (b["bbox_px"][3] - b["bbox_px"][1]) > 0 for b in found)


def test_dpi_scaling_still_detects():
    """Pixel tunables now rescale with dpi — a 400-dpi render must not lose the strike."""
    H, W = 800, 1600                                    # same page geometry at 2x resolution
    gray = np.full((H, W), 255, np.uint8)
    for x in range(200, 1000, 24):
        gray[364:420, x:x + 8] = 20
    gray[388:394, 200:1000] = 0                         # strike 2x thicker at 2x dpi
    found = st.strike_lines(gray, dpi=400)
    assert any(ln["len_in"] >= 1.5 and ln["angle_deg"] <= 5 for ln in found), found


def _strikeout_annot_doc(update=True, hidden=False):
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((50, 100), "keep deleted text here", fontsize=12)
    r = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}["deleted"]
    annot = page.add_strikeout_annot(r)
    if hidden:
        annot.set_flags(fitz.PDF_ANNOT_IS_HIDDEN)
    if update:
        annot.update()
    return doc, page


def test_acrobat_strikeout_annotation_is_detected():
    """Acrobat /StrikeOut annotations are caught only because PyMuPDF flows their appearance
    streams through get_drawings(); nothing else pins it, so an upstream change could silently
    remove the recall. Pin it (with and without annot.update()) until the explicit annotation
    pass lands."""
    for update in (True, False):
        doc, page = _strikeout_annot_doc(update=update)
        assert sorted(r["text"] for r in st.native_page_strikes(page, 0)) == ["deleted"], update
        assert sorted(r["text"] for r in st.native_flag_strikes(page, 0)) == ["deleted"], update
        doc.close()


def test_hidden_strikeout_annotation_is_not_detected():
    """A hidden /StrikeOut annotation paints no ink, so it must not be reported."""
    doc, page = _strikeout_annot_doc(update=True, hidden=True)
    assert st.native_page_strikes(page, 0) == []
    assert st.native_flag_strikes(page, 0) == []
    doc.close()


def test_low_dpi_still_detects():
    """72-dpi scans: floored run/stitch thresholds keep a thin 2-px strike (only upscaling was
    tested before, and the run gate had shrunk to '> 1.44 px', rejecting 2-px strokes)."""
    H, W = 144, 288
    gray = np.full((H, W), 255, np.uint8)
    for x in range(36, 180, 4):                         # faux glyph ticks
        gray[64:76, x:x + 2] = 20
    gray[69:71, 36:180] = 0                             # 2-px strike through the middle
    found = st.strike_lines(gray, dpi=72)
    assert any(ln["len_in"] >= 0.9 and ln["angle_deg"] <= 5 for ln in found), found


def test_to_gray_u8_rescales_16bit_scan():
    """16-bit scans used to saturate to all-white (clip to 255 -> zero detections); rescaled now."""
    a16 = np.full((20, 20), 65535, np.uint16)
    a16[8:12, :] = 0
    g = st.to_gray_u8(a16)
    assert g.dtype == np.uint8 and g[0, 0] == 255 and g[10, 0] == 0


def test_std_crop_handles_uint16_crop():
    """uint16 crops used to wrap mod-256 and score garbage; they are rescaled first."""
    struck = (_synthetic_struck_page(True)[170:220, 90:510].astype(np.uint16) * 257)
    clean = (_synthetic_struck_page(False)[170:220, 90:510].astype(np.uint16) * 257)
    ps, pc = st.score_crops([st.std_crop(struck), st.std_crop(clean)])
    assert ps > pc, (ps, pc)


def test_rotated_page_bbox_in_unit_range():
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 700), "keep gone here", fontsize=12)
    words = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}
    r = words["gone"]
    page.draw_line(fitz.Point(r.x0, (r.y0 + r.y1) / 2), fitz.Point(r.x1, (r.y0 + r.y1) / 2))
    page.set_rotation(90)
    rot = fitz.open("pdf", doc.tobytes())
    doc.close()
    recs = st.strikethroughs_in_pdf(rot)
    assert [x["text"] for x in recs] == ["gone"]
    assert all(0.0 <= v <= 1.0 for x in recs for v in x["bbox_frac"])
    res = st.detect_pdf(rot)
    assert "~~gone~~" in res["markdown"] and "gone" not in res["clean_text"]
    rot.close()


def test_sparse_native_page_not_dropped():
    """Pages with <5 words (signature pages etc.) used to be misrouted to blank/scanned."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "delete this now", fontsize=12)
    words = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}
    r = words["delete"]
    page.draw_line(fitz.Point(r.x0, (r.y0 + r.y1) / 2), fitz.Point(r.x1, (r.y0 + r.y1) / 2))
    res = st.detect_pdf(doc)
    assert res["page_sources"] == ["native"]
    assert res["n_struck_final"] == 1 and res["clean_text"] == "this now"
    doc.close()


def test_encrypted_pdf_raises_dedicated_error():
    import fitz
    doc = _synthetic_native_pdf()
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret")
    doc.close()
    for fn in (st.detect_pdf, st.strikethroughs_in_pdf):
        try:
            fn(data)
            assert False, "expected EncryptedPdfError"
        except st.EncryptedPdfError:
            pass


def test_model_dir_override_is_read_at_load_time(tmp_path):
    """PDF_STRIKETHROUGH_MODEL_DIR / set_model_dir take effect at load time — they used to be read
    once at import, so the documented 'set it then import' override silently no-op'd."""
    from pdf_strikethrough import cnn
    try:
        cnn.set_model_dir(str(tmp_path))              # empty dir: no model present
        try:
            cnn.get_model_meta()
            assert False, "expected FileNotFoundError pointing at the override dir"
        except FileNotFoundError as e:
            assert str(tmp_path) in str(e)
    finally:
        cnn.set_model_dir(None)                        # revert to the packaged model
    assert cnn.get_model_meta()["runtime"] in ("onnx", "torch")


def test_meta_geometry_mismatch_raises():
    """A retrained model shipping different crop/pad geometry must fail loudly, not silently
    preprocess off-distribution."""
    from pdf_strikethrough import cnn
    try:
        cnn._check_geometry({"crop_h": cnn.CROP_H + 1})
        assert False, "expected ValueError for mismatched crop_h"
    except ValueError:
        pass
    cnn._check_geometry({"crop_h": cnn.CROP_H, "crop_w": cnn.CROP_W,
                         "pad_x": cnn.PAD_X, "pad_y": cnn.PAD_Y})    # matching values: ok


def test_word_rejects_pixel_coordinate_bbox():
    """A Word given pixel coords (not [0,1] fractions) used to sail through and report all-clean;
    it now raises at construction."""
    try:
        Word("x", (100, 200, 500, 240), 0.9)
        assert False, "expected ValueError for a pixel-coordinate Word bbox"
    except ValueError:
        pass


def test_words_from_azure_di_requires_page_dims():
    """Missing/zero DI page dims used to fall back to 1.0 and silently detect nothing."""
    from pdf_strikethrough.ocr import words_from_azure_di
    page = {"words": [{"content": "x", "polygon": [1, 1, 3, 1, 3, 2, 1, 2], "confidence": 0.9}]}
    for bad in ({}, {"width": 0, "height": 11}, {"width": 8.5}):
        try:
            words_from_azure_di(dict(page, **bad))
            assert False, f"expected ValueError for dims {bad}"
        except ValueError:
            pass
    ok = words_from_azure_di(dict(page, width=8.5, height=11.0))
    assert ok and all(0 <= v <= 1 for v in ok[0].bbox)


def test_rapidocr_version_guard():
    """rapidocr < 3.2 predates the word_results shape this adapter reads."""
    from pdf_strikethrough.ocr import _require_rapidocr_3_2
    for bad in ("2.0.1", "3.1.9", "3"):
        try:
            _require_rapidocr_3_2(bad)
            assert False, f"expected RuntimeError for rapidocr {bad}"
        except RuntimeError:
            pass
    _require_rapidocr_3_2("3.2.0")                # ok
    _require_rapidocr_3_2("3.9.1")                # ok


def test_invisible_stroke_is_not_a_strike():
    """A white / opacity-0 line leaves no ink; the vector detector must not confirm it as a
    strike (it accepted paths on geometry alone before)."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=500, height=200)
    page.insert_text((50, 100), "keep deleted text here", fontsize=12)
    words = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}
    rd, rt = words["deleted"], words["text"]
    page.draw_line(fitz.Point(rd.x0, (rd.y0 + rd.y1) / 2),          # black, real strike
                   fitz.Point(rd.x1, (rd.y0 + rd.y1) / 2), width=1.0)
    page.draw_line(fitz.Point(rt.x0, (rt.y0 + rt.y1) / 2),          # white, invisible
                   fitz.Point(rt.x1, (rt.y0 + rt.y1) / 2), width=1.0, color=(1, 1, 1))
    assert sorted(r["text"] for r in st.native_page_strikes(page, 0)) == ["deleted"]
    doc.close()


def test_flag_grazing_overshoot_does_not_flag_neighbor():
    """A strike over one word overshooting a little into the next used to emit a spurious 1-char
    partial on the neighbor via the flag path (and 'both' kept it); the grazing guard drops it."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=500, height=200)
    page.insert_text((50, 100), "keep deleted text here", fontsize=12)
    words = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}
    rd, rt = words["deleted"], words["text"]
    ym = (rd.y0 + rd.y1) / 2
    page.draw_line(fitz.Point(rd.x0, ym),
                   fitz.Point(rt.x0 + 0.10 * (rt.x1 - rt.x0), ym), width=1.0)   # ~10% overshoot
    assert sorted(r["text"] for r in st.native_flag_strikes(page, 0)) == ["deleted"]
    assert sorted(r["text"] for r in st.page_strikes(page, 0, "both")) == ["deleted"]
    doc.close()


def test_sloped_strike_attributes_to_every_word():
    """A multi-word strike on a sloped (~7°) baseline: the stroke-y is interpolated at each
    word's x-midpoint, so every crossed word is attributed. A single global line-center (the old
    behavior) put the end words far out of band and dropped all but the middle one."""
    from pdf_strikethrough.scanned import classify_lines
    from pdf_strikethrough.ocr import Word
    pix_h, pix_w = 400, 800
    gray = np.full((pix_h, pix_w), 255, np.uint8)      # ink irrelevant: in-band hits skip the pixel test
    words = []
    for i in range(5):                                 # 5 words on a rising baseline
        cx, cy = 160 + i * 120, 150 + i * 15
        words.append(Word(f"w{i}", ((cx - 50) / pix_w, (cy - 14) / pix_h,
                                     (cx + 50) / pix_w, (cy + 14) / pix_h), 0.6))
    (x0, y0), (x1, y1) = (100.0, 144.0), (700.0, 216.0)   # stroke follows the same slope
    line = {"bbox_px": (int(x0), int(min(y0, y1)), int(x1), int(max(y0, y1))),
            "ends_px": ((x0, y0), (x1, y1)),
            "len_in": 3.0, "angle_deg": 6.8, "fill": 0.9, "run_px": 3.0}
    _tagged, struck = classify_lines([line], words, gray, config=ScanConfig.confidence_free())
    assert {h["text"] for h in struck} == {f"w{i}" for i in range(5)}, struck


def test_full_bleed_background_image_page_is_native():
    """A born-digital page with a full-bleed background image must stay native (image coverage is
    unioned, and visible text over real drawings beats the image) and still detect its strike."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 60, 84))
    pix.set_rect(pix.irect, (235, 235, 235))            # light-gray full-bleed background image
    page.insert_image(page.rect, pixmap=pix)
    page.insert_text((72, 100), "keep deleted here", fontsize=12)
    words = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}
    r = words["deleted"]
    page.draw_line(fitz.Point(r.x0, (r.y0 + r.y1) / 2), fitz.Point(r.x1, (r.y0 + r.y1) / 2))
    assert st.classify_page_source(page) == "native"
    res = st.detect_pdf(doc)
    assert res["page_sources"] == ["native"] and res["n_struck_final"] == 1
    doc.close()


def test_repeated_image_does_not_inflate_coverage():
    """Summing image bboxes made the same small image placed many times read as a full-page scan;
    the coarse-grid union keeps a text page with a repeated logo classified 'native'."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20))
    pix.set_rect(pix.irect, (200, 200, 200))
    for x in (50, 250, 450):                            # 3 copies of one small logo (~same area 3x)
        page.insert_image(fitz.Rect(x, 40, x + 120, 160), pixmap=pix)
    page.insert_text((72, 400), "plenty of real body text here", fontsize=12)
    assert st.classify_page_source(page) == "native"
    doc.close()


def test_partial_di_result_with_skip_does_not_raise():
    """A di_result covering fewer pages than the document used to abort with ValueError even under
    on_missing_ocr='skip'; it must skip the uncovered scanned page and keep the other results."""
    doc = _synthetic_native_pdf()
    pix = doc[0].get_pixmap(dpi=72)
    scan = doc.new_page(width=595, height=842)
    scan.insert_image(scan.rect, pixmap=pix)
    assert st.classify_page_source(doc[1]) == "scanned"
    di = {"pages": [{"width": 1, "height": 1, "words": []}]}     # covers page 0 only
    res = st.detect_pdf(doc, di_result=di, on_missing_ocr="skip")
    assert res["n_struck_final"] == 2 and res["warnings"]
    doc.close()


def test_cli_writes_non_cp1252_output_without_crashing(tmp_path, monkeypatch):
    """CLI file writes used the platform default (cp1252 on Windows) and crashed on any
    non-cp1252 char in the markdown / clean-text / JSON. All three opens are now utf-8."""
    import json
    from pdf_strikethrough import __main__ as cli
    nonlatin = "ﬁle café → π"                      # U+FB01, é, arrow, pi — not all in cp1252
    fake = {"source": None, "page_count": 1, "page_sources": ["native"], "n_struck_final": 1,
            "warnings": [], "markdown": f"~~{nonlatin}~~", "clean_text": nonlatin, "passages": [],
            "words": [{"page": 0, "text": nonlatin, "chars": nonlatin,
                       "char_span": (0, len(nonlatin)), "partial": False,
                       "bbox_frac": (0.1, 0.1, 0.2, 0.12), "tier": "vector",
                       "verdict": "struck", "final": True}]}
    monkeypatch.setattr(st, "detect_pdf", lambda *a, **k: fake)
    pdf = tmp_path / "x.pdf"
    doc = _synthetic_native_pdf(); pdf.write_bytes(doc.tobytes()); doc.close()
    md, ct, js = tmp_path / "o.md", tmp_path / "o.txt", tmp_path / "o.json"
    rc = cli.main(["detect", str(pdf), "--markdown", str(md),
                   "--clean-text", str(ct), "--json", str(js)])
    assert rc == 0
    assert md.read_text(encoding="utf-8") == f"~~{nonlatin}~~"
    assert ct.read_text(encoding="utf-8") == nonlatin
    assert json.loads(js.read_text(encoding="utf-8"))["words"][0]["text"] == nonlatin


def test_authenticated_encrypted_pdf_detects():
    """A doc encrypted then authenticated keeps needs_pass truthy but is_encrypted=False; the gate
    now lets it through, so the recover-and-retry workflow the error message recommends works."""
    import fitz
    doc = _synthetic_native_pdf()
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret")
    doc.close()
    enc = fitz.open("pdf", data)
    assert enc.needs_pass and enc.authenticate("secret") and not enc.is_encrypted
    res = st.detect_pdf(enc)
    assert res["n_struck_final"] == 2
    assert sorted(r["text"] for r in st.strikethroughs_in_pdf(enc)) == ["deleted", "text"]
    enc.close()


def test_missing_ocr_skip_keeps_native_results():
    """Mixed native+scanned without OCR used to abort and lose the native results."""
    doc = _synthetic_native_pdf()
    pix = doc[0].get_pixmap(dpi=72)                    # add a scanned (image-only) page
    scan = doc.new_page(width=595, height=842)
    scan.insert_image(scan.rect, pixmap=pix)
    assert st.classify_page_source(doc[1]) == "scanned"
    try:
        st.detect_pdf(doc)
        assert False, "expected OcrRequiredError with on_missing_ocr='raise'"
    except st.OcrRequiredError:
        pass
    res = st.detect_pdf(doc, on_missing_ocr="skip")
    assert res["n_struck_final"] == 2 and res["warnings"]
    doc.close()


def test_empty_word_text_filtered():
    gray = _synthetic_struck_page(True)
    H, W = gray.shape
    words = [Word("", (100 / W, 182 / H, 500 / W, 210 / H), 0.6),
             Word("   ", (0.7, 0.5, 0.8, 0.55), 0.6)]
    recs = st.detect_scanned_image(gray, words, config=ScanConfig.confidence_free())
    assert not recs                                     # nothing but blank text -> no records


def test_clean_text_immune_to_literal_tildes():
    from pdf_strikethrough import markdown as md
    items = [("x~~y", (0.10, 0.5, 0.18, 0.52), None),
             ("gone", (0.20, 0.5, 0.28, 0.52),
              {"final": True, "partial": False, "char_span": (0, 4)}),
             ("kept", (0.30, 0.5, 0.38, 0.52), None)]
    assert md.page_clean_text(items) == "x~~y kept"     # strip_struck() would corrupt this


# --------------------------------------------------------------------- v0.5.0 surface

def _mixed_native_scanned_pdf():
    """Page 0: native with a strike. Page 1: image-only (scanned)."""
    doc = _synthetic_native_pdf()
    scan = doc.new_page(width=595, height=842)
    scan.insert_image(scan.rect, pixmap=doc[0].get_pixmap(dpi=72))
    return doc


def test_strikethroughs_in_pdf_warns_on_scanned():
    """F1: a scanned page yields a silent [] for that page — warn so the caller knows to route it
    through detect_pdf(ocr=...)."""
    import warnings
    doc = _mixed_native_scanned_pdf()
    assert st.classify_page_source(doc[1]) == "scanned"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = st.strikethroughs_in_pdf(doc)
    assert sorted(r["text"] for r in out) == ["deleted", "text"]      # native page still detected
    assert rec and "scanned" in str(rec[0].message)
    doc.close()


def test_strikethroughs_in_pdf_no_warning_when_all_native():
    import warnings
    doc = _synthetic_native_pdf()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        st.strikethroughs_in_pdf(doc)
    assert not rec, [str(w.message) for w in rec]
    doc.close()


def _three_page_native_pdf():
    """Pages 0 and 2 carry a strike; page 1 is clean."""
    import fitz
    doc = fitz.open()
    for pno in range(3):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), "keep deleted text here", fontsize=12)
        if pno != 1:
            r = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}["deleted"]
            page.draw_line(fitz.Point(r.x0, (r.y0 + r.y1) / 2),
                           fitz.Point(r.x1, (r.y0 + r.y1) / 2))
    return doc


def test_detect_pdf_pages_subset_and_progress():
    """F5: pages= processes only the requested pages, progress= fires once per processed page,
    the result carries a `pages` key aligned to page_sources, and page_count stays the full doc."""
    doc = _three_page_native_pdf()
    seen = []
    res = st.detect_pdf(doc, pages=[0, 2], progress=lambda d, t, p: seen.append((d, t, p)))
    assert res["pages"] == [0, 2]
    assert res["page_sources"] == ["native", "native"]
    assert res["page_count"] == 3                       # full doc, not the subset size
    assert res["n_struck_final"] == 2                   # only the two struck pages processed
    assert {w["page"] for w in res["words"]} == {0, 2}
    assert seen == [(1, 2, 0), (2, 2, 2)]
    doc.close()


def test_detect_pdf_pages_negative_index_and_dedup():
    doc = _three_page_native_pdf()
    res = st.detect_pdf(doc, pages=[-1, -1, 2])         # -1 == 2; duplicates collapse
    assert res["pages"] == [2]
    doc.close()


def test_detect_pdf_pages_out_of_range_raises():
    doc = _three_page_native_pdf()
    for bad in ([9], [-9]):
        try:
            st.detect_pdf(doc, pages=bad)
            assert False, f"expected IndexError for pages={bad}"
        except IndexError:
            pass
    doc.close()


def test_detect_pdf_no_pages_key_when_full():
    """Backward compat: without pages=, the result has no `pages` key and page_sources spans all."""
    doc = _three_page_native_pdf()
    res = st.detect_pdf(doc)
    assert "pages" not in res
    assert len(res["page_sources"]) == 3
    doc.close()


def test_types_module_exports():
    """F2: the typed shapes are importable from the package and the types submodule."""
    from pdf_strikethrough import types as t
    assert st.StruckWord is t.StruckWord
    assert st.DetectResult is t.DetectResult and st.Passage is t.Passage
    # TypedDicts carry their documented keys in __annotations__
    assert {"page", "text", "tier", "final", "coverage", "cnn_prob"} <= set(t.StruckWord.__annotations__)
    assert {"page_count", "page_sources", "words", "pages"} <= set(t.DetectResult.__annotations__)


def test_py_typed_marker_shipped():
    import os
    import pdf_strikethrough
    marker = os.path.join(os.path.dirname(pdf_strikethrough.__file__), "py.typed")
    assert os.path.exists(marker)


# --------------------------------------------------------------------- CLI (F4)

def _write_three_page_pdf(tmp_path):
    doc = _three_page_native_pdf()
    p = tmp_path / "t.pdf"
    p.write_bytes(doc.tobytes())
    doc.close()
    return str(p)


def test_cli_version_flag():
    from pdf_strikethrough import __main__ as cli
    try:
        cli.main(["--version"])
        assert False, "expected SystemExit from --version"
    except SystemExit as e:
        assert e.code == 0


def test_cli_pages_and_fail_if_found(tmp_path):
    from pdf_strikethrough import __main__ as cli
    pdf = _write_three_page_pdf(tmp_path)
    # page 2 (1-based) is the clean middle page -> nothing found -> exit 0 even with --fail-if-found
    assert cli.main(["detect", pdf, "--pages", "2", "--fail-if-found"]) == 0
    # pages 1,3 both carry a strike -> --fail-if-found trips exit 3
    assert cli.main(["detect", pdf, "--pages", "1,3", "--fail-if-found"]) == 3
    # bad --pages spec -> usage error exit 1
    assert cli.main(["detect", pdf, "--pages", "0"]) == 1


def test_cli_json_has_schema_version_and_evidence(tmp_path):
    import json
    from pdf_strikethrough import __main__ as cli
    pdf = _write_three_page_pdf(tmp_path)
    out = tmp_path / "o.json"
    assert cli.main(["detect", pdf, "--json", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == cli.SCHEMA_VERSION
    assert payload["n_struck_final"] == 2 and "warnings" in payload
    w = payload["words"][0]
    assert "coverage" in w and w["tier"] == "vector"        # native evidence field present


def test_cli_missing_file_exits_1(tmp_path):
    from pdf_strikethrough import __main__ as cli
    assert cli.main(["detect", str(tmp_path / "nope.pdf")]) == 1


def test_confidence_free_ignores_confidence():
    """Under confidence_free(), identical geometry must score identically w/ and w/o conf."""
    gray = _synthetic_struck_page(True)
    H, W = gray.shape
    box = (100 / W, 182 / H, 500 / W, 210 / H)
    cfg = ScanConfig.confidence_free()
    a = st.detect_scanned_image(gray, [Word("gone", box, 0.55)], config=cfg)
    b = st.detect_scanned_image(gray, [Word("gone", box, None)], config=cfg)
    assert [r["score"] for r in a] == [r["score"] for r in b]
    assert [r["tier"] for r in a] == [r["tier"] for r in b]


# --------------------------------------------------------------------- F6: verdict consistency

def test_auto_record_verdict_not_struck_when_cnn_drops(monkeypatch):
    """F6: an 'auto' word the CNN votes down (final=False) must not keep verdict='struck' — that
    contradicts the ship decision. Report the CNN's read instead; a confirmed word stays 'struck'."""
    from pdf_strikethrough import cnn, detect
    gray = _synthetic_struck_page(True)
    H, W = gray.shape
    meta = {"p_hi": 0.85, "p_lo": 0.15}
    box = (100 / W, 182 / H, 500 / W, 210 / H)
    base = {"tier": "auto", "text": "gone", "chars": "gone", "char_span": (0, 4),
            "partial": False, "bbox_frac": box}

    monkeypatch.setattr(cnn, "score_crops", lambda crops: [0.05] * len(crops))   # CNN says clean
    dropped = detect.apply_cnn_verdict([dict(base)], gray, meta)[0]
    assert dropped["cnn_agrees"] is False and dropped["final"] is False
    assert dropped["verdict"] != "struck"                     # was misleadingly "struck" before

    monkeypatch.setattr(cnn, "score_crops", lambda crops: [0.99] * len(crops))   # CNN confirms
    kept = detect.apply_cnn_verdict([dict(base)], gray, meta)[0]
    assert kept["cnn_agrees"] is True and kept["final"] is True and kept["verdict"] == "struck"


def test_no_record_claims_struck_while_not_final():
    """Invariant across the scanned path: verdict=='struck' implies final is True."""
    for struck in (True, False):
        gray = _synthetic_struck_page(struck)
        H, W = gray.shape
        words = [Word("gone", (100 / W, 182 / H, 500 / W, 210 / H), 0.6)]
        recs = st.detect_scanned_image(gray, words, config=ScanConfig.confidence_free())
        for r in recs:
            if r.get("verdict") == "struck":
                assert r["final"], r


# --------------------------------------------------------------------- v0.6.0 surface

def test_detect_pdf_method_alias_deprecated_but_honored():
    """R-name: detect_pdf's native-page selector is now `method`; the old `native_method` still
    works but emits a DeprecationWarning, and passing both with different values raises."""
    import warnings
    doc = _synthetic_native_pdf()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = st.detect_pdf(doc, native_method="both")
    assert res["n_struck_final"] == 2
    assert any(issubclass(w.category, DeprecationWarning) for w in rec), [str(w.message) for w in rec]
    # method= is the new name and takes the same values
    assert st.detect_pdf(doc, method="vector")["n_struck_final"] == 2
    try:
        st.detect_pdf(doc, method="vector", native_method="flag")
        assert False, "expected ValueError when method / native_method disagree"
    except ValueError:
        pass
    doc.close()


def test_vector_records_carry_stroke_color_and_width():
    """R-forensics: native vector records report the dominant contributing stroke's color + width
    (RGB in [0,1], width in pt) — pen-color conventions are evidence in legal review."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "keep deleted here", fontsize=12)
    r = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}["deleted"]
    ym = (r.y0 + r.y1) / 2
    page.draw_line(fitz.Point(r.x0, ym), fitz.Point(r.x1, ym), width=1.5, color=(1, 0, 0))
    rec = st.native_page_strikes(page, 0)[0]
    assert rec["stroke_color"] == (1.0, 0.0, 0.0)
    assert rec["stroke_width"] == 1.5
    doc.close()


def test_vector_filled_bar_reports_fill_color():
    """A strike drawn as a thin FILLED bar reports the fill paint as stroke_color and the bar
    height as stroke_width."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "keep deleted here", fontsize=12)
    r = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}["deleted"]
    ym = (r.y0 + r.y1) / 2
    page.draw_rect(fitz.Rect(r.x0, ym - 1, r.x1, ym + 1), fill=(0, 0, 1), color=(0, 0, 1))
    rec = st.native_page_strikes(page, 0)[0]
    assert rec["stroke_color"] == (0.0, 0.0, 1.0) and rec["stroke_width"] > 0
    doc.close()


def test_annot_pass_detects_with_forensics():
    """R-annot: the explicit /StrikeOut annotation pass reports tier='annot' plus the annotation's
    author/date/color forensics."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((50, 100), "keep deleted text here", fontsize=12)
    r = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}["deleted"]
    a = page.add_strikeout_annot(r)
    a.set_info(title="J. Reviewer")
    a.set_colors(stroke=(1, 0, 0))
    a.update()
    recs = st.native_annot_strikes(page, 0)
    assert [x["text"] for x in recs] == ["deleted"]
    rec = recs[0]
    assert rec["tier"] == "annot" and rec["annot_author"] == "J. Reviewer"
    assert rec["annot_color"] == (1.0, 0.0, 0.0)
    doc.close()


def test_annot_pass_skips_hidden_annotation():
    """A hidden /StrikeOut annotation paints no ink, so the explicit pass must not report it."""
    doc, page = _strikeout_annot_doc(update=True, hidden=True)
    assert st.native_annot_strikes(page, 0) == []
    doc.close()


def test_page_strikes_annot_method_and_both_union():
    """method='annot' routes to the annotation pass; 'both' unions vector+flag+annot (deduped)."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((50, 100), "keep deleted text here", fontsize=12)
    r = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}["deleted"]
    page.add_strikeout_annot(r).update()
    assert [x["text"] for x in st.page_strikes(page, 0, "annot")] == ["deleted"]
    assert st.page_strikes(page, 0, "annot")[0]["tier"] == "annot"
    assert sorted(x["text"] for x in st.page_strikes(page, 0, "both")) == ["deleted"]  # no dupes
    # end-to-end through detect_pdf
    res = st.detect_pdf(doc, method="annot")
    assert res["n_struck_final"] == 1 and res["words"][0]["tier"] == "annot"
    doc.close()


def test_both_grafts_annotation_forensics_onto_covering_record():
    """A /StrikeOut annotation's appearance stream is also caught by the vector path, so under
    method='both' the vector record covers it. The union must GRAFT the annotation's author/color
    onto that record rather than drop it — otherwise the forensics are lost."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((50, 100), "keep deleted text here", fontsize=12)
    r = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}["deleted"]
    a = page.add_strikeout_annot(r)
    a.set_info(title="Counsel")
    a.update()
    recs = st.page_strikes(page, 0, "both")
    assert [x["text"] for x in recs] == ["deleted"]           # single record, not duplicated
    assert recs[0]["annot_author"] == "Counsel"               # forensics grafted on
    doc.close()


def test_page_strikes_rejects_unknown_method():
    doc = _synthetic_native_pdf()
    try:
        st.page_strikes(doc[0], 0, "nope")
        assert False, "expected ValueError for an unknown method"
    except ValueError:
        pass
    doc.close()


def test_render_overlay_boxes_struck_pages():
    """R-overlay: render_overlay returns one RGB image per struck page, boxing every final record."""
    doc = _three_page_native_pdf()          # pages 0 and 2 struck, page 1 clean
    pages = st.render_overlay(doc, dpi=100)
    assert [p["page"] for p in pages] == [0, 2]
    assert all(p["n_struck"] == 1 for p in pages)
    img = pages[0]["image"]
    assert img.mode == "RGB" and img.size[0] > 0 and img.size[1] > 0
    doc.close()


def test_render_overlay_empty_without_strikes():
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=300, height=120)
    page.insert_text((40, 60), "nothing struck here", fontsize=12)
    assert st.render_overlay(doc) == []
    doc.close()


def test_save_overlays_dir_and_prefix(tmp_path):
    """save_overlays writes overlay-p{n}.png into a directory, or {root}-p{n}{ext} for a filename."""
    doc = _synthetic_native_pdf()
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(doc.tobytes())
    doc.close()
    into_dir = st.save_overlays(str(pdf), str(tmp_path / "ov"), dpi=100)
    assert into_dir == [str(tmp_path / "ov" / "overlay-p0.png")]
    assert (tmp_path / "ov" / "overlay-p0.png").exists()
    as_prefix = st.save_overlays(str(pdf), str(tmp_path / "shot.png"), dpi=100)
    assert as_prefix == [str(tmp_path / "shot-p0.png")]
    assert (tmp_path / "shot-p0.png").exists()


def test_cli_overlay_writes_images(tmp_path):
    from pdf_strikethrough import __main__ as cli
    pdf = _write_three_page_pdf(tmp_path)
    outdir = tmp_path / "ov"
    assert cli.main(["detect", pdf, "--overlay", str(outdir), "--overlay-dpi", "100"]) == 0
    written = sorted(p.name for p in outdir.iterdir())
    assert written == ["overlay-p0.png", "overlay-p2.png"]   # the two struck pages


def test_cli_json_carries_forensic_evidence(tmp_path):
    """The CLI JSON now includes stroke_color/stroke_width (vector) and annot_* (annotation)."""
    import fitz
    import json
    from pdf_strikethrough import __main__ as cli
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((50, 100), "keep deleted here", fontsize=12)
    r = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}["deleted"]
    a = page.add_strikeout_annot(r)
    a.set_info(title="Counsel")
    a.update()
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(doc.tobytes())
    doc.close()
    out = tmp_path / "o.json"
    assert cli.main(["detect", str(pdf), "--method", "annot", "--json", str(out)]) == 0
    w = json.loads(out.read_text(encoding="utf-8"))["words"][0]
    assert w["tier"] == "annot" and w["annot_author"] == "Counsel"


def test_logger_has_null_handler_and_emits_debug(caplog):
    """R-log: the package attaches a NullHandler (silent by default) and logs pipeline
    diagnostics at DEBUG under the 'pdf_strikethrough' logger when a caller opts in."""
    import logging
    handlers = logging.getLogger("pdf_strikethrough").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)
    doc = _synthetic_native_pdf()
    with caplog.at_level(logging.DEBUG, logger="pdf_strikethrough"):
        st.detect_pdf(doc, method="vector")
    assert any("source=native" in r.message or "detect_pdf" in r.message for r in caplog.records)
    doc.close()
