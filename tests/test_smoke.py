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


def test_detect_pdf_native_end_to_end():
    doc = _synthetic_native_pdf()
    res = st.detect_pdf(doc, native_method="both")
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
    import fitz
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
    import fitz
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
