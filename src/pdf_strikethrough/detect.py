"""High-level orchestration: route a document to the right path and return struck words,
struck-aware markdown, the surviving clean text, and grouped deletion passages.

  native page   -> exact vector detection (no OCR, no model)
  scanned page  -> geometry + OCR words -> layer-1 classify -> layer-2 CNN verdict

The CNN verdict layer mirrors the corpus-validated gate: 'auto' words are kept only when the
CNN confirms at p_hi (crops too small to score are kept — no counter-evidence); on 'review'
words (the geometry can't decide) the CNN is the decider.
"""
from __future__ import annotations

import logging
import time
import warnings as _warnings

import pymupdf

from . import cnn, markdown as _md, native
from .ocr import words_from_azure_di
from .scanned import ScanConfig, analyze_scanned_page

log = logging.getLogger(__name__)   # "pdf_strikethrough.detect"; DEBUG diagnostics, opt-in

IMG_COVER_SCANNED = 0.70    # raster images cover >= this frac of the page -> scanned
RENDER_DPI = 200
HIGH_DPI_CAP = 300          # scanned rasters above this are worked at RENDER_DPI (see _working_dpi)
MAX_RENDER_MPIX = 128       # hard per-page raster ceiling; a bigger page is downsampled to fit
# Standalone raster inputs detect_image_file (and the CLI) accept — a photo/scan/fax that never
# was a PDF. Multi-page TIFFs are one frame per page.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


class OcrRequiredError(ValueError):
    """A scanned page was hit but no OCR backend (or DI result) was provided."""


class EncryptedPdfError(ValueError):
    """The PDF is password-protected; decrypt or authenticate before processing."""


def _raise_if_encrypted(doc):
    """The single encryption gate for the whole package (open_pdf and detect_pdf both route
    through it). Gates on ``is_encrypted``, which a successful ``doc.authenticate(password)``
    resets to False — NOT ``needs_pass``, which stays True forever after authenticate() and so
    made the exact recover-and-retry workflow the error message recommends fail."""
    if getattr(doc, "is_encrypted", False):
        raise EncryptedPdfError(
            "PDF is password-protected; open it with pymupdf and call doc.authenticate(password) "
            "(or save a decrypted copy) before processing")


def _open_doc(source):
    """Open `source` (path, bytes, or already-open fitz document) as a fitz document, applying the
    encryption gate. Returns ``(doc, owned)`` — `owned` is True when we opened it and the caller
    must close it. On EncryptedPdfError a doc WE opened is closed before raising, so no handle
    leaks (which would lock the file on Windows)."""
    if hasattr(source, "page_count"):
        doc, owned = source, False
    elif isinstance(source, (bytes, bytearray)):
        doc, owned = pymupdf.open(stream=bytes(source), filetype="pdf"), True
    else:
        doc, owned = pymupdf.open(source), True
    try:
        _raise_if_encrypted(doc)
    except EncryptedPdfError:
        if owned:
            doc.close()
        raise
    return doc, owned


def _image_coverage(page, grid=64):
    """Fraction of the page covered by raster images, UNIONED on a coarse boolean grid. Summing
    per-image bbox areas over-counts overlaps and repeats — the same 30%-of-page image placed
    three times would read as 90% coverage and misroute a born-digital page to 'scanned'."""
    import numpy as np
    R = page.rect
    pw, ph = R.width or 1.0, R.height or 1.0
    if R.get_area() <= 0:
        return 0.0
    cells = np.zeros((grid, grid), dtype=bool)
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & R
        if r.is_empty:
            continue
        cx0 = max(0, min(grid, int((r.x0 - R.x0) / pw * grid)))
        cx1 = max(0, min(grid, int(np.ceil((r.x1 - R.x0) / pw * grid))))
        cy0 = max(0, min(grid, int((r.y0 - R.y0) / ph * grid)))
        cy1 = max(0, min(grid, int(np.ceil((r.y1 - R.y0) / ph * grid))))
        cells[cy0:cy1, cx0:cx1] = True
    return float(cells.mean())


def _text_visibility(page):
    """(has_visible_text, has_invisible_text) from the text render modes reported by
    get_texttrace(). Render mode 3 (and zero opacity) is invisible — the hallmark of a scanned
    page's OCR text overlay; modes 0/1/2/4/5/6 actually paint ink and mean born-digital text."""
    visible = invisible = False
    for span in page.get_texttrace():
        if not any(chr(c[0]).strip() for c in span.get("chars", ())):
            continue
        if span.get("type", 0) == 3 or span.get("opacity", 1.0) == 0:
            invisible = True
        else:
            visible = True
        if visible and invisible:
            break
    return visible, invisible


def classify_page_source(page, words=None):
    """'native' | 'scanned' | 'blank' for one fitz page.

    Heavy raster-image coverage alone is not a scan: a born-digital page can carry a full-bleed
    background image behind real, visible text. It is routed 'scanned' only when that coverage
    coincides with an invisible OCR text overlay (render mode 3) or with no visible text over
    real vector drawings. Away from heavy image coverage, ANY real text means 'native' — sparse
    pages (signatures, cover sheets) stay native.

    ``words`` optionally supplies this page's ``get_text("words")`` output so a caller that also
    detects on the page extracts it once (default None = extract here, only on the light-image
    branch that needs it)."""
    img_cov = _image_coverage(page)
    if img_cov >= IMG_COVER_SCANNED:
        visible, invisible = _text_visibility(page)
        has_drawings = bool(page.get_drawings())
        # born-digital = visible text painted over real vector content, with no invisible overlay
        if visible and has_drawings and not invisible:
            return "native"
        return "scanned"
    if words is None:
        words = page.get_text("words")
    if any(w[4].strip() for w in words):
        return "native"
    return "blank" if img_cov < 0.05 else "scanned"


def _working_dpi(width_in, height_in, requested_dpi):
    """Effective raster DPI for one page, honoring both resolution guards. Returns
    ``(dpi, note_or_None)``: `note` is a caller-facing warning string set only when the pixel
    budget forced a resolution below what was requested (the perf normalization is silent — it is
    accuracy-neutral). See HIGH_DPI_CAP / MAX_RENDER_MPIX."""
    dpi = float(requested_dpi)
    if dpi > HIGH_DPI_CAP:                          # accuracy-neutral, so no warning
        log.debug("normalizing %g dpi down to %d", dpi, RENDER_DPI)
        dpi = float(RENDER_DPI)
    area_in = max(float(width_in), 0.0) * max(float(height_in), 0.0)
    budget_px = MAX_RENDER_MPIX * 1_000_000
    note = None
    if area_in > 0 and area_in * dpi * dpi > budget_px:
        capped = (budget_px / area_in) ** 0.5
        note = (f"page is {width_in:.0f}x{height_in:.0f} in; rendering at {capped:.0f} dpi instead "
                f"of {int(requested_dpi)} to stay under the {MAX_RENDER_MPIX} Mpix raster budget "
                f"(see SECURITY.md)")
        dpi = capped
    return max(1.0, dpi), note


def _downsample_gray(gray, dpi):
    """Apply the resolution guards to an already-loaded raster (the image-file path, where a
    frame is decoded before we know its size). Downsamples the array to the working DPI and
    returns ``(gray, dpi, note_or_None)``; a no-op when the raster is already within budget."""
    import numpy as np
    from PIL import Image
    h, w = gray.shape
    dpi = float(dpi)
    eff, note = _working_dpi(w / dpi, h / dpi, dpi)
    if eff < dpi - 0.5:
        scale = eff / dpi
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        gray = np.asarray(Image.fromarray(gray).resize((nw, nh), Image.LANCZOS), dtype=np.uint8)
        dpi = eff
    return gray, dpi, note


def _render_gray(page, dpi=RENDER_DPI):
    import numpy as np
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def _render_rgb(page, dpi=RENDER_DPI):
    import numpy as np
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


def apply_cnn_verdict(struck, gray, meta=None):
    """Score every non-weak struck record with the CNN and set verdict/final in place.
    Returns the list of records that survive (tier != 'weak'), each with 'final' bool.
    Gate semantics (corpus-validated): auto words need CNN confirmation at p_hi — a scored
    non-confirmation drops them; an unscoreable crop offers no counter-evidence and is kept.
    Review words are decided by the CNN outright."""
    if meta is None:
        meta = cnn.get_model_meta()
    kept, crops, owner = [], [], []
    for h in struck:
        if h.get("tier") == "weak":
            continue
        crop = cnn.word_crop_px(gray, h["bbox_frac"])
        if crop is not None:
            crops.append(cnn.std_crop(crop))
            owner.append(h)
        kept.append(h)
    probs = cnn.score_crops(crops) if crops else []
    for h, p in zip(owner, probs):
        h["cnn_prob"] = round(float(p), 3)
    for h in kept:
        p = h.get("cnn_prob")
        if h.get("tier") in ("vector", "flag"):            # native paths are exact; no CNN needed
            h["verdict"], h["final"] = "struck", True
        elif h["tier"] == "auto":
            h["cnn_agrees"] = (p >= meta["p_hi"]) if p is not None else None
            h["final"] = h["cnn_agrees"] is not False
            # verdict must not contradict final: a CNN-dropped 'auto' word used to keep
            # verdict="struck" while final=False (misleading). Report the CNN's actual read on a
            # drop; kept words (CNN confirmed, or crop too small to score) stay "struck".
            h["verdict"] = "struck" if h["final"] else cnn.verdict_of(p, meta)
        else:                                              # review (incl. orphans)
            h["verdict"] = cnn.verdict_of(p, meta) if p is not None else "unsure"
            h["final"] = h["verdict"] == "struck"
    return kept


def detect_scanned_image(gray, words, config=ScanConfig(), meta=None, dpi=RENDER_DPI):
    """Struck-word records for a single scanned page image + its OCR words. `gray` is a HxW
    grayscale array (RGB / float inputs are coerced); `dpi` must be the resolution the image
    was rendered/scanned at. Runs layer-1 classification then the CNN verdict."""
    from .lines import to_gray_u8
    gray = to_gray_u8(gray)
    _tagged, struck = analyze_scanned_page(gray, words, config=config, dpi=dpi)
    return apply_cnn_verdict(struck, gray, meta)


def _image_frames(source):
    """Yield ``(gray_uint8, dpi_or_None)`` for each frame of a raster image (path/bytes/PIL image).
    Multi-page TIFFs yield one item per frame; single-image formats yield one. ``dpi`` is the
    x-resolution recorded in the image metadata, or None when it carries none."""
    import io

    import numpy as np
    from PIL import Image, ImageSequence

    if hasattr(source, "seek") and hasattr(source, "mode"):     # already a PIL image
        img = source
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(source)))
    else:
        img = Image.open(source)
    frames = []
    for frame in ImageSequence.Iterator(img):
        gray = np.asarray(frame.convert("L"), dtype=np.uint8)
        dpi = frame.info.get("dpi") or img.info.get("dpi")
        xdpi = None
        if dpi:
            try:
                xdpi = int(round(float(dpi[0]))) or None
            except (TypeError, ValueError, IndexError):
                xdpi = None
        frames.append((gray, xdpi))
    return frames


def detect_image_file(source, ocr=None, words=None, words_by_page=None, scan_config=None,
                      dpi=None, meta=None, include_markdown=True):
    """Detect strikethroughs in a standalone raster image (``.png/.jpg/.tiff``, incl. multi-page
    TIFF) — a photo/scan/fax that never was a PDF.

    Every frame is a scanned page, so it needs OCR words: pass an `ocr` backend (run per frame),
    a `words` list (a single-frame image), or `words_by_page` (``{0-based frame: list[Word]}`` —
    e.g. ``words_from_textract(resp)``). DPI drives the geometry tunables: an explicit `dpi=`
    wins; otherwise it's read from the image metadata, falling back to 200.

    Returns the same dict shape as :func:`detect_pdf` (``page_sources`` all ``"scanned"``); there
    is no `pages` subset and no native path. A frame with no word source raises
    ``OcrRequiredError``.
    """
    import numpy as np

    from .lines import to_gray_u8
    frames = _image_frames(source)
    wbp = _normalize_words_by_page(words_by_page)
    if words is not None and len(frames) > 1 and wbp is None:
        raise ValueError("words= covers a single-frame image; for a multi-page TIFF pass an ocr "
                         "backend or words_by_page={frame: [...]}")
    if scan_config is None:
        # An image has no Azure-DI calibration source; every realistic word source here (rapidocr/
        # tesseract via ocr=, Textract/DocAI via words_by_page, a raw words= list) carries
        # confidences the classifier isn't calibrated to — so default to confidence-free.
        scan_config = ScanConfig.confidence_free()
    src_name = None if isinstance(source, (bytes, bytearray)) else str(source)

    all_words, warns, page_md, page_clean, passages = [], [], [], [], []
    for pno, (gray, meta_dpi) in enumerate(frames):
        use_dpi = dpi if dpi is not None else (meta_dpi or RENDER_DPI)
        gray, use_dpi, note = _downsample_gray(gray, use_dpi)
        if note:
            _warnings.warn(note, stacklevel=2)
            warns.append(note)
        if wbp is not None and pno in wbp:
            page_words = wbp[pno]
        elif words is not None and len(frames) == 1:
            page_words = words
        elif ocr is not None:
            t0 = time.perf_counter()
            page_words = ocr(np.stack([gray] * 3, axis=-1))
            log.debug("frame %d: OCR -> %d word(s) in %.0f ms",
                      pno, len(page_words), (time.perf_counter() - t0) * 1e3)
        else:
            raise OcrRequiredError(
                f"image frame {pno} has no OCR words; pass ocr=rapidocr_backend(), a words= list "
                f"(single-frame), or words_by_page={{{pno}: [...]}}")
        if meta is None:
            meta = cnn.get_model_meta()
        recs = detect_scanned_image(to_gray_u8(gray), page_words, config=scan_config,
                                    meta=meta, dpi=use_dpi)
        for r in recs:
            r["page"] = pno
        all_words.extend(recs)
        if include_markdown:
            by_key = {(r["bbox_frac"], r["text"]): r for r in recs}
            seq = [(w.text, w.bbox, by_key.get((w.bbox, w.text)))
                   for w in page_words if (w.text or "").strip()]
            page_md.append(_md.page_markdown(seq))
            page_clean.append(_md.page_clean_text(seq))
            for ps in _md.group_passages(seq):
                ps["page"] = pno
                passages.append(ps)

    result = {
        "source": src_name,
        "page_count": len(frames),
        "page_sources": ["scanned"] * len(frames),
        "words": all_words,
        "n_struck_final": sum(1 for w in all_words if w.get("final")),
        "warnings": warns,
    }
    if include_markdown:
        result["markdown"] = "\n\n".join(page_md)
        result["clean_text"] = "\n\n".join(c for c in page_clean if c).strip()
        result["passages"] = passages
    return result


def _native_word_seq(page, words=None):
    """Reading-order [(text, bbox_frac)] for a native page, boxes in rendered-page fractions.
    ``words`` optionally supplies this page's ``get_text("words")`` output (default None =
    extract here)."""
    if words is None:
        words = page.get_text("words")
    seq = []
    for (x0, y0, x1, y1, txt, *_r) in words:
        if txt.strip():
            seq.append((txt, native._bbox_frac(page, x0, y0, x1, y1)))
    return seq


def _match_native_seq(page, recs, words=None):
    """Reading-order [(text, bbox_frac, rec_or_None)] for a native page. Both detectors emit
    get_text('words') boxes, so records match by exact (bbox, text) key; a same-text spatial
    fallback (record center inside the word box) covers any residual float drift. ``words``
    optionally supplies this page's ``get_text("words")`` output (default None = extract here)."""
    by_key = {(r["bbox_frac"], r["text"]): r for r in recs}
    matched = set()
    seq = []
    for t, b in _native_word_seq(page, words):
        rec = by_key.get((b, t))
        if rec is None:
            for r in recs:
                if r["text"] != t or id(r) in matched:
                    continue
                cx = (r["bbox_frac"][0] + r["bbox_frac"][2]) / 2
                cy = (r["bbox_frac"][1] + r["bbox_frac"][3]) / 2
                if b[0] - 1e-3 <= cx <= b[2] + 1e-3 and b[1] - 2e-3 <= cy <= b[3] + 2e-3:
                    rec = r
                    break
        if rec is not None:
            matched.add(id(rec))
        seq.append((t, b, rec))
    return seq


def _di_pages(di_result):
    """Normalize a user-supplied Azure DI result to its pages list. Accepts the REST JSON dict,
    a {'analyzeResult': {...}} envelope, or an SDK object exposing .as_dict()."""
    if di_result is None:
        return None
    d = di_result
    if hasattr(d, "as_dict"):
        d = d.as_dict()
    if hasattr(d, "get"):
        if "analyzeResult" in d:
            d = d["analyzeResult"]
        pages = d.get("pages")
        if isinstance(pages, list):
            return pages
        raise ValueError("di_result has no 'pages' list — pass the analyze result JSON "
                         "(the dict with pages/paragraphs/...) or the SDK result's .as_dict()")
    raise TypeError(
        f"di_result must be the dict/JSON form of the Azure DI analyze result "
        f"(REST JSON or sdk_result.as_dict()), got {type(di_result).__name__}")


def _resolve_native_method(method, native_method):
    """Back-compat shim for the 0.6.0 R-name rename. detect_pdf's native-page selector is now
    ``method`` (matching ``strikethroughs_in_pdf``/``page_strikes`` and the CLI ``--method``);
    ``native_method`` is the deprecated alias, honored with a ``DeprecationWarning``. Passing both
    with different values is an error."""
    if native_method is not None:
        if method is not None and method != native_method:
            raise ValueError(
                "pass only one of method= / native_method= (native_method is the deprecated "
                "0.5.x alias; they disagree here)")
        _warnings.warn(
            "detect_pdf(native_method=...) is deprecated since 0.6.0; use method= (renamed to "
            "match strikethroughs_in_pdf/page_strikes and the CLI --method)",
            DeprecationWarning, stacklevel=3)
        return native_method
    return "vector" if method is None else method


def _normalize_pages(pages, page_count):
    """Coerce a user ``pages=`` value to a sorted list of unique, in-range 0-based indices.
    Accepts any iterable of ints (negatives index from the end, like list slicing). Out-of-range
    indices raise IndexError so a typo fails loudly instead of silently detecting nothing."""
    if pages is None:
        return list(range(page_count))
    if isinstance(pages, int):
        pages = [pages]
    out = set()
    for p in pages:
        q = int(p)
        if q < 0:
            q += page_count
        if not (0 <= q < page_count):
            raise IndexError(
                f"page index {p} is out of range for a {page_count}-page document")
        out.add(q)
    return sorted(out)


def _normalize_words_by_page(words_by_page):
    """Coerce a ``words_by_page`` value to a ``{0-based page: list[Word]}`` dict. Accepts that
    dict directly (e.g. from ``words_from_textract``/``words_from_docai``) or any sequence indexed
    by page. None passes through."""
    if words_by_page is None:
        return None
    if isinstance(words_by_page, dict):
        return {int(k): v for k, v in words_by_page.items()}
    return {i: v for i, v in enumerate(words_by_page)}


def detect_pdf(source, ocr=None, scan_config=None, dpi=RENDER_DPI, di_result=None,
               include_markdown=True, method=None, on_missing_ocr="raise",
               pages=None, progress=None, words_by_page=None, native_method=None):
    """Detect strikethroughs across a PDF, routing each page to native or scanned.

    Args:
        source: path, bytes, or open fitz document. Encrypted PDFs raise EncryptedPdfError.
        ocr: an OCR backend ``(image_ndarray) -> list[Word]`` (see ``pdf_strikethrough.ocr``).
            Required only if the document has scanned pages and `di_result` is not given.
        scan_config: ``ScanConfig`` for the scanned classifier (default: Azure-DI calibration;
            pass ``ScanConfig.confidence_free()`` for RapidOCR and other weak-confidence engines).
        dpi: raster resolution for scanned pages (detector tunables rescale automatically;
            calibrated/validated at 200).
        di_result: a pre-fetched Azure Document Intelligence analyze result as a DICT — the REST
            JSON, an {'analyzeResult': ...} envelope, or ``sdk_result.as_dict()``. When given,
            its per-page words are used instead of running `ocr` (and `scan_config` defaults to
            Azure-DI calibration).
        words_by_page: pre-fetched OCR words as ``{0-based page: list[Word]}`` (or a sequence
            indexed by page) — the provider-neutral counterpart to `di_result` for any cloud/OCR
            engine, e.g. ``words_from_textract(resp)`` / ``words_from_docai(doc)``. Used for the
            scanned pages it covers, in preference to running `ocr`. Since these confidences
            aren't calibrated to the classifier, `scan_config` defaults to
            ``ScanConfig.confidence_free()`` when it's given (and `di_result` isn't).
        include_markdown: also assemble struck-aware ``markdown``, surviving ``clean_text``, and
            grouped deletion ``passages`` (cheap; reuses the words already extracted).
        method: native-page detector — 'vector' (default), 'flag', 'annot', or 'both' — see
            :func:`pdf_strikethrough.native.page_strikes`. (Renamed from ``native_method`` in
            0.6.0 to match ``strikethroughs_in_pdf``/``page_strikes`` and the CLI ``--method``.)
        native_method: DEPRECATED alias for ``method``, kept for 0.5.x compatibility; emits a
            ``DeprecationWarning``. Passing both with different values raises ``ValueError``.
        on_missing_ocr: what to do when a scanned page is hit with no `ocr`/`di_result`:
            'raise' (default) raises OcrRequiredError; 'skip' skips the page, records a warning
            in the result, and still returns everything from the other pages.
        pages: restrict work to a subset — an iterable of 0-based page indices (negatives index
            from the end). Only those pages are classified and processed; a 300-page scan need
            not OCR every page. Out-of-range indices raise IndexError. When given, the result
            gains a ``pages`` key (the processed indices) and ``page_sources`` is aligned to it;
            ``di_result`` is still indexed by absolute page number. Default None = all pages.
        progress: optional callback ``progress(completed, total, page_index)`` invoked after each
            page is processed (`completed` counts from 1, `total` is the number of pages being
            processed, `page_index` is the just-finished 0-based page). Long OCR+CNN runs are
            otherwise silent and read as hangs; use this to drive a progress bar / stderr line.

    Returns a dict:
        {source, page_count, page_sources, words, n_struck_final, warnings,
         markdown, clean_text, passages}   (last three only if include_markdown;
         plus ``pages`` when a subset was requested). See ``pdf_strikethrough.types.DetectResult``.
    Each word record: page, text, chars, char_span, partial, bbox_frac, tier, verdict, final
    (+ cnn_prob / cnn_agrees on scanned records). ``clean_text`` is assembled from the word
    records (not by stripping the markdown), so the two always agree.
    """
    method = _resolve_native_method(method, native_method)
    doc, close = _open_doc(source)
    try:
        page_indices = _normalize_pages(pages, doc.page_count)
        sources = {p: classify_page_source(doc[p]) for p in page_indices}
        log.debug("detect_pdf: %d-page doc, processing %d page(s); native method=%r",
                  doc.page_count, len(page_indices), method)
        di_pages = _di_pages(di_result)
        wbp = _normalize_words_by_page(words_by_page)
        if scan_config is None:
            if di_pages is not None:
                scan_config = ScanConfig.azure_di()
            elif wbp is not None:
                scan_config = ScanConfig.confidence_free()
            else:
                scan_config = ScanConfig()
        meta = None
        words, warns = [], []
        page_md, page_clean, passages = [], [], []
        total = len(page_indices)
        for done, pno in enumerate(page_indices, start=1):
            page = doc[pno]
            recs, seq = [], []
            log.debug("page %d/%d (index %d): source=%s", done, total, pno, sources[pno])
            if sources[pno] == "native":
                nat_words = page.get_text("words")   # extract once; threaded through both detectors
                recs = native.page_strikes(page, pno, method, words=nat_words)
                for r in recs:
                    r["final"], r["verdict"] = True, "struck"
                if include_markdown:
                    seq = _match_native_seq(page, recs, words=nat_words)
                log.debug("page %d: native detector (%s) -> %d strike record(s)",
                          pno, method, len(recs))
            elif sources[pno] == "scanned":
                page_words = None
                R = page.rect
                page_dpi, note = _working_dpi(R.width / 72.0, R.height / 72.0, dpi)
                page_dpi = int(round(page_dpi))
                if note:
                    _warnings.warn(note, stacklevel=2)
                    warns.append(note)
                if di_pages is not None and pno < len(di_pages):
                    page_words = words_from_azure_di(di_pages[pno])
                elif wbp is not None and pno in wbp:
                    page_words = wbp[pno]
                elif ocr is not None:
                    t0 = time.perf_counter()
                    page_words = ocr(_render_rgb(page, page_dpi))
                    log.debug("page %d: OCR -> %d word(s) in %.0f ms",
                              pno, len(page_words), (time.perf_counter() - t0) * 1e3)

                if page_words is not None:
                    gray = _render_gray(page, page_dpi)
                    if meta is None:
                        meta = cnn.get_model_meta()
                    t0 = time.perf_counter()
                    recs = detect_scanned_image(gray, page_words, config=scan_config,
                                                meta=meta, dpi=page_dpi)
                    log.debug("page %d: geometry+CNN -> %d struck record(s) in %.0f ms",
                              pno, len(recs), (time.perf_counter() - t0) * 1e3)
                    for r in recs:
                        r["page"] = pno
                    if include_markdown:
                        by_key = {(r["bbox_frac"], r["text"]): r for r in recs}
                        seq = [(w.text, w.bbox, by_key.get((w.bbox, w.text)))
                               for w in page_words if (w.text or "").strip()]
                elif (nat_words := page.get_text("words")) and any(
                        w[4].strip() for w in nat_words):
                    # classified scanned but has extractable text (text-bearing scan / OCR-overlay
                    # / full-bleed born-digital) — run the native detector instead of raising
                    msg = (f"page {pno} classified as scanned but has extractable text; ran the "
                           f"native detector on it (no OCR backend / di_result was provided)")
                    _warnings.warn(msg, stacklevel=2)
                    warns.append(msg)
                    recs = native.page_strikes(page, pno, method, words=nat_words)
                    for r in recs:
                        r["final"], r["verdict"] = True, "struck"
                    if include_markdown:
                        seq = _match_native_seq(page, recs, words=nat_words)
                elif di_pages is not None and on_missing_ocr != "skip":
                    raise ValueError(
                        f"di_result has {len(di_pages)} pages but page {pno} of the "
                        f"{doc.page_count}-page document is scanned; pass a full di_result "
                        f"or an ocr backend, or use on_missing_ocr='skip'")
                elif on_missing_ocr == "skip":
                    warns.append(f"page {pno} is scanned but no OCR backend / di_result covered "
                                 f"it; skipped (no text or strikes reported for it)")
                else:
                    raise OcrRequiredError(
                        f"page {pno} is scanned and no OCR backend was provided; pass "
                        f"ocr=rapidocr_backend() (or another backend), di_result=/words_by_page=, "
                        f"or use on_missing_ocr='skip'")

            words.extend(recs)
            if include_markdown:
                page_md.append(_md.page_markdown(seq))
                page_clean.append(_md.page_clean_text(seq))
                for ps in _md.group_passages(seq):
                    ps["page"] = pno
                    passages.append(ps)
            if progress is not None:
                progress(done, total, pno)

        result = {
            "source": None if isinstance(source, (bytes, bytearray)) else (
                str(source) if close else getattr(source, "name", None)),
            "page_count": doc.page_count,
            "page_sources": [sources[p] for p in page_indices],
            "words": words,
            "n_struck_final": sum(1 for w in words if w.get("final")),
            "warnings": warns,
        }
        if pages is not None:
            result["pages"] = page_indices
        if include_markdown:
            result["markdown"] = "\n\n".join(page_md)
            result["clean_text"] = "\n\n".join(c for c in page_clean if c).strip()
            result["passages"] = passages
        return result
    finally:
        if close:
            doc.close()
