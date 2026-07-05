"""High-level orchestration: route a document to the right path and return struck words,
struck-aware markdown, the surviving clean text, and grouped deletion passages.

  native page   -> exact vector detection (no OCR, no model)
  scanned page  -> geometry + OCR words -> layer-1 classify -> layer-2 CNN verdict

The CNN verdict layer mirrors the corpus-validated gate: 'auto' words are kept only when the
CNN confirms at p_hi (crops too small to score are kept — no counter-evidence); on 'review'
words (the geometry can't decide) the CNN is the decider.
"""
from __future__ import annotations

import warnings as _warnings

import pymupdf

from . import cnn, markdown as _md, native
from .ocr import words_from_azure_di
from .scanned import ScanConfig, analyze_scanned_page

IMG_COVER_SCANNED = 0.70    # raster images cover >= this frac of the page -> scanned
RENDER_DPI = 200


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


def detect_pdf(source, ocr=None, scan_config=None, dpi=RENDER_DPI, di_result=None,
               include_markdown=True, native_method="vector", on_missing_ocr="raise",
               pages=None, progress=None):
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
        include_markdown: also assemble struck-aware ``markdown``, surviving ``clean_text``, and
            grouped deletion ``passages`` (cheap; reuses the words already extracted).
        native_method: 'vector' (default), 'flag', or 'both' — see
            :func:`pdf_strikethrough.native.page_strikes`.
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
    doc, close = _open_doc(source)
    try:
        page_indices = _normalize_pages(pages, doc.page_count)
        sources = {p: classify_page_source(doc[p]) for p in page_indices}
        di_pages = _di_pages(di_result)
        if scan_config is None:
            scan_config = ScanConfig.azure_di() if di_pages is not None else ScanConfig()
        meta = None
        words, warns = [], []
        page_md, page_clean, passages = [], [], []
        total = len(page_indices)
        for done, pno in enumerate(page_indices, start=1):
            page = doc[pno]
            recs, seq = [], []
            if sources[pno] == "native":
                nat_words = page.get_text("words")   # extract once; threaded through both detectors
                recs = native.page_strikes(page, pno, native_method, words=nat_words)
                for r in recs:
                    r["final"], r["verdict"] = True, "struck"
                if include_markdown:
                    seq = _match_native_seq(page, recs, words=nat_words)
            elif sources[pno] == "scanned":
                page_words = None
                if di_pages is not None and pno < len(di_pages):
                    page_words = words_from_azure_di(di_pages[pno])
                elif ocr is not None:
                    page_words = ocr(_render_rgb(page, dpi))

                if page_words is not None:
                    gray = _render_gray(page, dpi)
                    if meta is None:
                        meta = cnn.get_model_meta()
                    recs = detect_scanned_image(gray, page_words, config=scan_config,
                                                meta=meta, dpi=dpi)
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
                    recs = native.page_strikes(page, pno, native_method, words=nat_words)
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
                        f"ocr=rapidocr_backend() (or another backend) or di_result=..., or "
                        f"use on_missing_ocr='skip'")

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
