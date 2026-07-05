"""High-level orchestration: route a document to the right path and return struck words,
struck-aware markdown, the surviving clean text, and grouped deletion passages.

  native page   -> exact vector detection (no OCR, no model)
  scanned page  -> geometry + OCR words -> layer-1 classify -> layer-2 CNN verdict

The CNN verdict layer mirrors the corpus-validated gate: 'auto' words are kept only when the
CNN confirms at p_hi (crops too small to score are kept — no counter-evidence); on 'review'
words (the geometry can't decide) the CNN is the decider.
"""
from __future__ import annotations

import fitz

from . import cnn, markdown as _md, native
from .ocr import words_from_azure_di
from .scanned import ScanConfig, analyze_scanned_page

IMG_COVER_SCANNED = 0.70    # raster images cover >= this frac of the page -> scanned
RENDER_DPI = 200


class OcrRequiredError(ValueError):
    """A scanned page was hit but no OCR backend (or DI result) was provided."""


class EncryptedPdfError(ValueError):
    """The PDF is password-protected; decrypt or authenticate before processing."""


def _check_not_encrypted(doc):
    if getattr(doc, "needs_pass", False):
        raise EncryptedPdfError(
            "PDF is password-protected; open it with fitz and call doc.authenticate(password) "
            "(or save a decrypted copy) before processing")


def classify_page_source(page):
    """'native' | 'scanned' | 'blank' for one fitz page. Raster images covering most of the page
    mean scanned (an OCR text overlay over page images does not make a scan native); ANY real
    text otherwise means native — sparse pages (signature pages, cover sheets) are still native."""
    area = page.rect.get_area() or 1.0
    img_area = 0.0
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page.rect
        if not r.is_empty:
            img_area += r.get_area()
    if img_area / area >= IMG_COVER_SCANNED:
        return "scanned"
    if any(w[4].strip() for w in page.get_text("words")):
        return "native"
    return "blank" if img_area / area < 0.05 else "scanned"


def _render_gray(page, dpi=RENDER_DPI):
    import numpy as np
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def _render_rgb(page, dpi=RENDER_DPI):
    import numpy as np
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
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
            h["verdict"] = "struck"
            h["cnn_agrees"] = (p >= meta["p_hi"]) if p is not None else None
            h["final"] = h["cnn_agrees"] is not False
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


def _native_word_seq(page):
    """Reading-order [(text, bbox_frac)] for a native page, boxes in rendered-page fractions."""
    seq = []
    for (x0, y0, x1, y1, txt, *_r) in page.get_text("words"):
        if txt.strip():
            seq.append((txt, native._bbox_frac(page, x0, y0, x1, y1)))
    return seq


def _match_native_seq(page, recs):
    """Reading-order [(text, bbox_frac, rec_or_None)] for a native page. Both detectors emit
    get_text('words') boxes, so records match by exact (bbox, text) key; a same-text spatial
    fallback (record center inside the word box) covers any residual float drift."""
    by_key = {(r["bbox_frac"], r["text"]): r for r in recs}
    matched = set()
    seq = []
    for t, b in _native_word_seq(page):
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


def detect_pdf(source, ocr=None, scan_config=None, dpi=RENDER_DPI, di_result=None,
               include_markdown=True, native_method="vector", on_missing_ocr="raise"):
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

    Returns a dict:
        {source, page_count, page_sources, words, n_struck_final, warnings,
         markdown, clean_text, passages}   (last three only if include_markdown)
    Each word record: page, text, chars, char_span, partial, bbox_frac, tier, verdict, final
    (+ cnn_prob / cnn_agrees on scanned records). ``clean_text`` is assembled from the word
    records (not by stripping the markdown), so the two always agree.
    """
    close = not hasattr(source, "page_count")
    doc = source if hasattr(source, "page_count") else (
        fitz.open(stream=bytes(source), filetype="pdf") if isinstance(source, (bytes, bytearray))
        else fitz.open(source))
    try:
        _check_not_encrypted(doc)
        sources = [classify_page_source(doc[p]) for p in range(doc.page_count)]
        di_pages = _di_pages(di_result)
        if scan_config is None:
            scan_config = ScanConfig.azure_di() if di_pages is not None else ScanConfig()
        meta = None
        words, warnings = [], []
        page_md, page_clean, passages = [], [], []
        for pno in range(doc.page_count):
            page = doc[pno]
            recs, seq = [], []
            if sources[pno] == "native":
                recs = native.page_strikes(page, pno, native_method)
                for r in recs:
                    r["final"], r["verdict"] = True, "struck"
                if include_markdown:
                    seq = _match_native_seq(page, recs)
            elif sources[pno] == "scanned":
                page_words = None
                if di_pages is not None and pno < len(di_pages):
                    page_words = words_from_azure_di(di_pages[pno])
                elif ocr is not None:
                    page_words = ocr(_render_rgb(page, dpi))
                elif di_pages is not None:
                    raise ValueError(
                        f"di_result has {len(di_pages)} pages but page {pno} of the "
                        f"{doc.page_count}-page document is scanned; pass a full di_result "
                        f"or an ocr backend")
                elif on_missing_ocr == "skip":
                    warnings.append(f"page {pno} is scanned but no OCR backend was provided; "
                                    f"skipped (no text or strikes reported for it)")
                else:
                    raise OcrRequiredError(
                        f"page {pno} is scanned and no OCR backend was provided; pass "
                        f"ocr=rapidocr_backend() (or another backend) or di_result=..., or "
                        f"use on_missing_ocr='skip'")
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

            words.extend(recs)
            if include_markdown:
                page_md.append(_md.page_markdown(seq))
                page_clean.append(_md.page_clean_text(seq))
                for ps in _md.group_passages(seq):
                    ps["page"] = pno
                    passages.append(ps)

        result = {
            "source": None if isinstance(source, (bytes, bytearray)) else (
                str(source) if close else getattr(source, "name", None)),
            "page_count": doc.page_count,
            "page_sources": sources,
            "words": words,
            "n_struck_final": sum(1 for w in words if w.get("final")),
            "warnings": warnings,
        }
        if include_markdown:
            result["markdown"] = "\n\n".join(page_md)
            result["clean_text"] = "\n\n".join(c for c in page_clean if c).strip()
            result["passages"] = passages
        return result
    finally:
        if close:
            doc.close()
