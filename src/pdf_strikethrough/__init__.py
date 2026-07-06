"""pdf_strikethrough — detect struck-through (deleted) text in PDFs and scanned document images.

Quick start
-----------
Native / born-digital PDF (strikethroughs are vector drawings — detection is EXACT, no OCR):

    import pdf_strikethrough as st
    for w in st.strikethroughs_in_pdf("contract.pdf"):
        print(w["page"], repr(w["chars"]), "partial" if w["partial"] else "full")
    print(st.clean_markdown("contract.pdf"))          # surviving text, deletions removed

Any PDF (routes native/scanned per page; scanned needs an OCR backend):

    from pdf_strikethrough.ocr import rapidocr_backend
    from pdf_strikethrough.scanned import ScanConfig
    res = st.detect_pdf("scan.pdf", ocr=rapidocr_backend(),
                        scan_config=ScanConfig.confidence_free())
    struck = [w for w in res["words"] if w["final"]]

Low-level, on your own image (no PDF):

    lines = st.strike_lines(gray, dpi=200)            # OCR-free stroke geometry
    p = st.score_word(gray, (x0, y0, x1, y1))         # CNN strike probability for a word box
"""
import logging as _logging
import warnings as _warnings

from . import cnn, detect, lines, markdown, native, ocr, overlay, scanned, types
from .cnn import (get_model_meta, score_crops, score_word, std_crop, verdict_of, word_crop_px)
from .detect import (EncryptedPdfError, OcrRequiredError, apply_cnn_verdict,
                     classify_page_source, detect_pdf, detect_scanned_image)
from .lines import ink_mask, strike_lines, to_gray_u8
from .native import (native_annot_strikes, native_doc_strikes, native_flag_strikes,
                     native_markdown, native_page_strikes, page_strikes, strip_struck_markdown)
from .ocr import (Word, rapidocr_backend, tesseract_backend, words_from_azure_di)
from .overlay import render_overlay, save_overlays
from .scanned import ScanConfig, analyze_scanned_page
from .types import DetectResult, Passage, StruckWord

# Library logging etiquette: attach a NullHandler so importing the package never emits records on
# its own. Diagnostics (page routing, tier decisions, OCR/CNN timing) are logged at DEBUG under the
# "pdf_strikethrough" logger — a caller opts in with logging.getLogger("pdf_strikethrough").
# ``warnings`` stays reserved for caller-facing hazards (silent-[] on scans, scanned-fallback, ...).
_logging.getLogger("pdf_strikethrough").addHandler(_logging.NullHandler())

__version__ = "0.6.0"

__all__ = [
    # high-level
    "strikethroughs_in_pdf", "clean_markdown", "detect_pdf", "detect_scanned_image",
    "open_pdf", "render_page_gray", "render_overlay", "save_overlays",
    # native
    "native_page_strikes", "native_flag_strikes", "native_annot_strikes", "native_doc_strikes",
    "page_strikes", "native_markdown", "strip_struck_markdown",
    # scanned geometry + classifier
    "strike_lines", "ink_mask", "to_gray_u8", "analyze_scanned_page", "ScanConfig",
    "classify_page_source", "apply_cnn_verdict",
    # errors
    "OcrRequiredError", "EncryptedPdfError",
    # OCR
    "Word", "rapidocr_backend", "tesseract_backend", "words_from_azure_di",
    # CNN
    "score_word", "score_crops", "std_crop", "word_crop_px", "verdict_of", "get_model_meta",
    # typing
    "StruckWord", "DetectResult", "Passage",
    # submodules
    "cnn", "lines", "native", "ocr", "overlay", "scanned", "detect", "markdown", "types",
]


def open_pdf(source):
    """Open `source` (path, bytes, or an already-open fitz document) as a fitz document.
    Raises EncryptedPdfError for password-protected PDFs (routes through the same gate as
    ``detect_pdf``; an already-authenticated document passes)."""
    return detect._open_doc(source)[0]


def render_page_gray(page, dpi=lines.RENDER_DPI):
    """Render a fitz page to a grayscale uint8 (H, W) numpy array."""
    import numpy as np
    import pymupdf
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def strikethroughs_in_pdf(source, method="vector") -> "list[StruckWord]":
    """Struck-word records for a born-digital PDF (path/bytes/fitz doc), all pages, reading order.
    Exact — driven by the PDF's own strike drawings. `method`: 'vector' (stroke geometry, precise
    partial-char spans; default), 'flag' (MuPDF's strikeout span flag), or 'both' (union, maximum
    recall).

    Scanned PDFs have no vector strikes, so this returns [] for them — and emits a
    ``UserWarning`` naming the scanned pages, because a silent [] on a scan (the package's own
    README opens with one) is the most dangerous confusion it can produce. Route scans through
    ``detect_pdf(..., ocr=...)`` instead."""
    doc = open_pdf(source)
    try:
        scanned_pages = [p for p in range(doc.page_count)
                         if detect.classify_page_source(doc[p]) == "scanned"]
        if scanned_pages:
            shown = ", ".join(str(p) for p in scanned_pages[:10])
            more = "" if len(scanned_pages) <= 10 else f" (+{len(scanned_pages) - 10} more)"
            _warnings.warn(
                f"strikethroughs_in_pdf found {len(scanned_pages)} scanned page(s) "
                f"[{shown}{more}]: vector detection cannot see strikes on a scan and reports "
                f"nothing for them. Use detect_pdf(source, ocr=rapidocr_backend()) for scanned "
                f"pages.", stacklevel=2)
        return native.native_doc_strikes(doc, method)
    finally:
        if not hasattr(source, "page_count"):
            doc.close()


def clean_markdown(source) -> str:
    """Markdown for a born-digital PDF with struck (deleted) spans removed — the surviving text.
    Requires the ``[markdown]`` extra (pymupdf4llm); raises ImportError with the pip command
    otherwise. For an extra-free equivalent use ``detect_pdf(source)["clean_text"]``."""
    doc = open_pdf(source)
    try:
        if doc.page_count == 0:
            return ""
        return native.strip_struck_markdown(native.native_markdown(doc))
    finally:
        if not hasattr(source, "page_count"):
            doc.close()
