"""Render pages with the detected strikes drawn on top — a visual overlay.

For debugging / ScanConfig tuning and for the before/after documentation figure. Full strikes are
boxed in red, partial strikes in orange. No new dependency (PyMuPDF renders, Pillow draws).

    import pdf_strikethrough as st
    for pg in st.render_overlay("contract.pdf"):        # [{"page", "image", "n_struck"}]
        pg["image"].save(f"p{pg['page']}-overlay.png")

Also on the CLI: ``pdf-strikethrough detect FILE.pdf --overlay out/``.
"""
from __future__ import annotations

import os

import pymupdf
from PIL import Image, ImageDraw

from . import detect

FULL_COLOR = (214, 40, 40)       # red: fully struck word
PARTIAL_COLOR = (232, 138, 0)    # orange: partial strike
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _page_image(page, dpi):
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def render_overlay(source, result=None, *, dpi=150, pages=None,
                   full_color=FULL_COLOR, partial_color=PARTIAL_COLOR):
    """Render pages with the detected strike boxes drawn on them.

    Args:
        source: path, bytes, or open fitz document (same inputs as ``detect_pdf``).
        result: a ``detect_pdf(...)`` result dict whose ``words`` drive the boxes. When None,
            ``detect_pdf(source, include_markdown=False, pages=pages)`` is run (native pages need
            nothing extra; scanned pages come out empty without an OCR result — pass a `result`
            you computed with an OCR backend to overlay those).
        dpi: render resolution for the page images (default 150 — figure-friendly).
        pages: 0-based indices to render (negatives index from the end). Default None = every page
            that carries at least one struck word.
        full_color / partial_color: RGB outline colors for full vs partial strikes.

    Returns a list of ``{"page": int, "image": PIL.Image.Image, "n_struck": int}`` in page order.
    """
    doc, owned = detect._open_doc(source)
    try:
        if result is None:
            # skip (not raise on) scanned pages with no OCR: a native-only overlay of a mixed doc
            # should still render, leaving scanned pages un-boxed
            result = detect.detect_pdf(doc, include_markdown=False, pages=pages,
                                       on_missing_ocr="skip")
        by_page = {}
        for w in result.get("words", []):
            if w.get("final"):
                by_page.setdefault(w["page"], []).append(w)
        targets = (detect._normalize_pages(pages, doc.page_count) if pages is not None
                   else sorted(by_page))
        lw = max(1, round(dpi / 100))
        out = []
        for pno in targets:
            img = _page_image(doc[pno], dpi)
            draw = ImageDraw.Draw(img)
            W, H = img.size
            recs = by_page.get(pno, [])
            for w in recs:
                x0, y0, x1, y1 = w["bbox_frac"]
                box = [x0 * W - lw, y0 * H - lw, x1 * W + lw, y1 * H + lw]
                draw.rectangle(box, outline=partial_color if w.get("partial") else full_color,
                               width=lw)
            out.append({"page": pno, "image": img, "n_struck": len(recs)})
        return out
    finally:
        if owned:
            doc.close()


def save_overlays(source, out, result=None, *, dpi=150):
    """Render overlays for `source` and write one image per page to `out`, returning the written
    paths. If `out` has an image extension it is used as a filename prefix (``root-p{page}{ext}``);
    otherwise `out` is treated as a directory (created if needed) and PNGs are written into it."""
    root, ext = os.path.splitext(out)
    if ext.lower() in _IMG_EXTS:
        prefix, suffix = root, ext
    else:
        os.makedirs(out, exist_ok=True)
        prefix, suffix = os.path.join(out, "overlay"), ".png"
    written = []
    for item in render_overlay(source, result=result, dpi=dpi):
        path = f"{prefix}-p{item['page']}{suffix}"
        img = item["image"]
        if suffix.lower() in (".jpg", ".jpeg") and img.mode != "RGB":
            img = img.convert("RGB")
        img.save(path)
        written.append(path)
    return written
