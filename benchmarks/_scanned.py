"""Shared helpers for the scanned-recovery benchmark.

The confirmation-rate corpus is born-digital (vector strikes), so its pages route to the *native*
detector and never exercise the scanned path (OCR words + geometry + CNN). To benchmark that path
reproducibly we rasterize the born-digital pages into an image-only PDF — a genuine "scan" whose
*exact* strike set we already know from the native detector on the born-digital original. That known
set is the ground truth; the scanned path (fed DI or RapidOCR word boxes) is scored against it.

Rasterization is deterministic (fixed DPI, same pixmap), so a DI result captured once against the
image-only PDF stays aligned with a later rebuild of the same PDF.
"""
from __future__ import annotations

import fitz

import pdf_strikethrough as st

SCAN_DPI = 200   # the detector's calibration point; RENDER_DPI normalizes here anyway


def struck_pages(orig_path, top_k, method="both"):
    """The ``top_k`` original page indices carrying the most struck words, and the per-page ground
    truth. Returns ``(page_indices, gt_by_newindex)`` where ``gt_by_newindex[i]`` is the list of
    ground-truth struck ``bbox_frac`` on the i-th selected page (i = its index in the image PDF)."""
    by_page = _strikes_by_page(orig_path, method)
    ranked = sorted(by_page, key=lambda p: len(by_page[p]), reverse=True)[:top_k]
    page_indices = sorted(ranked)
    gt = {i: by_page[pno] for i, pno in enumerate(page_indices)}
    return page_indices, gt


def page_ground_truth(orig_path, page_indices, method="both"):
    """Ground truth for a FIXED page set (a manifest's ``scanned_pages``): ``gt_by_newindex[i]`` is
    the struck ``bbox_frac`` list on the i-th page of the image PDF. Scoring must use the pages the
    cached DI result was captured on; re-ranking with :func:`struck_pages` follows the current
    detector, so a change in per-page counts swapped pages and put the cached words on the wrong
    page images."""
    by_page = _strikes_by_page(orig_path, method)
    return {i: by_page.get(pno, []) for i, pno in enumerate(page_indices)}


def _strikes_by_page(orig_path, method):
    by_page: dict[int, list] = {}
    for r in st.strikethroughs_in_pdf(str(orig_path), method=method):
        by_page.setdefault(r["page"], []).append(r["bbox_frac"])
    return by_page


def build_scanned_pdf(orig_path, page_indices, dpi=SCAN_DPI):
    """Deterministically render ``page_indices`` of ``orig_path`` into an image-only PDF (no text
    layer, so every page classifies as ``scanned``). Page geometry is preserved, so page-fraction
    boxes from the born-digital original map 1:1 onto the rasterized pages."""
    src = fitz.open(str(orig_path))
    out = fitz.open()
    try:
        for pno in page_indices:
            page = src[pno]
            pix = page.get_pixmap(dpi=dpi)
            npage = out.new_page(width=page.rect.width, height=page.rect.height)
            npage.insert_image(npage.rect, stream=pix.tobytes("png"))
        return out.tobytes()
    finally:
        src.close()
        out.close()
