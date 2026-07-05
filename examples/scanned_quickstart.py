"""Scanned quick start — no assets to download.

Builds the same redline PDF, rasterizes it to an image-only page (a synthetic "scan" with no text
layer), then runs the geometry -> OCR -> CNN pipeline on it and prints the struck words.

    pip install "pdf-strikethrough-detect[rapidocr]"
    python examples/scanned_quickstart.py

Needs the [rapidocr] extra (a free, pip-only OCR backend). The detection geometry + CNN are
OCR-independent; OCR only supplies the word boxes to attribute strikes to.
"""
import sys

import pymupdf

import pdf_strikethrough as st
from pdf_strikethrough.scanned import ScanConfig

from native_quickstart import build_redline_pdf


def rasterize_to_scan(pdf_bytes: bytes, dpi: int = 200) -> bytes:
    """Render each page to an image and place it on a fresh page with NO text layer — i.e. what a
    flatbed scan of a printed redline looks like to the detector."""
    src = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    out = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi)
        scan = out.new_page(width=page.rect.width, height=page.rect.height)
        scan.insert_image(scan.rect, pixmap=pix)
    return out.tobytes()


def main() -> int:
    try:
        import rapidocr  # noqa: F401 — eager check; the backend imports it lazily otherwise
    except ImportError:
        print('this example needs the rapidocr extra: '
              'pip install "pdf-strikethrough-detect[rapidocr]"', file=sys.stderr)
        return 1
    from pdf_strikethrough.ocr import rapidocr_backend
    ocr = rapidocr_backend()

    scan = rasterize_to_scan(build_redline_pdf())
    print("page sources:", st.detect_pdf(scan, ocr=ocr,
                                         scan_config=ScanConfig.confidence_free())["page_sources"])

    res = st.detect_pdf(scan, ocr=ocr, scan_config=ScanConfig.confidence_free())
    struck = [w for w in res["words"] if w["final"]]
    print(f"\n{len(struck)} struck word(s) on the scan:")
    for w in struck:
        print(f"  p{w['page']} {w['chars']!r}  (tier={w['tier']}, score={w.get('score')}, "
              f"cnn_prob={w.get('cnn_prob')})")
    print("\nmarkdown:", res["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
