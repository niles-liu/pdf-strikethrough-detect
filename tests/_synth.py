"""Helpers shared by several test files. Synthetic fixtures only: no data files are checked in.

Imported by name (``from _synth import _blank``). ``pythonpath = ["tests"]`` in pyproject.toml
puts this folder on ``sys.path`` in every import mode; without that config, pytest's default mode
does too. Not a ``conftest.py``: ``benchmarks/frontier/tests`` has its own, and a session that
collects both folders imports only one module of that name.
"""
import numpy as np
import pymupdf

from pdf_strikethrough import cnn
from pdf_strikethrough.ocr import Word


# ----------------------------------------------------------------- scanned rasters, drawn in numpy

def _blank(h=500, w=900):
    return np.full((h, w), 255, np.uint8)


def _glyphs(gray, x0, x1, y0, y1, step=8, wide=3):
    """Fill a band with dense vertical 'glyph' ticks (ink)."""
    for x in range(x0, x1, step):
        gray[y0:y1, x:x + wide] = 20


def _saturated_cnn(monkeypatch):
    """StrikeNet pinned at prob 1.0, the faint-scan saturation it shows: it confirms every
    candidate."""
    monkeypatch.setattr(cnn, "score_crops", lambda crops, **kw: np.ones(len(crops)))


# ----------------------------------------------------------------- PDFs and images, drawn in MuPDF

def _synthetic_native_pdf():
    """Born-digital PDF: one line with a vector strike through the consecutive 'deleted text'."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "keep this deleted text here", fontsize=12)
    words = {w[4]: pymupdf.Rect(w[:4]) for w in page.get_text("words")}
    r1, r2 = words["deleted"], words["text"]
    ymid = (r1.y0 + r1.y1) / 2
    page.draw_line(pymupdf.Point(r1.x0 - 1, ymid), pymupdf.Point(r2.x1 + 1, ymid), width=1.0)
    return doc


def _mixed_native_scanned_pdf():
    """Page 0 native (struck); page 1 an image-only render of it (classified scanned)."""
    doc = _synthetic_native_pdf()
    scan = doc.new_page(width=595, height=842)
    scan.insert_image(scan.rect, pixmap=doc[0].get_pixmap(dpi=72))
    return doc


def _scanned_pdf(n=2):
    """`n` image-only pages of near-white pixels: scanned pages with nothing on them."""
    doc = pymupdf.open()
    for _ in range(n):
        page = doc.new_page(width=300, height=200)
        pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 60, 40), False)
        pix.set_rect(pix.irect, (250,))
        page.insert_image(page.rect, pixmap=pix)
    return doc.tobytes()


def _redline_image_png(dpi=200):
    """PNG bytes of a rendered page with a vector strike through 'struck', + its Word boxes."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=120)
    page.insert_text((20, 60), "keep struck", fontsize=20)
    b = {w[4]: pymupdf.Rect(w[:4]) for w in page.get_text("words")}
    r = b["struck"]
    ymid = (r.y0 + r.y1) / 2
    page.draw_line(pymupdf.Point(r.x0, ymid), pymupdf.Point(r.x1, ymid), width=1.5)
    png = page.get_pixmap(dpi=dpi).tobytes("png")
    W, H = page.rect.width, page.rect.height
    words = [Word(w[4], (w[0] / W, w[1] / H, w[2] / W, w[3] / H)) for w in page.get_text("words")]
    doc.close()
    return png, words


def _reopen(doc):
    """`doc` saved and opened again, as a reader of the file would see it."""
    return pymupdf.open("pdf", doc.tobytes())


# ----------------------------------------------------------------- results

def _final_chars(res):
    """The struck characters of every final record in a detect result."""
    return [w["chars"] for w in res["words"] if w["final"]]
