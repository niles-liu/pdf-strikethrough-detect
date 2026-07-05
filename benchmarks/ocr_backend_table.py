"""OCR-backend comparison table — reproduces the README's "Choosing an OCR backend" numbers.

For each scanned document in the corpus that has an Azure DI reference, run detection with each
available OCR backend and compare the struck regions each finds against the DI-reference result
(struck-region coverage + spatial agreement). The geometry + CNN carry detection; OCR only supplies
the word boxes, so this measures how much backend choice actually moves the result.

    pip install "pdf-strikethrough-detect[rapidocr,tesseract]"
    python benchmarks/ocr_backend_table.py

Inputs (per manifest entry): a `di_result` field pointing at the document's Azure DI analyze-result
JSON (relative to benchmarks/corpus/). Entries without `di_result` are skipped. Backends that
aren't installed are skipped with a note — so this runs partially with whatever you have.
"""
from __future__ import annotations

import json

import pdf_strikethrough as st
from pdf_strikethrough.scanned import ScanConfig

from _corpus import corpus_dir, iter_corpus


def _struck_boxes(result):
    return [w["bbox_frac"] for w in result["words"] if w.get("final")]


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def _coverage(reference, candidate, thr=0.3):
    """Fraction of reference struck boxes overlapped (IoU >= thr) by some candidate box."""
    if not reference:
        return 1.0
    return sum(any(_iou(r, c) >= thr for c in candidate) for r in reference) / len(reference)


def _available_backends():
    backends = {}
    cdir = corpus_dir()
    try:
        import rapidocr  # noqa: F401
        from pdf_strikethrough.ocr import rapidocr_backend
        backends["RapidOCR"] = (rapidocr_backend(), ScanConfig.confidence_free())
    except ImportError:
        print("  (RapidOCR not installed — skipping that column)")
    try:
        import pytesseract  # noqa: F401
        from pdf_strikethrough.ocr import tesseract_backend
        backends["Tesseract"] = (tesseract_backend(), ScanConfig.confidence_free())
    except ImportError:
        print("  (Tesseract not installed — skipping that column)")
    return backends, cdir


def main() -> None:
    backends, corpus_dir = _available_backends()
    if not backends:
        raise SystemExit('no OCR backends installed. pip install '
                         '"pdf-strikethrough-detect[rapidocr,tesseract]"')

    names = list(backends)
    print(f"\n{'document':<32} " + " ".join(f"{n:>18}" for n in names))
    print("(each cell: struck-region coverage vs the Azure DI reference)")
    print("-" * (32 + 19 * len(names)))
    ran = 0
    for entry, path in iter_corpus():
        di_file = entry.get("di_result")
        if not di_file:
            continue                                   # only docs with a DI reference are scored
        with open(corpus_dir / di_file, encoding="utf-8") as f:
            di = json.load(f)
        ref = _struck_boxes(st.detect_pdf(str(path), di_result=di))
        cells = []
        for name in names:
            ocr, cfg = backends[name]
            got = _struck_boxes(st.detect_pdf(str(path), ocr=ocr, scan_config=cfg))
            cells.append(f"{_coverage(ref, got):>17.1%}")
        print(f"{entry['name'][:32]:<32} " + " ".join(cells))
        ran += 1
    if not ran:
        print("\nNo manifest entries carry a `di_result` reference — nothing to compare. Add the "
              "Azure DI analyze-result JSON per document (see benchmarks/README.md).")


if __name__ == "__main__":
    main()
