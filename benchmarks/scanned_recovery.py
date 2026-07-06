"""Scanned-path recovery — how much of a known strike set survives the OCR + geometry + CNN path.

The confirmation-rate corpus is born-digital, so its strikes are found by the *native* detector.
We reuse that: rasterize the most-struck born-digital pages into an image-only PDF (a genuine
"scan"), take the native detector's strikes on the born-digital original as **ground truth**, then
run the scanned path over the raster with each available OCR backend and measure how many
ground-truth strikes it recovers (struck-region coverage, IoU >= 0.3) plus the total-count parity.

    python benchmarks/scanned_recovery.py

Inputs (per manifest entry, written by `prep_scanned_di.py`): `scanned_pages` (original page
indices) and, optionally, `scanned_di_result` (a cached Azure DI analyze-result JSON in corpus/).
Azure DI is read from that cached JSON (no cloud call); RapidOCR runs live if installed. Entries
without `scanned_pages` are skipped. This supersedes the DI-vs-original-pipeline `di_parity.py` with
a self-contained reference (the exact native truth) that needs no vanished pipeline.
"""
from __future__ import annotations

import json

import pdf_strikethrough as st
from pdf_strikethrough.scanned import ScanConfig

from _corpus import corpus_dir, iter_corpus
from _scanned import build_scanned_pdf, struck_pages


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def _coverage(reference, candidate, thr=0.3):
    if not reference:
        return 1.0
    return sum(any(_iou(r, c) >= thr for c in candidate) for r in reference) / len(reference)


def _struck_boxes(result):
    return [w["bbox_frac"] for w in result["words"] if w.get("final")]


def _backends():
    """Available scanned-word sources as ``name -> callable(orig, pages, gt_boxes) -> (cov, n)``.
    Azure DI is added per-entry (cached JSON); RapidOCR is added here if installed."""
    out = {}
    try:
        import rapidocr  # noqa: F401
        from pdf_strikethrough.ocr import rapidocr_backend
        eng, cfg = rapidocr_backend(), ScanConfig.confidence_free()

        def run_rapidocr(pdf_bytes):
            return st.detect_pdf(pdf_bytes, ocr=eng, scan_config=cfg)
        out["RapidOCR"] = run_rapidocr
    except ImportError:
        print("  (RapidOCR not installed — skipping that column; pip install 'rapidocr>=3.2')")
    return out


def main() -> None:
    cdir = corpus_dir()
    live = _backends()
    names = (["Azure DI"] if True else []) + list(live)
    print(f"\n{'document':<34} {'pages':>5} {'GT':>6} " +
          " ".join(f"{n:>20}" for n in names))
    print("(each backend cell: struck-region coverage vs native ground truth | predicted count)")
    print("-" * (34 + 13 + 21 * len(names)))

    tot_gt = 0
    tot_cov = {n: 0.0 for n in names}
    tot_pred = {n: 0 for n in names}
    ran = 0
    for entry, orig in iter_corpus():
        pages = entry.get("scanned_pages")
        if not pages:
            continue
        page_indices, gt = struck_pages(orig, len(pages))
        gt_boxes = [b for boxes in gt.values() for b in boxes]
        pdf_bytes = build_scanned_pdf(orig, page_indices)

        cells = []
        di_file = entry.get("scanned_di_result")
        if "Azure DI" in names:
            if di_file:
                di = json.loads((cdir / di_file).read_text(encoding="utf-8"))
                pred = _struck_boxes(st.detect_pdf(pdf_bytes, di_result=di))
                cov = _coverage(gt_boxes, pred)
            else:
                pred, cov = [], float("nan")
            tot_cov["Azure DI"] += cov * len(gt_boxes)
            tot_pred["Azure DI"] += len(pred)
            cells.append(f"{cov:>13.1%} | {len(pred):>4}")
        for n in live:
            pred = _struck_boxes(live[n](pdf_bytes))
            cov = _coverage(gt_boxes, pred)
            tot_cov[n] += cov * len(gt_boxes)
            tot_pred[n] += len(pred)
            cells.append(f"{cov:>13.1%} | {len(pred):>4}")

        print(f"{entry['name'][:34]:<34} {len(page_indices):>5} {len(gt_boxes):>6} " +
              " ".join(cells))
        tot_gt += len(gt_boxes)
        ran += 1

    if not ran:
        print("\nNo manifest entries carry `scanned_pages` — run `python benchmarks/"
              "prep_scanned_di.py` first (needs an Azure DI key) to build the scanned set.")
        return
    print("-" * (34 + 13 + 21 * len(names)))
    agg = " ".join(f"{(tot_cov[n] / tot_gt if tot_gt else 1.0):>13.1%} | {tot_pred[n]:>4}"
                   for n in names)
    print(f"{'TOTAL (coverage-weighted)':<34} {'':>5} {tot_gt:>6} {agg}")
    print(f"\n{tot_gt} known strikes across {ran} rasterized documents. Coverage = fraction of the "
          f"native ground-truth strikes the scanned path recovers.")


if __name__ == "__main__":
    main()
