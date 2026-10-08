"""Scanned-path recovery — how much of a known strike set survives the OCR + geometry + CNN path.

The confirmation-rate corpus is born-digital, so its strikes are found by the *native* detector.
We reuse that: rasterize the most-struck born-digital pages into an image-only PDF (a genuine
"scan"), take the native detector's strikes on the born-digital original as **ground truth**, then
run the scanned path over the raster with each available OCR backend and measure how many
ground-truth strikes it recovers (struck-region coverage, IoU >= 0.3) plus the total-count parity.

    python benchmarks/scanned_recovery.py

Inputs (per manifest entry, written by `prep_scanned_di.py`): `scanned_pages` (original page
indices) and, optionally, `scanned_di_result` (a cached Azure DI analyze-result JSON in corpus/,
downloaded with the PDFs by `fetch_corpus.py` and checked against `scanned_di_sha256`). Azure DI is
read from that cached JSON (no cloud call) and scored twice, with the default DI calibration and
with `ScanConfig.confidence_free()`; RapidOCR runs live if installed. Entries without
`scanned_pages` are skipped, and an entry without a DI result is n/a in the DI columns. This
supersedes the DI-vs-original-pipeline `di_parity.py` with a self-contained reference (the exact
native truth) that needs no vanished pipeline.
"""
from __future__ import annotations

import json

import pdf_strikethrough as st
from pdf_strikethrough.scanned import ScanConfig

from _corpus import check_file, corpus_dir, iter_corpus
from _scanned import build_scanned_pdf

# The cached DI words under each calibration the README quotes. None is detect_pdf's default with
# di_result, the Azure DI calibration.
DI_CONFIGS = {"Azure DI": None, "Azure DI conf-free": ScanConfig.confidence_free()}


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
    """OCR backends that run live, as ``name -> callable(pdf_bytes) -> detect_pdf result``. Azure
    DI is not one of them: it is read per entry from the cached JSON."""
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
    names = list(DI_CONFIGS) + list(live)
    print(f"\n{'document':<34} {'pages':>5} {'GT':>6} " +
          " ".join(f"{n:>20}" for n in names))
    print("(each backend cell: struck-region coverage vs native ground truth | predicted count)")
    print("-" * (34 + 13 + 21 * len(names)))

    tot_gt = 0
    tot_runs = {n: 0 for n in names}       # entries each backend ran on
    tot_scored = {n: 0 for n in names}     # ground truth on those entries
    tot_cov = {n: 0.0 for n in names}
    tot_pred = {n: 0 for n in names}
    ran = 0
    for entry, orig in iter_corpus():
        pages = entry.get("scanned_pages")
        if not pages:
            continue
        page_indices = sorted(pages)           # the pages the cached DI result was captured on
        by_page = {}
        for r in st.strikethroughs_in_pdf(str(orig), method="both"):
            by_page.setdefault(r["page"], []).append(r["bbox_frac"])
        gt_boxes = [b for pno in page_indices for b in by_page.get(pno, [])]
        pdf_bytes = build_scanned_pdf(orig, page_indices)

        results = {}
        di_file = entry.get("scanned_di_result")
        if di_file:
            di_path = cdir / di_file
            check_file(di_path, entry.get("scanned_di_sha256"), entry.get("scanned_di_url"))
            di = json.loads(di_path.read_text(encoding="utf-8"))
            for n, cfg in DI_CONFIGS.items():
                results[n] = st.detect_pdf(pdf_bytes, di_result=di, scan_config=cfg)
        for n, run in live.items():
            results[n] = run(pdf_bytes)

        cells = []
        for n in names:
            if n not in results:
                cells.append(f"{'n/a':>20}")
                continue
            pred = _struck_boxes(results[n])
            cov = _coverage(gt_boxes, pred)
            tot_runs[n] += 1
            tot_scored[n] += len(gt_boxes)
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
    agg = " ".join(f"{'n/a':>20}" if not tot_runs[n] else
                   f"{(tot_cov[n] / tot_scored[n] if tot_scored[n] else 1.0):>13.1%} | "
                   f"{tot_pred[n]:>4}" for n in names)
    print(f"{'TOTAL (coverage-weighted)':<34} {'':>5} {tot_gt:>6} {agg}")
    for n in names:
        if tot_runs[n] and tot_scored[n] != tot_gt:
            print(f"  ({n}: over the {tot_scored[n]} known strikes of the entries it ran on)")
    print(f"\n{tot_gt} known strikes across {ran} rasterized documents. Coverage = fraction of the "
          f"native ground-truth strikes the scanned path recovers.")


if __name__ == "__main__":
    main()
