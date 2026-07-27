"""Confidence-veto precision benchmark — scanned ruled-table over-flagging (issue #4).

Runs the scanned Azure-DI path over a directory of scanned, tightly-ruled table/form PDFs and
reports the struck-final count per document, split by OCR confidence: the high-confidence
population the 0.9.1 DI-confidence veto targets vs the low-confidence faint-scan residual left to a
post-1.0 CNN retrain. This is the reproduction harness for the v0.9.1 correctness patch.

The corpus is private and is NOT committed (see `.gitignore: /benchmarks/private/`). Point the
harness at a local copy — one subfolder per document, each holding a `*.pdf` and a `di-result.json`
(an Azure DI `prebuilt-layout` analyze result, so no cloud call is made):

    python benchmarks/confidence_veto.py [CORPUS_DIR]

CORPUS_DIR defaults to `benchmarks/private/ruled-tables/` or the PDF_STRIKETHROUGH_CORPUS_DIR env var.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pdf_strikethrough as st

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "private", "ruled-tables")
CLEAN_CONF = st.ScanConfig().max_clean_conf


def run(corpus_dir):
    docs = sorted(d for d in glob.glob(os.path.join(corpus_dir, "*")) if os.path.isdir(d))
    if not docs:
        print(f"no document subfolders under {corpus_dir!r}\n"
              "point the harness at a local corpus (PDF_STRIKETHROUGH_CORPUS_DIR=... or pass the "
              "path as an argument); each subfolder needs a *.pdf and a di-result.json.")
        return 1
    tot = hi = lo = 0
    print(f"{'document':28s} {'final':>6s} {'conf>=gate':>11s} {'conf<gate':>10s}")
    for d in docs:
        pdfs = glob.glob(os.path.join(d, "*.pdf"))
        di = os.path.join(d, "di-result.json")
        if not pdfs or not os.path.exists(di):
            continue
        with open(pdfs[0], "rb") as fh:
            pdf_bytes = fh.read()
        with open(di, encoding="utf-8") as fh:
            di_result = json.load(fh)
        res = st.detect_pdf(pdf_bytes, di_result=di_result, include_markdown=False)
        final = [w for w in res["words"] if w.get("final")]
        n_hi = sum(1 for w in final if (w.get("conf") or 0.0) >= CLEAN_CONF)
        n_lo = len(final) - n_hi
        tot += len(final); hi += n_hi; lo += n_lo
        print(f"{os.path.basename(d):28s} {len(final):6d} {n_hi:11d} {n_lo:10d}")
    print(f"{'TOTAL':28s} {tot:6d} {hi:11d} {lo:10d}")
    print(f"\ngate = ScanConfig.max_clean_conf = {CLEAN_CONF}. 'conf>=gate' is the high-confidence "
          "clean-text\nover-flag the 0.9.1 veto targets; 'conf<gate' is the faint-scan "
          "CNN-saturation residual\ntracked for a post-1.0 model retrain (issue #4·C/D).")
    return 0


if __name__ == "__main__":
    target = (sys.argv[1] if len(sys.argv) > 1
              else os.environ.get("PDF_STRIKETHROUGH_CORPUS_DIR", DEFAULT_DIR))
    raise SystemExit(run(target))
