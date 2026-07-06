"""Batch-mode helpers for the CLI (R-batch): per-file detection + the multiprocessing worker.

These live in a regular importable module, NOT in ``__main__``: a ``ProcessPoolExecutor`` worker is
pickled by its qualified name, and under ``python -m pdf_strikethrough`` the ``__main__`` module is
re-registered so a worker defined there can't be resolved in the spawned child. Defined here it
pickles the same way whether the CLI was launched via the console script or ``-m``.

The single-file CLI (``__main__``) imports ``SCHEMA_VERSION`` / ``_JSON_EVIDENCE`` / the OCR builders
from here too, so the shared payload shape has one home.
"""
from __future__ import annotations

import os

SCHEMA_VERSION = 1                 # bump when the --json / --jsonl payload shape changes

# evidence fields surfaced in --json/--jsonl — present-when-set, so a consumer sees *why* a word was
# flagged (native coverage/forensics, scanned score/cnn, docx change/author), not just that it was.
_JSON_EVIDENCE = ("page", "para", "text", "chars", "char_span", "partial", "bbox_frac", "tier",
                  "verdict", "coverage", "stroke_color", "stroke_width",
                  "annot_author", "annot_created", "annot_modified", "annot_color", "annot_id",
                  "docx_change", "docx_double", "docx_author", "docx_date", "docx_id",
                  "score", "cnn_prob", "cnn_agrees", "conf")


def _check_ocr_available(name):
    """Raise SystemExit with the pip hint if the chosen OCR extra isn't importable. Split from
    `_build_ocr` so batch mode can validate the extra once in the parent (before spawning workers)
    without paying to build a backend the parent won't use."""
    if name == "rapidocr":
        try:
            import rapidocr  # noqa: F401 — eager check so the error comes before any work
        except ImportError:
            raise SystemExit('--ocr rapidocr requires the rapidocr extra: '
                             'pip install "pdf-strikethrough-detect[rapidocr]"')
    elif name == "tesseract":
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            raise SystemExit('--ocr tesseract requires the tesseract extra: '
                             'pip install "pdf-strikethrough-detect[tesseract]" '
                             '(plus the tesseract system binary)')


def _build_ocr(name):
    if name == "none":
        return None
    _check_ocr_available(name)
    if name == "rapidocr":
        from .ocr import rapidocr_backend
        return rapidocr_backend()
    if name == "tesseract":
        from .ocr import tesseract_backend
        return tesseract_backend()
    # unreachable: argparse `choices` already rejects any other --ocr value before we get here.


_OCR_CACHE = {}


def _get_ocr_cached(name):
    """Build the OCR backend once per process; batch workers reuse it across every file they get."""
    if name not in _OCR_CACHE:
        _OCR_CACHE[name] = _build_ocr(name)
    return _OCR_CACHE[name]


def _batch_scan_config(opts):
    import pdf_strikethrough as st
    if opts["scan_config"] == "azure-di":
        return st.ScanConfig.azure_di()
    if opts["scan_config"] == "confidence-free" or opts["ocr"] != "none":
        return st.ScanConfig.confidence_free()
    return st.ScanConfig()


def _detect_payload(path, opts):
    """Detect on one file and return a JSON-serializable payload (or ``{'source', 'error'}``). Never
    raises — a batch of many files must not abort on one unreadable member (cloud-result flags,
    which are single-result, don't apply in batch, so this path only uses the --ocr backend)."""
    import pdf_strikethrough as st
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".docx":
            with open(path, "rb") as f:
                recs = st.strikethroughs_in_docx(f.read())
            final = [r for r in recs if r.get("final")]
            return {"schema_version": SCHEMA_VERSION, "source": path, "kind": "docx",
                    "n_struck_final": len(final),
                    "words": [{k: r[k] for k in _JSON_EVIDENCE if k in r} for r in recs]}
        ocr = _get_ocr_cached(opts["ocr"])
        sc = _batch_scan_config(opts)
        if ext in st.detect.IMAGE_SUFFIXES:
            res = st.detect_image_file(path, ocr=ocr, scan_config=sc, dpi=opts["dpi"])
        else:
            res = st.detect_pdf(path, ocr=ocr, scan_config=sc,
                                dpi=opts["dpi"] if opts["dpi"] is not None else 200,
                                method=opts["method"],
                                on_missing_ocr="skip" if opts["ocr"] == "none" else "raise")
        final = [w for w in res["words"] if w.get("final")]
        return {"schema_version": SCHEMA_VERSION, "source": path,
                "page_count": res["page_count"], "page_sources": res["page_sources"],
                "n_struck_final": len(final), "warnings": res.get("warnings", []),
                "words": [{k: w[k] for k in _JSON_EVIDENCE if k in w} for w in final]}
    except Exception as e:               # noqa: BLE001 — batch resilience: report, don't abort the run
        return {"source": path, "error": f"{type(e).__name__}: {e}"}


def _batch_worker(item):
    """Top-level (picklable) worker for the process pool: (path, opts) -> payload dict."""
    return _detect_payload(*item)
