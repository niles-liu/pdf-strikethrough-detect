"""CLI: `pdf-strikethrough detect FILE.pdf [--ocr rapidocr] [--json out.json]`.

Native PDFs need nothing extra. Scanned PDFs need an OCR backend (``--ocr rapidocr`` or
``--ocr tesseract``), or a pre-fetched Azure DI result (``--di-result result.json``); without
either, scanned pages are skipped with a warning while native pages are still fully processed.

Exit codes:
  0  success
  1  usage / file error (no such file, not a PDF, bad --pages)
  2  encrypted PDF, or a scanned page with no OCR backend / di_result
  3  --fail-if-found and at least one struck word was found (for CI gating)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

SCHEMA_VERSION = 1                 # bump when the --json payload shape changes


def _build_ocr(name):
    if name == "none":
        return None
    if name == "rapidocr":
        try:
            import rapidocr  # noqa: F401 — eager check so the error comes before any work
        except ImportError:
            raise SystemExit('--ocr rapidocr requires the rapidocr extra: '
                             'pip install "pdf-strikethrough-detect[rapidocr]"')
        from .ocr import rapidocr_backend
        return rapidocr_backend()
    if name == "tesseract":
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            raise SystemExit('--ocr tesseract requires the tesseract extra: '
                             'pip install "pdf-strikethrough-detect[tesseract]" '
                             '(plus the tesseract system binary)')
        from .ocr import tesseract_backend
        return tesseract_backend()
    # unreachable: argparse `choices` already rejects any other --ocr value before we get here.


def _parse_pages(spec):
    """'1-5,12,20-22' (1-based, inclusive) -> sorted 0-based list. Raises ValueError on garbage."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):                  # a range like 1-5 (not a bare negative)
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo < 1 or hi < lo:
                raise ValueError(f"bad page range {part!r} (use 1-based ascending, e.g. 1-5)")
            out.update(range(lo - 1, hi))
        else:
            n = int(part)
            if n < 1:
                raise ValueError(f"bad page number {part!r} (pages are 1-based)")
            out.add(n - 1)
    if not out:
        raise ValueError("no pages selected")
    return sorted(out)


def _load_di_result(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _open_out(path):
    """Return a text file handle for `path`, or stdout when path is '-'. The caller closes it
    (a no-op contextmanager wraps stdout so `with` never closes the real stdout)."""
    if path == "-":
        return contextlib.nullcontext(sys.stdout)
    return open(path, "w", encoding="utf-8")


def _cmd_detect(args):
    import pdf_strikethrough as st

    if args.pdf != "-" and not os.path.exists(args.pdf):
        print(f"error: no such file: {args.pdf}", file=sys.stderr)
        return 1

    try:
        page_subset = _parse_pages(args.pages) if args.pages else None
    except ValueError as e:
        print(f"error: --pages: {e}", file=sys.stderr)
        return 1

    di_result = None
    if args.di_result:
        try:
            di_result = _load_di_result(args.di_result)
        except (OSError, json.JSONDecodeError) as e:
            print(f"error: --di-result: cannot read {args.di_result}: {e}", file=sys.stderr)
            return 1

    ocr = _build_ocr(args.ocr)
    # ScanConfig: honor an explicit --scan-config; otherwise pick the calibration that matches the
    # source of words. Azure-DI confidences match the default calibration; neither RapidOCR nor
    # Tesseract do, so those run confidence-free and let geometry + the CNN decide.
    if args.scan_config == "azure-di":
        scan_config = st.ScanConfig.azure_di()
    elif args.scan_config == "confidence-free":
        scan_config = st.ScanConfig.confidence_free()
    elif di_result is not None:
        scan_config = st.ScanConfig.azure_di()
    elif args.ocr != "none":
        scan_config = st.ScanConfig.confidence_free()
    else:
        scan_config = st.ScanConfig()

    pdf_source = sys.stdin.buffer.read() if args.pdf == "-" else args.pdf
    # Open (and gate encryption) as its own step so its error handling doesn't swallow mid-run
    # RuntimeErrors from onnxruntime / PyMuPDF and mislabel them "cannot open FILE".
    try:
        doc = st.open_pdf(pdf_source)
    except st.EncryptedPdfError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:                # pymupdf file errors (corrupt/truncated/not a PDF)
        print(f"error: cannot open {args.pdf}: {e}", file=sys.stderr)
        return 1

    # Per-page progress on stderr when it's a TTY (long OCR+CNN runs otherwise read as a hang).
    def _progress(done, total, pno):
        print(f"\r  page {done}/{total} (p{pno})...", end="", file=sys.stderr, flush=True)
        if done == total:
            print("", file=sys.stderr, flush=True)          # newline after the last page
    progress = _progress if sys.stderr.isatty() else None

    # di_result-but-no-ocr on an uncovered scanned page raises with 'raise'; skip when the user
    # opted out of OCR entirely (--ocr none and no di_result), else let it raise so the gap is loud.
    on_missing_ocr = "skip" if (args.ocr == "none" and di_result is None) else "raise"
    try:
        res = st.detect_pdf(doc, ocr=ocr, scan_config=scan_config, dpi=args.dpi,
                            native_method=args.method, di_result=di_result,
                            pages=page_subset, progress=progress,
                            on_missing_ocr=on_missing_ocr)
    except st.OcrRequiredError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        doc.close()
    # open_pdf handed detect_pdf a doc, so re-attach a human-facing source label.
    res["source"] = "<stdin>" if args.pdf == "-" else args.pdf

    for w in res.get("warnings", []):
        print(f"warning: {w}", file=sys.stderr)
    final = [w for w in res["words"] if w.get("final")]

    if args.markdown:
        with _open_out(args.markdown) as f:
            f.write(res.get("markdown", ""))
        if args.markdown != "-":
            print(f"wrote struck-aware markdown to {args.markdown}")
    if args.clean_text:
        with _open_out(args.clean_text) as f:
            f.write(res.get("clean_text", ""))
        if args.clean_text != "-":
            print(f"wrote surviving clean text to {args.clean_text}")
    if args.json:
        # evidence fields (coverage on native, score/cnn_prob on scanned) are included when present
        # so a JSON consumer can see *why* a word was flagged, not just that it was.
        evidence = ("page", "text", "chars", "char_span", "partial", "bbox_frac", "tier",
                    "verdict", "coverage", "score", "cnn_prob", "cnn_agrees", "conf")
        payload = {"schema_version": SCHEMA_VERSION, "source": res["source"],
                   "page_count": res["page_count"], "page_sources": res["page_sources"],
                   "n_struck_final": len(final), "warnings": res.get("warnings", []),
                   "passages": res.get("passages", []),
                   "words": [{k: w[k] for k in evidence if k in w} for w in final]}
        if "pages" in res:
            payload["pages"] = res["pages"]
        with _open_out(args.json) as f:
            json.dump(payload, f, indent=2, default=list, ensure_ascii=False)
            if args.json == "-":
                f.write("\n")
        if args.json != "-":
            print(f"wrote {len(final)} struck words to {args.json}")

    if not (args.json or args.markdown or args.clean_text):
        print(f"{res['source']}: {res['page_count']} pages "
              f"({', '.join(sorted(set(res['page_sources'])))}), "
              f"{len(final)} struck words in {len(res.get('passages', []))} passages")
        for w in final[: args.limit]:
            kind = "partial" if w.get("partial") else "full"
            print(f"  p{w['page']:<3} {kind:<7} {w.get('chars', w.get('text'))!r}")
        if len(final) > args.limit:
            print(f"  ... and {len(final) - args.limit} more (use --json to dump all)")

    if args.fail_if_found and final:
        return 3
    return 0


def main(argv=None):
    # Console output can carry non-cp1252 chars (struck ﬁ, é, → ...). Never let a
    # print(...!r) crash on a narrow console encoding — replace unencodable chars instead.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass                             # non-reconfigurable stream (e.g. captured in tests)
    p = argparse.ArgumentParser(prog="pdf-strikethrough",
                                description="Detect struck-through text in PDFs.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {__import__('pdf_strikethrough').__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="detect strikethroughs in a PDF",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                       epilog="exit codes: 0 ok, 1 usage/file error, 2 encrypted / OCR required, "
                              "3 --fail-if-found matched")
    d.add_argument("pdf", help="path to the PDF ('-' reads from stdin)")
    d.add_argument("--ocr", default="none", choices=["none", "rapidocr", "tesseract"],
                   help="OCR backend for scanned pages (none = native pages only)")
    d.add_argument("--di-result", dest="di_result", metavar="PATH",
                   help="pre-fetched Azure Document Intelligence analyze result (JSON); used "
                        "instead of an OCR backend for scanned pages")
    d.add_argument("--scan-config", dest="scan_config", default="auto",
                   choices=["auto", "azure-di", "confidence-free"],
                   help="scanned-page calibration ('auto' picks azure-di with --di-result, "
                        "confidence-free with --ocr, else the default)")
    d.add_argument("--dpi", type=int, default=200, help="raster DPI for scanned pages")
    d.add_argument("--pages", metavar="SPEC",
                   help="process only these 1-based pages, e.g. '1-5,12' (default: all)")
    d.add_argument("--method", default="vector", choices=["vector", "flag", "both"],
                   help="native-page detector: vector geometry, MuPDF strikeout flag, or the "
                        "union of both")
    d.add_argument("--limit", type=int, default=25, help="max words to print (plain output)")
    d.add_argument("--json", metavar="PATH", help="write full results as JSON ('-' = stdout)")
    d.add_argument("--markdown", metavar="PATH",
                   help="write struck-aware markdown (~~deleted~~) ('-' = stdout)")
    d.add_argument("--clean-text", dest="clean_text", metavar="PATH",
                   help="write the surviving text with deletions removed ('-' = stdout)")
    d.add_argument("--fail-if-found", dest="fail_if_found", action="store_true",
                   help="exit 3 if any struck word is found (for CI gating)")
    d.set_defaults(func=_cmd_detect)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
