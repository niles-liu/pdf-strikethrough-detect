"""CLI: `pdf-strikethrough detect FILE.pdf [--ocr rapidocr] [--json out.json]`.

Native PDFs need nothing extra. Scanned PDFs need an OCR backend (``--ocr rapidocr`` or
``--ocr tesseract``); without one, scanned pages are skipped with a warning while native
pages are still fully processed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


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
    raise SystemExit(f"unknown --ocr backend: {name!r} (choose rapidocr, tesseract, or none)")


def _cmd_detect(args):
    import pdf_strikethrough as st

    if not os.path.exists(args.pdf):
        print(f"error: no such file: {args.pdf}", file=sys.stderr)
        return 1
    ocr = _build_ocr(args.ocr)
    # neither RapidOCR nor Tesseract confidences match the Azure-DI calibration the default
    # ScanConfig encodes — run confidence-free and let geometry + the CNN decide. API users with
    # a calibrated engine can pass their own ScanConfig.
    scan_config = st.ScanConfig.confidence_free() if args.ocr != "none" else st.ScanConfig()
    try:
        res = st.detect_pdf(args.pdf, ocr=ocr, scan_config=scan_config, dpi=args.dpi,
                            native_method=args.method,
                            on_missing_ocr="skip" if args.ocr == "none" else "raise")
    except st.EncryptedPdfError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except st.OcrRequiredError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:               # pymupdf file errors (corrupt/truncated/not a PDF)
        print(f"error: cannot open {args.pdf}: {e}", file=sys.stderr)
        return 1

    for w in res.get("warnings", []):
        print(f"warning: {w}", file=sys.stderr)
    final = [w for w in res["words"] if w.get("final")]
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(res.get("markdown", ""))
        print(f"wrote struck-aware markdown to {args.markdown}")
    if args.clean_text:
        with open(args.clean_text, "w") as f:
            f.write(res.get("clean_text", ""))
        print(f"wrote surviving clean text to {args.clean_text}")
    if args.json:
        payload = {"source": res["source"], "page_count": res["page_count"],
                   "page_sources": res["page_sources"], "n_struck_final": len(final),
                   "passages": res.get("passages", []),
                   "words": [{k: w[k] for k in ("page", "text", "chars", "char_span",
                                                "partial", "bbox_frac", "tier", "verdict")
                              if k in w} for w in final]}
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, default=list)
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
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="pdf-strikethrough",
                                description="Detect struck-through text in PDFs.")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="detect strikethroughs in a PDF")
    d.add_argument("pdf", help="path to the PDF")
    d.add_argument("--ocr", default="none", choices=["none", "rapidocr", "tesseract"],
                   help="OCR backend for scanned pages (default: none = native pages only)")
    d.add_argument("--dpi", type=int, default=200, help="raster DPI for scanned pages")
    d.add_argument("--method", default="vector", choices=["vector", "flag", "both"],
                   help="native-page detector: vector geometry (default), MuPDF strikeout flag, "
                        "or the union of both")
    d.add_argument("--limit", type=int, default=25, help="max words to print (plain output)")
    d.add_argument("--json", metavar="PATH", help="write full results to this JSON file")
    d.add_argument("--markdown", metavar="PATH", help="write struck-aware markdown (~~deleted~~)")
    d.add_argument("--clean-text", dest="clean_text", metavar="PATH",
                   help="write the surviving text with deletions removed")
    d.set_defaults(func=_cmd_detect)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
