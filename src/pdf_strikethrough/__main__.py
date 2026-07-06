"""CLI: `pdf-strikethrough detect FILE [--ocr rapidocr] [--json out.json]`.

FILE may be a PDF, a raster image (``.png/.jpg/.tiff``, incl. multi-page TIFF), or a Word
``.docx`` — the input kind is inferred from the extension. Native PDFs need nothing extra;
scanned PDFs and image files need an OCR backend (``--ocr rapidocr`` / ``--ocr tesseract``) or a
pre-fetched cloud result (``--di-result`` Azure DI, ``--textract-result`` AWS Textract,
``--docai-result`` Google DocAI); without any, scanned pages are skipped with a warning while
native pages are still fully processed. A ``.docx`` reads strike formatting + tracked deletions
straight from the markup (no OCR).

Exit codes:
  0  success
  1  usage / file error (no such file, unreadable, bad --pages)
  2  encrypted PDF, or a scanned page with no OCR backend / cloud result
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


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _open_out(path):
    """Return a text file handle for `path`, or stdout when path is '-'. The caller closes it
    (a no-op contextmanager wraps stdout so `with` never closes the real stdout)."""
    if path == "-":
        return contextlib.nullcontext(sys.stdout)
    return open(path, "w", encoding="utf-8")


# evidence fields surfaced in --json — present-when-set, so a consumer sees *why* a word was
# flagged (native coverage/forensics, scanned score/cnn, docx change/author), not just that it was.
_JSON_EVIDENCE = ("page", "para", "text", "chars", "char_span", "partial", "bbox_frac", "tier",
                  "verdict", "coverage", "stroke_color", "stroke_width",
                  "annot_author", "annot_created", "annot_modified", "annot_color", "annot_id",
                  "docx_change", "docx_double", "docx_author", "docx_date", "docx_id",
                  "score", "cnn_prob", "cnn_agrees", "conf")


def _cmd_detect(args):
    import pdf_strikethrough as st

    if args.pdf != "-" and not os.path.exists(args.pdf):
        print(f"error: no such file: {args.pdf}", file=sys.stderr)
        return 1

    ext = os.path.splitext(args.pdf)[1].lower() if args.pdf != "-" else ""
    if ext == ".docx":
        return _cmd_detect_docx(args)
    is_image = ext in st.detect.IMAGE_SUFFIXES

    try:
        page_subset = _parse_pages(args.pages) if args.pages else None
    except ValueError as e:
        print(f"error: --pages: {e}", file=sys.stderr)
        return 1

    # Word sources: at most one pre-fetched cloud result, else the --ocr backend.
    di_result = words_by_page = None
    for flag, path, adapter in (("--di-result", args.di_result, None),
                                ("--textract-result", args.textract_result, st.words_from_textract),
                                ("--docai-result", args.docai_result, st.words_from_docai)):
        if not path:
            continue
        try:
            data = _load_json(path)
        except (OSError, json.JSONDecodeError) as e:
            print(f"error: {flag}: cannot read {path}: {e}", file=sys.stderr)
            return 1
        if adapter is None:
            di_result = data
        else:
            words_by_page = adapter(data)

    ocr = _build_ocr(args.ocr)
    # ScanConfig: honor an explicit --scan-config; otherwise pick the calibration that matches the
    # word source. Azure-DI confidences match the default calibration; RapidOCR/Tesseract/Textract/
    # DocAI do not, so those run confidence-free and let geometry + the CNN decide.
    if args.scan_config == "azure-di":
        scan_config = st.ScanConfig.azure_di()
    elif args.scan_config == "confidence-free":
        scan_config = st.ScanConfig.confidence_free()
    elif di_result is not None:
        scan_config = st.ScanConfig.azure_di()
    elif words_by_page is not None or args.ocr != "none":
        scan_config = st.ScanConfig.confidence_free()
    else:
        scan_config = st.ScanConfig()

    if is_image:
        if page_subset is not None:
            print("warning: --pages does not apply to an image file; ignored", file=sys.stderr)
        try:
            res = st.detect_image_file(args.pdf, ocr=ocr, words_by_page=words_by_page,
                                       scan_config=scan_config, dpi=args.dpi)
        except st.OcrRequiredError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        res["source"] = args.pdf
        return _emit_result(args, res, overlay_source=None)   # overlay needs a PDF page renderer

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

    # no-word-source on an uncovered scanned page raises with 'raise'; skip when the user opted out
    # of OCR entirely (--ocr none and no cloud result), else let it raise so the gap is loud.
    have_words = di_result is not None or words_by_page is not None
    on_missing_ocr = "skip" if (args.ocr == "none" and not have_words) else "raise"
    try:
        res = st.detect_pdf(doc, ocr=ocr, scan_config=scan_config,
                            dpi=args.dpi if args.dpi is not None else 200,
                            method=args.method, di_result=di_result, words_by_page=words_by_page,
                            pages=page_subset, progress=progress,
                            on_missing_ocr=on_missing_ocr)
    except st.OcrRequiredError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        doc.close()
    # open_pdf handed detect_pdf a doc, so re-attach a human-facing source label.
    res["source"] = "<stdin>" if args.pdf == "-" else args.pdf
    return _emit_result(args, res, overlay_source=pdf_source)


def _emit_result(args, res, overlay_source):
    """Write the requested outputs for a detect result (PDF or image) and return the exit code.
    `overlay_source` is the input to re-render for --overlay, or None to skip it (image inputs)."""
    import pdf_strikethrough as st
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
    if args.provenance:
        with _open_out(args.provenance) as f:
            f.write(st.provenance_text(res))
        if args.provenance != "-":
            print(f"wrote audit-preserving (provenance) text to {args.provenance}")
    if args.overlay:
        if overlay_source is None:
            print("warning: --overlay is only supported for PDF input; skipped", file=sys.stderr)
        else:
            from . import overlay as _ov
            # reuse the results already computed; render reopens the source (doc handle is closed)
            written = _ov.save_overlays(overlay_source, args.overlay, result=res,
                                        dpi=args.overlay_dpi)
            print(f"wrote {len(written)} overlay image(s)" +
                  (f" to {args.overlay}" if written else " (no struck pages)"))
    if args.json:
        payload = {"schema_version": SCHEMA_VERSION, "source": res["source"],
                   "page_count": res["page_count"], "page_sources": res["page_sources"],
                   "n_struck_final": len(final), "warnings": res.get("warnings", []),
                   "passages": res.get("passages", []),
                   "words": [{k: w[k] for k in _JSON_EVIDENCE if k in w} for w in final]}
        if "pages" in res:
            payload["pages"] = res["pages"]
        with _open_out(args.json) as f:
            json.dump(payload, f, indent=2, default=list, ensure_ascii=False)
            if args.json == "-":
                f.write("\n")
        if args.json != "-":
            print(f"wrote {len(final)} struck words to {args.json}")

    if not (args.json or args.markdown or args.clean_text or args.overlay or args.provenance):
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


def _cmd_detect_docx(args):
    """Detect strikethroughs in a .docx (strike formatting + tracked deletions). No OCR/geometry,
    so PDF/image-only flags don't apply."""
    import pdf_strikethrough as st
    for val, flag in ((args.ocr != "none", "--ocr"), (args.overlay, "--overlay"),
                      (args.markdown, "--markdown"), (args.clean_text, "--clean-text"),
                      (args.provenance, "--provenance")):
        if val:
            print(f"warning: {flag} does not apply to a .docx; ignored", file=sys.stderr)
    try:
        with open(args.pdf, "rb") as f:
            recs = st.strikethroughs_in_docx(f.read())
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    final = [r for r in recs if r.get("final")]

    if args.json:
        payload = {"schema_version": SCHEMA_VERSION, "source": args.pdf, "kind": "docx",
                   "n_struck_final": len(final),
                   "words": [{k: r[k] for k in _JSON_EVIDENCE if k in r} for r in recs]}
        with _open_out(args.json) as f:
            json.dump(payload, f, indent=2, default=list, ensure_ascii=False)
            if args.json == "-":
                f.write("\n")
        if args.json != "-":
            print(f"wrote {len(final)} struck runs to {args.json}")
    else:
        print(f"{args.pdf}: {len(final)} struck run(s) (docx)")
        for r in final[: args.limit]:
            print(f"  para{r['para']:<3} {r['docx_change']:<8} {r.get('chars')!r}")
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
    d.add_argument("pdf", metavar="FILE",
                   help="path to a PDF, image (.png/.jpg/.tiff), or .docx ('-' reads a PDF "
                        "from stdin)")
    d.add_argument("--ocr", default="none", choices=["none", "rapidocr", "tesseract"],
                   help="OCR backend for scanned pages / image files (none = native pages only)")
    d.add_argument("--di-result", dest="di_result", metavar="PATH",
                   help="pre-fetched Azure Document Intelligence analyze result (JSON); used "
                        "instead of an OCR backend for scanned pages")
    d.add_argument("--textract-result", dest="textract_result", metavar="PATH",
                   help="pre-fetched AWS Textract result (JSON); used instead of an OCR backend")
    d.add_argument("--docai-result", dest="docai_result", metavar="PATH",
                   help="pre-fetched Google Document AI result (JSON); used instead of an OCR "
                        "backend")
    d.add_argument("--scan-config", dest="scan_config", default="auto",
                   choices=["auto", "azure-di", "confidence-free"],
                   help="scanned-page calibration ('auto' picks azure-di with --di-result, "
                        "confidence-free with --ocr, else the default)")
    d.add_argument("--dpi", type=int, default=None,
                   help="raster DPI for scanned pages (default: 200 for PDFs; read from the image "
                        "metadata for image files, else 200)")
    d.add_argument("--pages", metavar="SPEC",
                   help="process only these 1-based pages, e.g. '1-5,12' (default: all)")
    d.add_argument("--method", default="vector", choices=["vector", "flag", "annot", "both"],
                   help="native-page detector: vector geometry, MuPDF strikeout flag, explicit "
                        "/StrikeOut annotations, or the union of all three")
    d.add_argument("--limit", type=int, default=25, help="max words to print (plain output)")
    d.add_argument("--json", metavar="PATH", help="write full results as JSON ('-' = stdout)")
    d.add_argument("--markdown", metavar="PATH",
                   help="write struck-aware markdown (~~deleted~~) ('-' = stdout)")
    d.add_argument("--clean-text", dest="clean_text", metavar="PATH",
                   help="write the surviving text with deletions removed ('-' = stdout)")
    d.add_argument("--provenance", metavar="PATH",
                   help="write audit-preserving text: deletions kept as '[deleted: ...]' markers "
                        "instead of removed (for RAG/indexing) ('-' = stdout)")
    d.add_argument("--overlay", metavar="PATH",
                   help="write page images with the detected strikes boxed (red=full, "
                        "orange=partial); PATH is a directory, or a filename prefix if it ends in "
                        "an image extension. One image per struck page")
    d.add_argument("--overlay-dpi", dest="overlay_dpi", type=int, default=150,
                   help="render DPI for --overlay images")
    d.add_argument("--fail-if-found", dest="fail_if_found", action="store_true",
                   help="exit 3 if any struck word is found (for CI gating)")
    d.set_defaults(func=_cmd_detect)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
