"""CLI: `pdf-strikethrough detect FILE [--ocr rapidocr] [--json out.json]`.

FILE may be a PDF, a raster image (``.png/.jpg/.tiff``, incl. multi-page TIFF), or a Word
``.docx`` — the input kind is inferred from the extension. Native PDFs need nothing extra;
scanned PDFs and image files need an OCR backend (``--ocr rapidocr`` / ``--ocr tesseract``) or a
pre-fetched cloud result (``--di-result`` Azure DI, ``--textract-result`` AWS Textract,
``--docai-result`` Google DocAI); without any, scanned pages are skipped with a warning while
native pages are still fully processed. A ``.docx`` reads strike formatting + tracked deletions
straight from the markup (no OCR).

Batch: pass several files, a directory, or a glob (`detect *.pdf --jobs 4 --jsonl out.jsonl`), or
ask for ``--jsonl``, and each file is processed independently — JSONL output (one result object
per line, written as each file finishes), optional multiprocessing across files (``--jobs N``),
and one bad file never aborts the run (its line carries an ``error`` key). The mode follows the
form of the arguments, not the number of files they match: a directory holding one file is still
a batch. Per-file output flags (--markdown/--overlay/cloud results/--pages) are single-file only.

Results go to stdout (UTF-8 when written to '-') and status lines to stderr, so `--json -` and
`--markdown -` pipe cleanly.

Exit codes:
  0  success
  1  usage / file error (no such file, unreadable, bad --pages; in batch: >=1 file errored)
  2  encrypted PDF, or a scanned page with no OCR backend / cloud result
  3  --fail-if-found and at least one struck word was found (for CI gating)
  130  interrupted (Ctrl-C); in batch, the lines already written are kept
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

from ._batch import (SCHEMA_VERSION, _batch_worker, _build_ocr, _check_ocr_available,
                     _detect_payload, _JSON_EVIDENCE)


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


def _positive_int(text):
    """argparse type for --dpi / --jobs style values: a positive integer."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def _open_out(path):
    """Return a text file handle for `path`, or stdout when path is '-'. The caller closes it
    (a no-op contextmanager wraps stdout so `with` never closes the real stdout). Output to '-' is
    UTF-8 whatever the console code page."""
    if path == "-":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # keep main()'s handler
        except (AttributeError, ValueError):
            pass                             # non-reconfigurable stream (e.g. captured in tests)
        return contextlib.nullcontext(sys.stdout)
    return open(path, "w", encoding="utf-8")


def _is_glob(arg):
    """A FILE arg is a pattern when it has glob characters and is not itself an existing path
    ('report [final].pdf' is a file, not a character class)."""
    return any(c in arg for c in "*?[") and not os.path.exists(arg)


def _expand_inputs(patterns):
    """Expand the FILE args into a sorted, de-duplicated list of input paths. Each arg may be a
    literal file, a directory (its top-level PDFs / images / .docx), or a glob (``*.pdf``, and
    ``**`` recurses — expanded here too, so it works on shells that don't glob, e.g. Windows); a
    glob keeps only the files of those kinds, like a directory. '-' (stdin) passes through, alone.
    Returns None after printing an error when an arg matches nothing."""
    import glob as _glob

    import pdf_strikethrough as st
    exts = (".pdf", ".docx", *st.detect.IMAGE_SUFFIXES)
    if "-" in patterns and len(patterns) > 1:
        print("error: '-' reads one PDF from stdin and cannot be combined with other inputs",
              file=sys.stderr)
        return None
    out = []
    for pat in patterns:
        if pat == "-":
            out.append("-")
        elif os.path.isdir(pat):
            found = sorted(f for f in (os.path.join(pat, n) for n in os.listdir(pat))
                           if os.path.isfile(f) and os.path.splitext(f)[1].lower() in exts)
            if not found:
                print(f"error: no PDF/image/.docx files in directory {pat}", file=sys.stderr)
                return None
            out.extend(found)
        elif _is_glob(pat):
            hits = sorted(f for f in _glob.glob(pat, recursive=True)
                          if os.path.isfile(f) and os.path.splitext(f)[1].lower() in exts)
            if not hits:
                print(f"error: no PDF/image/.docx files match {pat}", file=sys.stderr)
                return None
            out.extend(hits)
        elif os.path.exists(pat):
            out.append(pat)
        else:
            print(f"error: no such file: {pat}", file=sys.stderr)
            return None
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _cmd_detect(args):
    """Dispatch by the form of the arguments, not the number of files matched: one literal file
    -> full single-file mode (every output flag); several files, a directory, a glob, or --jsonl
    -> batch mode (JSONL, optional --jobs parallelism)."""
    inputs = _expand_inputs(args.files)
    if inputs is None:
        return 1
    batch = (args.jsonl is not None or len(args.files) > 1
             or any(os.path.isdir(f) or _is_glob(f) for f in args.files))
    if batch and inputs == ["-"]:
        print("error: --jsonl takes file inputs; '-' (stdin) is single-file only", file=sys.stderr)
        return 1
    if not batch:
        return _cmd_detect_single(args, inputs[0])
    return _cmd_detect_batch(args, inputs)


def _cmd_detect_single(args, path):
    import pdf_strikethrough as st

    ext = os.path.splitext(path)[1].lower() if path != "-" else ""
    if ext == ".docx":
        return _cmd_detect_docx(args, path)
    is_image = ext in st.detect.IMAGE_SUFFIXES

    try:
        page_subset = _parse_pages(args.pages) if args.pages else None
    except ValueError as e:
        print(f"error: --pages: {e}", file=sys.stderr)
        return 1

    # Word sources: at most one pre-fetched cloud result, else the --ocr backend.
    given = [f for f, v in (("--di-result", args.di_result),
                            ("--textract-result", args.textract_result),
                            ("--docai-result", args.docai_result)) if v]
    if len(given) > 1:
        print(f"error: pass at most one cloud result ({' / '.join(given)} given)", file=sys.stderr)
        return 1
    di_result = words_by_page = None
    for flag, res_path, adapter in (("--di-result", args.di_result, None),
                                    ("--textract-result", args.textract_result,
                                     st.words_from_textract),
                                    ("--docai-result", args.docai_result, st.words_from_docai)):
        if not res_path:
            continue
        try:
            data = _load_json(res_path)
        except (OSError, json.JSONDecodeError) as e:
            print(f"error: {flag}: cannot read {res_path}: {e}", file=sys.stderr)
            return 1
        if adapter is None:
            di_result = data
        else:
            try:
                words_by_page = adapter(data)
            except (ValueError, TypeError, AttributeError) as e:
                print(f"error: {flag}: {res_path} is not a result this adapter reads: {e}",
                      file=sys.stderr)
                return 1

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

    if args.dump_crops:
        from . import active
        dc_source = sys.stdin.buffer.read() if path == "-" else path
        try:
            summary = active.dump_crops(
                dc_source, args.dump_crops, ocr=ocr, scan_config=scan_config, dpi=args.dpi,
                di_result=di_result, words_by_page=words_by_page,
                pages=None if is_image else page_subset, image=is_image)
        except (st.OcrRequiredError, st.EncryptedPdfError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except (OSError, RuntimeError, ValueError, IndexError) as e:
            print(f"error: {path}: {e}", file=sys.stderr)
            return 1
        print(f"wrote {summary['n_crops']} crop(s) to {summary['out_dir']} "
              f"(manifest: {summary['manifest']})", file=sys.stderr)
        return 0

    if is_image:
        from PIL import Image
        if page_subset is not None:
            print("warning: --pages does not apply to an image file; ignored", file=sys.stderr)
        try:
            res = st.detect_image_file(path, ocr=ocr, words_by_page=words_by_page,
                                       scan_config=scan_config, dpi=args.dpi)
        except st.OcrRequiredError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except (OSError, ValueError, Image.DecompressionBombError) as e:
            # unreadable / truncated / not an image / larger than PIL will decode
            print(f"error: cannot read {path}: {e}", file=sys.stderr)
            return 1
        res["source"] = path
        return _emit_result(args, res, overlay_source=None)   # overlay needs a PDF page renderer

    pdf_source = sys.stdin.buffer.read() if path == "-" else path
    # Open (and gate encryption) as its own step so its error handling doesn't swallow mid-run
    # RuntimeErrors from onnxruntime / PyMuPDF and mislabel them "cannot open FILE".
    try:
        doc = st.open_pdf(pdf_source)
    except st.EncryptedPdfError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:                # pymupdf file errors (corrupt/truncated/not a PDF)
        print(f"error: cannot open {path}: {e}", file=sys.stderr)
        return 1
    if page_subset is not None and page_subset[-1] >= doc.page_count:
        print(f"error: --pages: page {page_subset[-1] + 1} is past the end of the "
              f"{doc.page_count}-page document", file=sys.stderr)
        doc.close()
        return 1

    # Per-page progress on stderr when it's a TTY (long OCR+CNN runs otherwise read as a hang).
    def _progress(done, total, pno):
        print(f"\r  page {done}/{total} (p{pno + 1})...", end="", file=sys.stderr, flush=True)
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
    except ValueError as e:                  # a cloud result that does not fit this document
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        doc.close()
    # open_pdf handed detect_pdf a doc, so re-attach a human-facing source label.
    res["source"] = "<stdin>" if path == "-" else path
    return _emit_result(args, res, overlay_source=pdf_source)


def _emit_result(args, res, overlay_source):
    """Write the requested outputs for a detect result (PDF or image) and return the exit code.
    `overlay_source` is the input to re-render for --overlay, or None to skip it (image inputs).
    Status lines ("wrote ...") go to stderr, so they never mix with a result written to '-'."""
    import pdf_strikethrough as st
    for w in res.get("warnings", []):
        print(f"warning: {w}", file=sys.stderr)
    final = [w for w in res["words"] if w.get("final")]
    try:
        if args.markdown:
            with _open_out(args.markdown) as f:
                f.write(res.get("markdown", ""))
            if args.markdown != "-":
                print(f"wrote struck-aware markdown to {args.markdown}", file=sys.stderr)
        if args.clean_text:
            with _open_out(args.clean_text) as f:
                f.write(res.get("clean_text", ""))
            if args.clean_text != "-":
                print(f"wrote surviving clean text to {args.clean_text}", file=sys.stderr)
        if args.provenance:
            with _open_out(args.provenance) as f:
                f.write(st.provenance_text(res))
            if args.provenance != "-":
                print(f"wrote audit-preserving (provenance) text to {args.provenance}",
                      file=sys.stderr)
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
                print(f"wrote {len(final)} struck words to {args.json}", file=sys.stderr)
        if args.overlay:                     # last: a bad overlay path must not cost the text
            if overlay_source is None:
                print("warning: --overlay is only supported for PDF input; skipped",
                      file=sys.stderr)
            else:
                from . import overlay as _ov
                # reuse the results already computed; render reopens the source (doc is closed)
                written = _ov.save_overlays(overlay_source, args.overlay, result=res,
                                            dpi=args.overlay_dpi)
                print(f"wrote {len(written)} overlay image(s)" +
                      (f" to {args.overlay}" if written else " (no struck pages)"),
                      file=sys.stderr)
    except OSError as e:
        if "-" in (args.json, args.markdown, args.clean_text, args.provenance):
            _abandon_stdout()                # a closed pipe: do not fail again at exit
        print(f"error: cannot write output: {e}", file=sys.stderr)
        return 1

    if not (args.json or args.markdown or args.clean_text or args.overlay or args.provenance):
        print(f"{res['source']}: {res['page_count']} pages "
              f"({', '.join(sorted(set(res['page_sources'])))}), "
              f"{len(final)} struck words in {len(res.get('passages', []))} passages")
        for w in final[: args.limit]:
            kind = "partial" if w.get("partial") else "full"
            print(f"  p{w['page'] + 1:<3} {kind:<7} {w.get('chars', w.get('text'))!r}")
        if len(final) > args.limit:
            print(f"  ... and {len(final) - args.limit} more (use --json to dump all)")

    if args.fail_if_found and final:
        return 3
    return 0


def _cmd_detect_docx(args, path):
    """Detect strikethroughs in a .docx (strike formatting + tracked deletions). No OCR/geometry,
    so PDF/image-only flags don't apply."""
    import pdf_strikethrough as st
    for val, flag in ((args.ocr != "none", "--ocr"), (args.overlay, "--overlay"),
                      (args.markdown, "--markdown"), (args.clean_text, "--clean-text"),
                      (args.provenance, "--provenance"), (args.pages, "--pages"),
                      (args.di_result, "--di-result"), (args.textract_result, "--textract-result"),
                      (args.docai_result, "--docai-result"), (args.dump_crops, "--dump-crops")):
        if val:
            print(f"warning: {flag} does not apply to a .docx; ignored", file=sys.stderr)
    try:
        with open(path, "rb") as f:
            recs = st.strikethroughs_in_docx(f.read())
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    final = [r for r in recs if r.get("final")]

    if args.json:
        payload = {"schema_version": SCHEMA_VERSION, "source": path, "kind": "docx",
                   "n_struck_final": len(final),
                   "words": [{k: r[k] for k in _JSON_EVIDENCE if k in r} for r in recs]}
        try:
            with _open_out(args.json) as f:
                json.dump(payload, f, indent=2, default=list, ensure_ascii=False)
                if args.json == "-":
                    f.write("\n")
        except OSError as e:
            print(f"error: cannot write output: {e}", file=sys.stderr)
            return 1
        if args.json != "-":
            print(f"wrote {len(final)} struck runs to {args.json}", file=sys.stderr)
    else:
        print(f"{path}: {len(final)} struck run(s) (docx)")
        for r in final[: args.limit]:
            print(f"  para{r['para']:<3} {r['docx_change']:<8} {r.get('chars')!r}")
        if len(final) > args.limit:
            print(f"  ... and {len(final) - args.limit} more (use --json to dump all)")

    if args.fail_if_found and final:
        return 3
    return 0


# --- batch mode (R-batch): many files, optional multiprocessing, JSONL output -----------------
# The per-file worker + payload builder live in _batch.py (picklable under both the console script
# and `python -m`); this module only orchestrates output and the aggregate exit code.


_POLL_S = 0.2       # how often a batch wakes while waiting on a worker, so Ctrl-C is seen promptly


def _wait(fut):
    """``fut.result()``, waking every _POLL_S: a blocking wait on Windows notices Ctrl-C only when
    a result arrives."""
    from concurrent.futures import TimeoutError as FutureTimeout
    while True:
        try:
            return fut.result(timeout=_POLL_S)
        except FutureTimeout:
            continue


def _abandon_stdout():
    """After a write to a closed pipe (`... --jsonl - | head`), point stdout at the null device so
    the interpreter's own flush at exit does not raise again."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):
        pass                                 # no real file descriptor (e.g. captured in tests)


def _cmd_detect_batch(args, inputs):
    """Detect across many files. Writes JSONL (one payload per line, flushed as each file finishes)
    to --jsonl / --json, or prints a per-file summary; --jobs N spreads the files over N worker
    processes. The output is opened after the run's own checks and before any work. Exit: 130 if
    interrupted, else 3 if --fail-if-found matched any file, else 1 if any file errored or the
    output could not be written, else 0."""
    for val, flag in ((args.markdown, "--markdown"), (args.clean_text, "--clean-text"),
                      (args.provenance, "--provenance"), (args.overlay, "--overlay"),
                      (args.di_result, "--di-result"), (args.textract_result, "--textract-result"),
                      (args.docai_result, "--docai-result"), (args.pages, "--pages"),
                      (args.dump_crops, "--dump-crops")):
        if val:
            print(f"warning: {flag} is ignored in batch mode", file=sys.stderr)

    opts = {"ocr": args.ocr, "dpi": args.dpi, "method": args.method,
            "scan_config": args.scan_config}
    jobs = max(1, args.jobs)
    if sys.platform == "win32":
        jobs = min(jobs, 61)                 # the Windows ProcessPoolExecutor limit
    _check_ocr_available(args.ocr)          # a clean error before the output is truncated
    out_path = args.jsonl or args.json
    try:
        out = _open_out(out_path) if out_path else contextlib.nullcontext(None)
    except OSError as e:
        print(f"error: cannot write {out_path}: {e}", file=sys.stderr)
        return 1

    n = errors = hits = 0

    def emit(pl):                            # f is the open output, bound by the with below
        nonlocal n, errors, hits
        n += 1
        if f is not None:
            json.dump(pl, f, default=list, ensure_ascii=False)
            f.write("\n")
            f.flush()
        if "error" in pl:
            errors += 1
            print(f"error: {pl['source']}: {pl['error']}", file=sys.stderr)
        elif pl.get("n_struck_final"):
            hits += 1
        if not out_path:
            print(f"  {pl['source']}: ERROR" if "error" in pl else
                  f"  {pl['source']}: {pl.get('n_struck_final', 0)} struck "
                  f"({pl.get('page_count', '?')} pages)", flush=True)

    with out as f:
        try:
            if jobs <= 1:
                for path in inputs:
                    emit(_detect_payload(path, opts))
            else:
                from collections import deque
                from concurrent.futures import ProcessPoolExecutor
                from concurrent.futures.process import BrokenProcessPool
                pending = list(inputs)
                while pending:
                    # at most 2 * jobs files in flight, so an interrupt need not wait for the rest
                    ex = ProcessPoolExecutor(max_workers=jobs)
                    inflight, nxt, done, broke = deque(), 0, 0, False
                    try:
                        while done < len(pending):
                            while nxt < len(pending) and len(inflight) < 2 * jobs:
                                inflight.append(ex.submit(_batch_worker, (pending[nxt], opts)))
                                nxt += 1
                            emit(_wait(inflight.popleft()))
                            done += 1
                    except BrokenProcessPool:
                        broke = True
                    except KeyboardInterrupt:
                        # stop the workers now, or exit waits for every queued file (before
                        # Python 3.14's terminate_workers() only _processes reaches them)
                        if hasattr(ex, "terminate_workers"):
                            ex.terminate_workers()
                        else:
                            for proc in list((getattr(ex, "_processes", None) or {}).values()):
                                proc.terminate()
                        raise
                    finally:
                        for fut in inflight:
                            fut.cancel()
                        ex.shutdown(wait=False, cancel_futures=True)
                    if not broke:
                        break
                    # a worker crash breaks the whole pool without saying which file did it: re-run
                    # the first unreported file alone, then carry on with the rest in a new pool
                    suspect = pending[done]
                    with ProcessPoolExecutor(max_workers=1) as solo:
                        try:
                            emit(_wait(solo.submit(_batch_worker, (suspect, opts))))
                        except BrokenProcessPool:
                            emit({"schema_version": SCHEMA_VERSION, "source": suspect,
                                  "error": "BrokenProcessPool: processing this file crashed its "
                                           "worker process (a native crash in a PDF or image "
                                           "library)"})
                    pending = pending[done + 1:]
        except KeyboardInterrupt:
            print(f"\ninterrupted after {n} of {len(inputs)} file(s)"
                  + (f"; {out_path} holds those {n} line(s)" if out_path not in (None, "-")
                     else ""), file=sys.stderr)
            return 130
        except OSError as e:                 # the output went away mid-run (a closed pipe)
            if out_path in (None, "-"):
                _abandon_stdout()
            print(f"error: cannot write output after {n} file(s): {e}", file=sys.stderr)
            return 1

    if out_path and out_path != "-":
        print(f"wrote {n} result line(s) to {out_path}", file=sys.stderr)
    if out_path != "-":                     # human summary (skipped only when JSONL went to stdout)
        print(f"{n} file(s), {hits} with struck text, {errors} error(s)",
              file=sys.stderr if out_path else sys.stdout)

    if args.fail_if_found and hits:
        return 3
    return 1 if errors else 0


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
                       epilog="exit codes: 0 ok, 1 usage/file error (batch: >=1 file errored), "
                              "2 encrypted / OCR required, 3 --fail-if-found matched, "
                              "130 interrupted")
    d.add_argument("files", nargs="+", metavar="FILE",
                   help="one or more PDFs, images (.png/.jpg/.tiff), or .docx files; also a "
                        "directory (its top-level documents) or a glob like '*.pdf' ('**' "
                        "recurses). A single file gets full output (--markdown/--overlay/...); "
                        "several files, a directory or a glob run batch mode (JSONL). '-' reads "
                        "one PDF from stdin")
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
    d.add_argument("--dpi", type=_positive_int, default=None,
                   help="raster DPI for scanned pages (default: 200 for PDFs; read from the image "
                        "metadata for image files, else 200)")
    d.add_argument("--pages", metavar="SPEC",
                   help="process only these 1-based pages, e.g. '1-5,12' (default: all)")
    d.add_argument("--method", default="vector", choices=["vector", "flag", "annot", "both"],
                   help="native-page detector: vector geometry, MuPDF strikeout flag, explicit "
                        "/StrikeOut annotations, or the union of all three")
    d.add_argument("--limit", type=int, default=25, help="max words to print (plain output)")
    d.add_argument("--json", metavar="PATH", help="write full results as JSON ('-' = stdout); in "
                   "batch mode this is JSONL, one result object per file")
    d.add_argument("--jsonl", metavar="PATH",
                   help="batch mode (also for a single file): write one JSON result per line to "
                        "PATH as each file finishes ('-' = stdout)")
    d.add_argument("--jobs", type=_positive_int, default=1, metavar="N",
                   help="batch mode: process files across N worker processes (default 1)")
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
    d.add_argument("--overlay-dpi", dest="overlay_dpi", type=_positive_int, default=150,
                   help="render DPI for --overlay images")
    d.add_argument("--dump-crops", dest="dump_crops", metavar="DIR",
                   help="active-learning export: write each scored word crop (PNG) + a crops.jsonl "
                        "manifest under DIR for labeling (scanned PDF / image input; single file)")
    d.add_argument("--fail-if-found", dest="fail_if_found", action="store_true",
                   help="exit 3 if any struck word is found (for CI gating)")
    d.set_defaults(func=_cmd_detect)
    try:
        args = p.parse_args(argv)
    except SystemExit as e:
        if e.code == 2:                      # argparse's usage error; 2 here means OCR required
            raise SystemExit(1) from None
        raise
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
