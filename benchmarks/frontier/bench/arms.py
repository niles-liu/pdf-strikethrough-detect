"""The free arms (PREREG.md, Arms and inputs), run on every page and cached in
`runs/free/responses.jsonl`:

- B-naive: PyMuPDF `page.get_text()` on the born-digital page.
- B-pm4llm: `pymupdf4llm.to_markdown(doc, pages=[p], use_ocr=False)`.
- P-native: `detect_pdf(path, method="vector")` on the single-page born-digital PDF (C1).
- P-scan: `detect_pdf(scan, ocr=rapidocr_backend(params={"Global.use_cls": False}),
  scan_config=ScanConfig.confidence_free(), dpi=200)` on the page rasterized at 200 dpi: RapidOCR's
  text-line orientation classifier is off, since it turned struck and underlined lines upside down
  (deviation 2). P-scan-shipped is the same call with `rapidocr_backend()` as released, reported
  beside it; a record from before deviation 2 carries no `ocr_params` and is P-scan-shipped's. G5's
  pages are scans, so the two scan arms alone run there.

The package is scored from its word records, never its `~~` markdown: P-native's output is the
page's PyMuPDF words in extraction order, P-scan's the OCR backend's words (captured by a
pass-through wrapper) in the package's own reading order, each with the `char_span` of every
`final` record marked. On every page the unmarked words must equal the call's own `clean_text` as
a multiset of tokens, or the run stops.

Timing (cost and latency, PREREG.md, Operations): each arm times its own call and none of the
harness's checks: the package's arms from opening their one-page input to the result, the
baselines from loading the page of their already open source document (opening a whole document
is not one page's cost). The cost pass keeps the outputs and their CPU time with every onnxruntime
session pinned to one thread, so the CPU-seconds are the work and not idle threads spinning. The
latency pass, in a new process after the cost pass, times the same calls at the libraries'
default threads into `runs/free/latency.jsonl`, and stops if an output differs from the cost
pass's. In each, the arms run one after another over all their pages, the package's first, each
after one untimed call that builds the engines later calls reuse, and each timed call waits until
the process is idle. Either pass first turns Windows' power throttling off for its process: a run
with no window counts as background work, which Windows may move to slow cores, and stage B1 saw
P-scan take ten times the CPU-seconds that way. Run on an otherwise idle machine; `arms run`
holds a lock, so two never overlap. A cached output or timing stands only while its input's
sha256 is the one in `data/inputs/manifest.jsonl` and `requirements.txt` is the one it ran under
(a lock change is a new snapshot).

    python -m bench arms run --stage pilot           # the pages a stage sends
    python -m bench arms run --set G0 G1 --split dev # every dev page
    python -m bench arms run --stage latency --latency   # after the test stage's cost pass

`--workers N` spreads the cost pass over N processes, each pinned and warmed up as above, for a
dev run that must finish soon; their CPU-seconds share the machine, so the record says how many
ran (`workers`), and the costs that count come from a pass with one.
"""

import contextlib
import functools
import importlib
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import onnxruntime as ort
import pdf_strikethrough as st
import pymupdf
import rapidocr
from pdf_strikethrough import cnn, detect, markdown
from pdf_strikethrough.ocr import rapidocr_backend
from pdf_strikethrough.scanned import ScanConfig

from bench import render
from bench.common import (
    ROOT, RUNS, SETS, append_jsonl, guard_test_pages, harness_commit, load_pages, lock,
    read_jsonl, sha256_file, tags_in_force,
)  # fmt: skip

ARMS = ("B-naive", "B-pm4llm", "P-native", "P-scan", "P-scan-shipped")
ORDER = ("P-native", "P-scan", "P-scan-shipped", "B-naive", "B-pm4llm")  # the package's first
SCAN = ("P-scan", "P-scan-shipped")
INPUT = {"P-native": "c1.pdf", **dict.fromkeys(SCAN, "scan.pdf")}  # the B arms read the source PDF
OCR_PARAMS = {"P-scan": {"Global.use_cls": False}, "P-scan-shipped": {}}  # RapidOCR's, per scan arm
REFUSED = "must be normalized page fractions"  # the package's word check: an OCR box off the page
MAX_REFUSED = 10  # refusals in one run past which it is the backend, not the pages: stop
CACHE = RUNS / "free" / "responses.jsonl"
LATENCY = RUNS / "free" / "latency.jsonl"  # the latency pass's timings
ONNX_SHA256_PREFIX = "fac2c51b"
_pass: str | None = None  # this process's timing pass: a session keeps the threads it was made with


@functools.cache
def requirements_sha256() -> str:
    """The pinned environment the arms run in, as the lock file's sha256."""
    return sha256_file(ROOT / "requirements.txt")


def fingerprint() -> dict:
    """The package under test as frozen (PLAN.md, Package under test), and RapidOCR's models."""
    if os.environ.get("PDF_STRIKETHROUGH_MODEL_DIR"):
        raise RuntimeError("PDF_STRIKETHROUGH_MODEL_DIR is set; the shipped model must be used")
    onnx = sha256_file(Path(cnn.__file__).with_name("strike_verdict_cnn.onnx"))
    if not onnx.startswith(ONNX_SHA256_PREFIX):
        raise RuntimeError(f"strike_verdict_cnn.onnx is {onnx[:12]}…, not {ONNX_SHA256_PREFIX}…")
    models = Path(rapidocr.__file__).with_name("models")
    return {
        "version": st.__version__,
        "onnx_sha256": onnx,
        "model_meta": st.get_model_meta(),
        "rapidocr_models": {f.name: sha256_file(f) for f in sorted(models.glob("*.onnx"))},
    }


def _pin_threads() -> None:
    """Make every onnxruntime session created from now on run on one thread."""
    init = ort.InferenceSession.__init__

    def pinned(self, model, sess_options=None, *args, **kwargs):
        opts = sess_options or ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        init(self, model, opts, *args, **kwargs)

    ort.InferenceSession.__init__ = pinned


def full_speed() -> None:
    """Turn Windows' power throttling off for this process (elsewhere, nothing to do): the free
    arms' timing passes, and the models' latency sample (`harness submit latency`)."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):  # PROCESS_POWER_THROTTLING_STATE
        _fields_ = [(k, wintypes.ULONG) for k in ("version", "control", "state")]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = (
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    )  # fmt: skip
    off = State(1, 1, 0)  # version 1; execution speed: set here, and not throttled
    process, power_throttling = kernel32.GetCurrentProcess(), 4  # ProcessPowerThrottling
    if not kernel32.SetProcessInformation(process, power_throttling, ctypes.byref(off),
                                          ctypes.sizeof(off)):  # fmt: skip
        raise ctypes.WinError(ctypes.get_last_error())


def _start(name: str) -> None:
    """Fix this process's timing pass, at full speed, "cost" pinning onnxruntime first.
    pymupdf4llm is imported only then: its layout builds its sessions on import (importing it
    changes nothing else the other arms or the ground truth read, quad corrections apart, which
    bench.common sets)."""
    global _pass
    if _pass not in (None, name):
        raise RuntimeError(f"this process ran the {_pass} pass; run the {name} pass in a new one")
    if _pass is None:
        full_speed()
        if name == "cost":
            _pin_threads()
    _pass = name
    importlib.import_module("pymupdf4llm")


def _quiet(limit_s: float = 10.0) -> None:
    """Wait, up to `limit_s`, until the process burns under 10 ms of CPU in 0.1 s of sleep."""
    end = time.perf_counter() + limit_s
    while time.perf_counter() < end:
        c0 = time.process_time()
        time.sleep(0.1)
        if time.process_time() - c0 < 0.01:
            return


@contextlib.contextmanager
def _timed(out: dict):
    """Time the block once the process is idle, into `out` as CPU and wall seconds."""
    _quiet()
    t0, c0 = time.perf_counter(), time.process_time()
    yield
    out |= {"cpu_s": time.process_time() - c0, "wall_s": time.perf_counter() - t0}


def _marked(text: str, rec: dict | None) -> list[list[int]]:
    return [list(rec.get("char_span", (0, len(text))))] if rec and rec.get("final") else []


def _check_clean(words: list[tuple[str, list]], clean_text: str, page_id: str) -> None:
    left = ["".join(c for i, c in enumerate(t) if not any(s <= i < e for s, e in spans))
            for t, spans in words]  # fmt: skip
    if Counter(w for t in left for w in t.split()) != Counter(clean_text.split()):
        raise RuntimeError(f"{page_id}: unmarked words differ from clean_text; stopping")


def p_native(page: dict) -> dict:
    path, timing = render.path(page["page_id"], "c1.pdf"), {}
    with _timed(timing):
        res = st.detect_pdf(str(path), method="vector")
    with pymupdf.open(path) as doc:
        seq = detect._match_native_seq(doc[0], res["words"])
    words = [(t, _marked(t, r)) for t, _, r in seq]
    _check_clean(words, res["clean_text"], page["page_id"])
    return {"words": words, **timing}


@functools.cache
def _ocr(arm: str):
    """A scan arm's one RapidOCR engine in a run (built lazily on its first call)."""
    return rapidocr_backend(params=OCR_PARAMS[arm]) if OCR_PARAMS[arm] else rapidocr_backend()


def p_scan(page: dict, arm: str = "P-scan") -> dict:
    ocr, captured = _ocr(arm), []

    def capture(image):
        words = ocr(image)
        captured.append(words)
        return words

    path, timing, error = render.path(page["page_id"], "scan.pdf"), {}, None
    with _timed(timing):
        try:
            res = st.detect_pdf(str(path), ocr=capture, scan_config=ScanConfig.confidence_free(),
                                dpi=200)  # fmt: skip
        except ValueError as e:  # the package's own call refusing the page (deviation 2)
            if REFUSED not in str(e):
                raise  # a setting or a bug, not the page
            error = f"{type(e).__name__}: {e}"
    if error:  # no output: a failed page, scored worst case
        return {"words": [], "cnn_prob": [], "error": error, **timing}
    by_key = {(r["bbox_frac"], r["text"]): r for r in res["words"]}
    items = [(w.text, w.bbox, by_key.get((w.bbox, w.text))) for ws in captured for w in ws]
    rows = markdown.ordered_rows([it for it in items if (it[0] or "").strip()])
    seq = [it for row in rows for it in row]
    words = [(t, _marked(t, r)) for t, _, r in seq]
    _check_clean(words, res["clean_text"], page["page_id"])
    cnn_probs = [r.get("cnn_prob") if r else None for _, _, r in seq]
    return {"words": words, "cnn_prob": cnn_probs, **timing}


def b_naive(page: dict) -> dict:
    out = {}
    with pymupdf.open(ROOT / page["pdf"]) as doc, _timed(out):
        out["text"] = doc[page["page"]].get_text()
    return out


def b_pm4llm(page: dict) -> dict:
    import pymupdf4llm  # imported by _start, once the threads are set

    out = {}
    with pymupdf.open(ROOT / page["pdf"]) as doc, _timed(out):
        out["text"] = pymupdf4llm.to_markdown(doc, pages=[page["page"]], use_ocr=False,
                                              show_progress=False)  # fmt: skip
    return out


RUN = {"B-naive": b_naive, "B-pm4llm": b_pm4llm, "P-native": p_native, "P-scan": p_scan,
       "P-scan-shipped": functools.partial(p_scan, arm="P-scan-shipped")}  # fmt: skip


def applies(arm: str, page_id: str) -> bool:
    """Whether the arm runs on the page: G5's pages have no born-digital page, only a scan."""
    return arm in SCAN or not page_id.startswith("G5.")


def input_sha256(arm: str, page_id: str) -> str | None:
    """What the arm reads for the page, by its manifest hash: the source PDF or an input file."""
    row = render.manifest().get(page_id)
    return row and (row["files"][INPUT[arm]] if arm in INPUT else row["pdf_sha256"])


def _arm(rec: dict) -> str:
    """A record's arm: a P-scan record from before deviation 2 carries no `ocr_params` and ran the
    package's call as released, so it is P-scan-shipped's."""
    return "P-scan-shipped" if rec["arm"] == "P-scan" and "ocr_params" not in rec else rec["arm"]


def cached(path: Path | None = None) -> dict[tuple[str, str], dict]:
    """The latest record per arm and page in `path` (the cost pass's outputs by default, or
    LATENCY's timings), if it was made from the page's current input in the current environment,
    with the arm's OCR settings."""
    env = requirements_sha256()
    out = {}
    for r in read_jsonl(path or CACHE):
        arm = _arm(r)
        if (r.get("input_sha256") == input_sha256(arm, r["page_id"])
                and r.get("requirements_sha256") == env
                and r.get("ocr_params", {}) == OCR_PARAMS.get(arm, {})):  # fmt: skip
            out[arm, r["page_id"]] = r
    return out


def _output(rec: dict | None):
    """An arm's output as the cache holds it (JSON turns tuples into lists)."""
    return rec and json.loads(json.dumps(rec.get("words", rec.get("text"))))


def run(pages: list[dict], arms=ARMS, force: bool = False, latency: bool = False,
        workers: int = 1) -> int:  # fmt: skip
    """The cost pass, or with `latency` the latency pass (see the module's docstring)."""
    if latency and workers > 1:
        raise ValueError("the latency pass runs in one process: it times calls at default threads")
    if workers > 1 and any(p.get("split") == "test" for p in pages):
        raise ValueError("the test pages' cost pass runs in one process (PREREG, Timing)")
    with lock(CACHE.parent / ".lock"):
        return _run_pass(pages, arms, force, latency, workers)


_warm: set[str] = set()  # the arms this process has warmed up


def _work(arm: str, page: dict) -> tuple[str, dict]:
    """One timed call in a worker process of the cost pass, after its own warm-up."""
    _start("cost")
    if arm not in _warm:
        RUN[arm](page)  # untimed warm-up
        _warm.add(arm)
    return page["page_id"], RUN[arm](page)


def _run_pass(pages: list[dict], arms, force: bool, latency: bool, workers: int = 1) -> int:
    _start("latency" if latency else "cost")
    guard_test_pages(pages)
    render.require_current(pages)
    path, outputs = (LATENCY, cached()) if latency else (CACHE, None)
    have = cached(path)
    meta = {"package": fingerprint(), "requirements_sha256": requirements_sha256(),
            "commit": harness_commit(), "tags": tags_in_force(),
            "run_started": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    done, refused = 0, Counter()

    def record(arm: str, page_id: str, out: dict, ran: int = 1) -> None:
        nonlocal done
        key, error = (arm, page_id), out.get("error")
        status, refused[arm] = "error" if error else "ok", refused[arm] + bool(error)
        if latency:
            if key not in outputs:
                raise RuntimeError(f"{arm} {page_id}: run the cost pass first")
            if _output(out) != _output(outputs[key]):
                raise RuntimeError(f"{arm} {page_id}: the output at default threads differs "
                                   "from the cost pass's; stopping")  # fmt: skip
            out = {"cpu_s": out["cpu_s"], "wall_s": out["wall_s"]}
        ocr = {"ocr_params": OCR_PARAMS[arm]} if arm in SCAN else {}
        append_jsonl(path, {"arm": arm, "page_id": page_id, "status": status, "pass": _pass,
                            "input_sha256": input_sha256(*key), **ocr, **out, **meta,
                            "workers": ran})  # fmt: skip
        done += 1
        if refused[arm] > MAX_REFUSED:
            raise RuntimeError(f"{arm}: the package refused {refused[arm]} pages in this run, "
                               f"last {page_id} ({error}); stopping")  # fmt: skip

    for arm in (a for a in ORDER if a in arms):
        todo = [p for p in pages if (force or (arm, p["page_id"]) not in have)
                and applies(arm, p["page_id"])]  # fmt: skip
        if workers > 1 and len(todo) > 1:
            spawn, ran = multiprocessing.get_context("spawn"), min(workers, len(todo))
            with ProcessPoolExecutor(ran, mp_context=spawn) as pool:
                jobs = [pool.submit(_work, arm, p) for p in todo]
                try:
                    for job in as_completed(jobs):
                        record(arm, *job.result(), ran)  # each as it lands: a stop keeps the rest
                except BaseException:  # stop the queue too, or the pool works on, unrecorded
                    pool.shutdown(cancel_futures=True)
                    raise
            continue
        if todo:
            RUN[arm](todo[0])  # untimed warm-up
        for page in todo:
            record(arm, page["page_id"], RUN[arm](page))
    return done


def _run(args) -> None:
    pages = load_pages(*args.set, split=args.split)
    if args.stage:
        from bench.plans import items

        wanted = {i.page_id for i in items(args.stage)}
        pages = [p for p in load_pages(*SETS) if p["page_id"] in wanted]
    n = run(pages, args.arm, args.force, args.latency, args.workers)
    print(f"{n} free-arm records for {len(pages)} pages -> {LATENCY if args.latency else CACHE}")


def register(sub) -> None:
    p = sub.add_parser("arms", help="the free arms: B-naive, B-pm4llm, P-native and the scan arms")
    cmd = p.add_subparsers(required=True, metavar="action")
    r = cmd.add_parser("run")
    r.add_argument("--set", nargs="+", default=["G0", "G1"])
    r.add_argument("--split", choices=["dev", "test"], default="dev")
    r.add_argument("--stage", help="only the pages a stage sends (e.g. pilot), in place of --set")
    r.add_argument("--arm", nargs="+", choices=ARMS, default=list(ARMS))
    r.add_argument("--force", action="store_true", help="rerun pages already cached")
    r.add_argument("--latency", action="store_true",
                   help="the latency pass, at default threads (after the cost pass)")  # fmt: skip
    r.add_argument(
        "--workers",
        type=int,
        default=1,
        help="cost-pass processes (dev runs; the costs that count come from one)",
    )
    r.set_defaults(func=_run)
