"""Nothing built from an earlier ground truth is sent or scored: stale inputs are refused, test
pages must be the ones prereg-1 froze, and a free arm's output made from an earlier input reads
as missing. The arms run one at a time, the package's first, each after an untimed warm-up, and
each times its own call, unthrottled: the cost pass with onnxruntime pinned to one thread, the
latency pass at default threads and to the same output."""

import ctypes
import json
import sys
from pathlib import Path

import onnxruntime as ort
import pymupdf
import pytest
from pdf_strikethrough import cnn

from bench import arms, render

PAGE = {"page_id": "G1.doc.p3", "split": "dev", "page": 3, "pdf": "data/corpus/doc.pdf",
        "pdf_sha256": "src-1"}  # fmt: skip
ROW = {"pdf_sha256": "src-1", "page": 3, "files": {"c1.pdf": "c1-1", "scan.pdf": "scan-1"}}


@pytest.fixture(autouse=True)
def new_process(monkeypatch):
    """Each test starts as a fresh process would: no timing pass chosen, onnxruntime unpinned."""
    monkeypatch.setattr(arms, "_pass", None)
    monkeypatch.setattr(ort.InferenceSession, "__init__", ort.InferenceSession.__init__)


@pytest.fixture
def manifest(monkeypatch, tmp_path):
    rows = {PAGE["page_id"]: json.loads(json.dumps(ROW))}
    sources = {PAGE["pdf"]: "src-1"}
    monkeypatch.setattr(render, "manifest", lambda: rows)
    monkeypatch.setattr(render, "source_sha256", sources.__getitem__)
    monkeypatch.setattr(render, "PREREG1", tmp_path / "prereg1.json")  # nothing frozen yet
    monkeypatch.setattr(render, "PREREG2", tmp_path / "prereg2.json")
    monkeypatch.setattr(render, "GT", tmp_path)  # no G1 corrections
    monkeypatch.setattr(render, "tags_in_force", lambda: [])  # nor any tag
    monkeypatch.setattr(render, "tag_holds", lambda tag, path: True)  # each on its lists' commit
    return rows, sources


def test_inputs_from_another_source_or_page_are_refused(manifest):
    rows, sources = manifest
    render.require_current([PAGE])
    for change in ({"pdf_sha256": "src-0"}, {"page": 4}):
        rows[PAGE["page_id"]].update(change)
        with pytest.raises(ValueError, match="stale"):
            render.require_current([PAGE])
        rows[PAGE["page_id"]].update(ROW)
    with pytest.raises(ValueError, match="missing"):
        render.require_current([{**PAGE, "page_id": "G1.doc.p9"}])
    sources[PAGE["pdf"]] = "src-2"  # the PDF on disk is no longer the ground truth's
    with pytest.raises(ValueError, match="rebuild the ground truth"):
        render.require_current([PAGE])


def test_once_frozen_a_test_page_must_be_what_prereg_1_tags(manifest):
    rows, sources = manifest
    g1 = {**PAGE, "set": "G1", "split": "test", "doc": "doc"}
    g0 = {**g1, "set": "G0", "pdf": "data/gt/G0/test.pdf"}
    g2 = {**g1, "set": "G2", "pdf": "data/bills/b.pdf", "doc": "b", "xml_sha256": "xml-1"}
    g5 = {**g1, "page_id": "G5.p.p3", "set": "G5", "pdf": "data/bressay/pages/p.png",
          "transcription_sha256": "t-1"}  # fmt: skip
    rows[g5["page_id"]] = dict(ROW)
    sources |= {g0["pdf"]: "src-1", g2["pdf"]: "src-1", g5["pdf"]: "src-1"}
    pages = [g1, g0, g2, g5, {**PAGE, "set": "G1"}]  # the dev page is never checked
    render.require_current(pages)  # nothing frozen yet
    frozen = {"g0_test": {g0["pdf"]: "src-1"}, "g1_test_pages": [g1["page_id"]],
              "g1_documents": {"doc": "src-1"}, "g2_bills": {"b": ["src-1", "xml-1"]},
              "g2_pages": [g2["page_id"]],
              "g5_pages": {g5["page_id"]: ["src-1", "t-1"]}}  # fmt: skip
    for change in ({}, {"g1_test_pages": []}, {"g1_documents": {"doc": "src-0"}},
                   {"g0_test": {g0["pdf"]: "src-0"}}, {"g2_bills": {"b": ["src-0", "xml-1"]}},
                   {"g2_bills": {"b": ["src-1", "xml-0"]}}, {"g2_pages": []},
                   {"g5_pages": {g5["page_id"]: ["src-1", "t-0"]}}):  # fmt: skip
        render.PREREG1.write_text(json.dumps(frozen | change), encoding="utf-8")
        if not change:
            render.require_current(pages)
            continue
        with pytest.raises(ValueError, match="plans freeze"):
            render.require_current(pages)


def test_the_prereg_1_tag_without_its_lists_is_refused(manifest, monkeypatch):
    monkeypatch.setattr(render, "tags_in_force", lambda: ["prereg-1"])
    with pytest.raises(FileNotFoundError, match="prereg1.json is missing"):
        render.require_current([PAGE])


def test_the_prereg_2_tag_without_its_lists_is_refused(manifest, monkeypatch):
    render.PREREG1.write_text(json.dumps({"g1_test_pages": [], "g1_documents": {}}), "utf-8")
    monkeypatch.setattr(render, "tags_in_force", lambda: ["prereg-1", "prereg-2"])
    with pytest.raises(FileNotFoundError, match="prereg2.json is missing"):
        render.require_current([PAGE])


def test_a_tag_on_a_commit_without_its_lists_is_refused(manifest, monkeypatch):
    render.PREREG1.write_text(json.dumps({"g1_test_pages": [], "g1_documents": {}}), "utf-8")
    render.PREREG2.write_text(json.dumps({"ground_truth": {}}), "utf-8")
    monkeypatch.setattr(render, "tags_in_force", lambda: ["prereg-1", "prereg-2"])
    render.require_current([PAGE])  # each tag on the commit that holds its lists
    monkeypatch.setattr(render, "tag_holds", lambda tag, path: tag == "prereg-1")
    with pytest.raises(FileNotFoundError, match="prereg-2 tag's commit doesn't hold"):
        render.require_current([PAGE])  # tagged first, committed after: refused


def test_once_prereg_2_is_frozen_a_test_pages_ground_truth_must_match(manifest, tmp_path):
    rows, sources = manifest
    g1 = {**PAGE, "set": "G1", "split": "test", "doc": "doc", "words": [{"text": "a"}]}
    dev = {**PAGE, "set": "G1", "words": [{"text": "changed"}]}  # dev pages are never checked
    render.PREREG1.write_text(json.dumps({"g1_test_pages": [g1["page_id"]],
                                          "g1_documents": {"doc": "src-1"}}), encoding="utf-8")  # fmt: skip
    digest = render.sha256_bytes(json.dumps(g1, sort_keys=True, ensure_ascii=False).encode())
    render.PREREG2.write_text(json.dumps({"ground_truth": {g1["page_id"]: digest}}), "utf-8")
    render.require_current([g1, dev])
    relabelled = {**g1, "words": [{"text": "a", "struck": [[0, 1]]}]}
    with pytest.raises(ValueError, match="ground truth differs from prereg2.json"):
        render.require_current([relabelled])
    fix = {"page_id": g1["page_id"], "word": 0, "text": "a", "label": "struck", "chars": "a"}
    (tmp_path / "G1-corrections.jsonl").write_text(json.dumps(fix) + "\n", encoding="utf-8")
    render.require_current([relabelled])  # the relabel is the one sanctioned change


def test_a_cached_output_stands_only_while_its_input_and_environment_are_current(
    manifest, tmp_path, monkeypatch
):
    rows, _ = manifest
    cache = tmp_path / "responses.jsonl"
    monkeypatch.setattr(arms, "CACHE", cache)
    env = {"requirements_sha256": arms.requirements_sha256()}
    lines = [
        {"arm": "P-native", "page_id": PAGE["page_id"], "input_sha256": "c1-0", "n": 1, **env},
        {"arm": "P-native", "page_id": PAGE["page_id"], "input_sha256": "c1-1", "n": 2, **env},
        {"arm": "B-naive", "page_id": PAGE["page_id"], "input_sha256": "src-1", "n": 3, **env},
        {"arm": "P-scan", "page_id": PAGE["page_id"], "n": 4, **env},  # inputs not yet hashed
        {"arm": "B-pm4llm", "page_id": PAGE["page_id"], "input_sha256": "src-1", "n": 5,
         "requirements_sha256": "another lock"},
    ]  # fmt: skip
    cache.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    got = {k: r["n"] for k, r in arms.cached().items()}
    assert got == {("P-native", PAGE["page_id"]): 2, ("B-naive", PAGE["page_id"]): 3}
    rows[PAGE["page_id"]]["files"]["c1.pdf"] = "c1-2"  # C1 rebuilt from a new source
    assert ("P-native", PAGE["page_id"]) not in arms.cached()


def test_the_arms_run_one_at_a_time_package_first_each_after_a_warm_up(manifest, tmp_path,
                                                                        monkeypatch):  # fmt: skip
    calls = []
    monkeypatch.setattr(arms, "CACHE", tmp_path / "responses.jsonl")
    monkeypatch.setattr(arms, "fingerprint", lambda: {})
    monkeypatch.setattr(arms, "harness_commit", lambda: "test")
    monkeypatch.setattr(arms, "RUN", {a: (lambda p, a=a: calls.append((a, p["page"])) or {})
                                      for a in arms.ARMS})  # fmt: skip
    pages = [PAGE, {**PAGE, "page_id": "G1.doc.p4", "page": 4}]
    rows, sources = manifest
    rows["G1.doc.p4"] = {**ROW, "page": 4}
    n = len(arms.ORDER)
    assert arms.run(pages) == 2 * n
    order = [a for a, _ in calls]
    assert order == [a for a in arms.ORDER for _ in range(3)]  # warm-up, then both pages
    assert [p for _, p in calls] == [3, 3, 4] * n
    assert arms.run(pages) == 0 and len(calls) == 3 * n  # cached: nothing runs, not a warm-up


def test_a_scan_arms_output_stands_only_under_its_own_ocr_settings(manifest, tmp_path,
                                                                   monkeypatch):  # fmt: skip
    cache = tmp_path / "responses.jsonl"
    monkeypatch.setattr(arms, "CACHE", cache)
    env = {"requirements_sha256": arms.requirements_sha256(), "input_sha256": "scan-1"}
    lines = [
        {"arm": "P-scan", "page_id": PAGE["page_id"], "n": 1, **env},  # before deviation 2
        {"arm": "P-scan", "page_id": PAGE["page_id"], "n": 2, "ocr_params": {}, **env},
        {"arm": "P-scan", "page_id": PAGE["page_id"], "n": 3,
         "ocr_params": {"Global.use_cls": False}, **env},
    ]  # fmt: skip
    cache.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    got = {k: r["n"] for k, r in arms.cached().items()}
    assert got == {("P-scan-shipped", PAGE["page_id"]): 1, ("P-scan", PAGE["page_id"]): 3}
    assert arms.OCR_PARAMS["P-scan"] == {"Global.use_cls": False}
    assert arms.OCR_PARAMS["P-scan-shipped"] == {} and arms.applies("P-scan-shipped", "G5.x.p0")


def test_workers_spread_the_cost_pass_and_say_so(manifest, tmp_path, monkeypatch):
    cache = tmp_path / "responses.jsonl"
    monkeypatch.setattr(arms, "CACHE", cache)
    monkeypatch.setattr(arms, "fingerprint", lambda: {})
    monkeypatch.setattr(arms, "harness_commit", lambda: "test")

    class Pool:  # in-process stand-in for the process pool
        def __init__(self, n, mp_context=None):
            self.n = n

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, *args):
            from concurrent.futures import Future

            f = Future()
            f.set_result((args[1]["page_id"], {"words": [], "cpu_s": 1.0, "wall_s": 1.0}))
            return f

    monkeypatch.setattr(arms, "ProcessPoolExecutor", Pool)
    rows, _ = manifest
    rows["G1.doc.p4"] = {**ROW, "page": 4}
    pages = [PAGE, {**PAGE, "page_id": "G1.doc.p4", "page": 4}]
    assert arms.run(pages, ["P-scan"], workers=4) == 2
    recs = [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines()]
    assert {r["workers"] for r in recs} == {2}  # the pool that ran: two pages, two workers
    assert {json.dumps(r["ocr_params"]) for r in recs} == {'{"Global.use_cls": false}'}
    with pytest.raises(ValueError, match="one process"):
        arms.run(pages, ["P-scan"], latency=True, workers=4)
    with pytest.raises(ValueError, match="test pages' cost pass"):
        arms.run([{**PAGE, "split": "test"}], ["P-scan"], workers=4)
    monkeypatch.setitem(arms.RUN, "P-scan", lambda page: {"words": [], "cpu_s": 1.0, "wall_s": 1.0})
    rows["G1.doc.p5"] = {**ROW, "page": 5}
    assert arms.run([{**PAGE, "page_id": "G1.doc.p5", "page": 5}], ["P-scan"], workers=4) == 1
    last = json.loads(cache.read_text(encoding="utf-8").splitlines()[-1])
    assert last["workers"] == 1  # one page left: run in this process, and stamped so


def test_two_runs_never_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(arms, "CACHE", tmp_path / "responses.jsonl")
    (tmp_path / ".lock").write_text("1234")  # a run still going
    with pytest.raises(RuntimeError, match="another run holds"):
        arms.run([PAGE])


def test_the_clean_text_check_stops_a_run_on_a_mismatch():
    arms._check_clean([("kept", []), ("gone", [[0, 4]])], "kept", "P")
    with pytest.raises(RuntimeError, match="differ from clean_text"):
        arms._check_clean([("kept", []), ("gone", [[0, 2]])], "kept", "P")


def test_the_baselines_time_their_own_calls(tmp_path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "Hello there")
        doc.save(pdf)
    monkeypatch.setattr(arms, "_quiet", lambda: None)
    for arm in (arms.b_naive, arms.b_pm4llm):
        out = arm({"pdf": str(pdf), "page": 0})  # an absolute path stands in for the corpus's
        assert "Hello there" in out["text"] and out["cpu_s"] >= 0 and out["wall_s"] > 0


def test_the_cost_pass_pins_onnxruntime_and_a_process_runs_one_pass():
    arms._start("cost")
    onnx = Path(cnn.__file__).with_name("strike_verdict_cnn.onnx")
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    assert sess.get_session_options().intra_op_num_threads == 1
    with pytest.raises(RuntimeError, match="in a new one"):
        arms._start("latency")  # this process's sessions are pinned


def _throttling(new: tuple[int, int] | None = None) -> list[int]:
    """This process's power-throttling control and state masks, after setting them to `new`."""
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    process = ctypes.c_void_p(kernel32.GetCurrentProcess())
    masks = (ctypes.c_ulong * 3)(1, *(new or (0, 0)))  # version 1
    call = kernel32.SetProcessInformation if new else kernel32.GetProcessInformation
    assert call(process, 4, masks, ctypes.sizeof(masks))  # 4: ProcessPowerThrottling
    return list(masks)[1:]


@pytest.mark.skipif(sys.platform != "win32", reason="power throttling is Windows'")
def test_a_timing_pass_turns_windows_power_throttling_off():
    _throttling((0, 0))  # Windows decides, as in a new process
    arms._start("latency")
    assert _throttling() == [1, 0]  # execution speed: the process's call, and not throttled


def test_the_latency_pass_needs_the_cost_pass_and_its_output(manifest, tmp_path, monkeypatch):
    for name, file in (("CACHE", "responses.jsonl"), ("LATENCY", "latency.jsonl")):
        monkeypatch.setattr(arms, name, tmp_path / file)
    monkeypatch.setattr(arms, "fingerprint", lambda: {})
    monkeypatch.setattr(arms, "harness_commit", lambda: "test")
    out = {"text": "live", "cpu_s": 1.0, "wall_s": 2.0}
    monkeypatch.setattr(arms, "RUN", {"B-naive": lambda page: dict(out)})
    with pytest.raises(RuntimeError, match="run the cost pass first"):
        arms.run([PAGE], ["B-naive"], latency=True)
    for latency in (False, True):  # each pass in a process of its own
        monkeypatch.setattr(arms, "_pass", None)
        assert arms.run([PAGE], ["B-naive"], latency=latency) == 1
    (row,) = arms.cached(arms.LATENCY).values()
    assert row["pass"] == "latency" and row["wall_s"] == 2.0 and "text" not in row
    out["text"] = "changed"
    with pytest.raises(RuntimeError, match="differs from the cost pass's"):
        arms.run([PAGE], ["B-naive"], force=True, latency=True)


def test_a_scan_call_the_package_refuses_is_a_failed_page_not_a_stop(manifest, monkeypatch):
    def refuse(*a, **k):
        raise ValueError("Word.bbox must be normalized page fractions in [0,1]")

    monkeypatch.setattr(arms.st, "detect_pdf", refuse)
    monkeypatch.setattr(arms, "_ocr", lambda arm: lambda image: [])
    monkeypatch.setattr(arms, "_quiet", lambda: None)
    monkeypatch.setattr(arms.render, "path", lambda pid, name: Path(name))
    out = arms.p_scan(PAGE)
    assert out["words"] == [] and out["error"].startswith("ValueError") and "cpu_s" in out

    def broken(*a, **k):  # any other ValueError is a setting or a bug: it stops the run
        raise ValueError("Global.use_clss is not a valid key.")

    monkeypatch.setattr(arms.st, "detect_pdf", broken)
    with pytest.raises(ValueError, match="not a valid key"):
        arms.p_scan(PAGE)


def test_a_run_the_package_refuses_page_after_page_stops(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(arms, "CACHE", tmp_path / "responses.jsonl")
    monkeypatch.setattr(arms, "fingerprint", lambda: {})
    monkeypatch.setattr(arms, "harness_commit", lambda: "test")
    monkeypatch.setattr(arms, "MAX_REFUSED", 1)
    monkeypatch.setitem(arms.RUN, "P-scan", lambda page: {"words": [], "error": "ValueError: x",
                                                          "cpu_s": 1.0, "wall_s": 1.0})  # fmt: skip
    rows, _ = manifest
    rows["G1.doc.p4"] = {**ROW, "page": 4}
    with pytest.raises(RuntimeError, match="refused 2 pages"):
        arms.run([PAGE, {**PAGE, "page_id": "G1.doc.p4", "page": 4}], ["P-scan"])
