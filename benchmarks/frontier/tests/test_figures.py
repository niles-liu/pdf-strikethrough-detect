"""The test run's figures: the scatter's points and drawing, and the gallery's runs and page."""

import math
import re

import pytest

import figures
from bench import report
from bench.score import Token


def w(text, x, struck=False, y=0.1):
    return {"text": text, "bbox": [x, y - 0.005, x + 0.05, y + 0.005], "line": [0, 1],
            "struck": [[0, len(text)]] if struck else [], "masked": False, "furniture": False}  # fmt: skip


def live(text):
    return Token(text, [False] * len(text))


def test_arms_parse_into_model_rung_and_input_and_models_read_short():
    assert figures.parse_arm("opus-5-5:R1:C2:p2") == ("opus-5-5", "R1p2", "C2")
    assert figures.parse_arm("gemini-4-argon:R5:C2-tiled") == ("gemini-4-argon", "R5", "C2-tiled")
    assert figures.parse_arm("P-native") == ("P-native", "", "C1")
    assert figures.parse_arm("opus-5-5:R0:C1@v2") == ("opus-5-5", "R0@v2", "C1")  # by version
    assert [figures.short(m) for m in ("gpt-6-astra", "grok-4-7", "opus-5-5", "P-scan")] == [
        "Astra", "Grok 4.7", "Opus 5.5", "P-scan"]  # fmt: skip


def test_a_square_root_axis_puts_more_ticks_near_zero():
    assert figures._ticks(1.0, root=True) == [0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
    assert figures._ticks(0.3, root=True)[-2:] == [0.25, 0.3]  # the top itself closes the axis
    assert figures._ticks(0.5, root=False) == pytest.approx([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    assert figures._top(0.031) == 0.05 and figures._top(0.97) == 1.0


def test_runs_are_consecutive_sub_tokens_sharing_one_failure():
    rows = [{"struck": True, "outcome": "live"}, {"struck": True, "outcome": "live"},
            {"struck": False, "outcome": "live"},  # read right: ends the leak
            {"struck": False, "outcome": "marked"}, {"struck": False, "outcome": "absent"},
            {"struck": True, "outcome": "absent"},  # a missed strike: no run of its own
            {"struck": True, "outcome": "live"}]  # fmt: skip
    runs = figures.failure_runs(rows)
    assert [(k, len(v)) for k, v in runs] == [("leak", 2), ("over-deletion", 2), ("leak", 1)]


def test_the_worst_runs_are_each_pages_longest_ranked_and_failures_listed(monkeypatch):
    long = {"page_id": "G0.a.p0", "set": "G0",
            "words": [w("drop", 0.1, True), w("this", 0.2, True), w("clause", 0.3, True)]}  # fmt: skip
    short = {"page_id": "G0.b.p0", "set": "G0", "words": [w("old", 0.1, True), w("new", 0.2)]}
    g1 = {"page_id": "G1.c.p0", "set": "G1", "words": [w("gone", 0.1, True)]}
    monkeypatch.setattr(figures, "load_pages", lambda *s, **k: [long, short, g1])
    outs = [
        {"arm": "m:R0:C2", "page_id": "G0.b.p0", "tokens": [live("old"), live("new")]},
        {"arm": "m:R0:C2", "page_id": "G0.a.p0",
         "tokens": [live("drop"), live("this"), live("clause")]},
        {"arm": "m:R0:C2", "page_id": "G1.c.p0", "tokens": None, "status": "refusal"},
        {"arm": "m:QA:C2", "page_id": "G0.a.p0", "qa_id": "q1", "tokens": [live("drop")]},
        {"arm": "P-native", "page_id": "G1.c.p0", "tokens": [live("gone")]},  # by construction
    ]  # fmt: skip
    monkeypatch.setattr(figures.harness, "outputs", lambda stage: outs)
    found = figures.worst("test", per_kind=1)
    assert set(found) == {"m:R0:C2"}  # the QA probe and P-native on G1 are left out
    arm = found["m:R0:C2"]
    assert arm["pages"] == 3 and arm["failed"] == [("G1.c.p0", "refusal")]
    assert [(r["page"]["page_id"], len(r["rows"])) for r in arm["leak"]] == [("G0.a.p0", 3)]
    assert arm["over-deletion"] == []


def test_the_points_pool_each_arm_on_each_set_as_the_tables_do(monkeypatch):
    def row(arm, set_, struck_live, cost, strict=False, listed=None):
        return {"arm": arm, "set": set_, "page_id": f"{set_}.{arm}.{struck_live}", "strict": strict,
                "struck": 10, "struck_live": struck_live, "struck_marked": 10 - struck_live,
                "live": 20, "live_live": 20, "cost_usd": cost,
                "cost_list_usd": cost if listed is None else listed}  # fmt: skip

    rows = [row("m:R0:C2", "G2", 1, 0.01, listed=0.02), row("m:R0:C2", "G2", 3, 0.03, strict=True),
            row("P-native", "G1", 0, 0.0), row("P-native", "G2", 2, 0.0001)]  # fmt: skip
    monkeypatch.setattr(figures.report, "scores",
                        lambda stage: rows + [r | {"set": "G2-strict"} for r in rows if r["strict"]])  # fmt: skip
    pts = figures.points("test")
    assert [(p["arm"], p["set"], p["by_construction"]) for p in pts] == [
        ("P-native", "G1", True), ("P-native", "G2", False), ("m:R0:C2", "G2", False)]  # fmt: skip
    p = pts[2]
    m = report.metrics(rows[:2])
    assert (p["y"], p["x"], p["x_list"]) == (
        m["leakage"],
        m["usd_per_1000"],
        m["usd_per_1000_list"],
    )
    assert p["y"] == pytest.approx(0.2) and p["x"] == pytest.approx(20.0)
    assert p["x_list"] == pytest.approx(25.0)  # list price where a promotion applied, else recorded


def test_the_scatter_rings_list_price_and_names_what_it_cannot_draw():
    pts = [{"arm": "gemini-4-argon:R5:C2", "set": "G2", "y": 0.05, "x": 50.0, "x_list": 100.0,
            "pages": 51, "failure_rate": 0.0},
           {"arm": "P-scan", "set": "G2", "y": 0.2, "x": 0.1, "x_list": 0.1, "pages": 51,
            "failure_rate": 0.0},
           {"arm": "opus-5-5:R0:C2", "set": "G2", "y": 0.0, "x": math.nan, "x_list": math.nan,
            "pages": 51, "failure_rate": 0.0}]  # fmt: skip
    out = figures.svg(pts, "leakage", "test")
    assert "($100.00 at list price)" in out and 'fill="#fff" stroke=' in out  # the ring
    assert "Not drawn: opus-5-5:R0:C2 (no price)" in out
    assert "$0.1</text>" in out and "$100</text>" in out  # the log axis spans both prices
    assert out.count("<title>") == 2
    with pytest.raises(ValueError, match="nothing to draw"):
        figures.svg(pts[2:], "leakage", "test")


def test_the_notes_name_every_arm_left_off_on_lines_that_fit():
    base = {"y": 0.1, "x": 1.0, "x_list": 1.0, "pages": 50, "failure_rate": 0.0}
    pts = [base | {"arm": "P-scan", "set": "G5"},
           base | {"arm": "P-native", "set": "G1", "by_construction": True},
           base | {"arm": "B-naive", "set": "G1", "x": 0.0, "x_list": 0.0},
           *(base | {"arm": f"opus-5-5:R{k}:C2-tiled", "set": "G5", "y": math.nan}
             for k in range(18))]  # fmt: skip
    out = figures.svg(pts, "leakage", "test")
    assert "P-native (agrees with the ground truth by its construction)" in out
    assert "B-naive (costs $0, off the log axis)" in out
    notes = [t.split(">", 1)[1].split("<", 1)[0] for t in out.split("<text ") if "#666" in t]
    assert len(notes) > 2 and all(f"R{k}:C2-tiled" in out for k in range(18))
    assert max(map(len, notes)) <= (figures.W - figures.R - figures.L) / 5.6  # inside the panel


def test_a_label_keeps_off_the_line_to_its_list_price():
    marks = [{"x": 100.0, "y": 100.0, "ring": 200.0, "text": "R0"}]
    figures._labels(marks, (0, 0, 400, 400))
    m = marks[0]
    box = (m["lx"], m["ly"] - 8, m["lx"] + figures.CHAR * 2, m["ly"] + 1)
    assert not figures._overlaps(box, (100, 98, 200, 102))


def test_the_gallery_keeps_a_shared_crop_once_and_escapes_the_text(monkeypatch):
    page = {"page_id": "G0.a.p0", "set": "G0", "words": [w("<b>", 0.1, True), w("kept", 0.2)]}
    rows = [{"word": 0, "span": [0, 3], "struck": True, "outcome": "live", "tokens": [0]}]
    item = {"page": page, "rows": rows, "tokens": [live("<b>"), live("kept")], "details": rows}
    found = {arm: {"pages": 1, "failed": [], "leak": [item], "over-deletion": []}
             for arm in ("opus-5-5:R0:C2", "sonnet-5-5:R0:C2")}  # fmt: skip
    crops = []
    monkeypatch.setattr(figures, "_scored", lambda stage: [])
    monkeypatch.setattr(figures, "worst", lambda stage, per_kind, scored: found)
    monkeypatch.setattr(figures, "_crop", lambda *a: crops.append(a) or "data:image/jpeg;base64,AA")
    out = figures.gallery("test")
    assert len(crops) == 1 and out.count('data-crop="c0"') == 2  # one crop, both arms
    assert '"c0": "data:image/jpeg;base64,AA"' in out
    assert "<mark><del>&lt;b&gt;</del></mark>" in out  # the page's word, escaped and struck
    assert "<b>&lt;b&gt;</b> kept" in out  # the output's, escaped and in bold
    assert out.index("opus-5-5:R0:C2") < out.index("sonnet-5-5:R0:C2")  # config order


def toks(n):
    return [live(f"t{k}") for k in range(n)]


def test_each_stretch_of_output_gets_an_excerpt_and_none_repeats():
    far = {"rows": [{"word": j, "tokens": [k]} for j, k in enumerate((5, 6, 7, 190))],
           "tokens": toks(200), "details": []}  # fmt: skip
    out = figures._output_text(far, [0, 1, 2, 3])  # one word paired far off
    assert "<b>t5</b>" in out and "<b>t190</b>" in out and " … " in out and "t100" not in out
    long = {"rows": [{"word": k, "tokens": [k]} for k in range(100)], "tokens": toks(120),
            "details": []}  # fmt: skip
    shown = re.sub("<[^>]+>", "", figures._output_text(long, list(range(100)))).split()
    assert "…" in shown and max(shown.count(f"t{k}") for k in range(120)) == 1


def test_a_run_left_out_shows_its_gap_by_its_nearest_words_output():
    details = [{"word": 0, "tokens": [50]}, {"word": 1, "tokens": [3]},
               {"word": 2, "tokens": []}, {"word": 3, "tokens": [4]}]  # fmt: skip
    item = {"rows": [details[2]], "tokens": toks(60), "details": details}
    assert "t3 <b>▯</b> t4" in figures._output_text(item, [2])  # not after word 0's far t50


def test_the_gallery_shows_only_what_the_scores_counted(monkeypatch):
    page = {"page_id": "G0.a.p0", "set": "G0", "words": [w("old", 0.1, True), w("new", 0.2)]}
    monkeypatch.setattr(figures, "load_pages", lambda *s, **k: [page])
    outs = [{"arm": "m:R0:C2", "page_id": "G0.a.p0", "tokens": [live("old"), live("new")]},
            {"arm": "m:R1:C2", "page_id": "G0.a.p0", "tokens": [], "status": "ok"}]  # fmt: skip
    monkeypatch.setattr(figures.harness, "outputs", lambda stage: outs)
    row = {"arm": "m:R0:C2", "page_id": "G0.a.p0", "struck": 1, "struck_live": 1, "live": 1}
    empty = {"arm": "m:R1:C2", "page_id": "G0.a.p0", "struck": 1, "struck_live": 1, "live": 1,
             "live_absent": 1, "failed": True}  # fmt: skip
    found = figures.worst("test", 1, [row, empty])
    assert found["m:R1:C2"]["failed"] == [("G0.a.p0", "no usable output")]
    with pytest.raises(ValueError, match="rescore"):  # the ground truth moved since scoring
        figures.worst("test", 1, [row | {"struck_live": 0}, empty])
    with pytest.raises(ValueError, match="by version"):
        figures.worst("test", 1, [row | {"arm": "m:R0:C2@v2"}])
    with pytest.raises(ValueError, match="at least 1"):
        figures.worst("test", 0)
