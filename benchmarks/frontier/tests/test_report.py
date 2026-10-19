"""The report's decisions on hand-made score rows: a claim's largest p and its insufficient-data
floors, every endpoint's claims as PREREG states them (a golden test), R-best and each
flagship's rung, the image condition once C2-tiled has run, G2's strict subset as a set of its
own, the calibration, the cascade, and the canary's drift and window."""

import json
import math
from datetime import datetime, timedelta

import pytest

from bench import report


def test_a_claim_takes_its_largest_p_and_reads_insufficient_under_the_floors():
    tests = [{"p": 0.01, "struck": 500, "clusters": 9}, {"p": 0.04, "struck": 300, "clusters": 7}]
    c = report.claim(1, "x", "G0", tests)
    assert (c["p"], c["struck"], c["clusters"], c["insufficient"]) == (0.04, 300, 7, False)
    for struck, clusters in ((199, 9), (500, 5)):
        test = {"p": 0.01, "struck": struck, "clusters": clusters}
        assert report.claim(1, "x", "G0", [test])["insufficient"]


def row(rung: str, unmarked: int, over: int = 0, input_: str = "C2", page: str = "p1",
        failed: bool = False, model: str = "m") -> dict:  # fmt: skip
    """One page: 100 struck sub-tokens, `unmarked` of them left live; 100 live, `over` marked. A
    failed page is scored worst case."""
    if failed:
        unmarked, over = 100, 0
    return {"model": model, "rung": rung, "input": input_, "page_id": page, "struck": 100,
            "struck_live": unmarked, "struck_absent": 0, "struck_marked": 100 - unmarked,
            "live": 100, "live_marked": over, "live_absent": 100 if failed else 0,
            "live_live": 100 - over - (100 if failed else 0), "failed": failed,
            "cost_usd": 0.01, "priced": True}  # fmt: skip


@pytest.fixture
def ladder(monkeypatch, tmp_path):
    rows: list[dict] = []
    monkeypatch.setattr(report, "RUNS", tmp_path)
    monkeypatch.setattr(report, "scores", lambda stage: rows)
    models = {"m": {"role": "ladder", "provider": "a"}}
    monkeypatch.setattr(report, "config", {"models": models}.__getitem__)
    return rows


def test_r_best_marks_most_within_a_point_of_r0s_over_deletion(ladder):
    ladder += [row("R0", 90), row("R2", 30), row("R3", 10, over=2), row("R4", 20, over=1),
               row("R5", 50), row("R6", 40)]  # fmt: skip
    out = report.rbest()["a"]  # R3 marks most, but 2 points above R0: out
    assert (out["rbest"], out["runner_up"], out["fallback"]) == ("R4", "R2", False)
    assert out["eligible"] == ["R2", "R4", "R5", "R6"] and out["image"] is None


def test_a_cheaper_rung_within_two_points_of_the_fewest_unmarked_wins(ladder):
    ladder += [row("R0", 90), row("R2", 22), row("R3", 21), row("R4", 20), row("R6", 15)]
    out = report.rbest()["a"]  # R6 leaves fewest; R2–R4 lie more than 2 points above it
    assert (out["rbest"], out["runner_up"]) == ("R6", "R4")
    ladder[:] = [row("R0", 90), row("R2", 22), row("R3", 21), row("R4", 20), row("R6", 19)]
    out = report.rbest()["a"]  # every rung within 2 points of R6's 19: R2 and R3 tie on cost,
    assert (out["rbest"], out["runner_up"]) == ("R3", "R6")  # R3 leaving fewer of the two
    ladder[:] = [row("R0", 90), row("R4", 20), row("R5", 22)]
    assert report.rbest()["a"]["rbest"] == "R4"  # R4 is cheaper than R5: no tie on cost here


def test_a_rung_failing_more_pages_than_r0_plus_one_is_out(ladder):
    pages = [f"p{i}" for i in range(4)]
    ladder += [row("R0", 90, page=p) for p in pages]
    ladder += [row("R2", 30, page=p) for p in pages]
    ladder += [row("R4", 5, page=p, failed=p in ("p0", "p1")) for p in pages]  # two failures
    ladder += [row("R5", 10, page=p, failed=p == "p0") for p in pages]  # one: kept
    out = report.rbest()["a"]
    assert out["out_for_failures"] == ["R4"] and out["failed"] == {"R2": 0, "R4": 2, "R5": 1}
    # R5 is compared on the three pages none failed: 10%, so it beats R2 by 20 points
    assert (out["rbest"], out["runner_up"], out["unmarked"]["R5"]) == ("R5", "R2", 0.1)


def test_with_no_rung_within_the_limits_r_best_is_the_nearest_and_alone(ladder):
    ladder += [row("R0", 90)] + [row(f"R{i}", 10 * i, over=5 + i) for i in range(2, 7)]
    out = report.rbest()["a"]
    assert (out["rbest"], out["runner_up"], out["fallback"], out["eligible"]) == (
        "R2", None, True, [],
    )  # fmt: skip


def test_the_image_condition_is_chosen_once_c2_tiled_has_run(ladder):
    ladder += [row("R0", 90), row("R2", 30), row("R2", 20, input_="C2-tiled")]
    assert report.rbest()["a"]["image"] == "C2-tiled"
    ladder[-1] = row("R2", 29, input_="C2-tiled")  # within 2 points: the cheaper C2 stands
    assert report.rbest()["a"]["image"] == "C2"


def test_each_flagship_keeps_r_best_or_the_runner_up_by_the_same_rule(monkeypatch):
    rows: list[dict] = []
    monkeypatch.setattr(report, "scores", lambda stage: rows)
    models = {"f": {"role": "flagship", "provider": "a"}}
    monkeypatch.setattr(report, "config", {"models": models}.__getitem__)
    choices = {"rbest": "R4", "runner_up": "R2"}
    monkeypatch.setattr(report.plans, "choice", lambda prov, key: choices[key])
    rows += [row("R4", 20, model="f"), row("R2", 21, model="f")]
    assert report.flagship_check()["f"]["flagship"] == "R2"  # within 2 points, and cheaper
    rows[:] = [row("R4", 20, model="f"), row("R2", 30, model="f")]
    assert report.flagship_check()["f"]["flagship"] == "R4"
    rows[:] = [row("R4", 20, model="f"), row("R2", 10, over=2, model="f")]
    assert report.flagship_check()["f"]["flagship"] == "R4"  # R2 over-deletes past R-best's
    choices["runner_up"] = "none"
    assert report.flagship_check()["f"] == {"rbest": "R4", "runner_up": None, "flagship": "R4"}
    choices["runner_up"] = "R2"
    rows[:] = [row("R4", 20, model="f") | {"partial": True}, row("R2", 21, model="f")]
    with pytest.raises(ValueError, match="--partial"):  # a binding choice stands on whole stages
        report.flagship_check()


def test_r_best_refuses_partial_scores_and_a_missing_base_chooses_nothing(ladder):
    ladder += [row("R0", 90), row("R2", 30) | {"partial": True}]
    with pytest.raises(ValueError, match="--partial"):
        report.rbest()
    cells = {("R2", "C2"): [row("R2", 30)]}  # no R0 to judge against: no choice, no crash
    assert report._choose(cells, ("R0", "C2"), [("R2", "C2")])["choice"] is None


def test_a_partial_tiled_stage_is_named_as_such(monkeypatch, tmp_path):
    staged = {"ladder": [row("R0", 90), row("R2", 30)],
              "ladder-tiled": [row("R2", 20, input_="C2-tiled") | {"partial": True}]}  # fmt: skip
    (tmp_path / "ladder-tiled").mkdir()
    (tmp_path / "ladder-tiled" / "scores.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(report, "RUNS", tmp_path)
    monkeypatch.setattr(report, "scores", lambda stage: staged[stage])
    monkeypatch.setattr(
        report, "config", {"models": {"m": {"role": "ladder", "provider": "a"}}}.__getitem__
    )
    with pytest.raises(ValueError, match="ladder-tiled was scored with --partial"):
        report.rbest()


def test_g2s_strict_bills_are_a_set_of_their_own_and_partial_scores_are_refused(monkeypatch):
    rows = [{"set": "G2", "strict": True, "page_id": "a"},
            {"set": "G2", "strict": False, "page_id": "b"},
            {"set": "G0", "page_id": "c"}]  # fmt: skip
    monkeypatch.setattr(report, "read_jsonl", lambda path: rows)
    got = [(r["set"], r["page_id"]) for r in report.scores("test")]
    assert got == [("G2", "a"), ("G2", "b"), ("G0", "c"), ("G2-strict", "a")]
    rows.append({"set": "G0", "page_id": "d", "partial": True})
    with pytest.raises(ValueError, match="--partial"):
        report.endpoints("test")


def test_calibration_reads_each_side_alone_p_scan_over_every_word_and_both_together():
    where = {"page_id": "p1", "set": "G0", "cluster": "p1"}
    pscan = {"arm": "P-scan", **where, "probs": [[0, 0, 0.05], [1, 1, 0.95], [2, 0, 0.5],
                                                 [3, 1, None]]}  # fmt: skip
    model = {"arm": "m:R4:C2", **where, "probs": [[0, 0, 0.1], [1, 1, 0.7], [2, 0, None],
                                                  [3, 1, 0.2]]}  # fmt: skip
    blank = {"arm": "m:R5:C2", **where, "probs": [[0, 0, None], [1, 1, None]]}  # 0 to 100, say
    markdown = {"arm": "m:R2:C2", **where}  # no probabilities: left out
    summary, bins = report.calibration([pscan, model, blank, markdown])
    got = {(r["arm"], r["view"], r["side"]): r for r in summary}
    assert sorted(got) == [
        ("P-scan", "all", "P-scan"), ("P-scan", "own", "P-scan"),
        ("m:R4:C2", "own", "model"), ("m:R4:C2", "paired", "P-scan"),
        ("m:R4:C2", "paired", "model"), ("m:R5:C2", "own", "model"),
        ("m:R5:C2", "paired", "P-scan"), ("m:R5:C2", "paired", "model"),
    ]  # fmt: skip
    own = got[("m:R4:C2", "own", "model")]
    assert (own["words"], own["coverage"], own["struck_coverage"]) == (3, 0.75, 1.0)
    assert (own["ece"], own["brier"]) == pytest.approx((0.4, 0.74 / 3))
    mine = got[("P-scan", "own", "P-scan")]
    assert (mine["words"], mine["struck_coverage"], mine["ece"]) == (3, 0.5, pytest.approx(0.2))
    every = got[("P-scan", "all", "P-scan")]  # the word P-scan gave none counted at 0
    assert (every["words"], every["brier"]) == (4, pytest.approx(1.255 / 4))
    pair = got[("m:R4:C2", "paired", "model")]  # words 0 and 1: the model gave 2 none, P-scan 3
    assert pair["words"] == 2 and (pair["ece"], pair["brier"]) == pytest.approx((0.2, 0.05))
    theirs = got[("m:R4:C2", "paired", "P-scan")]
    assert (theirs["ece"], theirs["brier"]) == pytest.approx((0.05, 0.0025))
    assert pair["brier_gap"] == pytest.approx(0.0475) and pair["ece_lo"] is None  # one cluster
    empty = got[("m:R5:C2", "own", "model")]  # no usable probability: shown, not dropped
    assert (empty["words"], empty["coverage"]) == (0, 0.0) and math.isnan(empty["ece"])
    assert len({tuple(r) for r in summary}) == 1  # one set of columns, for the CSV
    mine = [(b["bin"], b["words"], b["struck_share"]) for b in bins
            if b["arm"] == "m:R4:C2" and b["view"] == "paired" and b["side"] == "model"]  # fmt: skip
    assert mine == [(0.1, 1, 0.0), (0.7, 1, 1.0)]


def test_calibration_intervals_come_from_six_clusters_and_bins_from_ten_words():
    rows = [{"arm": "P-scan", "page_id": f"p{i}", "set": "G0", "cluster": f"p{i}",
             "probs": [[0, 0, 0.2], [1, 1, 0.8]]} for i in range(6)]  # fmt: skip
    summary, bins = report.calibration(rows)
    own = next(r for r in summary if r["view"] == "own")
    assert own["clusters"] == 6 and own["brier"] == pytest.approx(0.04)
    assert (own["brier_lo"], own["brier_hi"]) == pytest.approx((0.04, 0.04))
    low = next(b for b in bins if b["view"] == "own" and b["bin"] == 0.2)
    assert (low["words"], low["read"]) == (6, False) and low["struck_lo"] == 0.0
    assert 0 < low["struck_hi"] < 0.5


def test_the_cascade_pairs_each_model_with_p_scan_on_its_uncertain_words():
    theirs = {"struck": 2, "struck_live": 2, "live": 3, "live_live": 3}
    page = {"struck": 10, "struck_live": 4, "struck_marked": 6, "live": 100, "live_live": 100}
    pscan = {"arm": "P-scan", "page_id": "p1", "set": "G0", **page,
             "subsets": {"uncertain": theirs}}  # fmt: skip
    model = {"arm": "m:R0:C2", "page_id": "p1", "set": "G0",
             "subsets": {"uncertain": {"struck": 2, "struck_marked": 1, "struck_live": 1,
                                       "live": 3, "live_marked": 1, "live_live": 2}}}  # fmt: skip
    calm = {"page_id": "p2", "subsets": {"uncertain": {}}}  # nothing uncertain on p2
    elsewhere = model | {"page_id": "p3"}  # P-scan has no row for p3: left out
    (row,) = report.cascade([pscan, pscan | calm, model, model | calm, elsewhere])
    assert (row["pages"], row["struck"], row["live"]) == (1, 2, 3)
    assert (row["leakage"], row["p_scan_leakage"]) == (0.5, 1.0)
    assert (row["over_marked"], row["p_scan_over_marked"]) == pytest.approx((1 / 3, 0.0))
    # whole pages: the model answers the uncertain words and P-scan the rest (on p2, all of it)
    assert row["cascade_page_leakage"] == pytest.approx(7 / 20)  # (4 - 2 + 1) + 4, of 10 + 10
    assert row["p_scan_page_leakage"] == pytest.approx(8 / 20)
    assert row["cascade_page_over_marked"] == pytest.approx(1 / 200)
    assert report.cascade([pscan | calm, model | calm]) == []  # no all-NaN row


def test_reliability_bins_take_the_tenth_a_probability_names():
    diagram, _ = report._calibrated([("c", 0, 1 - 0.9), ("c", 1, 1.0)])
    assert [b["bin"] for b in diagram] == [0.1, 0.9]  # 0.0999… is 0.1; 1.0 in the last bin


def test_an_empty_table_removes_the_csv_an_earlier_run_wrote(tmp_path):
    path = tmp_path / "cascade.csv"
    report._csv(path, [{"arm": "m", "pages": 1}])
    assert path.exists()
    report._csv(path, [])
    assert not path.exists()


def test_the_ablation_reads_low_against_the_high_runs_pooled_and_their_range(monkeypatch):
    def run(name: str, leaked: int, page: str = "p1", **extra) -> dict:
        return {"arm": f"m:R4:C2:{name}", "page_id": page, "set": "G0", "struck": 100,
                "struck_live": leaked, "struck_marked": 100 - leaked, "live": 100,
                "live_live": 100, "failed": False, "status": "ok", "cost_usd": 0.01,
                **extra}  # fmt: skip

    rows = [run("r1", 10), run("r2", 20), run("r3", 30), run("low", 40)]
    rows += [run(n, 0, "p2") for n in ("r1", "r2", "r3")]
    rows += [run("low", 100, "p2", failed=True, status="length")]  # low ran out on p2
    monkeypatch.setattr(report, "scores", lambda stage: rows)
    models = {"m": {"role": "ladder", "provider": "a"}}
    monkeypatch.setattr(report, "config", {"models": models}.__getitem__)
    monkeypatch.setattr(report.plans, "choice", lambda provider, key: "R4")
    out = report.ablation()["m"]
    assert (out["pages"], out["low"]["leakage"]) == (2, 0.7)  # the failure counted, worst case
    assert out["failures"] == {"m:R4:C2:r1": {}, "m:R4:C2:r2": {}, "m:R4:C2:r3": {},
                               "m:R4:C2:low": {"length": 1}}  # fmt: skip
    alone = out["without_failures"]  # p1 alone
    assert (alone["pages"], alone["low"]["leakage"]) == (1, 0.4)
    assert alone["high"]["leakage"] == pytest.approx(0.2)
    assert alone["high_range"]["leakage"] == [0.1, 0.3]
    assert alone["low_minus_high"]["leakage"] == pytest.approx(0.2)
    assert alone["low"]["usd_per_1000"] == pytest.approx(10.0)
    # p1 (10, 20, 30 | 40) and p2 (0, 0, 0 | 100): low's total, 140, is the highest of 16
    assert out["p"]["leakage"] == pytest.approx(2 / 16) and alone["p"]["over_marked"] == 1.0
    rows.pop(0)  # a run without its page is fine; a run without any score is not
    assert report.ablation()["m"]["pages"] == 1
    rows[:] = [r for r in rows if r["arm"] != "m:R4:C2:r2"]
    with pytest.raises(ValueError, match="no scores for m:R4:C2:r2"):
        report.ablation()
    rows.append(run("r2", 0) | {"partial": True})
    with pytest.raises(ValueError, match="--partial"):
        report.ablation()


@pytest.mark.parametrize("end, drift", [
    ([1] * 10, False),  # as at the start
    ([1] * 9 + [2], False),  # outside the start runs' range, which alone is no drift
    ([5] * 10, True),  # every page leaks more
])  # fmt: skip
def test_the_canary_tests_the_end_run_against_the_start_runs_page_by_page(
    monkeypatch, tmp_path, end, drift
):
    def run(leaked: list[int]) -> list[dict]:
        return [{"arm": "m:R0:C2", "model": "m", "page_id": f"p{i}", "struck": 10,
                 "struck_live": x, "struck_marked": 10 - x}
                for i, x in enumerate(leaked)]  # fmt: skip

    runs = {1: run([1] * 10), 2: run([1] * 10), 3: run([1] * 10), 4: run(end)}
    for n in runs:
        (tmp_path / f"canary-{n}").mkdir()
        (tmp_path / f"canary-{n}" / "scores.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(report, "RUNS", tmp_path)
    monkeypatch.setattr(report, "scores", lambda stage: runs[int(stage.split("-")[1])])
    (row,) = report.canary()
    assert row["drift"] is drift and row["pages"] == 10
    assert row["p"]["live_marked"] is None  # no live text on these pages: that count untested
    for r in runs[4]:  # the end run marks live words too: drift either way
        r |= {"live": 100, "live_marked": 5}
    for r in (r for n in (1, 2, 3) for r in runs[n]):
        r |= {"live": 100, "live_marked": 0}
    assert report.canary()[0]["drift"] is True
    runs[4] = []  # not run yet: untested, not "no drift"
    assert report.canary()[0]["drift"] is None
    runs[4] = run(end) + [{"page_id": "p0", "partial": True, "arm": "x", "model": "x"}]
    with pytest.raises(ValueError, match="--partial"):
        report.canary()


# --------------------------------------------------------------------- the endpoints, golden

BILLS = [f"b{i}" for i in range(1, 9)]  # G2's clusters; b1–b6 strict
STRUCK, LIVE = 200, 1000  # per page: 1 point is 2 struck or 10 live sub-tokens


def page(arm: str, set_: str, cluster: str, sl: int = 0, lm: int = 0, la: int = 0,
         failed: bool = False, **extra) -> dict:  # fmt: skip
    """A score row: `sl` struck sub-tokens left live (the rest marked), `lm` live ones marked and
    `la` absent; a failed page scored worst case."""
    if failed:
        sl, lm, la = STRUCK, 0, LIVE
    return {"arm": arm, "model": arm.split(":")[0], "page_id": f"{set_}.{cluster}.p0",
            "set": set_, "cluster": cluster, "strict": set_ == "G2" and cluster <= "b6",
            "struck": STRUCK, "struck_live": sl, "struck_marked": STRUCK - sl, "struck_absent": 0,
            "live": LIVE, "live_marked": lm, "live_absent": la, "live_live": LIVE - lm - la,
            "failed": failed, "status": "length" if failed else "ok", **extra}  # fmt: skip


def golden_rows() -> list[dict]:
    """f1 beats the package nowhere at R0 and matches it at R1 in every wording; f2 matches it in
    three wordings but marks 1.2 points of live text in the fourth. f1's PDF route failed one
    page; f2's reads text only."""
    rows = []
    for b in BILLS:
        rows += [page("P-native", "G2", b), page("P-scan", "G2", b, sl=40)]
        rows += [page("f1:R0:C1", "G2", b, sl=150, failed=b == "b8"),
                 page("f1:R0:C2", "G2", b, sl=100),
                 page("f1:R4:C2", "G2", b, sl=10)]  # fmt: skip
        rows += [page(report.r1_arm("f1", w), "G2", b, sl=1, lm=5, la=5) for w in report.WORDINGS]
        rows += [page("f2:R0:C1", "G2", b, sl=200),
                 page("f2:R0:C2", "G2", b, sl=0 if b in ("b6", "b7", "b8") else 120),
                 page("f2:R4:C2", "G2", b, sl=60)]  # fmt: skip
        rows += [page(report.r1_arm("f2", w), "G2", b, sl=1, lm=12 if w == 3 else 5, la=5)
                 for w in report.WORDINGS]  # fmt: skip
    for set_, cluster in (("G0", "g0"), ("G1", "g1")):
        rows += [page("P-native", set_, cluster), page("f1:R0:C1", set_, cluster, sl=50),
                 page("f1:R0:C2", set_, cluster, sl=20)]  # fmt: skip
    return rows


@pytest.fixture
def golden(monkeypatch):
    models = {"f1": {"role": "flagship", "provider": "a", "c1_route": "image+text"},
              "f2": {"role": "flagship", "provider": "b", "c1_route": "text"},
              "s": {"role": "ladder", "provider": "a"}}  # fmt: skip
    monkeypatch.setattr(report, "config", {"models": models}.__getitem__)
    monkeypatch.setattr(
        report.plans, "choice", lambda prov, key: {"flagship": "R4", "image": "C2"}[key]
    )
    rows = golden_rows()
    monkeypatch.setattr(report, "read_jsonl", lambda path: rows)
    out = report.endpoints("test")
    return {(r["endpoint"], r["claim"], r["set"]): r for r in out}, out


def test_the_endpoints_claims_tests_and_families_as_prereg_states_them(golden):
    claims, out = golden
    g2 = {(e, c): r for (e, c, s), r in claims.items() if s == "G2"}
    e1 = "P-native leaks less than {} R0, on C1 and C2"
    # 1: f1's failed page counts worst case in one analysis and leaves both sides in the other;
    # the claim takes the larger p: 1/256 with all eight bills, 1/128 with seven
    r = g2[(1, e1.format("f1"))]
    assert (r["p_worst_case"], r["p_without_failures"], r["p"]) == (1 / 256, 1 / 128, 1 / 128)
    assert r["failures"]["f1:R0:C1"] == {"length": 1} and r["failures"]["P-native"] == {}
    # f2 leaks no more than the package on three bills: 8 of 256 patterns reach its total
    assert g2[(1, e1.format("f2"))]["p"] == 8 / 256
    assert g2[(1, e1.format("f1"))]["p_holm"] == pytest.approx(2 / 128)  # Holm over two
    assert g2[(1, e1.format("f2"))]["p_holm"] == pytest.approx(8 / 256)
    assert g2[(1, e1.format("f1"))]["holds"] and not g2[(1, e1.format("f2"))]["holds"]
    # 2: within 1 point on leakage and marked over-deletion against P-native, and on absent
    # over-deletion against the same model at R0, in all four wordings: twelve tests
    e2 = "{} R1 C2 is no worse than P-native, in every wording"
    assert g2[(2, e2.format("f1"))]["p"] == 1 / 256 and g2[(2, e2.format("f1"))]["holds"]
    assert g2[(2, e2.format("f2"))]["p"] == 1.0  # one wording marks 1.2 points: no match
    assert not g2[(2, e2.format("f2"))]["holds"]
    assert set(g2[(2, e2.format("f1"))]["ci"]) == {f"wording {w}" for w in report.WORDINGS}
    # 3: each direction its own hypothesis, Holm over all four
    e3 = {
        "f1 above": "f1 R4 C2 F1 above P-scan",
        "above f1": "P-scan F1 above f1 R4 C2",
        "f2 above": "f2 R4 C2 F1 above P-scan",
        "above f2": "P-scan F1 above f2 R4 C2",
    }
    assert {k: g2[(3, v)]["p"] for k, v in e3.items()} == {
        "f1 above": 1 / 256,
        "above f1": 1.0,
        "f2 above": 1.0,
        "above f2": 1 / 256,
    }
    assert {k for k, v in e3.items() if g2[(3, v)]["holds"]} == {"f1 above", "above f2"}
    assert g2[(3, e3["f1 above"])]["p_holm"] == pytest.approx(4 / 256)
    # 4: C1 leaks more than C2; f2's PDF route passes text alone, so its claim says so
    e4 = "{} R0 C1 leaks more than C2"
    assert (g2[(4, e4.format("f1"))]["p"], g2[(4, e4.format("f2"))]["p"]) == (1 / 128, 1 / 256)
    assert g2[(4, e4.format("f2"))]["by_construction"]
    assert not g2[(4, e4.format("f1"))]["by_construction"]
    # G2-strict: tested, never adjusted; G0 pooled with an interval and G1 per cluster, untested
    strict = [r for (e, c, s), r in claims.items() if s == report.STRICT and "p" in r]
    assert len(strict) == 10 and all("p_holm" not in r and not r["insufficient"] for r in strict)
    assert claims[(1, e1.format("f1"), report.STRICT)]["ci"] is None
    shown = [r for r in out if r["set"] in ("G0", "G1")]
    assert shown and all("p" not in r and "p_holm" not in r for r in shown)
    g0 = next(r for r in shown if r["set"] == "G0" and r["endpoint"] == 1 and "f1" in r["claim"])
    assert g0["ci"]["f1:R0:C1"][0] == pytest.approx(0.25)  # 50 of 200 leaked, P-native none
    g1 = next(r for r in shown if r["set"] == "G1" and r["endpoint"] == 1 and "f1" in r["claim"])
    assert g1["by_cluster"]["f1:R0:C2"] == {"pooled": 0.1, "per_cluster": {"g1": 0.1}}
    held = sorted((r["endpoint"], r["claim"]) for r in out if r.get("holds"))
    assert len(held) == 6 and all(r["set"] == "G2" for r in out if "p_holm" in r)


def test_a_flagship_answered_by_two_versions_is_shown_outside_the_families(monkeypatch):
    models = {"f1": {"role": "flagship", "provider": "a", "c1_route": "image+text"},
              "f2": {"role": "flagship", "provider": "b", "c1_route": "text"}}  # fmt: skip
    monkeypatch.setattr(report, "config", {"models": models}.__getitem__)
    monkeypatch.setattr(
        report.plans, "choice", lambda prov, key: {"flagship": "R4", "image": "C2"}[key]
    )
    rows = golden_rows()
    for r in rows:
        if r["arm"] == "f2:R0:C2" and r["cluster"] == "b1":
            r["arm"] = "f2:R0:C2@f2-v2"  # scored by version: another version answered this page
    monkeypatch.setattr(report, "read_jsonl", lambda path: rows)
    g2 = [r for r in report.endpoints("test") if r["set"] == "G2"]
    assert {r["model"] for r in g2} == {"f1", "f2"}
    assert all(r["split_versions"] and "p_holm" not in r for r in g2 if r["model"] == "f2")
    assert all("p_holm" in r and not r["split_versions"] for r in g2 if r["model"] == "f1")
    e1 = next(r for r in g2 if r["endpoint"] == 1 and r["model"] == "f1")
    assert e1["p_holm"] == e1["p"]  # f1 alone in endpoint 1's family


def test_the_floors_read_both_analyses(monkeypatch):
    """f's PDF route fails three of the eight bills: without them its test counts five clusters,
    under the floor, so its claim leaves the family, and g is adjusted alone."""
    models = {"f": {"role": "flagship", "provider": "a", "c1_route": "image+text"},
              "g": {"role": "flagship", "provider": "b", "c1_route": "image+text"}}  # fmt: skip
    monkeypatch.setattr(report, "config", {"models": models}.__getitem__)
    monkeypatch.setattr(
        report.plans, "choice", lambda prov, key: {"flagship": "R4", "image": "C2"}[key]
    )
    rows = []
    for b in BILLS:
        rows += [page("P-native", "G2", b), page("f:R0:C2", "G2", b, sl=100),
                 page("f:R0:C1", "G2", b, sl=150, failed=b in ("b6", "b7", "b8")),
                 page("g:R0:C1", "G2", b, sl=150), page("g:R0:C2", "G2", b, sl=100)]  # fmt: skip
    monkeypatch.setattr(report, "read_jsonl", lambda path: rows)
    e1 = {
        r["model"]: r for r in report.endpoints("test") if r["endpoint"] == 1 and r["set"] == "G2"
    }
    assert (e1["f"]["clusters"], e1["f"]["clusters_without_failures"]) == (5, 5)
    assert e1["f"]["insufficient"] and "p_holm" not in e1["f"]
    assert e1["g"]["p_holm"] == e1["g"]["p"] == 1 / 256  # alone in the family


def test_endpoint_2_fails_a_one_line_prompt_that_drops_live_text(monkeypatch):
    """R1 leaks no more and marks no more than the package, but leaves out 1.5 points of live
    text that the same model at R0 kept: no match."""
    models = {"f": {"role": "flagship", "provider": "a", "c1_route": "image+text"}}
    monkeypatch.setattr(report, "config", {"models": models}.__getitem__)
    monkeypatch.setattr(
        report.plans, "choice", lambda prov, key: {"flagship": "R4", "image": "C2"}[key]
    )
    rows = []
    for b in BILLS:
        rows += [page("P-native", "G2", b), page("f:R0:C2", "G2", b, sl=100)]
        rows += [page(report.r1_arm("f", w), "G2", b, la=15) for w in report.WORDINGS]
    monkeypatch.setattr(report, "read_jsonl", lambda path: rows)
    (e2,) = [r for r in report.endpoints("test") if r["endpoint"] == 2 and r["set"] == "G2"]
    assert e2["p"] == 1.0 and not e2["holds"]
    for r in rows:
        if ":R1:" in r["arm"]:
            r |= {"live_absent": 5, "live_live": LIVE - 5}  # half a point: within the bound
    (e2,) = [r for r in report.endpoints("test") if r["endpoint"] == 2 and r["set"] == "G2"]
    assert e2["p"] == 1 / 256 and e2["holds"]


def test_reading_leakage_and_list_prices_and_the_by_construction_flag(monkeypatch, tmp_path):
    handwriting = {"struck": 10, "struck_live": 2, "struck_marked": 6, "struck_absent": 2,
                   "cost_usd": 0.01, "cost_list_usd": 0.02}  # fmt: skip
    m = report.metrics([handwriting])
    assert (m["leakage"], m["read_leakage"]) == (0.2, 0.25)  # absent words read nothing
    assert (m["usd_per_1000"], m["usd_per_1000_list"]) == pytest.approx((10.0, 20.0))
    rows = [{"arm": a, "set": "G1", "cluster": "d", "page_id": "p", "struck": 1, "struck_marked": 1}
            for a in ("P-native", "B-pm4llm", "P-scan")]  # fmt: skip
    monkeypatch.setattr(report, "RUNS", tmp_path)
    monkeypatch.setattr(report, "scores", lambda stage: rows)
    (tmp_path / "test").mkdir()
    flags = {r["arm"]: r["by_construction"] for r in report.tables("test")}
    assert flags == {"B-pm4llm": True, "P-native": True, "P-scan": False}


def test_the_cascade_reports_each_band_its_hand_off_and_its_price():
    pscan = {"arm": "P-scan", "page_id": "p1", "set": "G0", "struck": 10, "struck_marked": 10,
             "live": 90, "live_live": 90, "cost_usd": 0.001, "priced": True,
             "subsets": {"uncertain": {"struck": 1, "struck_live": 1},
                         "uncertain 0.40-0.60": {}}}  # fmt: skip
    other = pscan | {"page_id": "p2", "subsets": {"uncertain": {}, "uncertain 0.40-0.60": {}}}
    model = {"arm": "m:R0:C2", "page_id": "p1", "set": "G0", "cost_usd": 0.01, "priced": True,
             "subsets": {"uncertain": {"struck": 1, "struck_marked": 1},
                         "uncertain 0.40-0.60": {}}}  # fmt: skip
    model2 = model | {"page_id": "p2", "subsets": other["subsets"]}
    (row,) = report.cascade([pscan, other, model, model2])  # the narrow band hands off nothing
    assert row["band"] == "uncertain" and (row["pages"], row["handoff_pages"]) == (1, 0.5)
    assert row["handoff_sub_tokens"] == pytest.approx(1 / 200)
    # P-scan on both pages, and the model on the one handed off
    assert row["cascade_usd_per_1000"] == pytest.approx(1000 * (0.002 + 0.01) / 2)


def test_the_window_holds_the_latency_sample_and_a_model_with_no_canary(monkeypatch, tmp_path):
    day = datetime(2026, 11, 2, 9)
    sent = {"canary-1": {"m": [day]}, "canary-4": {"m": [day + timedelta(days=6)]},
            "test": {"m": [day + timedelta(hours=1)]},
            "latency": {"m": [day + timedelta(days=7)]}}  # fmt: skip
    assert report.window("m", sent)["outside"] == 1  # timed after the end canary
    del sent["canary-4"]
    assert report.window("m", sent)["outside"] == 0  # the window still open
    assert report.window("x", {"test": {"x": [day]}})["outside"] == 1  # no canary: all outside
    (tmp_path / "test").mkdir()
    reply = {"model": "x", "started": day.isoformat(), "status": "ok"}
    (tmp_path / "test" / "responses.jsonl").write_text(json.dumps(reply) + "\n", "utf-8")
    monkeypatch.setattr(report, "RUNS", tmp_path)
    (row,) = report.canary()  # sent test requests, never a canary: listed, untested
    assert (row["model"], row["drift"], row["window"]["outside"]) == ("x", None, 1)


def test_the_canary_reports_each_models_window_and_what_falls_outside(monkeypatch, tmp_path):
    day = datetime(2026, 11, 2, 9)
    sent = {"canary-1": {"m": [day]}, "canary-4": {"m": [day + timedelta(days=8)]},
            "test": {"m": [day + timedelta(hours=1), day + timedelta(days=9)]}}  # fmt: skip
    w = report.window("m", sent)
    assert w["overrun"] and w["outside"] == 1 and w["days"] == pytest.approx(8.0)
    sent["canary-4"]["m"] = [day + timedelta(days=6)]
    sent["test"]["m"] = [day + timedelta(hours=1), day + timedelta(days=5)]
    w = report.window("m", sent)
    assert not w["overrun"] and w["outside"] == 0
    assert report.window("m", {"canary-1": {"m": [day]}}) is None  # no test request yet
    (tmp_path / "test").mkdir()
    batch = {"batch_id": "b", "submitted": day.isoformat()}
    (tmp_path / "test" / "batches.jsonl").write_text(json.dumps(batch) + "\n", "utf-8")
    reply = {
        "model": "m",
        "batch_id": "b",
        "started": "",
        "recorded": (day + timedelta(days=3)).isoformat(),
    }
    (tmp_path / "test" / "responses.jsonl").write_text(json.dumps(reply) + "\n", "utf-8")
    monkeypatch.setattr(report, "RUNS", tmp_path)
    assert report._sent_times()["test"]["m"] == [day]  # a batched request: sent when submitted


def test_the_strike_columns_set_finding_strikes_apart_from_reading():
    row = {"struck": 10, "struck_marked": 6, "struck_live": 2, "struck_absent": 2, "live": 20,
           "live_marked": 2}  # fmt: skip
    m = report.metrics([row])
    assert (m["strikes_found"], m["strikes_read"]) == (0.6, 0.8)
    assert (m["strikes_found_of_read"], m["mark_precision"]) == (0.75, 0.75)
    assert m["strikes_found"] == pytest.approx(1 - m["unmarked"])
    assert m["strikes_found_of_read"] == pytest.approx(1 - m["read_leakage"])


def _route_row(arm, i, sl=0, sm=20, sa=0, lm=0, la=0, usd=0.0, failed=False):
    """One G0 page of 20 struck and 100 live sub-tokens, for the routing tests."""
    return {"arm": arm, "page_id": f"G0.p{i}", "set": "G0", "cluster": f"G0.p{i}", "struck": 20,
            "struck_live": sl, "struck_marked": sm, "struck_absent": sa, "live": 100,
            "live_live": 100 - lm - la, "live_marked": lm, "live_absent": la, "failed": failed,
            "priced": True, "cost_usd": usd, "cost_list_usd": usd}  # fmt: skip


def test_routes_guard_the_reference_and_price_the_paid_option(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "RUNS", tmp_path)  # no latency sample
    rows = []
    for i in range(8):
        rows += [
            _route_row("P-scan", i, sl=4, sm=16, usd=0.0001),
            _route_row("good:R2:C2", i, usd=0.02),
            _route_row("cheap:R2:C2", i, sl=int(i == 0), sm=20 - int(i == 0), usd=0.005),
            _route_row("markall:R2:C2", i, lm=100, usd=0.001),  # no misses: every word marked
            _route_row("dropper:R2:C2", i, sm=0, sa=20, usd=0.001),  # no leaks: strikes left out
            _route_row("partial:R2:C2:p1", i, usd=0.0001),  # a paraphrase is not a system
        ]  # fmt: skip
    monkeypatch.setattr(report, "scores", lambda stage: rows)
    picked, guards = report.routes("test")
    got = {r["role"]: r["system"] for r in picked if r["kind"] == "images of print"}
    assert got == {"free default": "P-scan", "reference": "good:R2:C2",
                   "paid option": "cheap:R2:C2"}  # fmt: skip
    assert not any(g["system"] == "partial:R2:C2:p1" for g in guards)
    marked = [g for g in guards if g["system"] == "markall:R2:C2" and g["rate"] == "over_marked"]
    assert marked[0]["against"] == "P-scan" and marked[0]["shown_worse"]


def test_a_cheaper_cascade_replaces_the_paid_option_and_failures_are_each_systems_own(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(report, "RUNS", tmp_path)
    (tmp_path / "latency").mkdir()  # the latency sample times the model alone
    (tmp_path / "latency" / "scores.jsonl").write_text(
        json.dumps({"arm": "good:R2:C2", "latency_s": 12.0}) + "\n", encoding="utf-8"
    )
    rows = []
    for i in range(8):  # P-scan misses only the one word it leaves uncertain, on pages 6 and 7
        hard = i >= 6
        rows += [
            _route_row("P-scan", i, sl=int(hard), sm=20 - int(hard), usd=0.0001)
            | {"inserted": 6, "subsets": {"uncertain": {"struck": 1, "struck_live": 1} if hard else {}}},
            _route_row("good:R2:C2", i, usd=0.02, failed=i == 6)
            | {"subsets": {"uncertain": {"struck": 1, "struck_marked": 1} if hard else {}}},
        ]  # fmt: skip
    monkeypatch.setattr(report, "scores", lambda stage: rows)
    picked, _ = report.routes("test")
    got = {r["role"]: r for r in picked if r["kind"] == "images of print"}
    assert got["paid option"]["system"] == "cascade:good:R2:C2"
    assert got["replaced paid option"]["system"] == "good:R2:C2"
    assert got["reference"]["failure_rate"] == 1 / 8  # its own failure, not the fallback's
    assert got["paid option"]["failure_rate"] == 1 / 8  # the page it handed off and failed
    assert got["paid option"]["halluc_per_1000"] == pytest.approx(1000 * 6 / 120)  # P-scan's text
    assert got["reference"]["s_per_page"] == 12.0
    for r in rows:  # an unpriced model: no cascade can be shown to cost half as much
        if r["arm"] == "good:R2:C2":
            r["priced"] = False
    picked, _ = report.routes("test")
    got = {r["role"]: r for r in picked if r["kind"] == "images of print"}
    assert "replaced paid option" not in got and got["paid option"]["system"] == "good:R2:C2"
    assert math.isnan(got["paid option"]["usd_per_1000_list"])


def test_a_failed_page_reads_none_of_its_strikes():
    ok = {"struck": 10, "struck_marked": 8, "struck_live": 2, "live": 10, "live_live": 10}
    failed = {"struck": 10, "struck_live": 10, "live": 10, "live_absent": 10, "failed": True}
    m = report.metrics([ok, failed])
    assert m["strikes_read"] == 0.5 and m["strikes_found_of_read"] == 0.4


def test_a_paid_systems_failed_page_scores_as_the_free_defaults_output():
    free = {"p": {"struck_live": 4, "failed": False}}
    paid = {"p": {"struck_live": 20, "failed": True}}
    assert report._fallback(paid, free) == free
    assert report._fallback(free, paid) == free  # pages it answered stay its own
