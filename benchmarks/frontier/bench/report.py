"""Results (PREREG.md, Primary endpoints, Statistics, Prompts): the four endpoints, R-best and the
flagship check, the effort ablation, the results tables and the canary's drift and window checks.

    python -m bench report endpoints [test]          # the claims, each tested with and without
                                                     # failed pages; G0 and G1 beside them
    python -m bench report rbest [ladder]            # R-best, runner-up and image (ladder-tiled)
    python -m bench report flagship-check            # each flagship's rung: R-best or runner-up
    python -m bench report ablation                  # low effort against the repeats' runs at high
    python -m bench report tables <stage>            # per arm and set, cluster, G0 stratum, the
                                                     # calibration and the cascade -> CSV
    python -m bench report canary                    # drift between each model's canary runs,
                                                     # and each model's test window

The endpoints are tested on G2 alone. G2's strict subset, the bills introduced after the cutoff,
is reported beside G2 as the set `G2-strict` (PREREG, G2), with its p-values but outside the Holm
families; G0 is reported pooled with its intervals, and G1 per cluster, neither tested.
"""

import csv
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from fractions import Fraction

import numpy as np

from bench import plans
from bench.common import RUNS, config, provenance, read_jsonl
from bench.score import COUNTS
from bench.stats import bootstrap_ci, f1_sign_flip_p, holm, relabel_p, sign_flip_p, wilson

MIN_STRUCK, MIN_CLUSTERS, ALPHA, POINT = 200, 6, 0.025, 0.01
DELTA = Fraction(2, 100)  # rungs within 2 points of the fewest unmarked count as tied
COST = {"R2": 0, "R3": 0, "R4": 1, "R5": 2, "R6": 3, "C2": 0, "C2-tiled": 1}  # cost tiers
WORDINGS = (0, 1, 2, 3)  # endpoint 2: R1's base wording and its three paraphrases
DRIFT_P = 0.025  # each of the canary's two tests (PREREG, Operations)
WINDOW = timedelta(days=7)  # each model's test requests, between its canaries
CANARY = {"struck_live": "struck", "live_marked": "live"}  # each count, and what it is out of
STRICT = "G2-strict"
# arms that agree with G1's ground truth by its construction: reported as such, never as results
BY_CONSTRUCTION = {("P-native", "G1"), ("B-pm4llm", "G1")}
QA_ANSWERS = ("live", "struck", "both", "neither", "failed")
BINS = 10  # the reliability diagrams' equal-width bins
MIN_BIN = 10  # a bin with fewer words is shown, not read
# a calibration cluster's sums: words, squared error, then per bin its words, probabilities, struck
NB, SP, SY = (slice(2 + i * BINS, 2 + (i + 1) * BINS) for i in range(3))
KEEP = ("struck", "leakage", "unmarked", "over_marked", "over_absent", "f1")  # a subset's metrics
OUTCOMES = ("struck", "struck_live", "struck_marked", "struck_absent",
            "live", "live_live", "live_marked", "live_absent")  # fmt: skip
WHOLE = ("leakage", "unmarked", "over_marked", "over_absent", "f1")  # the cascade's whole pages
# the effort ablation's tests: per page, the error counts and the sub-tokens they can occur in
TESTED = {"leakage": (("struck_live",), "struck"),
          "unmarked": (("struck_live", "struck_absent"), "struck"),
          "over_marked": (("live_marked",), "live")}  # fmt: skip


def scores(stage: str) -> list[dict]:
    """A stage's score rows, G2's strict bills repeated as the set G2-strict."""
    rows = read_jsonl(RUNS / stage / "scores.jsonl")
    if not rows:
        raise FileNotFoundError(
            f"no scores for {stage}; run `python -m bench harness score {stage}`"
        )
    return rows + [r | {"set": STRICT} for r in rows if r.get("strict")]


def _rate(num: float, den: float) -> float:
    return num / den if den else float("nan")


def _refuse_partial(rows: list[dict], what: str) -> None:
    if any(r.get("partial") for r in rows):
        raise ValueError(f"{what} was scored with --partial; score it complete first")


def metrics(rows: list[dict]) -> dict:
    """Pooled over the rows' sub-tokens (PREREG, Scoring); cost and seconds per page are Q5's,
    the cost unknown (NaN) when any row lacks a price, never a silent zero. `read_leakage` is
    leakage among the struck sub-tokens the output read (live or marked), for handwriting, where
    a system that can't read a struck word leaves it absent rather than live. The strike columns
    (exploratory, deviation 3) say how well a system finds strikes apart from how well it reads:
    the share of struck sub-tokens marked, the share it read at all, the share marked among those
    read, and the share of its marks that fall on struck text. On a failed page the first three
    take their worst (its struck text unread in strikes read, read but missed in found of those
    read); it makes no marks, so mark precision leaves it out."""
    c = {k: sum(r.get(k, 0) for r in rows) for k in COUNTS}
    unread = sum(r.get("struck_live", 0) for r in rows if r.get("failed"))
    hits, misses = c["struck_marked"], c["struck_live"] + c["struck_absent"]
    tokens, n = c["struck"] + c["live"], len(rows)
    priced = all(r.get("priced", True) for r in rows)
    usd = sum(r.get("cost_usd") or 0 for r in rows) if priced else math.nan
    listed = (
        sum(r.get("cost_list_usd", r.get("cost_usd")) or 0 for r in rows) if priced else math.nan
    )
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    return {
        "pages": n,
        "struck": c["struck"],
        "dropped": c["dropped"],  # sub-tokens that normalize to nothing, counted not scored
        "leakage": _rate(c["struck_live"], c["struck"]),
        "read_leakage": _rate(c["struck_live"], c["struck_live"] + c["struck_marked"]),
        "over_marked": _rate(c["live_marked"], c["live"]),
        "over_absent": _rate(c["live_absent"], c["live"]),
        "unmarked": _rate(misses, c["struck"]),
        "f1": _rate(2 * hits, 2 * hits + misses + c["live_marked"]),
        "strikes_found": _rate(hits, c["struck"]),
        "strikes_read": _rate(hits + c["struck_live"] - unread, c["struck"]),
        "strikes_found_of_read": _rate(hits, hits + c["struck_live"]),
        "mark_precision": _rate(hits, hits + c["live_marked"]),
        "halluc_per_1000": _rate(1000 * c["inserted"], tokens),
        "halluc_digit_per_1000": _rate(1000 * c["inserted_digit"], tokens),
        "cer": _rate(c["cer_num"], c["cer_den"]),
        "wer": _rate(c["wer_num"], c["wer_den"]),
        "failure_rate": _rate(sum(bool(r.get("failed")) for r in rows), n),
        "refusal_rate": _rate(sum(r.get("status") == "refusal" for r in rows), n),
        "parse_failure_rate": _rate(sum(r.get("status") == "parse_error" for r in rows), n),
        "usd_per_1000": _rate(1000 * usd, n),
        "usd_per_1000_list": _rate(1000 * listed, n),  # at list price, once promotions end
        "s_per_page": _rate(sum(lat), len(lat)),
    }


# --------------------------------------------------------------------------------- endpoints

RATE = ("struck", "struck_live", "live", "live_marked", "live_absent")  # the rate claims' columns
F1 = ("struck_marked", "struck_live", "struck_absent", "live_marked")


def clusters(rows: list[dict], a: str, b: str, set_: str, keys: tuple, drop_failed=False,
             ids: bool = False):  # fmt: skip
    """Per-cluster sums over the pages both arms scored in one set: `keys` of a, then of b; with
    `drop_failed`, a page either arm failed leaves both sides (PREREG, Scoring: failures).
    Clusters enter in their IDs' order, so the random sign patterns reproduce; with `ids`, the IDs
    come back too."""
    by = {(r["arm"], r["page_id"]): r for r in rows if r["set"] == set_ and "struck" in r}
    sums: dict[str, np.ndarray] = {}
    for arm, pid in by:
        if arm != a or (b, pid) not in by:
            continue
        x, y = by[(a, pid)], by[(b, pid)]
        if drop_failed and (x["failed"] or y["failed"]):
            continue
        row = np.array([x[k] for k in keys] + [y[k] for k in keys], dtype=float)
        sums[x["cluster"]] = sums.get(x["cluster"], 0) + row
    order = sorted(sums)
    out = np.array([sums[k] for k in order]).reshape(-1, 2 * len(keys))
    return (out, order) if ids else out


def rate_test(c: np.ndarray, err: str, den: str, bound: float, above: bool) -> dict:
    """One-sided: rate_A − rate_B lies above (or below) `bound`, by the sign-flip test on
    E_A − E_B − bound·S per cluster, clusters with S = 0 left out."""
    k, e, s = len(RATE), RATE.index(err), RATE.index(den)
    d = c[:, e] - c[:, k + e] - bound * c[:, s]
    keep = c[:, s] > 0
    return {"p": sign_flip_p(d[keep] if above else -d[keep]), "clusters": int(keep.sum()),
            "struck": int(c[:, 0].sum())}  # fmt: skip


def rate_ci(c: np.ndarray, err: str, den: str) -> list[float]:
    """rate_A − rate_B, pooled, with its paired cluster bootstrap interval."""
    k, e, s = len(RATE), RATE.index(err), RATE.index(den)
    est = _rate(c[:, e].sum() - c[:, k + e].sum(), c[:, s].sum())
    return [est, *bootstrap_ci(c, lambda x: (x[..., e] - x[..., k + e]) / x[..., s])]


def rate_by_cluster(c: np.ndarray, order: list[str], err: str, den: str) -> dict:
    """rate_A − rate_B pooled and per cluster, with no interval: for sets with too few clusters to
    read one from (G1, G2-strict)."""
    k, e, s = len(RATE), RATE.index(err), RATE.index(den)
    return {"pooled": _rate(c[:, e].sum() - c[:, k + e].sum(), c[:, s].sum()),
            "per_cluster": {cid: _rate(x[e] - x[k + e], x[s]) for cid, x in zip(order, c)}}  # fmt: skip


def _f1(x: np.ndarray, at: int) -> np.ndarray:
    """Pooled F1 from (marked, live, absent, false-alarm) columns starting at `at`; 0 when there
    is nothing to score, as in the F1 test."""
    hits, misses, fa = x[..., at], x[..., at + 1] + x[..., at + 2], x[..., at + 3]
    den = 2 * hits + misses + fa
    return np.where(den > 0, 2 * hits / np.where(den > 0, den, 1), 0.0)


def f1_tests(c: np.ndarray) -> tuple[dict, dict, list[float]]:
    """Endpoint 3's two hypotheses (A's pooled F1 above B's, and B's above A's), and F1_A − F1_B
    with its interval."""
    a = np.c_[c[:, 0], c[:, 1] + c[:, 2], c[:, 3]]
    b = np.c_[c[:, 4], c[:, 5] + c[:, 6], c[:, 7]]
    n = {"clusters": len(c), "struck": int(c[:, 0:3].sum())}
    return {"p": f1_sign_flip_p(a, b), **n}, {"p": f1_sign_flip_p(b, a), **n}, f1_ci(c)


def f1_ci(c: np.ndarray) -> list[float]:
    """F1_A − F1_B, pooled, with its paired cluster bootstrap interval."""
    diff = float(_f1(c.sum(axis=0), 0) - _f1(c.sum(axis=0), 4)) if len(c) else math.nan
    return [diff, *bootstrap_ci(c, lambda x: _f1(x, 0) - _f1(x, 4))]


def claim(endpoint: int, text: str, set_: str, tests: list[dict], ci=None) -> dict:
    """A claim needing every test to hold: the largest p-value; "insufficient data" under 200
    struck sub-tokens or 6 clusters (PREREG, Primary endpoints)."""
    struck, n = min(t["struck"] for t in tests), min(t["clusters"] for t in tests)
    return {"endpoint": endpoint, "claim": text, "set": set_, "p": max(t["p"] for t in tests),
            "struck": struck, "clusters": n, "ci": ci,
            "insufficient": struck < MIN_STRUCK or n < MIN_CLUSTERS}  # fmt: skip


def r1_arm(model: str, wording: int) -> str:
    """Endpoint 2's model side in one wording: R1's base (0) or a paraphrase."""
    return f"{model}:R1:C2" + (f":p{wording}" if wording else "")


def endpoints(stage: str = "test") -> list[dict]:
    """Every claim on G2, its p-value and its Holm-adjusted p-value within its endpoint, with the
    numbers reported beside it. A claim's p is the larger of two analyses (PREREG, the decision
    rule): failed pages scored worst case, and failed pages left out of both sides; both are
    shown, with each arm's failures by kind. G2-strict's claims are shown unadjusted, outside the
    families; G0 is reported pooled, with intervals, and G1 per cluster, neither tested. A
    flagship whose answers were scored by version (`--split-versions`, PREREG, Operations) has its
    claims shown on the pinned version's pages, outside the families too. A stage scored with
    `--partial` is refused."""
    rows = scores(stage)
    _refuse_partial(rows, stage)
    models = config("models")
    split = {r["model"] for r in rows if "@" in r["arm"]}  # answers from two versions of a model
    out: list[dict] = []

    def pair(a: str, b: str, set_: str, keys: tuple = RATE, drop: bool = False) -> np.ndarray:
        return clusters(rows, a, b, set_, keys, drop)

    failed: dict[tuple, Counter] = {}
    for r in rows:
        if r.get("failed"):
            failed.setdefault((r["arm"], r["set"]), Counter())[r["status"]] += 1

    def failures(arms: list[str], set_: str) -> dict:
        return {a: dict(failed.get((a, set_), {})) for a in dict.fromkeys(arms)}

    def tested(endpoint: int, text: str, set_: str, tests, arms: list[str], ci, **extra) -> dict:
        worst, without = tests(False), tests(True)
        # the floors take both analyses' tests: a claim stands on both, so neither may be thin
        c = claim(endpoint, text, set_, worst + without, ci)
        p_worst, p_without = max(t["p"] for t in worst), max(t["p"] for t in without)
        thin = {"struck_without_failures": min(t["struck"] for t in without),
                "clusters_without_failures": min(t["clusters"] for t in without)}  # fmt: skip
        return c | thin | {"p_worst_case": p_worst, "p_without_failures": p_without,
                           "p": max(p_worst, p_without), "failures": failures(arms, set_),
                           **extra}  # fmt: skip

    for k in (k for k, m in models.items() if m["role"] == "flagship"):
        first = len(out)
        prov = models[k]["provider"]
        rung, image = plans.choice(prov, "flagship"), plans.choice(prov, "image")
        best = f"{k}:{rung}:{image}"
        c1, c2 = f"{k}:R0:C1", f"{k}:R0:C2"
        r0 = c2  # endpoint 2's own baseline: the same model at R0 on C2
        text_only = models[k].get("c1_route") == "text"
        for set_ in ("G2", STRICT):
            interval = set_ == "G2"  # G2-strict's few bills are shown per cluster instead
            # 1: the package leaks less than the flagship at R0, on C1 and on C2
            out.append(tested(
                1, f"P-native leaks less than {k} R0, on C1 and C2", set_,
                lambda d, s=set_: [rate_test(pair(x, "P-native", s, drop=d), "struck_live",
                                             "struck", 0.0, above=True) for x in (c1, c2)],
                [c1, c2, "P-native"],
                {x: rate_ci(pair(x, "P-native", set_), "struck_live", "struck")
                 for x in (c1, c2)} if interval else None,
            ))  # fmt: skip

            # 2: R1 is no worse than the package, in every wording: leakage and marked
            # over-deletion within 1 point of P-native's, and absent over-deletion within 1 point
            # of the same model's at R0
            def e2(d: bool, s: str = set_) -> list[dict]:
                tests = []
                for w in WORDINGS:
                    c = pair(r1_arm(k, w), "P-native", s, drop=d)
                    tests += [rate_test(c, "struck_live", "struck", POINT, above=False),
                              rate_test(c, "live_marked", "live", POINT, above=False),
                              rate_test(pair(r1_arm(k, w), r0, s, drop=d), "live_absent", "live",
                                        POINT, above=False)]  # fmt: skip
                return tests

            ci = {f"wording {w}": {
                "leakage": rate_ci(pair(r1_arm(k, w), "P-native", set_), "struck_live", "struck"),
                "over_marked": rate_ci(pair(r1_arm(k, w), "P-native", set_), "live_marked", "live"),
                "over_absent_vs_r0": rate_ci(pair(r1_arm(k, w), r0, set_), "live_absent", "live"),
            } for w in WORDINGS} if interval else None  # fmt: skip
            arms = [*(r1_arm(k, w) for w in WORDINGS), r0, "P-native"]
            out.append(tested(2, f"{k} R1 C2 is no worse than P-native, in every wording", set_,
                              e2, arms, ci))  # fmt: skip

            # 3: F1 at the flagship's R-best against P-scan, each direction its own hypothesis
            f1 = {d: f1_tests(pair(best, "P-scan", set_, F1, d)) for d in (False, True)}
            native = pair(best, "P-native", set_, F1)  # the same pages
            ci = {"F1 model − P-scan": f1[False][2],
                  "P-native F1": float(_f1(native.sum(axis=0), 4)) if len(native) else math.nan}  # fmt: skip
            ci = ci if interval else None
            for side, text in ((0, f"{k} {rung} {image} F1 above P-scan"),
                               (1, f"P-scan F1 above {k} {rung} {image}")):  # fmt: skip
                out.append(tested(3, text, set_, lambda d, x=side: [f1[d][x]],
                                  [best, "P-scan"], ci))  # fmt: skip

            # 4: the PDF route leaks more than the image, at R0
            out.append(tested(
                4, f"{k} R0 C1 leaks more than C2", set_,
                lambda d, s=set_: [rate_test(pair(c1, c2, s, drop=d), "struck_live", "struck",
                                             0.0, above=True)],
                [c1, c2], rate_ci(pair(c1, c2, set_), "struck_live", "struck") if interval else None,
                by_construction=text_only,
            ))  # fmt: skip
        out += _reported(rows, k, best, c1, c2, r0)
        for r in out[first:]:
            r |= {"model": k, "split_versions": k in split}
    for e in (1, 2, 3, 4):
        family = [r for r in out if r["endpoint"] == e and r["set"] == "G2"
                  and not r["insufficient"] and not r["split_versions"]]  # fmt: skip
        for r, adj in zip(family, holm([r["p"] for r in family])):
            r["p_holm"], r["holds"] = adj, adj < ALPHA
    return out


def _reported(rows: list[dict], k: str, best: str, c1: str, c2: str, r0: str) -> list[dict]:
    """The endpoints' differences on G0, pooled with their intervals, and on G1 and G2-strict per
    cluster, with no interval: reported, never tested."""
    out = []
    for set_ in ("G0", "G1", STRICT):

        def diff(a: str, b: str, err: str, den: str, keys: tuple = RATE, s: str = set_):
            c, order = clusters(rows, a, b, s, keys, ids=True)
            return rate_ci(c, err, den) if s == "G0" else rate_by_cluster(c, order, err, den)

        def f1_diff(a: str, b: str, s: str = set_):
            c, order = clusters(rows, a, b, s, F1, ids=True)
            if s == "G0":
                return f1_ci(c)
            pooled = float(_f1(c.sum(axis=0), 0) - _f1(c.sum(axis=0), 4)) if len(c) else math.nan
            return {"pooled": pooled,
                    "per_cluster": {cid: float(_f1(x, 0) - _f1(x, 4)) for cid, x in zip(order, c)}}  # fmt: skip

        found = {
            1: {x: diff(x, "P-native", "struck_live", "struck") for x in (c1, c2)},
            2: {f"wording {w}": {
                "leakage": diff(r1_arm(k, w), "P-native", "struck_live", "struck"),
                "over_marked": diff(r1_arm(k, w), "P-native", "live_marked", "live"),
                "over_absent_vs_r0": diff(r1_arm(k, w), r0, "live_absent", "live"),
            } for w in WORDINGS},
            3: {"F1 model − P-scan": f1_diff(best, "P-scan")},
            4: {"C1 − C2 leakage": diff(c1, c2, "struck_live", "struck")},
        }  # fmt: skip
        out += [{"endpoint": e, "claim": f"{k}: reported, not tested", "set": set_,
                 "ci" if set_ == "G0" else "by_cluster": v} for e, v in found.items()]  # fmt: skip
    return out


# ------------------------------------------------------------------------------------- R-best


def _cells(rows: list[dict], model: str, keys: list[tuple]) -> dict[tuple, list[dict]]:
    """A model's scored rows for the (rung, input) cells in `keys`, each restricted to the pages
    all of those cells scored, so their rates compare the same pages. G2-strict's copies of G2's
    rows are left out."""
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        if (
            r["model"] == model
            and "struck" in r
            and r.get("set") != STRICT
            and ((r["rung"], r["input"]) in keys)
        ):
            cells.setdefault((r["rung"], r["input"]), []).append(r)
    common = (
        set.intersection(*({r["page_id"] for r in v} for v in cells.values())) if cells else set()
    )
    return {key: [r for r in v if r["page_id"] in common] for key, v in cells.items()}


def _frac(rows: list[dict], num: tuple[str, ...], den: str) -> Fraction | None:
    d = sum(r[den] for r in rows)
    return Fraction(sum(r[k] for r in rows for k in num), d) if d else None


def _excess(cells: dict, base: tuple, key: tuple) -> Fraction:
    """How far `key`'s over-deletion, marked or absent, goes past 1 point above `base`'s; 0 or
    less when both stay within."""
    return max((_frac(cells[key], (k,), "live") or 0) - (_frac(cells[base], (k,), "live") or 0)
               - Fraction(1, 100) for k in ("live_marked", "live_absent"))  # fmt: skip


def _unmarked(rows: list[dict]) -> Fraction:
    """The share of struck sub-tokens left unmarked (live or absent); 2, worse than any share,
    when there are none to read."""
    u = _frac(rows, ("struck_live", "struck_absent"), "struck")
    return u if u is not None else Fraction(2)


def _cost(option: tuple) -> int:
    rung, input_ = option
    return COST.get(rung, 0) + COST.get(input_, 0)


def _on(cells: dict, keys: list[tuple]) -> dict[tuple, list[dict]]:
    """The keys' rows on the pages every one of them scored and none of them failed."""
    pages = set.intersection(*({r["page_id"] for r in cells[k]} for k in keys))
    pages -= {r["page_id"] for k in keys for r in cells[k] if r.get("failed")}
    return {k: [r for r in cells[k] if r["page_id"] in pages] for k in keys}


def _choose(cells: dict, base: tuple, options: list[tuple]) -> dict:
    """The rule R-best, the image condition and each flagship's rung are chosen by (PREREG,
    Prompts). An option failing more pages than `base` does, plus one, is out; `base` and the
    rest are compared on the pages none of them failed. An option is eligible when its
    over-deletion, marked and absent, each stays no more than 1 point above base's there. The
    choice is the cheapest eligible option leaving at most DELTA more struck sub-tokens unmarked
    than the eligible option leaving fewest, among equally cheap ones the one leaving fewer; the
    runner-up is the eligible option, other than the choice, leaving fewest. With none eligible,
    the choice is the option with the fewest failed pages, then the least excess over the limits
    (each against base on the pages both answered), and there is no runner-up."""
    present = [o for o in options if o in cells]
    out = {"choice": None, "runner_up": None, "fallback": False, "eligible": [], "failed": {},
           "unmarked": {}}  # fmt: skip
    if base not in cells or not present:
        return out
    failed = {o: sum(bool(r.get("failed")) for r in cells[o]) for o in [base, *present]}
    out["failed"] = {o: failed[o] for o in present}
    kept = [o for o in present if failed[o] <= failed[base] + 1]
    out["out_for_failures"] = [o for o in present if o not in kept]
    on = _on(cells, [base, *dict.fromkeys(kept)])
    u = {o: _unmarked(on[o]) for o in kept}
    eligible = [o for o in kept if _excess(on, base, o) <= 0]
    out["unmarked"] = {o: float(v) for o, v in u.items()}
    out["eligible"] = eligible
    if eligible:
        floor = min(u[o] for o in eligible)
        near = [o for o in eligible if u[o] <= floor + DELTA]
        pick = min(near, key=lambda o: (_cost(o), u[o], options.index(o)))
        rest = [o for o in eligible if o != pick]
        out["choice"] = pick
        out["runner_up"] = min(rest, key=lambda o: (u[o], _cost(o), options.index(o)), default=None)
        return out

    def excess(o: tuple) -> Fraction:
        pair = _on(cells, [base, *dict.fromkeys([o])])
        return _excess(pair, base, o)

    out["choice"] = min(present, key=lambda o: (failed[o], excess(o), options.index(o)))
    out["fallback"] = True
    return out


def rbest(stage: str = "ladder") -> dict[str, dict]:
    """PREREG, Prompts: per provider, on its ladder model, R-best among R2–R6 against R0 by
    `_choose`'s rule, and the runner-up; then, at R-best, the image condition (C2 or C2-tiled) by
    the same rule, once `<stage>-tiled` has run R-best on C2-tiled (None until then). Exact
    arithmetic, each choice on the pages its cells all scored. A stage scored with `--partial`
    is refused: a choice prereg-2 fixes stands on complete stages."""
    rows = scores(stage)
    _refuse_partial(rows, stage)
    if (RUNS / f"{stage}-tiled" / "scores.jsonl").exists():
        tiled = scores(f"{stage}-tiled")
        _refuse_partial(tiled, f"{stage}-tiled")
        rows += tiled
    out = {}
    for k, m in config("models").items():
        if m["role"] != "ladder":
            continue
        rungs = [(f"R{i}", "C2") for i in range(2, 7)]
        cells = _cells(rows, k, [("R0", "C2"), *rungs])
        if ("R0", "C2") not in cells:
            continue
        pick = _choose(cells, ("R0", "C2"), rungs)
        best = pick["choice"][0] if pick["choice"] else None
        image = None
        tiled = _cells(rows, k, [("R0", "C2"), (best, "C2"), (best, "C2-tiled")]) if best else {}
        if (best, "C2-tiled") in tiled:
            image = _choose(tiled, ("R0", "C2"), [(best, "C2"), (best, "C2-tiled")])["choice"][1]
        out[m["provider"]] = {
            "model": k, "rbest": best, "runner_up": (pick["runner_up"] or (None,))[0],
            "fallback": pick["fallback"], "image": image,
            "eligible": [r for r, _ in pick["eligible"]],
            "out_for_failures": [r for r, _ in pick.get("out_for_failures", [])],
            "unmarked": {r: v for (r, _), v in pick["unmarked"].items()},
            "failed": {r: v for (r, _), v in pick["failed"].items()},
        }  # fmt: skip
    return out


def flagship_check(stage: str = "flagship-check") -> dict[str, dict]:
    """Each flagship's rung (PREREG, Prompts): it runs R-best and the runner-up on the ladder's
    dev pages and keeps whichever `_choose`'s rule ranks first, with R-best in R0's place; that
    is `flagship` in config/choices.toml. A provider whose runner-up is `none` keeps R-best. A
    stage scored with `--partial` is refused."""
    rows = scores(stage)
    _refuse_partial(rows, stage)
    out = {}
    for k, m in config("models").items():
        if m["role"] != "flagship":
            continue
        best, runner = (
            plans.choice(m["provider"], "rbest"),
            plans.choice(m["provider"], "runner_up"),
        )
        if runner == "none":
            out[k] = {"rbest": best, "runner_up": None, "flagship": best}
            continue
        cells = _cells(rows, k, [(best, "C2"), (runner, "C2")])
        if len(cells) < 2:
            continue
        pick = _choose(cells, (best, "C2"), [(best, "C2"), (runner, "C2")])
        out[k] = {"rbest": best, "runner_up": runner, "flagship": pick["choice"][0],
                  "unmarked": {r: v for (r, _), v in pick["unmarked"].items()},
                  "failed": {r: v for (r, _), v in pick["failed"].items()}}  # fmt: skip
    return out


def ablation(stage: str = "ablation") -> dict[str, dict]:
    """PREREG, Models: per ladder model, R-best at low effort against the repeats' three runs at
    high, on the pages all four scored: low's rates and cost minus the high runs' pooled ones,
    beside the high runs' range, with the canary's relabeling test per error (`p`); each run's
    failures by kind; and the same without the pages any run failed (`without_failures`). Low's
    own run-to-run spread is not measured. A stage scored with `--partial` is refused."""
    every = scores(stage)
    _refuse_partial(every, stage)
    rows = [r for r in every if "struck" in r and r["set"] != STRICT]
    out = {}
    for k, m in config("models").items():
        if m["role"] != "ladder":
            continue
        best = plans.choice(m["provider"], "rbest")
        high, low = [f"{k}:{best}:C2:r{n}" for n in (1, 2, 3)], f"{k}:{best}:C2:low"
        runs = {a: [r for r in rows if r["arm"] == a] for a in (*high, low)}
        if missing := [a for a, v in runs.items() if not v]:
            raise ValueError(f"{stage} has no scores for {', '.join(missing)}")
        pages = set.intersection(*({r["page_id"] for r in v} for v in runs.values()))
        failed = {r["page_id"] for v in runs.values() for r in v if r["failed"]}
        out[k] = {"rbest": best, **_low_vs_high(runs, high, low, pages),
                  "failures": {a: dict(Counter(r["status"] for r in v if r["failed"]))
                               for a, v in runs.items()},
                  "without_failures": _low_vs_high(runs, high, low, pages - failed)}  # fmt: skip
    return out


def _low_vs_high(runs: dict[str, list[dict]], high: list[str], low: str, pages: set) -> dict:
    """The ablation on `pages`: low's rates and cost, the high runs' pooled and their range, and
    per error the canary's relabeling test of low's counts against the high runs', over the pages
    holding sub-tokens it can occur in (`p`; none without such a page)."""
    by = {a: {r["page_id"]: r for r in v if r["page_id"] in pages} for a, v in runs.items()}
    each = [metrics(list(by[a].values())) for a in high]
    pooled = metrics([r for a in high for r in by[a].values()])
    mine = metrics(list(by[low].values()))
    keys = (*KEEP[1:], "usd_per_1000")
    p = {}
    for name, (errors, den) in TESTED.items():
        held = sorted(pid for pid, r in by[low].items() if r.get(den))
        x = [[_count(by[a][pid], errors) for a in (*high, low)] for pid in held]
        p[name] = relabel_p(np.array(x)) if held else None
    return {"pages": len(pages), "struck": mine["struck"],
            "low": {x: mine[x] for x in keys}, "high": {x: pooled[x] for x in keys},
            "high_range": {x: [min(e[x] for e in each), max(e[x] for e in each)] for x in keys},
            "low_minus_high": {x: mine[x] - pooled[x] for x in keys}, "p": p}  # fmt: skip


def _count(row: dict, errors: tuple[str, ...]) -> int:
    return sum(row.get(e, 0) for e in errors)


# ------------------------------------------------------------------------------------- tables


def calibration(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """PREREG, Scoring (exploratory): per arm giving words a probability of being struck (R4–R6
    answers and P-scan, whose score rows list `probs`) and set, three views: `own`, the arm over
    every word it gives one, with that share of the scored and of the struck words (`coverage`,
    `struck_coverage`); `all`, P-scan over every word, a word it gives none counted at 0; and
    `paired`, a model arm and P-scan (`side`) on the words both give one, conditional on P-scan's
    geometry, with the model's Brier score minus P-scan's (`brier_gap`). Each has its expected
    calibration error and Brier score with cluster bootstrap intervals (none under MIN_CLUSTERS
    clusters); the second list holds the reliability diagrams."""
    pscan = {r["page_id"]: {j: p for j, _, p in r["probs"] if p is not None}
             for r in rows if r["arm"] == "P-scan" and "probs" in r}  # fmt: skip
    listed: dict[tuple, list[tuple]] = {}  # (arm, set) -> [(cluster, struck, p or None)]
    paired: dict[tuple, list[tuple]] = {}  # (arm, set) -> [(cluster, struck, model p, P-scan p)]
    for r in rows:
        if "probs" not in r:
            continue
        key, theirs = (r["arm"], r["set"]), pscan.get(r["page_id"], {})
        listed.setdefault(key, []).extend((r["cluster"], y, p) for _, y, p in r["probs"])
        if r["arm"] != "P-scan":
            paired.setdefault(key, []).extend((r["cluster"], y, p, theirs[j])
                                              for j, y, p in r["probs"]
                                              if p is not None and j in theirs)  # fmt: skip
    summary, bins = [], []
    unscored = {"coverage": None, "struck_coverage": None}
    no_gap = {"brier_gap": None, "brier_gap_lo": None, "brier_gap_hi": None}

    def add(arm: str, set_: str, view: str, side: str, points: list[tuple], extra: dict) -> None:
        diagram, row = _calibrated(points)
        names = {"arm": arm, "set": set_, "view": view, "side": side}
        summary.append(names | row | extra)
        bins.extend(names | b for b in diagram)

    for (arm, set_), got in sorted(listed.items()):
        given = [x for x in got if x[2] is not None]
        side = "P-scan" if arm == "P-scan" else "model"
        cover = {"coverage": _rate(len(given), len(got)),
                 "struck_coverage": _rate(sum(y for _, y, _ in given), sum(y for _, y, _ in got))}  # fmt: skip
        add(arm, set_, "own", side, given, cover | no_gap)
        if arm == "P-scan":
            every = [(c, y, 0.0 if p is None else p) for c, y, p in got]
            add(arm, set_, "all", side, every, unscored | no_gap)
            continue
        pairs = paired.get((arm, set_), [])
        add(arm, set_, "paired", "model", [(c, y, m) for c, y, m, _ in pairs],
            unscored | _brier_gap(pairs))  # fmt: skip
        add(arm, set_, "paired", "P-scan", [(c, y, q) for c, y, _, q in pairs], unscored | no_gap)
    return summary, bins


def _ece(s: np.ndarray) -> np.ndarray:
    return np.abs(s[..., SP] - s[..., SY]).sum(axis=-1) / s[..., 0]


def _brier(s: np.ndarray) -> np.ndarray:
    return s[..., 1] / s[..., 0]


def _interval(per: np.ndarray, stat) -> tuple:
    """A statistic's cluster bootstrap interval, none under MIN_CLUSTERS clusters."""
    return bootstrap_ci(per, stat) if len(per) >= MIN_CLUSTERS else (None, None)


def _calibrated(points: list[tuple]) -> tuple[list[dict], dict]:
    """(cluster, struck, probability) points: the reliability diagram in BINS equal-width bins
    (1.0 in the last, empty bins left out), each bin's struck share with its Wilson interval and
    whether it holds MIN_BIN words, enough to read; and the words, struck words and clusters,
    with the expected calibration error and Brier score and their intervals."""
    sums: dict[str, np.ndarray] = {}
    for c, y, p in points:
        b = min(int(round(p * BINS, 9)), BINS - 1)  # rounded first, so 1 - 0.9 lands in 0.1's bin
        v = sums.setdefault(c, np.zeros(2 + 3 * BINS))
        v[[0, 1, NB.start + b, SP.start + b, SY.start + b]] += (1, (p - y) ** 2, 1, p, y)
    per = np.array([sums[c] for c in sorted(sums)]).reshape(-1, 2 + 3 * BINS)
    tot = per.sum(axis=0)
    diagram = []
    for b in range(BINS):
        n, struck = tot[NB][b], tot[SY][b]
        if n:
            lo, hi = wilson(round(struck), round(n))
            diagram.append({"bin": b / BINS, "words": round(n), "mean_p": tot[SP][b] / n,
                            "struck_share": struck / n, "struck_lo": lo, "struck_hi": hi,
                            "read": n >= MIN_BIN})  # fmt: skip
    row = {"words": round(tot[0]), "struck": round(tot[SY].sum()), "clusters": len(per)}
    for name, stat in (("ece", _ece), ("brier", _brier)):
        row[name] = float(stat(tot)) if tot[0] else math.nan
        row[f"{name}_lo"], row[f"{name}_hi"] = _interval(per, stat)
    return diagram, row


def _brier_gap(pairs: list[tuple]) -> dict:
    """The model's Brier score minus P-scan's on the same words, with its cluster bootstrap
    interval, from (cluster, struck, model p, P-scan p) points."""
    sums: dict[str, np.ndarray] = {}
    for c, y, m, q in pairs:
        sums[c] = sums.get(c, 0) + np.array([1.0, (m - y) ** 2, (q - y) ** 2])
    per = np.array([sums[c] for c in sorted(sums)]).reshape(-1, 3)
    tot = per.sum(axis=0)

    def gap(s: np.ndarray) -> np.ndarray:
        return (s[..., 1] - s[..., 2]) / s[..., 0]

    lo, hi = _interval(per, gap)
    return {"brier_gap": float(gap(tot)) if tot[0] else math.nan,
            "brier_gap_lo": lo, "brier_gap_hi": hi}  # fmt: skip


def cascade(rows: list[dict]) -> list[dict]:
    """PREREG, Scoring (exploratory): per model arm, set and band (`uncertain` is the model meta's,
    the others the fixed grid), the arm's outcomes on the words P-scan leaves uncertain (a token's
    `cnn_prob` strictly inside the band), against P-scan's on the same words of the same pages;
    the whole pages with the model answering those words and P-scan the rest (`cascade_page_*`),
    against P-scan's alone (`p_scan_page_*`); how much is handed off (the uncertain sub-tokens'
    share of P-scan's, and the share of pages with any, `handoff_*`); and its $ per 1,000 pages,
    P-scan on every page and the model on the pages handed off. A page-level proxy for the app's
    second look, which works on crops."""
    pscan = {r["page_id"]: r for r in rows if r["arm"] == "P-scan" and r.get("subsets")}
    acc: dict[tuple, dict[str, list]] = {}
    for r in rows:
        if ":" not in r["arm"] or not r.get("subsets") or r["page_id"] not in pscan:
            continue
        alone = pscan[r["page_id"]]
        for band, mine in r["subsets"].items():
            theirs = alone["subsets"].get(band)
            if theirs is None:
                continue
            a = acc.setdefault((r["arm"], r["set"], band), {"mine": [], "theirs": [], "joint": [],
                                                            "alone": [], "model": []})  # fmt: skip
            a["mine"].append(mine)
            a["theirs"].append(theirs)
            a["alone"].append(alone)
            a["model"].append(r)
            a["joint"].append({k: alone.get(k, 0) - theirs.get(k, 0) + mine.get(k, 0)
                               for k in OUTCOMES})  # fmt: skip
    out = []
    for (arm, set_, band), a in sorted(acc.items()):
        if not any(a["theirs"]):  # P-scan left nothing uncertain on these pages
            continue
        m, t = metrics(a["mine"]), metrics(a["theirs"])
        joint, alone = metrics(a["joint"]), metrics(a["alone"])
        handed = [bool(x.get("struck", 0) + x.get("live", 0)) for x in a["theirs"]]
        scored = sum(x.get("struck", 0) + x.get("live", 0) for x in a["alone"])
        words = sum(x.get("struck", 0) + x.get("live", 0) for x in a["theirs"])
        priced = all(x.get("priced", True) for x in a["alone"] + a["model"])
        usd = (sum(x.get("cost_usd") or 0 for x in a["alone"])
               + sum(x.get("cost_usd") or 0 for x, h in zip(a["model"], handed) if h))  # fmt: skip
        out.append({"arm": arm, "set": set_, "band": band, "pages": sum(handed),
                    "struck": m["struck"], "live": sum(c.get("live", 0) for c in a["mine"]),
                    "handoff_sub_tokens": _rate(words, scored),
                    "handoff_pages": _rate(sum(handed), len(handed)),
                    "cascade_usd_per_1000": _rate(1000 * usd, len(handed)) if priced else math.nan,
                    **{k: m[k] for k in KEEP[1:]}, **{f"p_scan_{k}": t[k] for k in KEEP[1:]},
                    **{f"cascade_page_{k}": joint[k] for k in WHOLE},
                    **{f"p_scan_page_{k}": alone[k] for k in WHOLE}})  # fmt: skip
    return out


# ---------------------------------------------------------------------------------- routing

ROUTE_KINDS = {  # PREREG, Routing: each page kind's sets, its inputs, and the free arms reading it
    "born-digital": (("G0", "G2"), ("C1", "C1p"), ("P-native", "B-naive", "B-pm4llm")),
    "images of print": (("G0", "G2"), ("C2", "C2-tiled"), ("P-scan",)),  # not P-scan-shipped,
    "handwriting": (("G5",), ("C2", "C2-tiled"), ("P-scan",)),  # in no endpoint or route
}
FREE = {a for _, _, free in ROUTE_KINDS.values() for a in free}
MARGIN = 0.01  # one point: R-best's tolerance and endpoint 2's
GUARD = ("struck", "struck_live", "struck_absent", "live", "live_marked", "live_absent")
GUARDED = {  # each guarded rate: the counts it sums, and what it is out of
    "misses": (("struck_live", "struck_absent"), "struck"),
    "leakage": (("struck_live",), "struck"),
    "over_marked": (("live_marked",), "live"),
    "over_absent": (("live_absent",), "live"),
}
SHOWN = ("unmarked", "leakage", "over_marked", "over_absent", "strikes_found_of_read",
         "halluc_per_1000", "halluc_digit_per_1000", "failure_rate", "usd_per_1000_list",
         "s_per_page")  # a route's columns, beside it  # fmt: skip


def _systems(rows: list[dict], kind: str) -> dict[str, dict[str, dict]]:
    """A kind's systems: its free arms, and each cell on one of its inputs (an arm
    `model:rung:input`, so no paraphrase, effort or repeat), that scored every one of its pages;
    per system, its rows by page."""
    sets, inputs, free = ROUTE_KINDS[kind]
    mine = [r for r in rows if r["set"] in sets]
    pages = {r["page_id"] for r in mine}
    by: dict[str, dict[str, dict]] = {}
    for r in mine:
        parts = r["arm"].split(":")
        if r["arm"] in free or (len(parts) == 3 and parts[2] in inputs):
            by.setdefault(r["arm"], {})[r["page_id"]] = r
    return {a: v for a, v in sorted(by.items()) if set(v) == pages}


def _fallback(rows: dict[str, dict], free: dict[str, dict]) -> dict[str, dict]:
    """A paid system's pages, each it failed scored as the free default's output on that page:
    what the app shows when a paid call fails."""
    return {pid: (free[pid] if r.get("failed") else r) for pid, r in rows.items()}


def _cascades(systems: dict[str, dict[str, dict]], pscan: dict[str, dict]) -> dict:
    """Each paid system's cascade in the model meta's band: per page, P-scan's outcomes with the
    words it leaves uncertain the model's instead, priced as P-scan plus the model wherever any
    word is handed off; a page the model failed is P-scan's alone, and counts as the cascade's
    failure (`own_failed`). Its text, and so its inserted tokens, is P-scan's."""
    out = {}
    for arm, pages in systems.items():
        if arm in FREE or not all(r.get("subsets") for r in pages.values()):
            continue
        rows = {}
        for pid, r in pages.items():
            alone = pscan[pid]
            theirs, mine = alone["subsets"]["uncertain"], r["subsets"]["uncertain"]
            sent = bool(theirs.get("struck", 0) + theirs.get("live", 0))
            take = sent and not r.get("failed")
            rows[pid] = {k: alone.get(k, 0) + take * (mine.get(k, 0) - theirs.get(k, 0))
                         for k in OUTCOMES}  # fmt: skip
            rows[pid] |= {"page_id": pid, "cluster": r["cluster"], "failed": alone.get("failed"),
                          "own_failed": bool(alone.get("failed") or (sent and r.get("failed"))),
                          "inserted": alone.get("inserted", 0),
                          "inserted_digit": alone.get("inserted_digit", 0),
                          "priced": alone.get("priced", True)
                          and (not sent or r.get("priced", True)),
                          "cost_list_usd": (alone.get("cost_list_usd") or 0)
                          + sent * (r.get("cost_list_usd") or 0)}  # fmt: skip
        out[f"cascade:{arm}"] = rows
    return out


def _paired(a: dict[str, dict], b: dict[str, dict]) -> np.ndarray:
    """Per-cluster sums of the GUARD counts on the same pages: a's, then b's."""
    sums: dict[str, np.ndarray] = {}
    for pid, x in a.items():
        row = np.array([x.get(k, 0) for k in GUARD] + [b[pid].get(k, 0) for k in GUARD], float)
        sums[x["cluster"]] = sums.get(x["cluster"], 0) + row
    return np.array([sums[k] for k in sorted(sums)]).reshape(-1, 2 * len(GUARD))


def _worse(c: np.ndarray, rate: str) -> dict:
    """a's rate minus b's, pooled, with its paired cluster-bootstrap interval; a is shown to be
    more than a point worse when the interval's lower end is at or above +1 point."""
    errs, den = GUARDED[rate]
    k, s, e = len(GUARD), GUARD.index(den), [GUARD.index(x) for x in errs]

    def diff(x: np.ndarray) -> np.ndarray:
        a = x[..., e].sum(axis=-1) / x[..., s]
        return a - x[..., [i + k for i in e]].sum(axis=-1) / x[..., s + k]

    with np.errstate(divide="ignore", invalid="ignore"):
        est = float(diff(c.sum(axis=0))) if len(c) else math.nan
    lo, hi = bootstrap_ci(c, diff)
    return {"diff": est, "lo": lo, "hi": hi, "shown_worse": bool(lo >= MARGIN)}


def routes(stage: str = "test") -> tuple[list[dict], list[dict]]:
    """PREREG, Routing: per page kind, the free default (the free system with the fewest misses),
    the reference (the fewest misses among the systems not shown to be more than a point worse
    than the free default on either over-deletion), the paid option (the cheapest paid system at
    list price not shown to be more than a point worse than the reference on misses, leakage and
    both over-deletions, ties to the faster) and a cascade that passes the same test at no more
    than half the paid option's price, which it then replaces. Paid systems' failed pages score
    as the free default's output, but a route's failure rate is the system's own. Returns each
    route with its columns (routes.csv) and every system's guards (route_guards.csv); a route's
    accuracy is "as selected", chosen on the same pages. An unpriced system sorts last on price,
    and no cascade replaces a paid option unless both are priced."""
    rows = [r for r in scores(stage) if r["set"] != STRICT and "struck" in r]
    _refuse_partial(rows, stage)
    timed = _speeds(stage)
    out, guards = [], []
    for kind, (_, _, free_arms) in ROUTE_KINDS.items():
        systems = _systems(rows, kind)
        if not (free := [a for a in systems if a in free_arms]):
            continue
        cost = {a: metrics(list(v.values())) for a, v in systems.items()}
        stats = dict(cost)

        def price(a: str) -> float:
            v = cost[a]["usd_per_1000_list"]
            return math.inf if math.isnan(v) else v

        def speed(a: str) -> float:  # a model's seconds come from the latency sample alone
            v = cost[a]["s_per_page"]
            v = timed.get(a, math.nan) if math.isnan(v) else v
            return math.inf if math.isnan(v) else v

        def failures(a: str) -> float:  # its own, not the fallback's
            v = list((systems.get(a) or cascades[a]).values())
            return _rate(sum(bool(r.get("own_failed", r.get("failed"))) for r in v), len(v))

        default = min(free, key=lambda a: (stats[a]["unmarked"], price(a), a))
        scored = {a: (v if a in FREE else _fallback(v, systems[default]))
                  for a, v in systems.items()}  # fmt: skip
        stats |= {a: metrics(list(v.values())) for a, v in scored.items() if a not in FREE}
        cascades = _cascades(systems, systems["P-scan"]) if "P-scan" in systems else {}
        for a, v in cascades.items():
            cost[a] = stats[a] = metrics(list(v.values()))

        def test(a: str, b: str, rates) -> dict[str, dict]:
            c = _paired(scored.get(a) or cascades[a], scored[b])
            res = {r: _worse(c, r) for r in rates}
            guards.extend({"kind": kind, "system": a, "against": b, "rate": r, **g,
                           "clusters": len(c)} for r, g in res.items())  # fmt: skip
            return res

        eligible = [a for a in systems
                    if not any(g["shown_worse"] for g in
                               test(a, default, ("over_marked", "over_absent")).values())]  # fmt: skip
        reference = min(eligible, key=lambda a: (stats[a]["unmarked"], price(a), speed(a), a))
        passes = {a: not any(g["shown_worse"] for g in test(a, reference, GUARDED).values())
                  for a in [*systems, *cascades] if a != reference}  # fmt: skip
        passes[reference] = True
        paid = min((a for a in systems if a not in FREE and passes[a]),
                   key=lambda a: (price(a), speed(a), a), default=None)  # fmt: skip
        half = paid and math.isfinite(price(paid)) and price(paid) / 2
        cascade = half and min((a for a in cascades
                                if passes[a] and math.isfinite(price(a)) and price(a) <= half),
                               key=lambda a: (price(a), speed(a), a), default=None)  # fmt: skip
        roles = {"free default": default, "reference": reference,
                 "paid option": cascade or paid, "replaced paid option": cascade and paid}  # fmt: skip
        for role, a in roles.items():
            if a:
                s = speed(a)
                out.append({"kind": kind, "role": role, "system": a, "pages": stats[a]["pages"],
                            "within_a_point": passes[a], **{k: stats[a][k] for k in SHOWN},
                            "failure_rate": failures(a), "s_per_page": s if s < math.inf else math.nan,
                            "usd_per_1000_list": cost[a]["usd_per_1000_list"]})  # fmt: skip
    return out, guards


def _speeds(stage: str) -> dict[str, float]:
    """Seconds per page per arm from the latency sample's scores, for a stage whose model rows
    carry none (Q5: elsewhere calls run in parallel, batched or queued)."""
    path = RUNS / "latency" / "scores.jsonl"
    if stage == "latency" or not path.exists():
        return {}
    by: dict[str, list[dict]] = {}
    for r in read_jsonl(path):
        by.setdefault(r["arm"], []).append(r)
    return {a: metrics(v)["s_per_page"] for a, v in by.items()}


def tables(stage: str) -> list[dict]:
    """Per arm and set, every metric (results.csv), a row that agrees with G1's ground truth by
    its construction flagged, and per cluster beside it (clusters.csv); per G0 stratum
    (strata.csv); the QA probe's answers per arm (qa.csv); the calibration (calibration.csv, its
    diagrams in calibration_bins.csv) and the cascade (cascade.csv). The analysis without failed
    pages is part of `endpoints`."""
    every = scores(stage)
    rows = [r for r in every if "struck" in r]
    out = []
    for arm in sorted({r["arm"] for r in rows}):
        for set_ in sorted({r["set"] for r in rows}):
            if mine := [r for r in rows if r["arm"] == arm and r["set"] == set_]:
                flag = {"by_construction": (arm, set_) in BY_CONSTRUCTION}
                out.append({"arm": arm, "set": set_, **flag, **metrics(mine)})
    _csv(RUNS / stage / "results.csv", out)
    by_cluster: dict[tuple, list[dict]] = {}
    for r in rows:
        by_cluster.setdefault((r["arm"], r["set"], r["cluster"]), []).append(r)
    per_cluster = [{"arm": a, "set": s, "cluster": c, **metrics(v)}
                   for (a, s, c), v in sorted(by_cluster.items())]  # fmt: skip
    _csv(RUNS / stage / "clusters.csv", per_cluster)
    answers: dict[str, Counter] = {}
    for r in every:
        if "answer" in r:
            answers.setdefault(r["arm"], Counter())[r["answer"]] += 1
    _csv(RUNS / stage / "qa.csv", [{"arm": a, **{k: c[k] for k in QA_ANSWERS}}
                                   for a, c in sorted(answers.items())])  # fmt: skip
    strata: dict[tuple, dict] = {}
    for r in rows:
        for name, c in (r.get("strata") or {}).items():
            acc = strata.setdefault((r["arm"], name), {})
            for key, v in c.items():
                acc[key] = acc.get(key, 0) + v
    per_stratum = [{"arm": arm, "stratum": name, **{k: metrics([c])[k] for k in KEEP}}
                   for (arm, name), c in sorted(strata.items())]  # fmt: skip
    _csv(RUNS / stage / "strata.csv", per_stratum)
    summary, bins = calibration(rows)
    _csv(RUNS / stage / "calibration.csv", summary)
    _csv(RUNS / stage / "calibration_bins.csv", bins)
    _csv(RUNS / stage / "cascade.csv", cascade(rows))
    return out


def _csv(path, rows: list[dict]) -> None:
    """Write the rows, or with none, remove what an earlier run wrote there."""
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # the BOM lets Excel read P→M
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def canary() -> list[dict]:
    """Drift (PREREG, Operations): per model, its end-of-window canary run (canary-4) against its
    three start runs, on the pages all four scored, for two counts (CANARY): struck sub-tokens left
    live, on the pages holding a strike, and live sub-tokens marked, on those holding live text.
    Each is `relabel_p`'s exact test, and p below DRIFT_P for either reads as drift. A model
    missing a run is untested (`drift` None). Each run's leakage is reported beside it, and the
    model's test window (`window`). Runs scored with `--partial` are refused."""
    runs = {n: scores(f"canary-{n}") for n in (1, 2, 3, 4)
            if (RUNS / f"canary-{n}" / "scores.jsonl").exists()}  # fmt: skip
    _refuse_partial([r for rs in runs.values() for r in rs], "a canary run")
    sent = _sent_times()
    out = []
    canaried = {r["model"] for rs in runs.values() for r in rs if ":" in r["arm"]}
    tested_models = {m for s in TEST_STAGES for m in sent.get(s, {})}  # a model with no canary too
    for model in sorted(canaried | tested_models):
        mine = {
            n: v
            for n, rs in runs.items()
            if (v := [r for r in rs if r["model"] == model and "struck" in r])
        }
        common = set.intersection(*({r["page_id"] for r in v} for v in mine.values())) if (
            mine
        ) else set()  # fmt: skip
        leak = {
            n: metrics([r for r in v if r["page_id"] in common])["leakage"] for n, v in mine.items()
        }
        p = None
        if len(mine) == 4:
            by = {n: {r["page_id"]: r for r in v} for n, v in mine.items()}
            p = {}
            for count, den in CANARY.items():
                held = sorted(pid for pid in common if by[4][pid].get(den))
                x = [[by[n][pid].get(count, 0) for n in (1, 2, 3, 4)] for pid in held]
                p[count] = relabel_p(np.array(x)) if held else None
        tested = [v for v in (p or {}).values() if v is not None]
        out.append({"model": model, "pages": len(common), "leakage": leak, "p": p,
                    "drift": min(tested) < DRIFT_P if tested else None,
                    "window": window(model, sent)})  # fmt: skip
    return out


# a model's test requests, which its canaries bracket (the latency sample's included)
TEST_STAGES = ("test", "fable", "qa", "latency")


def _sent_times() -> dict[str, dict[str, list[datetime]]]:
    """Per stage and model, when each request recorded was sent: its own start where the reply
    records one, else its batch's submission, else when it was recorded."""
    out: dict[str, dict[str, list[datetime]]] = {}
    for stage in (*(f"canary-{n}" for n in (1, 2, 3, 4)), *TEST_STAGES):
        submitted = {b["batch_id"]: b["submitted"]
                     for b in read_jsonl(RUNS / stage / "batches.jsonl") if "submitted" in b}  # fmt: skip
        for r in read_jsonl(RUNS / stage / "responses.jsonl"):
            when = r.get("started") or submitted.get(r.get("batch_id")) or r.get("recorded")
            if when:
                out.setdefault(stage, {}).setdefault(r["model"], []).append(
                    datetime.fromisoformat(when)
                )
    return out


def window(model: str, sent: dict[str, dict[str, list[datetime]]]) -> dict | None:
    """A model's test window (PREREG, Operations): from its first start canary sent to its last
    end canary sent, within 7 days, its test requests sent inside it. An overrun, or a test
    request outside the canaries, is reported beside the model's results; with no start canary,
    every test request is outside; None before any test request. While the end canary is unsent
    the window is open, and only the requests before its start are counted outside."""
    tests = [t for s in TEST_STAGES for t in sent.get(s, {}).get(model, [])]
    if not tests:
        return None
    start = [t for n in (1, 2, 3) for t in sent.get(f"canary-{n}", {}).get(model, [])]
    end = sent.get("canary-4", {}).get(model, [])
    first, last = min(tests), max(tests)
    out = {"test_first": first.isoformat(), "test_last": last.isoformat(),
           "start": min(start).isoformat() if start else None,
           "end": max(end).isoformat() if end else None}  # fmt: skip
    if not start:
        return out | {"outside": len(tests)}
    if not end:
        return out | {"outside": sum(t < min(start) for t in tests)}
    span = max(end) - min(start)
    outside = sum(t < min(start) or t > max(end) for t in tests)
    return out | {"days": span.total_seconds() / 86400, "overrun": span > WINDOW,
                  "outside": outside}  # fmt: skip


def _json(x) -> str:
    """JSON with NaN written as null, which strict parsers accept."""

    def clean(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, dict):
            return {k: clean(w) for k, w in v.items()}
        return [clean(w) for w in v] if isinstance(v, (list, tuple)) else v

    return json.dumps(clean(x), indent=2, ensure_ascii=False)


def register(sub) -> None:
    p = sub.add_parser(
        "report", help="endpoints, R-best, the flagship check, the ablation, tables, the canary"
    )
    cmd = p.add_subparsers(required=True, metavar="action")
    e = cmd.add_parser("endpoints")
    e.add_argument("stage", nargs="?", default="test")
    e.set_defaults(
        func=lambda a: print(_json({"provenance": provenance(), "claims": endpoints(a.stage)}))
    )
    for name, fn, default in (
        ("rbest", rbest, "ladder"),
        ("flagship-check", flagship_check, "flagship-check"),
        ("ablation", ablation, "ablation"),
    ):
        c = cmd.add_parser(name)
        c.add_argument("stage", nargs="?", default=default)
        c.set_defaults(
            func=lambda a, fn=fn, name=name: print(
                _json({"provenance": provenance(), name: fn(a.stage)})
            )
        )
    t = cmd.add_parser("tables")
    t.add_argument("stage")
    t.set_defaults(func=_tables)
    r = cmd.add_parser("routes", help="the app's routing default per page kind (PREREG, Routing)")
    r.add_argument("stage", nargs="?", default="test")
    r.set_defaults(func=_routes)
    cmd.add_parser("canary").set_defaults(
        func=lambda a: print(_json({"provenance": provenance(), "canary": canary()}))
    )


def _routes(args) -> None:
    """`report routes`: routes.csv and route_guards.csv, with the code beside them."""
    picked, guards = routes(args.stage)
    _csv(RUNS / args.stage / "routes.csv", picked)
    _csv(RUNS / args.stage / "route_guards.csv", guards)
    (RUNS / args.stage / "routes-provenance.json").write_text(
        _json(provenance()) + "\n", encoding="utf-8", newline="\n"
    )
    for r in picked:
        print(f"{r['kind']:16} {r['role']:13} {r['system']:28} misses {r['unmarked']:.3f} "
              f"leakage {r['leakage']:.3f} ${r['usd_per_1000_list']:.2f}/1,000")  # fmt: skip


def _tables(args) -> None:
    """`report tables`, with the code it ran at beside the CSVs (PREREG, Operations: Deviations)."""
    rows = tables(args.stage)
    (RUNS / args.stage / "tables-provenance.json").write_text(
        _json(provenance()) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{len(rows)} rows -> {RUNS / args.stage}")
