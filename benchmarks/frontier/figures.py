"""The test run's two figures (PREREG.md, Stage B3, step 4): the leakage-against-cost scatter and
the failure gallery. They live outside `bench/`, so every score keeps recording the code `prereg-2`
froze, and draw with what is already pinned: a plotting library would change `requirements.txt`,
which makes every cached free-arm output stale.

    python figures.py scatter <stage> [--metric leakage] [--out FILE]   # runs/<stage>/scatter.svg
    python figures.py gallery <stage> [--per-kind 2] [--out FILE]       # runs/<stage>/gallery.html

Both read a stage once it is scored (`python -m bench harness score <stage>`) and write only their
own file. The scatter puts each arm's metric on each set, pooled over its pages (leakage unless
`--metric` names another), against its $ per 1,000 pages on a log axis: a filled mark at the price
recorded, a ring at list price where a promotion makes the two differ. One panel per set; G2-strict
and the rows that agree with G1's ground truth by its construction are left out. The gallery shows
each arm's worst failures: its longest runs of consecutive sub-tokens that are struck but kept as
current text (leaks), or current but marked deleted or left out (over-deletions), at most one run
per page, each with the page crop and the output around it, then its failed pages. G5's ground
truth has no word boxes, so its runs are shown as text.
"""

import argparse
import base64
import html
import io
import itertools
import json
import math
import re
import textwrap
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from bench import harness, render, report
from bench.audit import _kind, _render
from bench.common import RUNS, config, load_pages, provenance
from bench.score import score_page, sub_tokens

SETS = ("G2", "G0", "G1", "G5")  # G2 first: the tested set
TITLES = {"G2": "G2: US Senate bills, the tested set", "G0": "G0: synthetic redlines",
          "G1": "G1: public redlines, ground truth from the package's detectors",
          "G5": "G5: handwritten pages"}  # fmt: skip
METRICS = {  # a metric of `report.metrics`: its name, and what it measures
    "leakage": ("Leakage", "the share of struck sub-tokens kept as current text"),
    "unmarked": ("Unmarked share", "the share of struck sub-tokens kept as current or left out"),
    "read_leakage": ("Read leakage", "leakage among the struck sub-tokens the output read"),
    "over_marked": ("Marked over-deletion", "the share of current sub-tokens marked deleted"),
    "over_absent": ("Absent over-deletion", "the share of current sub-tokens left out"),
    "f1": ("Struck-word F1", "F1 of marking the struck sub-tokens"),
}
SHAPES = {"C1": "square", "C1p": "diamond", "C2": "circle", "C2-tiled": "triangle"}
INPUTS = {"C1": "PDF (C1)", "C1p": "image + text (C1′)", "C2": "image (C2)",
          "C2-tiled": "tiled image (C2-tiled)"}  # fmt: skip
FREE_INPUT = {"P-native": "C1", "B-naive": "C1", "B-pm4llm": "C1", "P-scan": "C2",
              "P-scan-shipped": "C2"}  # the born-digital arms take C1's square  # fmt: skip
FREE = {"P-native": "#000000", "P-scan": "#525252", "P-scan-shipped": "#8c8c8c",
        "B-naive": "#a8a8a8", "B-pm4llm": "#a8a8a8"}  # fmt: skip
SHADES = {  # per provider, darkest first: its flagship, then its ceiling, ladder and other models
    "anthropic": ("#8c2d04", "#cc4c02", "#ec7014", "#f59e3c"),
    "google": ("#08519c", "#4292c6"),
    "openai": ("#006d2c", "#41ab5d"),
    "xai": ("#54278f", "#807dba"),
}
ROLES = ("flagship", "ceiling", "ladder", "other")


def short(model: str) -> str:
    """`gpt-6-astra` reads Astra, `grok-4-7` Grok 4.7; a free arm keeps its name."""
    if model in FREE:
        return model
    parts = model.split("-")
    named = [p for p in parts[1:] if p.isalpha()]
    return named[-1].title() if named else f"{parts[0].title()} {'.'.join(parts[1:])}"


def colours() -> dict[str, str]:
    """A hue per provider, darker for its flagship; greys for the free arms."""
    models = config("models")
    out = dict(FREE)
    for provider, shades in SHADES.items():
        mine = sorted((k for k, m in models.items() if m["provider"] == provider),
                      key=lambda k: (ROLES.index(models[k]["role"]), k))  # fmt: skip
        out |= {k: shades[min(i, len(shades) - 1)] for i, k in enumerate(mine)}
    return out


def parse_arm(arm: str) -> tuple[str, str, str]:
    """(model, rung with its paraphrase and any version, input) of a model's arm, which
    `--split-versions` scoring names `<arm>@<version>`; a free arm is its own model."""
    arm, _, version = arm.partition("@")
    if arm in FREE_INPUT:
        return arm, "", FREE_INPUT[arm]
    parts = arm.split(":")
    model, rung, input_ = (parts + ["", ""])[:3]
    return model, rung + "".join(parts[3:]) + (f"@{version}" if version else ""), input_


def _model_key(model: str, order: list[str]) -> tuple[int, int]:
    """The package's arms first, then the models in config order."""
    if model in FREE:
        return 0, list(FREE).index(model)
    return 1, order.index(model) if model in order else len(order)


def _arm_key(arm: str, order: list[str]) -> tuple:
    model, rung, input_ = parse_arm(arm)
    return *_model_key(model, order), rung, input_


def _usd(v: float) -> str:
    return f"${v:,.2f}" if v >= 0.01 else f"${v:.{1 - math.floor(math.log10(v))}f}"


def _scored(stage: str) -> list[dict]:
    """The stage's score rows, refused unless it is scored whole."""
    rows = report.scores(stage)  # raises until `harness score <stage>` has run
    report._refuse_partial(rows, stage)
    return rows


# ---------------------------------------------------------------------------------- scatter

W, PH = 860, 520  # the figure's width and a panel's height
L, R, T, B = 66, 28, 40, 64  # a panel's margins: its title above, ticks and a note below
CHAR = 5.4  # a label's average character width at 9.5 px


def points(stage: str, metric: str = "leakage") -> list[dict]:
    """Each arm on each set: the metric pooled over its pages, and its $ per 1,000 pages; a row
    that agrees with G1's ground truth by its construction is flagged, to be named, not drawn."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in _scored(stage):
        if "struck" in r and r["set"] != report.STRICT:
            groups[r["arm"], r["set"]].append(r)
    out = []
    for (arm, set_), rows in sorted(groups.items()):
        m = report.metrics(rows)
        out.append({"arm": arm, "set": set_, "y": m[metric], "x": m["usd_per_1000"],
                    "x_list": m["usd_per_1000_list"], "pages": m["pages"],
                    "failure_rate": m["failure_rate"],
                    "by_construction": (arm, set_) in report.BY_CONSTRUCTION})  # fmt: skip
    return out


def _off(p: dict, name: str) -> str | None:
    """Why a point is not drawn, or None when it is."""
    if p.get("by_construction"):
        return "agrees with the ground truth by its construction"
    if math.isnan(p["y"]):
        return f"no {name.lower()}"
    if math.isnan(p["x"]):
        return "no price"
    return None if p["x"] > 0 else "costs $0, off the log axis"


def _mark(shape: str, x: float, y: float, colour: str, filled: bool = True, r: float = 4.5) -> str:
    paint = (f'fill="{colour}" stroke="#fff" stroke-width="0.8"' if filled
             else f'fill="#fff" stroke="{colour}" stroke-width="1.4"')  # fmt: skip
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {paint}/>'
    if shape == "square":
        r *= 0.9
        return f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r:.1f}" height="{2 * r:.1f}" {paint}/>'
    corners = ([(0, -1.3), (1.3, 0), (0, 1.3), (-1.3, 0)] if shape == "diamond"
               else [(0, -1.25), (1.15, 0.85), (-1.15, 0.85)])  # fmt: skip
    return f'<polygon points="{" ".join(f"{x + a * r:.1f},{y + b * r:.1f}" for a, b in corners)}" {paint}/>'


def _overlaps(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _labels(marks: list[dict], plot: tuple[float, float, float, float]) -> None:
    """Each mark's label beside it (`lx`, `ly`), nudged off the marks and labels already placed and
    kept inside the plot (x0, y0, x1, y1); where no place is free, the first."""
    x0, y0, x1, y1 = plot
    taken = [(x - 5, m["y"] - 5, x + 5, m["y"] + 5)
             for m in marks for x in (m["x"], m["ring"]) if x is not None]  # fmt: skip
    taken += [(min(m["x"], m["ring"]), m["y"] - 2, max(m["x"], m["ring"]), m["y"] + 2)
              for m in marks if m["ring"] is not None]  # a label over the line reads as struck  # fmt: skip
    for m in sorted(marks, key=lambda m: (m["y"], m["x"])):
        w = CHAR * len(m["text"])
        tries = [(dx, dy) for dy in (3.5, -6.5, 13.5, -16.5, 23.5, -26.5, 33.5)
                 for dx in (7, -7 - w)]  # fmt: skip
        boxes = [
            (m["x"] + dx, m["y"] + dy - 8, m["x"] + dx + w, m["y"] + dy + 1) for dx, dy in tries
        ]
        inside = [k for k, b in enumerate(boxes)
                  if x0 <= b[0] and b[2] <= x1 and y0 <= b[1] and b[3] <= y1] or [0]  # fmt: skip
        k = next((k for k in inside if not any(_overlaps(boxes[k], t) for t in taken)), inside[0])
        taken.append(boxes[k])
        m["lx"], m["ly"] = m["x"] + tries[k][0], m["y"] + tries[k][1]


def _top(v: float) -> float:
    """The y axis's top: the smallest round rate above the highest mark."""
    return next((t for t in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8) if t >= v * 1.05), 1.0)


def _ticks(top: float, root: bool) -> list[float]:
    """Round rates up to the top: on a square-root axis, closer together near zero."""
    if not root:
        return [top * i / 5 for i in range(6)]
    return sorted({t for t in (0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0) if t <= top} | {top})


def svg(pts: list[dict], metric: str, stage: str) -> str:
    """The scatter: one panel per set, a shared log axis of cost and a shared metric axis."""
    tone, order = colours(), list(config("models"))
    name, meaning = METRICS[metric]
    on = [p for p in pts if _off(p, name) is None]
    if not on:
        raise ValueError(f"nothing to draw for {stage}: no arm has both a {metric} and a price")
    xs = [v for p in on for v in (p["x"], p["x_list"]) if v > 0]
    lo, hi = math.floor(math.log10(min(xs))), math.ceil(math.log10(max(xs)))
    if hi == lo:
        hi += 1
    top = _top(max(p["y"] for p in on))
    root = metric != "f1"  # an error rate: the best arms' low rates stay apart
    sets = [s for s in SETS if any(p["set"] == s for p in pts)]
    sets += sorted({p["set"] for p in pts} - set(sets))
    models = sorted({parse_arm(p["arm"])[0] for p in on}, key=lambda m: _model_key(m, order))
    inputs = [i for i in SHAPES if any(parse_arm(p["arm"])[2] == i for p in on)]

    # the legend: what a mark means, then each model's colour and each input's shape
    head = [f'<text x="{L}" y="24" font-size="16" font-weight="600">{html.escape(name)} against '
            f"cost, stage {html.escape(stage)}</text>"]  # fmt: skip
    lines = [
        f"{name}: {meaning}. Each mark is one arm on one set, pooled over its pages.",
        "Cost: $ per 1,000 pages on a log axis, filled at the price recorded and ringed at list "
        "price where a promotion makes them differ.",
        "Labels give the rung (p1–p3: R1's paraphrases); hover a mark for its numbers.",
    ]
    if root:
        lines.insert(2, f"{name} is on a square-root axis, so the low rates the best arms reach "
                        "stay apart.")  # fmt: skip
    for k, line in enumerate(lines):
        head.append(f'<text x="{L}" y="{43 + 14 * k}" font-size="11" fill="#444">'
                    f"{html.escape(line)}</text>")  # fmt: skip
    x, y = L, 58 + 14 * len(lines)
    for m in models:
        w = 22 + CHAR * 1.15 * len(short(m))
        if x + w > W - R:
            x, y = L, y + 18
        head.append(_mark("circle", x + 5, y - 4, tone.get(m, "#999")))
        head.append(f'<text x="{x + 14}" y="{y}" font-size="11">{html.escape(short(m))}</text>')
        x += w
    x, y = L, y + 20
    for i in inputs:
        head.append(_mark(SHAPES[i], x + 5, y - 4, "#555"))
        head.append(f'<text x="{x + 14}" y="{y}" font-size="11">{html.escape(INPUTS[i])}</text>')
        x += 30 + CHAR * 1.15 * len(INPUTS[i])
    legend = y + 14

    body, x0, x1, oy = [], L, W - R, legend
    for set_ in sets:
        y0, y1 = oy + T, oy + PH - B
        why = ", ".join(f"{p['arm']} ({_off(p, name)})"
                        for p in pts if p["set"] == set_ and p not in on)  # fmt: skip
        note = textwrap.wrap(f"Not drawn: {why}", int((x1 - x0) / 5.6)) if why else []

        def px(v: float) -> float:
            return x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)

        def py(v: float) -> float:
            return y1 - (math.sqrt(v / top) if root else v / top) * (y1 - y0)

        body.append(f'<text x="{x0}" y="{oy + T - 12}" font-size="13" font-weight="600">'
                    f"{html.escape(TITLES.get(set_, set_))}</text>")  # fmt: skip
        body.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" '
                    'stroke="#bbb"/>')  # fmt: skip
        for e in range(lo, hi + 1):
            for d in range(1, 10) if e < hi else (1,):
                gx = px(d * 10.0**e)
                body.append(f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y1}" '
                            f'stroke="{"#ddd" if d == 1 else "#f2f2f2"}"/>')  # fmt: skip
            tick = f"${10**e:,}" if e >= 0 else f"${10.0**e:.{-e}f}"
            body.append(f'<text x="{px(10.0**e):.1f}" y="{y1 + 15}" font-size="10" '
                        f'text-anchor="middle" fill="#444">{tick}</text>')  # fmt: skip
        for v in _ticks(top, root):
            label = f"{v:.1%}" if 0 < v < 0.1 and round(v * 100) != v * 100 else f"{v:.0%}"
            body.append(f'<line x1="{x0}" y1="{py(v):.1f}" x2="{x1}" y2="{py(v):.1f}" '
                        'stroke="#eee"/>')  # fmt: skip
            body.append(f'<text x="{x0 - 6}" y="{py(v) + 3.5:.1f}" font-size="10" '
                        f'text-anchor="end" fill="#444">{label}</text>')  # fmt: skip
        mid = (y0 + y1) / 2
        body.append(f'<text x="{(x0 + x1) / 2}" y="{y1 + 32}" font-size="11" text-anchor="middle" '
                    'fill="#222">$ per 1,000 pages (log)</text>')  # fmt: skip
        body.append(f'<text x="{x0 - 46}" y="{mid}" font-size="11" text-anchor="middle" '
                    f'fill="#222" transform="rotate(-90 {x0 - 46} {mid})">{html.escape(name)}'
                    f"{' (square root)' if root else ''}</text>")  # fmt: skip

        marks = []
        for p in (p for p in on if p["set"] == set_):
            model, rung, input_ = parse_arm(p["arm"])
            ring = p["x_list"] > 0 and abs(p["x_list"] - p["x"]) > 0.005 * p["x"]
            marks.append({"p": p, "x": px(p["x"]), "y": py(p["y"]),
                          "ring": px(p["x_list"]) if ring else None,
                          "colour": tone.get(model, "#999"), "shape": SHAPES.get(input_, "circle"),
                          "text": rung or p["arm"]})  # fmt: skip
        _labels(marks, (x0, y0, x1, y1))
        for m in marks:
            p = m["p"]
            tip = (f"{p['arm']} on {set_}: {name.lower()} {p['y']:.1%}; {_usd(p['x'])} per 1,000 "
                   "pages" + (f" ({_usd(p['x_list'])} at list price)" if m["ring"] else "")
                   + f"; {p['pages']} pages, {p['failure_rate']:.0%} failed")  # fmt: skip
            parts = [f"<title>{html.escape(tip)}</title>"]
            if m["ring"]:
                parts.append(f'<line x1="{m["x"]:.1f}" y1="{m["y"]:.1f}" x2="{m["ring"]:.1f}" '
                             f'y2="{m["y"]:.1f}" stroke="{m["colour"]}"/>')  # fmt: skip
                parts.append(_mark(m["shape"], m["ring"], m["y"], m["colour"], filled=False))
            parts.append(_mark(m["shape"], m["x"], m["y"], m["colour"]))
            parts.append(f'<text x="{m["lx"]:.1f}" y="{m["ly"]:.1f}" font-size="9.5" '
                         f'fill="{m["colour"]}">{html.escape(m["text"])}</text>')  # fmt: skip
            body.append(f"<g>{''.join(parts)}</g>")

        for k, line in enumerate(note):
            body.append(f'<text x="{x0}" y="{y1 + 50 + 13 * k}" font-size="10" fill="#666">'
                        f"{html.escape(line)}</text>")  # fmt: skip
        oy += PH + 13 * max(0, len(note) - 1)

    height = oy
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
            f'viewBox="0 0 {W} {height}" font-family="system-ui, -apple-system, Segoe UI, '
            f'sans-serif"><desc>{html.escape(str(provenance()))}</desc>'
            f'<rect width="100%" height="100%" fill="#fff"/>{"".join(head)}{"".join(body)}</svg>')  # fmt: skip


def scatter(stage: str, metric: str = "leakage") -> str:
    return svg(points(stage, metric), metric, stage)


# ---------------------------------------------------------------------------------- gallery

KINDS = ("leak", "over-deletion")
WHAT = {"leak": "struck text kept as current text",
        "over-deletion": "current text marked deleted or left out"}  # fmt: skip
BOX = {"leak": "#d62728", "over-deletion": "#e67e00"}
CLIP = 30  # a long run shows its first and last words this many at a time


def failure_runs(details: list[dict]) -> list[tuple[str, list[dict]]]:
    """Runs of consecutive scored sub-tokens, in the page's word order, sharing one failure."""

    def kind(r: dict) -> str | None:
        k = _kind(r)
        return k if k in KINDS else None

    return [(k, list(g)) for k, g in itertools.groupby(details, key=kind) if k]


CHECKED = ("struck", "struck_live", "struck_absent", "live", "live_marked", "live_absent")


def worst(stage: str, per_kind: int = 2, scored: list[dict] | None = None) -> dict[str, dict]:
    """Per arm: its pages, its failed pages with their status, and per kind its `per_kind` longest
    runs, at most one per page, ties to the page ID. With the stage's score rows (`scored`),
    every page must score as its row does, so the gallery shows what the scores counted (after a
    `g1 build`, rescore first)."""
    if per_kind < 1:
        raise ValueError("per_kind must be at least 1")
    counted = {(r["arm"], r["page_id"]): r for r in scored or () if "struck" in r}
    if any("@" in arm for arm, _ in counted):
        raise ValueError(f"{stage} is scored by version (--split-versions): the gallery is not")
    pages = {p["page_id"]: p for p in load_pages(*SETS)}
    out: dict[str, dict] = {}
    for o in harness.outputs(stage):
        page = pages.get(o["page_id"])
        if o.get("qa_id") or page is None or (o["arm"], page["set"]) in report.BY_CONSTRUCTION:
            continue
        a = out.setdefault(o["arm"], {"pages": 0, "failed": [], **{k: [] for k in KINDS}})
        a["pages"] += 1
        res = score_page(page, o["tokens"], details=True) if o["tokens"] is not None else None
        failed = res is None or res["failed"]
        if counted and ((row := counted.get((o["arm"], o["page_id"]))) is None
                     or bool(row.get("failed")) != failed
                     or (not failed and any(row.get(k, 0) != res[k] for k in CHECKED))):  # fmt: skip
            raise ValueError(f"{o['arm']} on {o['page_id']} does not score as {stage}'s scores "
                             "say: rescore the stage first")  # fmt: skip
        if failed:
            a["failed"].append((o["page_id"], o["status"] if res is None else "no usable output"))
            continue
        longest: dict[str, list[dict]] = {}
        for k, rows in failure_runs(res["details"]):
            if len(rows) > len(longest.get(k, ())):
                longest[k] = rows
        for k, rows in longest.items():
            a[k].append({"page": page, "rows": rows, "tokens": o["tokens"],
                         "details": res["details"]})  # fmt: skip
            a[k] = sorted(a[k], key=lambda r: (-len(r["rows"]), r["page"]["page_id"]))[:per_kind]
    return out


def _crop(page: dict, run: list[int], colour: str, width: int = 900) -> str:
    """The whole text lines around the run, its words boxed, as a JPEG data URL no wider than
    `width`."""
    image = Image.open(render.path(page["page_id"], "c2.png")).convert("RGB")
    w, h = image.size
    boxes = [page["words"][j]["bbox"] for j in run]
    lh = max(b[3] - b[1] for b in boxes)
    top, bottom = min(b[1] for b in boxes) - 2 * lh, max(b[3] for b in boxes) + 2 * lh
    every = [(b, (b[1] + b[3]) / 2) for word in page["words"] if (b := word.get("bbox"))]
    near = [b for b, mid in every if top <= mid <= bottom]  # the lines two either side
    y0, y1 = min(b[1] for b in near), max(b[3] for b in near)
    # a third of a line's margin, stopping at the next line out: word boxes nearly touch
    up = max((b[3] for b, mid in every if mid < top), default=0)
    down = min((b[1] for b, mid in every if mid > bottom), default=1)
    area = (max(0, min(b[0] for b in near) - 0.01) * w, max(y0 - lh / 3, min(up, y0)) * h,
            min(1, max(b[2] for b in near) + 0.01) * w, min(y1 + lh / 3, max(down, y1)) * h)  # fmt: skip
    out = image.crop(tuple(round(v) for v in area))
    draw = ImageDraw.Draw(out)
    for b in boxes:
        draw.rectangle([b[0] * w - area[0] - 3, b[1] * h - area[1] - 3, b[2] * w - area[0] + 3,
                        b[3] * h - area[1] + 3], outline=colour, width=3)  # fmt: skip
    if out.width > width:
        out = out.resize((width, round(out.height * width / out.width)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _clip(items: list, n: int = CLIP) -> list:
    """A long list's first and last `n`, with None where the middle was left out."""
    return items if len(items) <= 2 * n else [*items[:n], None, *items[-n:]]


def _page_text(page: dict, run: list[int], context: int = 8) -> str:
    """The page's words around the run, struck characters struck through, the run highlighted."""
    words, inside = page["words"], set(run)
    js = range(max(0, run[0] - context), min(len(words), run[-1] + context + 1))
    out = []
    for j in _clip([j for j in js if not words[j]["furniture"]], CLIP + context):
        if j is None:
            out.append("…")
            continue
        w = words[j]
        text = "".join(f"<del>{html.escape(w['text'][s:e])}</del>" if struck
                       else html.escape(w["text"][s:e]) for s, e, struck in sub_tokens(w))  # fmt: skip
        out.append(f"<mark>{text}</mark>" if j in inside else text)
    return " ".join(out)


SPREAD, SHOWN = 16, 6  # output tokens further apart start a new excerpt; tokens shown either side


def _excerpts(ks: list[int]) -> list[list[int]]:
    """Sorted output token indices split wherever the next is more than SPREAD further on: the
    scorer can align a word to a token far from its neighbours'."""
    out: list[list[int]] = []
    for k in ks:
        if out and k - out[-1][-1] <= SPREAD:
            out[-1].append(k)
        else:
            out.append([k])
    return out


def _output_text(item: dict, run: list[int]) -> str:
    """The output around the run, as the audit renders it, the run's tokens in bold: an excerpt
    per stretch of output its words align to (the first and last three where there are more), a
    long one cut in the middle; a run left out entirely shows a gap mark where it would be, by
    the output of the page's nearest words either side (a farther word may be paired far off)."""
    tokens = item["tokens"]
    focus = sorted({k for r in item["rows"] for k in r["tokens"]})
    if not focus:
        by_word: dict[int, list[int]] = defaultdict(list)
        for r in item["details"]:
            by_word[r["word"]] += r["tokens"]
        before = max((j for j, ks in by_word.items() if j < run[0] and ks), default=None)
        after = min((j for j, ks in by_word.items() if j > run[-1] and ks), default=None)
        if before is not None:  # right after the output of the word before the run
            return _render(tokens, [], (max(by_word[before]), None), width=SHOWN)
        if after is not None and (k := min(by_word[after])) > 0:  # else right before the next's
            return _render(tokens, [], (k - 1, None), width=SHOWN)
        return _render(tokens, [], (None, None if after is None else 0), width=SHOWN)
    groups = _excerpts(focus)
    if len(groups) > 6:
        groups = [*groups[:3], f"({len(groups) - 6} more)", *groups[-3:]]
    parts = []
    for g in groups:
        if isinstance(g, str):
            parts.append(g)
        elif len(g) > 2 * (CLIP + SHOWN):  # two windows that cannot overlap
            parts.append(_render(tokens, g[:CLIP], width=SHOWN) + " … "
                         + _render(tokens, g[-CLIP:], width=SHOWN))  # fmt: skip
        else:
            parts.append(_render(tokens, g, width=SHOWN))
    return " … ".join(parts)


def _run_html(kind: str, item: dict, crops: dict[tuple, tuple[str, str]]) -> str:
    """One run: its crop (kept once in `crops` for every arm sharing it), the page and the output."""
    page, rows = item["page"], item["rows"]
    run = sorted({r["word"] for r in rows})
    if all(page["words"][j].get("bbox") for j in run):
        key = (page["page_id"], tuple(run), kind)
        if key not in crops:
            crops[key] = (f"c{len(crops)}", _crop(page, run, BOX[kind]))
        img = f'<img data-crop="{crops[key][0]}" alt="the page around the run">'
    else:
        img = '<p class="note">This set has no word boxes: the text alone.</p>'
    n = len(rows)
    how = ""
    if kind == "over-deletion":
        marked = sum(r["outcome"] == "marked" for r in rows)
        how = f": {marked} marked deleted, {n - marked} left out"
    return (f'<div class="run {kind}"><h3>{html.escape(page["page_id"])} <small>{n} '
            f"sub-token{'s' * (n != 1)} of {WHAT[kind]}{how}</small></h3>{img}"
            f"<p><strong>The page:</strong> {_page_text(page, run)}</p>"
            f'<p class="out"><strong>The output:</strong> {_output_text(item, run)}</p></div>')  # fmt: skip


PAGE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>body{{font:15px/1.45 system-ui,sans-serif;max-width:980px;margin:2em auto;padding:0 16px;
background:#fff;color:#111}}h2{{border-top:2px solid #333;padding-top:.8em;margin-top:2em}}
h2 small,h3 small{{color:#555;font-weight:normal}}h3{{font-size:15px;margin:1.2em 0 .4em}}
img{{max-width:100%;border:1px solid #ddd}}.out{{font-family:ui-monospace,monospace;font-size:13px;
background:#f5f5f5;padding:.5em;overflow-wrap:anywhere}}.out b{{background:#ffd84d;padding:0 .1em}}
mark{{background:#ffe9a8}}del{{color:#b00;text-decoration-thickness:2px}}.note{{color:#666}}
table{{border-collapse:collapse;font-size:13px}}td,th{{border-bottom:1px solid #ddd;padding:.2em .6em;
text-align:right}}td:first-child,th:first-child{{text-align:left}}.wrap{{overflow-x:auto}}
footer{{color:#777;font-size:12px;margin:3em 0 1em;overflow-wrap:anywhere}}</style>
<h1>{title}</h1>{intro}<div class="wrap"><table><tr><th>Arm</th><th>Pages</th>
<th>Longest leak</th><th>Longest over-deletion</th><th>Failed pages</th></tr>{toc}</table></div>
{body}<footer>{provenance}</footer>"""
INTRO = """<p>Each arm's worst failures on stage {stage}, the package's arms included: its longest runs
of consecutive sub-tokens (the scorer's unit, in the page's word order) that are <b>struck but kept
as current text</b> (red boxes) or <b>current but marked deleted or left out</b> (orange boxes), at
most one run per page, the {per_kind} longest of each kind, ties to the page ID; then its failed
pages, scored worst case. On the page, struck text is <del>struck through</del> and the run
<mark>highlighted</mark>; in the output, ⟦text⟧ is text it marks as deleted, the run's words are in
<b>bold</b>, ▯ is where it has nothing, and … joins the stretches of output the run's words align
to, or cuts a long one. G5's ground truth has no word boxes, so its runs show
text alone. Missed strikes (struck text left out) and added words are counted in the results
tables, not shown here; so are the rows that agree with G1's ground truth by its construction.</p>"""


def gallery(stage: str, per_kind: int = 2) -> str:
    """The gallery page: a table of each arm's longest runs, then each arm's runs and failures."""
    found = worst(stage, per_kind, _scored(stage))
    order = list(config("models"))
    toc, body, crops = [], [], {}
    for arm in sorted(found, key=lambda a: _arm_key(a, order)):
        a, anchor = found[arm], html.escape("arm-" + re.sub(r"[^\w.-]", "-", arm))
        longest = {k: len(a[k][0]["rows"]) if a[k] else 0 for k in KINDS}
        toc.append(f'<tr><td><a href="#{anchor}">{html.escape(arm)}</a></td><td>{a["pages"]}</td>'
                   f'<td>{longest["leak"]}</td><td>{longest["over-deletion"]}</td>'
                   f'<td>{len(a["failed"])}</td></tr>')  # fmt: skip
        parts = [f'<h2 id="{anchor}">{html.escape(arm)} <small>{a["pages"]} pages</small></h2>']
        for k in KINDS:
            parts += ([_run_html(k, item, crops) for item in a[k]]
                      or [f'<p class="note">No {WHAT[k]}.</p>'])  # fmt: skip
        if a["failed"]:
            parts.append("<p><b>Failed pages</b>, scored worst case: " + ", ".join(
                f"{html.escape(p)} ({html.escape(s)})" for p, s in a["failed"]) + "</p>")  # fmt: skip
        body.append("".join(parts))
    # each crop once, however many arms share it: the page fills its images in when it loads
    images = json.dumps(dict(crops.values()))
    body.append(f"<script>const C={images};document.querySelectorAll('img[data-crop]')"
                ".forEach(i => i.src = C[i.dataset.crop])</script>")  # fmt: skip
    title = f"Failure gallery: {stage}"
    return PAGE.format(title=html.escape(title), toc="".join(toc), body="".join(body),
                       intro=INTRO.format(stage=html.escape(stage), per_kind=per_kind),
                       provenance=html.escape(str(provenance())))  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(required=True, dest="figure", metavar="figure")
    s = sub.add_parser("scatter", help="the metric against cost per arm and set, as SVG")
    s.add_argument("stage")
    s.add_argument("--metric", default="leakage", choices=list(METRICS))
    s.add_argument("--out", type=Path)
    g = sub.add_parser("gallery", help="each arm's worst failures, as HTML")
    g.add_argument("stage")
    g.add_argument("--per-kind", type=int, default=2)
    g.add_argument("--out", type=Path)
    a = parser.parse_args()
    if a.figure == "gallery" and a.per_kind < 1:
        parser.error("--per-kind must be at least 1")
    if a.figure == "scatter":
        text = scatter(a.stage, a.metric)
        name = "scatter.svg" if a.metric == "leakage" else f"scatter-{a.metric}.svg"
    else:
        text, name = gallery(a.stage, a.per_kind), "gallery.html"
    out = a.out or RUNS / a.stage / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
