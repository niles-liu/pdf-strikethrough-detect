"""What each stage sends (PREREG.md, Prompts and Operations), the cap in force, and what the tags
freeze.

    python -m bench plans draw              # the dev pages (seed 20261013) -> config/dev_pages.json
    python -m bench plans show <stage>
    python -m bench plans freeze            # what prereg-1 tags -> config/prereg1.json
    python -m bench plans freeze --stage 2  # what prereg-2 tags -> config/prereg2.json

Stages: smoke and smoke-batch (stage B1), pilot, ladder, ladder-tiled, paraphrases, flagship-check,
repeats, ablation and cache-probe (B2), canary-1 to canary-4 (three at the start of a model's test window, one
at its end), test, fable, qa and latency (B3), and b4. `b2` and `b3` name those groups (GROUPS):
each has its own cap. The repeats and the effort ablation, sent together, answer an identical
request once (SHARED): the ablation's runs at high are the repeats'.
"""

import contextlib
import functools
import json
import math
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from bench.common import (
    CONFIG, DATA, GT, PREREG1, PREREG2, ROOT, RUNS, SETS, config, dirty, guard_test_pages,
    harness_commit, load_pages, read_jsonl, seed, sha256_bytes, sha256_file,
)  # fmt: skip
from bench.score import score_page
from bench.stats import largest_remainder

DEV_PAGES = CONFIG / "dev_pages.json"
PREREG = ROOT / "PREREG.md"
FEWSHOT_PAGE = "G0.dev-002.p0"  # R6's worked examples are cut from it, so no dev list draws it
# The dev lists (PREREG, Prompts): pages from G0-dev and G1-dev, how many of them strike-free,
# and pages from G2-dev, three struck and one strike-free. The pilot draws first; the others
# draw from the pilot's pages.
COMPOSITION = {"pilot": (30, 2, 4), "ladder": (20, 2, 4), "repeats": (20, 2, 0),
               "canary": (10, 0, 0)}  # fmt: skip
FABLE_PAGES, QA_ITEMS, LATENCY_PAGES = 50, 40, 30
SEQUENTIAL = {"smoke", "latency"}
# every other stage sends its own: the ladder's R2 too, as R-best compares it with rungs sent then
SHARED = ("repeats", "ablation")
GROUPS = {
    "b2": ("pilot", "ladder", "ladder-tiled", "paraphrases", "flagship-check", "repeats",
           "ablation", "cache-probe"),
    "b3": ("canary-1", "canary-2", "canary-3", "test", "fable", "qa", "latency", "canary-4"),
}  # fmt: skip
# Stages sent with Azure's implicit prompt cache on, before the cache probe: a model's
# `prompt_cache` setting leaves their requests, and so their custom IDs, as they were sent.
CACHE_ON = ("smoke", "smoke-batch", *GROUPS["b2"][:-1])
# The cells the cap may drop, in the order config/budget.toml's drop_order gives: the flagships'
# C1 at R1 and R-best, their C1′, the Fable subset and the QA probe (PREREG, Operations).
DROPS = ("test:C1 marks", "test:C1prime", "fable", "qa")
# Choices not yet made, at their costliest, for pricing stage B2 before they are made: R6 (JSON
# and three worked examples) on C2-tiled as well as C2, and R5 as the runner-up.
COSTLIEST = {"rbest": "R6", "runner_up": "R5", "image": "C2-tiled", "flagship": "R6"}
_assumed: set[tuple[str, str]] | None = None  # while pricing: the choices taken at their costliest


@dataclass(frozen=True)
class Cell:
    """One arm: a model at one prompt, input and setting. `rung` is R0–R6, QA-R0/QA-R1 (the QA
    probe's two framings), M+P or P→M; `input` is C1, C1p (C1′), C2, C2-tiled, C1-scan (the
    smoke test's image-only PDF) or text."""

    model: str
    rung: str
    input: str
    para: int = 0
    effort: str = "high"
    repeat: int = 0

    @property
    def arm(self) -> str:
        parts = [self.model, self.rung, self.input]
        if self.para:
            parts.append(f"p{self.para}")
        if self.effort != "high":
            parts.append(self.effort)
        if self.repeat:
            parts.append(f"r{self.repeat}")
        return ":".join(parts)


@dataclass(frozen=True)
class Item:
    cell: Cell
    page_id: str
    qa_id: str | None = None


# ------------------------------------------------------------------------------------ dev pages


def page_kind(page: dict) -> str | None:
    """`struck` when the page holds a struck sub-token as scored, `free` when it holds no strike
    at all, else None (its only strikes normalize to nothing)."""
    if score_page(page, None)["struck"]:
        return "struck"
    return None if any(w["struck"] for w in page["words"] if not w["furniture"]) else "free"


def _parts(n: int, free: int, docs: list[str]) -> dict[str, tuple[int, int]]:
    """A list's G0 and G1 pages as (struck, strike-free) per part: half of `n` from G0-dev, with
    `free` strike-free; the rest split between G1-dev's documents, the first by name taking the
    odd page, each with half of `free` strike-free."""
    half, rest = n // 2, n - n // 2
    out = {"G0": (half - free, free)}
    for i, doc in enumerate(docs):
        k = rest // len(docs) + (i < rest % len(docs))
        out[doc] = (k - free // 2, free // 2)
    return out


def draw_dev() -> dict[str, list[str]]:
    """PREREG, Prompts: one generator draws the pilot's pages, then from those the ladder's, the
    repeats' (the effort ablation's too) and the canary's, each on its own. Per part (G0-dev, each
    G1-dev document, G2-dev) and kind (struck or strike-free), pages are drawn uniformly; R6's
    example page never is."""
    rng = random.Random(seed("dev_pages"))
    pages = {p["page_id"]: p for p in load_pages("G0", "G1", "G2", split="dev")}
    part = {pid: p["doc"] if p["set"] == "G1" else p["set"] for pid, p in pages.items()}
    kind = {pid: page_kind(p) for pid, p in pages.items()}
    docs = sorted({p["doc"] for p in pages.values() if p["set"] == "G1"})
    out: dict[str, list[str]] = {}
    for name, (n, free, g2) in COMPOSITION.items():
        want = _parts(n, free, docs) | ({"G2": (g2 - 1, 1)} if g2 else {})
        pool = sorted(pages.keys() - {FEWSHOT_PAGE}) if name == "pilot" else out["pilot"]
        picked = []
        for where, counts in want.items():
            for k, n_k in zip(("struck", "free"), counts):
                ids = [pid for pid in pool if part[pid] == where and kind[pid] == k]
                if len(ids) < n_k:
                    raise ValueError(f"{name}: {where} has {len(ids)} {k} pages, not {n_k}")
                picked += rng.sample(ids, n_k)
        out[name] = sorted(picked)
    return out


def dev_pages() -> dict[str, list[str]]:
    if not DEV_PAGES.exists():
        raise FileNotFoundError("no dev pages; run `python -m bench plans draw`")
    return json.loads(DEV_PAGES.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------- choices and the cap


def choice(provider: str, key: str) -> str:
    value = config("choices")[provider][key]
    if value.startswith("TBD"):
        if _assumed is not None:
            _assumed.add((provider, key))
            return COSTLIEST[key]
        raise ValueError(f"{provider}'s {key} is not chosen yet (config/choices.toml)")
    return value


@contextlib.contextmanager
def costliest_choices() -> Iterator[set[tuple[str, str]]]:
    """While pricing, a choice not yet made reads as its costliest (COSTLIEST); yields the
    (provider, key) pairs so read."""
    global _assumed
    _assumed = set()
    try:
        yield _assumed
    finally:
        _assumed = None


def dropped() -> list[str]:
    """The cells dropped for the cap (PREREG, Operations): `dropped` in config/budget.toml, which
    must be the first items of its `drop_order`, in that order; the drop order is PREREG's
    (DROPS), so a typo or a reordering is refused rather than frozen."""
    budget = config("budget")
    out, order = budget.get("dropped", []), budget["drop_order"]
    if order != list(DROPS):
        raise ValueError(f"config/budget.toml: drop_order is {order}, not PREREG's {list(DROPS)}")
    if out != order[: len(out)]:
        raise ValueError(f"config/budget.toml: `dropped` must start drop_order, in order: {order}")
    return out


def group(stage: str) -> str:
    """A stage's group for the cap: b1 (the smoke tests), b2, b3, or later (stage B4, and G3's
    and G4's addenda)."""
    if stage.startswith("smoke"):
        return "b1"
    return next((g for g, stages in GROUPS.items() if stage in stages), "later")


def cap(stage: str, budget: dict | None = None) -> float | None:
    """The cap in force for a stage, on everything spent so far, the smoke tests included
    (PREREG, Operations): stages B1–B2's for stage B2, B1–B3's for stage B3, and that plus the
    reserve for what comes later. None for the smoke tests, which come before any cap, and for a
    cap still TBD (no stage can send under it); a cap that is neither is an error, as a quoted
    number would silently turn the cap off."""
    g = group(stage)
    if g == "b1":
        return None
    budget = budget or config("budget")
    key = "cap_b2_usd" if g == "b2" else "cap_b3_usd"
    value, reserve = budget[key], budget.get("reserve_usd", 0)
    if isinstance(value, str) and value.startswith("TBD"):
        return None
    for name, x in ((key, value), ("reserve_usd", reserve)):
        # a quoted number, nan or inf would turn the cap off as surely as no cap at all
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 <= x < math.inf:
            raise ValueError(f"config/budget.toml: {name} is {x!r}, not a number of dollars")
    return value + (reserve if g == "later" else 0)


# --------------------------------------------------------------------------------------- items


def _subset(rng: random.Random, pages: list[dict], n: int) -> list[str]:
    """n pages stratified by set, in proportion to each set's size."""
    sets = sorted({p["set"] for p in pages})
    by_set = [sorted(p["page_id"] for p in pages if p["set"] == s) for s in sets]
    shares = largest_remainder([len(x) for x in by_set], min(n, len(pages)))
    return [pid for ids, k in zip(by_set, shares) for pid in sorted(rng.sample(ids, k))]


def items(stage: str, guard: bool = True) -> list[Item]:
    """Every (cell, page) the stage sends, in a fixed order. `guard` False lists test pages before
    prereg-2, for the dry run alone, which reads only local files (PREREG, Operations)."""
    models = config("models")
    role = {k: m["role"] for k, m in models.items()}
    prov = {k: m["provider"] for k, m in models.items()}
    ladder = [k for k in models if role[k] == "ladder"]
    flagships = [k for k in models if role[k] == "flagship"]
    runnable = [k for k in models if role[k] != "ceiling"]
    out: list[Item] = []

    def add(pages, cells) -> None:
        out.extend(Item(c, p) for c in cells for p in pages)

    def best(k: str) -> str:
        """A model's R-best: a flagship its own (R-best or the runner-up, by the flagship check),
        every other model its provider's."""
        return choice(prov[k], "flagship" if role[k] == "flagship" else "rbest")

    def tiled(k: str) -> list[Cell]:
        return [Cell(k, best(k), "C2-tiled")] * (choice(prov[k], "image") == "C2-tiled")

    if stage == "smoke":  # every route a later dry run prices, once per model
        page = dev_pages()["pilot"][0]
        for k in models:
            cells = [Cell(k, "R0", i) for i in ("C2", "C1", "C1p", "C2-tiled")]
            cells += [Cell(k, "R6", "C2"), Cell(k, "R6", "C2-tiled")]
            # an image-only PDF: whether the model reads a PDF's pages or its text alone
            cells += [Cell(k, "R0", "C1-scan")] * (prov[k] in ("xai", "openai"))
            cells += [Cell(k, "R0", "C2", effort="low")] * (role[k] == "ladder")  # the ablation's
            cells += [Cell(k, "M+P", "C2")] * (role[k] == "flagship")
            # a flagship's C1 at its rung, R6's worked examples with a PDF: priced from nothing
            # else, as no other stage before the test run sends it
            cells += [Cell(k, "R6", "C1")] * (role[k] == "flagship")
            add([page], cells)
    elif stage == "smoke-batch":  # PREREG, stage B1 step 1: an image and a PDF, batched or at Flex
        add(dev_pages()["pilot"][:1], [Cell(k, "R0", i) for k in models
                                       if mode(stage, models[k]) != "sequential"
                                       for i in ("C2", "C1")])  # fmt: skip
    elif stage == "pilot":
        add(dev_pages()["pilot"], [Cell(k, "R2", "C2") for k in ladder])
    elif stage == "ladder":  # R-best is chosen on C2, then tried on C2-tiled (ladder-tiled)
        add(dev_pages()["ladder"], [Cell(k, f"R{i}", "C2") for k in ladder for i in range(7)])
    elif stage == "ladder-tiled":
        add(dev_pages()["ladder"], [Cell(k, best(k), "C2-tiled") for k in ladder])
    elif stage == "paraphrases":
        add(dev_pages()["ladder"], [Cell(k, r, "C2", para=p) for k in ladder
                                    for r in dict.fromkeys(("R0", "R1", best(k))) for p in (1, 2, 3)])  # fmt: skip
    elif stage == "flagship-check":  # a runner-up of `none`: R-best alone qualified, or none did
        add(dev_pages()["ladder"], [Cell(k, r, "C2") for k in flagships
                                    for r in (choice(prov[k], x) for x in ("rbest", "runner_up"))
                                    if r != "none"])  # fmt: skip
    elif stage == "repeats":
        add(dev_pages()["repeats"], [Cell(k, r, "C2", repeat=n) for k in ladder
                                     for r in dict.fromkeys(("R0", best(k))) for n in (1, 2, 3)])  # fmt: skip
    elif stage == "ablation":  # low effort against the repeats' three runs at high, reused
        add(dev_pages()["repeats"], [Cell(k, best(k), "C2", **x) for k in ladder
                                     for x in ({"repeat": 1}, {"repeat": 2}, {"repeat": 3},
                                               {"effort": "low"})])  # fmt: skip
    elif stage == "cache-probe":  # whether Azure serves GPT-6 with its prompt cache off
        add(dev_pages()["pilot"][:1], [Cell(k, "R0", "C2") for k in models if prov[k] == "openai"])
    elif stage.startswith("canary-"):
        n = int(stage.removeprefix("canary-"))
        gone = "fable" in dropped()  # a dropped subset has no window to bracket
        add(dev_pages()["canary"], [Cell(k, "R0", "C2", repeat=n) for k in models
                                    if not (gone and role[k] == "ceiling")])  # fmt: skip
    elif stage == "test":
        pages = test_pages(guard)
        every = [p["page_id"] for p in pages]
        g0_g2 = [p["page_id"] for p in pages if p["set"] in ("G0", "G2")]
        g2 = [p["page_id"] for p in pages if p["set"] == "G2"]
        scans = g5_pages(guard)
        core = [p["page_id"] for p in scans if p["core"]]
        drop = dropped()
        for k in runnable:
            r, flagship = best(k), role[k] == "flagship"
            cells = [Cell(k, x, "C2") for x in dict.fromkeys(("R0", "R1", r))] + tiled(k)
            cells.append(Cell(k, "R0", "C1"))
            # a flagship's C1 at R1 and R-best
            if flagship and "test:C1 marks" not in drop:
                cells += [Cell(k, x, "C1") for x in dict.fromkeys(("R1", r))]
            add(every, cells)
            if flagship and "test:C1prime" not in drop:
                add(g0_g2, [Cell(k, "R0", "C1p")])
            if flagship:  # endpoint 2's other wordings
                add(g2, [Cell(k, "R1", "C2", para=p) for p in (1, 2, 3)])
            add(core, [Cell(k, "R0", "C2")])  # G5: R0 on the core, R-best on every page
            add([p["page_id"] for p in scans], [Cell(k, r, "C2"), *tiled(k)])
    elif stage in ("fable", "qa") and stage in dropped():
        pass  # dropped for the cap
    elif stage == "fable":
        rng = random.Random(seed("fable_subset"))
        pages = _subset(rng, test_pages(guard), FABLE_PAGES)
        for k in (k for k in models if role[k] == "ceiling"):
            add(pages, [Cell(k, x, "C2") for x in dict.fromkeys(("R0", "R1", best(k)))])
    elif stage == "qa":
        qa = read_jsonl(GT / "G0-qa.jsonl")
        test_ids = {p["page_id"] for p in test_pages(guard) if p["set"] == "G0"}
        pool = sorted((q for q in qa if q["page_id"] in test_ids), key=lambda q: q["qa_id"])
        picks = random.Random(seed("qa_probe")).sample(pool, min(QA_ITEMS, len(pool)))
        for k in flagships:
            out.extend(Item(Cell(k, f"QA-{f}", "C2"), q["page_id"], q["qa_id"])
                       for f in ("R0", "R1") for q in picks)  # fmt: skip
    elif stage == "latency":  # every test set, G5's core included
        rng = random.Random(seed("latency"))
        scans = [p for p in g5_pages(guard) if p["core"]]
        pages = _subset(rng, test_pages(guard) + scans, LATENCY_PAGES)
        add(pages, [Cell(k, best(k), "C2") for k in runnable])  # the prompt the app would use
    elif stage == "b4":  # M+P: P-scan's reading of the page's scan, on G0 and G2
        pages = [p["page_id"] for p in test_pages(guard) if p["set"] in ("G0", "G2")]
        add(pages, [Cell(k, "M+P", "C2") for k in flagships])
    else:
        raise ValueError(f"unknown stage {stage!r}")
    if guard:
        splits = _splits()
        guard_test_pages({"split": splits[i.page_id]} for i in out)  # dev_pages.json included
    return out


@functools.cache
def _splits() -> dict[str, str]:
    return {p["page_id"]: p["split"] for p in load_pages(*SETS)}


def test_pages(guard: bool = True) -> list[dict]:
    """G0–G2's test pages; G5's run their own cells (g5_pages)."""
    pages = load_pages("G0", "G1", "G2", split="test")
    if guard:
        guard_test_pages(pages)
    return pages


def g5_pages(guard: bool = True) -> list[dict]:
    pages = load_pages("G5")
    if guard:
        guard_test_pages(pages)
    return pages


def mode(stage: str, model: dict) -> str:
    """How a stage sends a model's requests: batch; flex, Azure's half-price tier for GPT-6, sent
    like sequential requests; or sequential, at the standard tier (the smoke test and latency)."""
    if stage in SEQUENTIAL:
        return "sequential"
    if model.get("service_tier") == "flex":
        return "flex"
    return "batch" if model["batch"] else "sequential"


# ------------------------------------------------------------------------------------- freezing


def freeze() -> dict:
    """PREREG, What each tag fixes: what prereg-1 tags, from the built ground truth, with the
    prompts' hashes as drafted before the pilot. Refuses while any of it is TBD (B1), while
    PREREG.md holds a TBD (B1), on a tree that differs from its commit, while a model has no
    smoke-test answer reporting its `expect`, and when the held-back seed's digest is not the
    one PREREG.md publishes."""
    from bench.g2 import cutoff
    from bench.prompts import hashes

    models = config("models")
    _refuse_dirty(PREREG1)
    if n := len(_placeholders("B1")):
        raise ValueError(f"PREREG.md still holds {n} TBD (B1); fill them before prereg-1")
    if unconfirmed := _unconfirmed(models):
        raise ValueError(f"no smoke-test answer reports the pinned `expect` for: {unconfirmed}")
    secret = DATA / "private" / "g0_heldback_seed.txt"
    digest = _seed_digest(secret)
    if digest != (published := _published_digest()):
        found = (f"the held-back seed's sha256 is {digest[:12]}…" if secret.exists()
                 else f"no held-back seed at {secret} (drawn once, by `g0 heldback`)")  # fmt: skip
        raise ValueError(f"{found}; PREREG.md publishes {published[:12]}…")
    cap("pilot")  # a cap that is not a number, nor TBD, is refused here too
    if not isinstance(config("budget")["cap_b2_usd"], (int, float)):
        raise ValueError("config/budget.toml: cap_b2_usd must be set before prereg-1")
    dropped()  # the drop order the tag fixes is PREREG's
    pages = {s: load_pages(s) for s in SETS}
    g1, g2 = pages["G1"], pages["G2"]
    g2_test = [p for p in g2 if p["split"] == "test"]
    out = {
        "commit": harness_commit(exclude=(_tag_file(PREREG1),)),
        "prereg_sha256": sha256_file(PREREG),
        "models": {
            k: {
                "id": m["id"],
                "expect": m["expect"],
                "cutoff": m["cutoff"],
                "source": m["cutoff_source"],
                "c1_route": m["c1_route"],
            }
            for k, m in models.items()
        },  # fmt: skip
        "g2_cutoff": cutoff().isoformat(),
        "g0_test": {p["pdf"]: p["pdf_sha256"] for p in pages["G0"] if p["split"] == "test"},
        "g0_heldback_seed_sha256": digest,
        "g1_dev_documents": sorted({p["doc"] for p in g1 if p["split"] == "dev"}),
        "g1_documents": {p["doc"]: p["pdf_sha256"] for p in g1},
        "g1_test_pages": [p["page_id"] for p in g1 if p["split"] == "test"],
        "g2_bills": {p["doc"]: [p["pdf_sha256"], p["xml_sha256"]] for p in g2_test},
        "g2_pages": [p["page_id"] for p in g2_test],
        "g2_dev_bills": {
            p["doc"]: [p["pdf_sha256"], p["xml_sha256"]] for p in g2 if p["split"] == "dev"
        },  # fmt: skip
        "g2_dev_pages": [p["page_id"] for p in g2 if p["split"] == "dev"],
        "g5_pages": {
            p["page_id"]: [p["pdf_sha256"], p["transcription_sha256"]] for p in pages["G5"]
        },
        "g5_core": [p["page_id"] for p in pages["G5"] if p["core"]],
        "dev_pages": dev_pages(),
        "prompts": hashes(),
        "budget": config("budget"),
    }
    if tbd := list(_tbd(out, later="TBD (B2)")):
        raise ValueError(f"still TBD, to fill before prereg-1: {', '.join(tbd)}")
    return out


def freeze_2() -> dict:
    """PREREG, What each tag fixes: what prereg-2 tags. The sha256 of every file the run's rules
    live in (the code, the prompts, the config, PREREG.md and the environment), of every test
    page's ground truth, of G1's error-rate sample and of G0's QA items, beside the choices, the
    budget and the prompts as tuned. Refuses on a tree that differs from its commit, while
    anything is TBD, and before prereg-1's lists exist."""
    from bench.prompts import hashes

    _refuse_dirty(PREREG2)
    if not PREREG1.exists():
        raise FileNotFoundError(f"no {PREREG1.name}: freeze prereg-1's lists first")
    if n := len(_placeholders("B1", "B2")):
        raise ValueError(f"PREREG.md still holds {n} TBD; fill them before prereg-2")
    from bench.harness import spent

    budget = config("budget")
    cap("test")  # a quoted cap, nan or a negative one is refused as such
    if not isinstance(budget["cap_b3_usd"], (int, float)) or isinstance(budget["cap_b3_usd"], bool):
        raise ValueError("config/budget.toml: cap_b3_usd must be set before prereg-2")
    if budget["cap_b3_usd"] <= (so_far := spent()):
        raise ValueError(f"config/budget.toml: cap_b3_usd ${budget['cap_b3_usd']} is not above "
                         f"the ${so_far:.2f} already spent")  # fmt: skip
    dropped()
    out = {
        "commit": harness_commit(exclude=(_tag_file(PREREG2),)),
        "prereg_sha256": sha256_file(PREREG),
        "files": _file_hashes(),
        "ground_truth": ground_truth_hashes(),
        "g1_error_sample": sha256_file(GT / "G1-error-sample.jsonl"),
        "g0_qa": sha256_file(GT / "G0-qa.jsonl"),
        "g2_coverage": sha256_file(GT / "G2-coverage.jsonl"),  # what the alignment rule kept
        "prompts": hashes(),
        "choices": config("choices"),
        "models": config("models"),
        "budget": budget,
    }
    if tbd := list(_tbd(out)):
        raise ValueError(f"still TBD, to fill before prereg-2: {', '.join(tbd)}")
    return out


def ground_truth_hashes() -> dict[str, str]:
    """Each test page's ground-truth record (G0–G2's test pages and G5's), by its canonical JSON."""
    return {p["page_id"]: record_sha256(p) for p in load_pages(*SETS, split="test")}


def record_sha256(record: dict) -> str:
    return sha256_bytes(json.dumps(record, sort_keys=True, ensure_ascii=False).encode())


def _file_hashes() -> dict[str, str]:
    """The files the run's rules live in, by path relative to the benchmark root."""
    paths = [p for folder in ("bench", "prompts", "config") for p in (ROOT / folder).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p != PREREG2]  # fmt: skip
    paths += [ROOT / f for f in ("PREREG.md", "requirements.in", "requirements.txt",
                                 "corpus_manifest.json")]  # fmt: skip
    return {_rel(p): sha256_file(p) for p in sorted(paths)}


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _tag_file(path: Path) -> str:
    """A tag's lists, as git names them: always in config/, where `plans freeze` writes them."""
    return f"config/{path.name}"


def _refuse_dirty(written: Path) -> None:
    """A tag's lists name a commit, so the tree must match it: everything but the file being
    written is committed first."""
    if listed := dirty(exclude=(_tag_file(written),)):
        raise ValueError(f"commit first; the benchmark's files differ from HEAD:\n{listed}")
    if harness_commit() == "unknown":
        raise ValueError("no git commit to name")


def _unconfirmed(models: dict) -> list[str]:
    """Models with no `ok` smoke-test answer reporting the model ID pinned as `expect` (PREREG,
    stage B1 step 1). An answer first recorded as another model's counts once `expect` is
    re-pinned to the model that served it, as the harness then counts it."""
    seen = {(r["model"], r.get("reported_model")) for s in ("smoke", "smoke-batch")
            for r in read_jsonl(RUNS / s / "responses.jsonl")
            if r.get("status") == "ok"
            or (r.get("status") == "model_mismatch" and r.get("answer_status") == "ok")}  # fmt: skip
    return [k for k, m in models.items() if (k, m["expect"]) not in seen]


def _tbd(x, at: str = "", later: str | None = None) -> Iterator[str]:
    """Where in `x` a value still reads TBD; with `later`, a value reading it is left for the
    next tag (prereg-1 leaves `TBD (B2)` to prereg-2)."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _tbd(v, f"{at}.{k}" if at else k, later)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _tbd(v, f"{at}[{i}]", later)
    elif isinstance(x, str) and x.startswith("TBD") and not (later and x.startswith(later)):
        yield at


def _placeholders(*stages: str) -> list[str]:
    """PREREG.md's unfilled values for those stages: each `TBD (B1)` or `TBD (B2)` not quoted in
    backticks, as the file's own note on them is, a line break inside one included."""
    names = "|".join(stages)
    return re.findall(rf"(?<!`)TBD\s+\((?:{names})\)", PREREG.read_text(encoding="utf-8"))


def _seed_digest(path: Path) -> str:
    return (
        sha256_bytes(path.read_text(encoding="utf-8").strip().encode())
        if path.exists()
        else "TBD (B1)"
    )


def _published_digest() -> str:
    """The held-back seed's sha256 as PREREG.md publishes it (G0, Seeds)."""
    found = re.search(r"decimal string is published:\s*`([0-9a-f]{64})`",
                      PREREG.read_text(encoding="utf-8"))  # fmt: skip
    if not found:
        raise ValueError("PREREG.md publishes no held-back seed digest where G0's seeds are")
    return found.group(1)


def _draw(args) -> None:
    pages = draw_dev()
    DEV_PAGES.write_text(json.dumps(pages, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("  ".join(f"{k}: {len(v)}" for k, v in pages.items()), f"-> {DEV_PAGES}")


def _show(args) -> None:
    its = items(args.stage)
    arms = sorted({i.cell.arm for i in its})
    print(
        f"{args.stage}: {len(its)} requests, {len(arms)} arms, {len({i.page_id for i in its})} pages"
    )
    for arm in arms:
        print(f"  {arm}: {sum(i.cell.arm == arm for i in its)}")


def _freeze(args) -> None:
    path, frozen = (PREREG2, freeze_2()) if args.stage == 2 else (PREREG1, freeze())
    path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"-> {path}")


def register(sub) -> None:
    p = sub.add_parser("plans", help="the dev pages, each stage's requests, what the tags fix")
    cmd = p.add_subparsers(required=True, metavar="action")
    cmd.add_parser("draw", help="draw the dev pages").set_defaults(func=_draw)
    s = cmd.add_parser("show", help="list a stage's arms")
    s.add_argument("stage")
    s.set_defaults(func=_show)
    f = cmd.add_parser("freeze", help="write config/prereg1.json, or prereg2.json with --stage 2")
    f.add_argument("--stage", type=int, choices=(1, 2), default=1)
    f.set_defaults(func=_freeze)
