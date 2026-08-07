"""The §3.1 config control — does the real over-flagging regime survive `confidence_free()`?

`degrade_ladder.py` ran its whole gate under ``ScanConfig.confidence_free()``, while every
ruled-forms false-positive figure this repo quotes was measured under ``ScanConfig()``. That is a
confound, and it is the one that can invalidate the ladder outright: ``confidence_gating=False``
makes the chain gate — **and its over-flagging escape** — inert, so the ladder may have exercised a
decision path structurally incapable of the failure it went looking for.

This holds document class and OCR backend fixed (the real private corpus, its cached Azure DI
results) and moves only the config, which is the missing cell. It costs nothing: the DI results are
on disk, so there is no cloud call and no OCR pass.

Read the result three ways, pre-committed here so the reading isn't fitted to the number:

* **`confidence_free` still over-flags** (FP stays the same order of magnitude) → the config is
  exonerated, the ladder tested a live path, and its negative verdict stands. The remaining
  candidate causes are the fade model and the missing confusers (POSITIVES.md §3).
* **`confidence_free` under-detects** (FP collapses toward zero along with recall) → the ladder was
  scoring a path that cannot over-flag. Its "wrong failure mode" verdict says nothing about whether
  degradation reproduces the regime, and the gate must be re-run with gating **ON** before the
  degrade-and-relabel route is called dead.
* **Neither cleanly** → report the split rather than picking a story.

    python benchmarks/gating_control.py [CORPUS_DIR]

CORPUS_DIR defaults the same way `confidence_veto.py`'s does.
"""
from __future__ import annotations

import os
import sys

import pdf_strikethrough as st
from confidence_veto import DEFAULT_DIR, _analyze, _docs_in, _load, load_ground_truth, score

CONFIGS = {
    "default": st.ScanConfig(),                    # gating ON — the config every FP figure used
    "confidence_free": st.ScanConfig.confidence_free(),   # gating OFF — the config the ladder used
}


def run(corpus_dir):
    docs = _docs_in(corpus_dir)
    if not docs:
        print(f"no document subfolders under {corpus_dir!r}\n"
              "point the harness at a local corpus (PDF_STRIKETHROUGH_CORPUS_DIR=... or pass the "
              "path as an argument); each subfolder needs a *.pdf and a di-result.json.")
        return 1
    gt = load_ground_truth(corpus_dir)
    if gt is None:
        print("!! no ground-truth.json at the corpus root: this control needs labels, because a "
              "struck-final\n   count is not a false-positive count.")
        return 1

    totals = {k: dict(n=0, fp=0, tp=0, fn=0) for k in CONFIGS}
    unlabeled = []
    print(f"{'document':26s} " + "  ".join(f"{k:>19s}" for k in CONFIGS))
    print(f"{'':26s} " + "  ".join(f"{'final':>6s}{'FP':>7s}{'TP':>6s}" for _ in CONFIGS))
    for name, pdf, di in docs:
        pdf_bytes, di_result = _load(pdf, di)
        if name not in gt:
            unlabeled.append(name)
        truth = gt.get(name, {"strikes": []})
        cells = []
        for k, cfg in CONFIGS.items():
            final = _analyze(pdf_bytes, di_result, cfg)
            tp, fp, fn = score(final, truth)
            t = totals[k]
            t["n"] += len(final); t["fp"] += fp; t["tp"] += tp; t["fn"] += fn
            cells.append(f"{len(final):6d}{fp:7d}{tp:6d}")
        print(f"{name:26s} " + "  ".join(cells))
    print(f"{'TOTAL':26s} " + "  ".join(
        f"{totals[k]['n']:6d}{totals[k]['fp']:7d}{totals[k]['tp']:6d}" for k in CONFIGS))

    base, free = totals["default"], totals["confidence_free"]
    strikes = base["tp"] + base["fn"]
    print()
    for k, t in totals.items():
        print(f"{k:16s} struck-final={t['n']:<5d} FP={t['fp']:<5d} recall={t['tp']}/{strikes}")

    d_fp = 100.0 * (free["fp"] - base["fp"]) / max(base["fp"], 1)
    print(f"\nmoving ONLY the config, gating ON -> OFF: FP {base['fp']} -> {free['fp']} "
          f"({d_fp:+.0f}%), recall {base['tp']} -> {free['tp']} of {strikes}")

    # The pre-committed reading. "Collapse" = the FP population the ladder was supposed to
    # reproduce is mostly gone, which is exactly what a structurally-inert path looks like.
    # Every branch is a ratio against the baseline FP count, so a baseline of zero has no reading
    # at all -- and would otherwise satisfy the first branch trivially and report an exoneration
    # from a corpus that never over-flagged in the first place.
    if base["fp"] == 0:
        print("\n=> NO READING. The default config produced no false positives on this corpus, so\n"
              "   there is no over-flagging regime here to be exonerated or implicated. Point this\n"
              "   at the corpus the 113-FP figure came from.")
    elif free["fp"] >= 0.5 * base["fp"]:
        print("\n=> CONFIG EXONERATED. The over-flagging regime survives confidence_free(), so the\n"
              "   ladder ran a live path. Its negative verdict stands; look to the fade model and\n"
              "   the missing confusers (POSITIVES.md section 3).")
    elif free["fp"] <= 0.1 * base["fp"]:
        print("\n=> CONFIG IMPLICATED. The regime does NOT survive confidence_free(): the ladder was\n"
              "   scoring a path that cannot produce the failure it looked for. Its verdict says\n"
              "   nothing about degradation. Re-run the gate with gating ON before calling the\n"
              "   degrade-and-relabel route dead.")
    else:
        print("\n=> PARTIAL. The regime is attenuated but not gone. Report the split; do not round it\n"
              "   to either story. The gate is still worth re-running with gating ON, but the fade\n"
              "   and confuser hypotheses are not cleared either.")
    if unlabeled:
        print(f"\n!! {len(unlabeled)} document(s) absent from ground-truth.json, scored as having NO "
              f"real strikes: {', '.join(unlabeled)}. Label them or these counts are wrong.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(sys.argv[1:]) != len(args):
        print(f"{__file__}: takes no options\nusage: gating_control.py [CORPUS_DIR]")
        raise SystemExit(2)
    target = args[0] if args else os.environ.get("PDF_STRIKETHROUGH_CORPUS_DIR", DEFAULT_DIR)
    raise SystemExit(run(target))
