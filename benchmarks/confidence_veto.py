"""Confidence-veto precision benchmark — scanned ruled-table over-flagging (issues #4, #7).

Runs the scanned Azure-DI path over a directory of scanned, tightly-ruled table/form PDFs.

Default mode reports the struck-final count per document, split by OCR confidence: the
high-confidence population the 0.9.1 DI-confidence veto targets vs the low-confidence faint-scan
residual left to a CNN retrain. This is the reproduction harness for the v0.9.1 correctness patch.

``--ab`` mode is the reproduction harness for the **v0.10.0 printed-rule veto** (issue #7): it runs
each document twice, with ``ScanConfig()`` and with ``ScanConfig.ruled_forms()``, and reports the
change. When the corpus ships a `ground-truth.json` it scores TP / FP / FN against it instead of
quoting raw counts — a raw struck-final count is NOT a false-positive count, because it includes the
real strikes. Quote the number this prints; do not quote a remembered one.

The corpus is private and is NOT committed (see `.gitignore: /benchmarks/private/`). Point the
harness at a local copy — one subfolder per document, each holding a `*.pdf` and a `di-result.json`
(an Azure DI `prebuilt-layout` analyze result, so no cloud call is made), plus an optional
`ground-truth.json` at the corpus root (schema documented in its own `_schema` key):

``--switches`` scores all four combinations of the two provisional precision switches
(``veto_printed_rules`` and ``rescue_clean_chains``), since they are independent and opted into
separately — it reports what each buys alone and how much they overlap.

    python benchmarks/confidence_veto.py [CORPUS_DIR] [--ab | --switches]

CORPUS_DIR defaults to `benchmarks/private/ruled-tables/` or the PDF_STRIKETHROUGH_CORPUS_DIR env var.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pdf_strikethrough as st

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "private", "ruled-tables")
CLEAN_CONF = st.ScanConfig().max_clean_conf
MATCH_IOU = 0.30          # a reported word matches a ground-truth strike above this box overlap


def _docs_in(corpus_dir):
    out = []
    for d in sorted(x for x in glob.glob(os.path.join(corpus_dir, "*")) if os.path.isdir(x)):
        pdfs = glob.glob(os.path.join(d, "*.pdf"))
        di = os.path.join(d, "di-result.json")
        if pdfs and os.path.exists(di):
            out.append((os.path.basename(d), pdfs[0], di))
    return out


def _load(pdf_path, di_path):
    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()
    with open(di_path, encoding="utf-8") as fh:
        return pdf_bytes, json.load(fh)


def _analyze(pdf_bytes, di_result, config=None):
    res = st.detect_pdf(pdf_bytes, di_result=di_result, scan_config=config,
                        include_markdown=False)
    return [w for w in res["words"] if w.get("final")]


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(corpus_dir):
    """The corpus label set, or None when the corpus doesn't ship one."""
    path = os.path.join(corpus_dir, "ground-truth.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("_docs", {})


def score(final, truth):
    """(tp, fp, fn) for one document's reported words against its ground-truth strikes."""
    strikes = list(truth.get("strikes", []))
    hit = [False] * len(strikes)
    tp = 0
    for w in final:
        box = w.get("bbox_frac")
        for i, s in enumerate(strikes):
            if hit[i] or w.get("page") != s["page"] or not box:
                continue
            if _iou(box, s["bbox"]) >= MATCH_IOU:
                hit[i] = True
                tp += 1
                break
    return tp, len(final) - tp, len(strikes) - sum(hit)


def run_conf_split(docs):
    tot = hi = lo = 0
    print(f"{'document':28s} {'final':>6s} {'conf>=gate':>11s} {'conf<gate':>10s}")
    for name, pdf, di in docs:
        final = _analyze(*_load(pdf, di))
        n_hi = sum(1 for w in final if (w.get("conf") or 0.0) >= CLEAN_CONF)
        n_lo = len(final) - n_hi
        tot += len(final); hi += n_hi; lo += n_lo
        print(f"{name:28s} {len(final):6d} {n_hi:11d} {n_lo:10d}")
    print(f"{'TOTAL':28s} {tot:6d} {hi:11d} {lo:10d}")
    print(f"\ngate = ScanConfig.max_clean_conf = {CLEAN_CONF}. 'conf>=gate' is the high-confidence "
          "clean-text\nover-flag the 0.9.1 veto targets; 'conf<gate' is the faint-scan "
          "CNN-saturation residual\ntracked for the CNN retrain (issue #4·C/D).")
    return 0


def run_ab(corpus_dir, docs):
    gt = load_ground_truth(corpus_dir)
    if gt is None:
        print("!! no ground-truth.json at the corpus root: reporting RAW struck-final counts.\n"
              "   A raw count is not a false-positive count — it includes the real strikes.\n")
    head = f"{'document':28s} {'off':>5s} {'on':>5s}"
    print(head + (f" {'FP off':>7s} {'FP on':>6s} {'TP':>4s} {'FN':>4s}" if gt else ""))

    t = dict(off=0, on=0, fp_off=0, fp_on=0, tp_off=0, tp_on=0, fn_on=0, strikes=0)
    unlabeled = []
    for name, pdf, di in docs:
        pdf_bytes, di_result = _load(pdf, di)
        off = _analyze(pdf_bytes, di_result, None)
        on = _analyze(pdf_bytes, di_result, st.ScanConfig.ruled_forms())
        t["off"] += len(off); t["on"] += len(on)
        row = f"{name:28s} {len(off):5d} {len(on):5d}"
        if gt is not None:
            # An unlabeled document is UNKNOWN, not known-clean. Scoring it as all-false-positive
            # would quietly inflate the headline number, so say so instead of assuming.
            if name not in gt:
                unlabeled.append(name)
            truth = gt.get(name, {"strikes": []})
            tp_off, fp_off, _ = score(off, truth)
            tp_on, fp_on, fn_on = score(on, truth)
            t["fp_off"] += fp_off; t["fp_on"] += fp_on
            t["tp_off"] += tp_off; t["tp_on"] += tp_on; t["fn_on"] += fn_on
            t["strikes"] += len(truth.get("strikes", []))
            flag = "  <-- RECALL LOSS" if tp_on < tp_off else ""
            row += f" {fp_off:7d} {fp_on:6d} {tp_on:4d} {fn_on:4d}{flag}"
        print(row)

    print(f"{'TOTAL':28s} {t['off']:5d} {t['on']:5d}"
          + (f" {t['fp_off']:7d} {t['fp_on']:6d} {t['tp_on']:4d} {t['fn_on']:4d}" if gt else ""))
    delta = 100.0 * (t["on"] - t["off"]) / max(t["off"], 1)
    print(f"\nstruck-final words: {t['off']} -> {t['on']} ({delta:+.0f}%) with "
          "ScanConfig.ruled_forms()")
    if gt is not None:
        d_fp = 100.0 * (t["fp_on"] - t["fp_off"]) / max(t["fp_off"], 1)
        print(f"false positives:    {t['fp_off']} -> {t['fp_on']} ({d_fp:+.0f}%)")
        print(f"true positives:     {t['tp_off']} -> {t['tp_on']} of {t['strikes']} real strikes "
              f"in the corpus  (recall {t['tp_on']}/{t['strikes']})")
        if unlabeled:
            print(f"\n!! {len(unlabeled)} document(s) are absent from ground-truth.json and were "
                  f"scored as having NO real strikes: {', '.join(unlabeled)}.\n"
                  "   Label them, or the false-positive counts above are wrong.")
        if t["tp_on"] < t["tp_off"]:
            print("\n!! the veto COST RECALL on this corpus — it must not ship in that state.")
    return 0


def run_switches(corpus_dir, docs):
    """Every combination of the two provisional precision switches, scored the same way as --ab.

    They are independent and opted into separately, so the useful question is what each buys ALONE
    and whether they overlap. One row per document, then per-configuration totals.
    """
    gt = load_ground_truth(corpus_dir)
    if gt is None:
        print("!! no ground-truth.json at the corpus root: this mode needs labels to report FP.")
        return 1
    configs = {
        "default": st.ScanConfig(),
        "veto only": st.ScanConfig.ruled_forms(),
        "chain only": st.ScanConfig(rescue_clean_chains=False),
        "both": st.ScanConfig.ruled_forms(rescue_clean_chains=False),
    }
    totals = {k: dict(n=0, fp=0, tp=0, fn=0) for k in configs}
    print(f"{'document':26s} " + " ".join(f"{k:>11s}" for k in configs) + "   (false positives)")
    unlabeled = []
    for name, pdf, di in docs:
        pdf_bytes, di_result = _load(pdf, di)
        if name not in gt:
            unlabeled.append(name)
        truth = gt.get(name, {"strikes": []})
        cells = []
        for k, cfg in configs.items():
            final = _analyze(pdf_bytes, di_result, cfg)
            tp, fp, fn = score(final, truth)
            t = totals[k]
            t["n"] += len(final); t["fp"] += fp; t["tp"] += tp; t["fn"] += fn
            cells.append(fp)
        print(f"{name:26s} " + " ".join(f"{c:11d}" for c in cells))
    print(f"{'TOTAL FP':26s} " + " ".join(f"{totals[k]['fp']:11d}" for k in configs))

    base = totals["default"]["fp"]
    print()
    for k, t in totals.items():
        delta = 100.0 * (t["fp"] - base) / max(base, 1)
        print(f"{k:12s} struck-final={t['n']:<4d} FP={t['fp']:<4d} ({delta:+.0f}% vs default)  "
              f"recall={t['tp']}/{t['tp'] + t['fn']}")
    apart = (base - totals["veto only"]["fp"]) + (base - totals["chain only"]["fp"])
    together = base - totals["both"]["fp"]
    print(f"\nseparately the two switches remove {apart} FPs, together {together} — "
          f"an overlap of {apart - together}.")
    if unlabeled:
        print(f"\n!! {len(unlabeled)} document(s) absent from ground-truth.json, scored as having NO "
              f"real strikes: {', '.join(unlabeled)}. Label them or these counts are wrong.")
    if min(t["tp"] for t in totals.values()) < totals["default"]["tp"]:
        print("\n!! a switch COST RECALL on this corpus — it must not ship in that state.")
    return 0


def run(corpus_dir, ab=False, switches=False):
    docs = _docs_in(corpus_dir)
    if not docs:
        print(f"no document subfolders under {corpus_dir!r}\n"
              "point the harness at a local corpus (PDF_STRIKETHROUGH_CORPUS_DIR=... or pass the "
              "path as an argument); each subfolder needs a *.pdf and a di-result.json.")
        return 1
    if switches:
        return run_switches(corpus_dir, docs)
    return run_ab(corpus_dir, docs) if ab else run_conf_split(docs)


if __name__ == "__main__":
    KNOWN = {"--ab", "--switches"}
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    if set(flags) - KNOWN:
        print(f"{__file__}: unknown option(s) {' '.join(sorted(set(flags) - KNOWN))}\n"
              "usage: confidence_veto.py [CORPUS_DIR] [--ab | --switches]")
        raise SystemExit(2)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = args[0] if args else os.environ.get("PDF_STRIKETHROUGH_CORPUS_DIR", DEFAULT_DIR)
    raise SystemExit(run(target, ab="--ab" in flags, switches="--switches" in flags))
