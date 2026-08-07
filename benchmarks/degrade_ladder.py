"""Does synthetic degradation reproduce the regime that actually breaks the detector?

    python benchmarks/degrade_ladder.py                 # RapidOCR, whole ladder
    python benchmarks/degrade_ladder.py --levels L0_clean,L3_fax
    python benchmarks/degrade_ladder.py --docs copyright   # substring-match the manifest `file`
    python benchmarks/degrade_ladder.py --write          # also cache the degraded PDFs

``--write`` drops ``<doc>.<level>.pdf`` into ``corpus/`` alongside the originals. They are ignored by
the manifest loader, but anything else globbing that directory will pick them up; delete them when
you are done looking.

Budget it before starting: this is ``pages x levels`` OCR passes at a second or several each, so the
full ladder over the whole corpus is a ~25-minute job. OCR is the whole cost — each document is
rasterized once and every rung degrades that one raster — so ``--docs`` / ``--levels`` are the only
lever, and worth using while you are still deciding what to look at.

This is the **entry gate for the degrade-and-relabel route**, not a headline benchmark. The route
assumes that fading, resampling, skewing and JPEG-ing a born-digital redline lands it in the same
regime as a photocopied form. That assumption is testable here, cheaply and with no cloud call: walk
the ladder over the public redline corpus, score each rung against the native detector's exact
strike set, and watch what the curve does.

Read the output as a **direction test**, not a measurement:

- Recall falls and the predicted count climbs as the rungs get worse -> degradation is at least
  pushing toward the failing regime, and L2/L3 crops are worth dumping as training negatives.
- Recall falls but the count *also* falls -> the ladder is destroying ink, not manufacturing
  confusion. That is a different failure from the one on the SOF corpus and the route is not
  reproducing it.
- Nothing moves much -> the ladder is too gentle to be the regime, and the domain gap is doing the
  work. Say so and stop, rather than spending Azure DI calls on it.

⚠ The scanned SOF corpus over-flags **printed rules and handwriting** on ruled forms. Government
redlines have neither. So even a textbook curve here leaves the ruled-form part of the gap
unmodelled — see the caveat in `_degrade.py` and `POSITIVES.md` §3.
"""
from __future__ import annotations

import sys

import pdf_strikethrough as st

from _corpus import corpus_dir, iter_corpus
from _degrade import BY_NAME, LADDER, build_degraded_pdf, degrade_frac_boxes, render_pages
from _scanned import struck_pages
from scanned_recovery import _coverage, _struck_boxes


def _opt(argv, name):
    """Comma-separated values for ``--name=a,b`` or ``--name a,b``; None if the flag is absent.

    An empty or missing value is an error rather than a silent fall-through to the default: a run
    scoped with ``--docs`` that quietly scored the whole corpus would be a 25-minute surprise, and
    the resulting table would look like the one that was asked for.
    """
    for i, a in enumerate(argv):
        raw = None
        if a == f"--{name}":
            raw = argv[i + 1] if i + 1 < len(argv) else ""
        elif a.startswith(f"--{name}="):
            raw = a.split("=", 1)[1]
        if raw is not None:
            vals = [v.strip() for v in raw.split(",") if v.strip()]
            if not vals:
                raise SystemExit(f"--{name} needs a comma-separated value")
            return vals
    return None


def _levels(argv):
    names = _opt(argv, "levels")
    if names is None:
        return list(LADDER)
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        raise SystemExit(f"unknown level(s) {', '.join(unknown)}; "
                         f"choose from {', '.join(BY_NAME)}")
    return [BY_NAME[n] for n in names]


def _rapidocr():
    try:
        import rapidocr  # noqa: F401
        from pdf_strikethrough.ocr import rapidocr_backend
        from pdf_strikethrough.scanned import ScanConfig
    except ImportError:
        raise SystemExit("this script needs RapidOCR: pip install 'rapidocr>=3.2'") from None
    eng, cfg = rapidocr_backend(), ScanConfig.confidence_free()
    return lambda pdf_bytes: st.detect_pdf(pdf_bytes, ocr=eng, scan_config=cfg)


def main() -> None:
    argv = sys.argv[1:]
    levels = _levels(argv)
    only = _opt(argv, "docs")
    write = "--write" in argv
    run = _rapidocr()
    cdir = corpus_dir()

    print(f"\n{'document':<34} {'pg':>3} {'GT':>4} " +
          " ".join(f"{lv.name:>22}" for lv in levels))
    print("(each cell: recall of the native strike set | predicted struck count)")
    print("-" * (34 + 9 + 23 * len(levels)))

    tot_gt = 0
    tot_cov = {lv.name: 0.0 for lv in levels}
    tot_pred = {lv.name: 0 for lv in levels}
    ran = 0
    for entry, orig in iter_corpus():
        pages = entry.get("scanned_pages")
        if not pages:
            continue
        if only and not any(s.lower() in entry["file"].lower() for s in only):
            continue
        page_indices, gt = struck_pages(orig, len(pages))
        # Boxes are pooled across the document's pages, matching how scanned_recovery.py scores:
        # _struck_boxes flattens predictions the same way, so a rung's recall is page-blind on both
        # sides. Fine for a direction test between rungs; do not read a cell as per-page accuracy.
        gt_boxes = [b for boxes in gt.values() for b in boxes]
        if not gt_boxes:
            continue
        rendered = render_pages(orig, page_indices)     # once per document, reused by every rung
        _, w, h = rendered[0]
        aspect = w / h                                  # pages of one document share a page size

        cells = []
        for lv in levels:
            pdf_bytes = build_degraded_pdf(rendered, lv)
            if write:
                (cdir / f"{orig.stem}.{lv.name}.pdf").write_bytes(pdf_bytes)
            pred = _struck_boxes(run(pdf_bytes))
            cov = _coverage(degrade_frac_boxes(gt_boxes, lv, aspect), pred)
            tot_cov[lv.name] += cov * len(gt_boxes)
            tot_pred[lv.name] += len(pred)
            cells.append(f"{cov:>15.1%} | {len(pred):>4}")

        print(f"{entry['name'][:34]:<34} {len(page_indices):>3} {len(gt_boxes):>4} " +
              " ".join(cells))
        tot_gt += len(gt_boxes)
        ran += 1

    if not ran:
        if only:
            print(f"\nNo manifest entry with `scanned_pages` matched --docs {','.join(only)}.")
        else:
            print("\nNo manifest entry carries `scanned_pages`. Run `python benchmarks/"
                  "prep_scanned_di.py` first, or add the field by hand — the ladder needs only the\n"
                  "page list, not a DI result, so no Azure key is required to pick pages.")
        return

    print("-" * (34 + 9 + 23 * len(levels)))
    agg = " ".join(f"{(tot_cov[lv.name] / tot_gt if tot_gt else 1.0):>15.1%} | "
                   f"{tot_pred[lv.name]:>4}" for lv in levels)
    print(f"{'TOTAL (recall-weighted)':<34} {'':>3} {tot_gt:>4} {agg}")
    print(f"\n{tot_gt} known strikes across {ran} documents. Predicted counts above the GT column "
          f"are\nover-flagging: on this corpus every extra detection is a false positive.")
    if write:
        print(f"Degraded PDFs cached under {cdir}/ as <doc>.<level>.pdf")


if __name__ == "__main__":
    main()
