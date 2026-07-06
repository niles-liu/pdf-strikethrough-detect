"""DI-parity (legacy) — reproduces the "1477 vs 1484 struck words (99.5%)" claim.

Superseded on this repo by ``scanned_recovery.py``, which scores the scanned path against the
native detector's *known* strikes and so needs no vanished pipeline. This script remains for anyone
who has a genuinely scanned corpus AND the original pipeline's recorded reference counts.

The package's scanned classifier is a from-scratch reimplementation of the original Azure
Document Intelligence-based pipeline. This script feeds a stored DI analyze-result through
``detect_pdf(di_result=...)`` and compares the number of struck words this package reports against
the reference count the original DI pipeline produced, per document — the "N vs M" parity figure.

    python benchmarks/di_parity.py

Inputs (per manifest entry): `di_result` (path to the DI analyze-result JSON, relative to
benchmarks/corpus/) and `di_reference_struck` (the struck-word count the original DI pipeline
reported for that document). Entries missing either are skipped. Needs no OCR backend or cloud
access — the DI words come from the stored JSON.
"""
from __future__ import annotations

import json

import pdf_strikethrough as st

from _corpus import corpus_dir, iter_corpus


def main() -> None:
    cdir = corpus_dir()
    total_ours = total_ref = 0
    ran = 0
    print(f"{'document':<40} {'ours':>6} {'DI ref':>7} {'parity':>7}")
    print("-" * 64)
    for entry, path in iter_corpus():
        di_file = entry.get("di_result")
        ref = entry.get("di_reference_struck")
        if not di_file or ref is None:
            continue
        with open(cdir / di_file, encoding="utf-8") as f:
            di = json.load(f)
        res = st.detect_pdf(str(path), di_result=di)
        ours = res["n_struck_final"]
        parity = min(ours, ref) / max(ours, ref) if max(ours, ref) else 1.0
        print(f"{entry['name'][:40]:<40} {ours:>6} {ref:>7} {parity:>6.1%}")
        total_ours += ours
        total_ref += ref
        ran += 1
    print("-" * 64)
    if not ran:
        print("No manifest entries carry both `di_result` and `di_reference_struck` — nothing to "
              "compare (see benchmarks/README.md).")
        return
    overall = min(total_ours, total_ref) / max(total_ours, total_ref) if max(total_ours, total_ref) else 1.0
    print(f"{'TOTAL':<40} {total_ours:>6} {total_ref:>7} {overall:>6.1%}")
    print(f"\n{total_ours} vs {total_ref} struck words -> {overall:.1%} parity with the original "
          f"Azure-DI pipeline.")


if __name__ == "__main__":
    main()
