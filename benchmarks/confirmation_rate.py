"""Confirmation rate — the headline native-detection claim, made reproducible.

For every PDF in the corpus, run the vector detector and the flag detector independently and
report what fraction of vector-detected struck words are also flagged by MuPDF's own strikeout
signal (`FZ_STEXT_STRIKEOUT`). The two are independent evidence sources, so a high agreement rate
is a strong falsifiable check on the vector path.

    python benchmarks/confirmation_rate.py

Backs the "99.8% of vector detections are independently confirmed by the flag signal, on 10
public redline PDFs (54.7k struck words; 92.5-100% per document)" claim in the README and
native.py. Needs only the PDFs — no OCR, no cloud.
"""
from __future__ import annotations

import pdf_strikethrough as st

from _corpus import iter_corpus


def _keys(records):
    """Match-insensitive-to-float-noise key set for a page's struck words."""
    return {(r["page"], r["text"], round(r["bbox_frac"][0], 2), round(r["bbox_frac"][1], 2))
            for r in records}


def main() -> None:
    total_vec = total_confirmed = total_flag_extra = 0
    print(f"{'document':<40} {'vector':>7} {'confirmed':>10} {'rate':>7} {'flag-only':>10}")
    print("-" * 78)
    for entry, path in iter_corpus():
        vec = st.strikethroughs_in_pdf(str(path), method="vector")
        flag = st.strikethroughs_in_pdf(str(path), method="flag")
        vk, fk = _keys(vec), _keys(flag)
        confirmed = len(vk & fk)
        rate = confirmed / len(vk) if vk else 1.0
        print(f"{entry['name'][:40]:<40} {len(vk):>7} {confirmed:>10} {rate:>6.1%} "
              f"{len(fk - vk):>10}")
        total_vec += len(vk)
        total_confirmed += confirmed
        total_flag_extra += len(fk - vk)
    print("-" * 78)
    overall = total_confirmed / total_vec if total_vec else 1.0
    print(f"{'TOTAL':<40} {total_vec:>7} {total_confirmed:>10} {overall:>6.1%} "
          f"{total_flag_extra:>10}")
    print(f"\n{total_vec} vector detections across the corpus; {overall:.2%} confirmed by the flag "
          f"signal; {total_flag_extra} additional words flagged only (captured by method='both').")


if __name__ == "__main__":
    main()
