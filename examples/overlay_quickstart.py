"""Overlay + forensics quick start — no assets to download.

Builds a small redline PDF (a vector strike plus an explicit /StrikeOut annotation with an author),
then (1) prints the evidence each record carries — stroke color/width for the vector strike, and
author/color for the annotation — and (2) writes a before/after overlay image pair showing the
detected strikes boxed. This is the figure the README/launch post is built around.

    python examples/overlay_quickstart.py [OUT_DIR]

Only needs the base install (pymupdf + pillow ship with the package).
"""
import sys

import pymupdf

import pdf_strikethrough as st


def build_redline_pdf() -> bytes:
    """One page: a vector strike through 'brown', and an Acrobat-style /StrikeOut annotation
    (author 'J. Reviewer', red) through 'lazy'."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 120), "The quick brown fox jumps over the lazy dog", fontsize=14)
    boxes = {w[4]: pymupdf.Rect(w[:4]) for w in page.get_text("words")}
    r = boxes["brown"]
    ymid = (r.y0 + r.y1) / 2
    page.draw_line(pymupdf.Point(r.x0, ymid), pymupdf.Point(r.x1, ymid), width=1.2, color=(0.8, 0, 0))
    annot = page.add_strikeout_annot(boxes["lazy"])
    annot.set_info(title="J. Reviewer")
    annot.set_colors(stroke=(1, 0, 0))
    annot.update()
    return doc.tobytes()


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    pdf = build_redline_pdf()

    print("struck words with evidence (union of vector + flag + annotation):")
    for w in st.strikethroughs_in_pdf(pdf, method="both"):
        bits = []
        if w.get("stroke_color") is not None:
            bits.append(f"stroke={w['stroke_color']} width={w.get('stroke_width')}")
        if w.get("annot_author"):
            bits.append(f"author={w['annot_author']!r} annot_color={w.get('annot_color')}")
        print(f"  p{w['page']} {w['chars']!r:<10} tier={w['tier']:<7} {'  '.join(bits)}")

    written = st.save_overlays(pdf, out_dir, dpi=150)
    print("\noverlay image(s):", written)


if __name__ == "__main__":
    main()
