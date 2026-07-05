"""Native (born-digital) quick start — no assets to download.

Builds a small redline PDF in memory (a line with two vector strikethroughs), then runs the exact
native detector on it and prints the struck words, the surviving clean text, and the ~~markdown~~.

    python examples/native_quickstart.py

Only needs the base install (pymupdf ships with the package).
"""
import pymupdf

import pdf_strikethrough as st


def build_redline_pdf() -> bytes:
    """A one-page PDF: 'The quick brown fox jumps over the lazy dog' with 'brown' and 'lazy' struck
    through with vector lines — exactly what a redline/track-changes export produces."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 120), "The quick brown fox jumps over the lazy dog", fontsize=14)
    boxes = {w[4]: pymupdf.Rect(w[:4]) for w in page.get_text("words")}
    for word in ("brown", "lazy"):
        r = boxes[word]
        ymid = (r.y0 + r.y1) / 2
        page.draw_line(pymupdf.Point(r.x0, ymid), pymupdf.Point(r.x1, ymid), width=1.0)
    return doc.tobytes()


def main() -> None:
    pdf = build_redline_pdf()

    print("struck words (exact, no OCR):")
    for w in st.strikethroughs_in_pdf(pdf):
        kind = "partial" if w["partial"] else "full"
        print(f"  p{w['page']} {kind:<7} {w['chars']!r}  (tier={w['tier']}, "
              f"coverage={w['coverage']})")

    res = st.detect_pdf(pdf)
    print("\nmarkdown  :", res["markdown"])
    print("clean_text:", res["clean_text"])
    print("passages  :", [(p["text"], p["n_words"]) for p in res["passages"]])


if __name__ == "__main__":
    main()
