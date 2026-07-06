"""RAG provenance quick start — keep deleted text visible-as-deleted, don't silently drop it.

Struck text that flows into a vector index is a real failure mode: the retriever surfaces a
sentence the document explicitly *deleted*. This package's ``clean_text`` removes it — but for an
audit-preserving index you often want the deletion recorded, not erased. ``provenance_text`` keeps
each struck span as a ``[deleted: ...]`` marker so a downstream chunk still shows that something
was struck.

    python examples/rag_provenance.py

Only needs the base install (pymupdf ships with the package).
"""
import pymupdf

import pdf_strikethrough as st


def build_redline_pdf() -> bytes:
    """One page striking 'the old rate of 5%' out of a sentence."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 120), "The fee is the old rate of 5% billed monthly.", fontsize=14)
    words = {w[4]: pymupdf.Rect(w[:4]) for w in page.get_text("words")}
    for token in ("the", "old", "rate", "of", "5%"):
        r = words[token]
        ymid = (r.y0 + r.y1) / 2
        page.draw_line(pymupdf.Point(r.x0, ymid), pymupdf.Point(r.x1, ymid),
                       width=1.2, color=(0.8, 0, 0))
    return doc.tobytes()


def main() -> None:
    pdf = build_redline_pdf()
    result = st.detect_pdf(pdf, method="vector")

    print("clean_text  (deletions removed - the default):")
    print("   ", repr(result["clean_text"]))
    print("\nprovenance_text (deletions kept as markers - audit-preserving for RAG):")
    print("   ", repr(st.provenance_text(result)))
    print("\ncustom marker template:")
    print("   ", repr(st.provenance_text(result, template="[REDACTED:{}]")))

    # A minimal integration sketch: index the provenance text so the deletion is searchable-as-
    # deleted rather than lost. Swap `chunks` into LlamaIndex / LangChain / your vector store.
    chunks = st.provenance_text(result).split(". ")
    print("\nchunks an index would receive:")
    for c in chunks:
        print("   ", repr(c))


if __name__ == "__main__":
    main()
