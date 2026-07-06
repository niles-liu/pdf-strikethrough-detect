"""Assemble struck-aware text from a page's words + strike decisions.

Reading-order words (each optionally carrying a struck record) become:
  - ``markdown`` with deletions wrapped as ``~~struck~~`` (partial strikes wrap only the struck
    chars in place), and
  - ``clean_text`` = the surviving text with those spans removed.

This reflects THIS package's own strike decisions (from vector geometry on native pages, or
geometry+OCR+CNN on scanned pages) — it does not depend on any external markdown engine, so the
text and the per-word records always agree.
"""
from __future__ import annotations

import re

import numpy as np

_STRIKE_SPAN = re.compile(r"~~(.*?)~~", re.S)


def mark_word(text, rec):
    """Markdown for one word given its struck record (or None). full -> ~~word~~; partial ->
    only the struck chars wrapped in place; not struck -> the word unchanged."""
    if rec is None or not rec.get("final"):
        return text
    c0, c1 = rec.get("char_span", (0, len(text)))
    if not rec.get("partial") or (c0 == 0 and c1 >= len(text)):
        return f"~~{text}~~"
    return f"{text[:c0]}~~{text[c0:c1]}~~{text[c1:]}"


def cluster_rows(items):
    """Group [(text, bbox_frac, rec), ...] into visual rows (by y-center), each sorted left→right,
    rows top→bottom. bbox_frac is (x0, y0, x1, y1) in page fractions."""
    if not items:
        return []
    med_h = float(np.median([b[3] - b[1] for _, b, _ in items])) or 1e-3
    rows, row, row_y = [], [], None
    for it in sorted(items, key=lambda t: ((t[1][1] + t[1][3]) / 2, t[1][0])):
        yc = (it[1][1] + it[1][3]) / 2
        if row_y is None or abs(yc - row_y) <= 0.6 * med_h:
            row.append(it)
            row_y = yc if row_y is None else 0.8 * row_y + 0.2 * yc
        else:
            rows.append(sorted(row, key=lambda t: t[1][0]))
            row, row_y = [it], yc
    if row:
        rows.append(sorted(row, key=lambda t: t[1][0]))
    return rows


def page_markdown(items):
    """Struck-aware markdown for one page from [(text, bbox_frac, rec), ...] (rec may be None)."""
    return "\n".join(" ".join(mark_word(t, r) for t, _b, r in row) for row in cluster_rows(items))


def page_clean_text(items):
    """Surviving (non-deleted) text for one page from [(text, bbox_frac, rec), ...]: struck
    words are dropped (partial strikes keep only the un-struck chars), assembled row by row
    from the word records directly — immune to literal '~~' in the document text."""
    rows_out = []
    for row in cluster_rows(items):
        toks = []
        for t, _b, r in row:
            if r is None or not r.get("final"):
                toks.append(t)
                continue
            c0, c1 = r.get("char_span", (0, len(t)))
            if not r.get("partial") or (c0 == 0 and c1 >= len(t)):
                continue                       # fully struck: drop the word
            rem = t[:c0] + t[c1:]
            if rem:
                toks.append(rem)
        if toks:
            rows_out.append(" ".join(toks))
    return "\n".join(rows_out)


def strip_struck(md):
    """Remove ~~struck~~ spans from markdown -> the surviving (non-deleted) text.

    Caveat: text that itself contains a literal '~~' is ambiguous in markdown form; prefer
    :func:`page_clean_text`, which works from the word records and has no such ambiguity."""
    clean = _STRIKE_SPAN.sub("", md)
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    return re.sub(r"\n{3,}", "\n\n", clean).strip()


def mark_provenance(md, template="[deleted: {}]"):
    """Rewrite ``~~struck~~`` spans as audit markers (default ``[deleted: X]``) instead of removing
    them — audit-preserving text for RAG / indexing, where silently dropping deleted text loses the
    fact that something *was* deleted (a documented LlamaIndex failure mode). `template` is a
    ``str.format`` pattern receiving the struck text.

    Consecutive struck words separated only by spaces (a deletion passage) collapse into one
    marker — ``~~the~~ ~~old~~ ~~rate~~`` -> ``[deleted: the old rate]`` — rather than one marker
    per word. Shares :func:`strip_struck`'s caveat: text with a literal ``~~`` is ambiguous in
    markdown form (prefer building this from ``detect_pdf(...)['markdown']``, which the package
    emits)."""
    matches = list(_STRIKE_SPAN.finditer(md))
    out, pos, i = [], 0, 0
    while i < len(matches):
        m = matches[i]
        out.append(md[pos:m.start()])
        texts, end, j = [m.group(1)], m.end(), i + 1
        while j < len(matches) and set(md[end:matches[j].start()]) <= {" ", "\t"}:
            texts.append(md[end:matches[j].start()] + matches[j].group(1))   # keep the separator
            end, j = matches[j].end(), j + 1
        out.append(template.format("".join(texts)))
        pos, i = end, j
    out.append(md[pos:])
    return "".join(out)


def group_passages(items):
    """Maximal runs of consecutive final-struck words in reading order -> deletion passages.
    Returns [{text, n_words, bbox_frac}], one per contiguous struck section."""
    passages, run = [], []

    def flush():
        if not run:
            return
        boxes = [b for _, b, _ in run]
        text = " ".join((r["chars"] if r.get("partial") else t) for t, _, r in run)
        passages.append({
            "text": text, "n_words": len(run),
            "bbox_frac": [round(min(b[0] for b in boxes), 5), round(min(b[1] for b in boxes), 5),
                          round(max(b[2] for b in boxes), 5), round(max(b[3] for b in boxes), 5)],
        })
        run.clear()

    for row in cluster_rows(items):
        for t, b, r in row:
            if r is not None and r.get("final"):
                run.append((t, b, r))
            else:
                flush()
    flush()
    return passages
