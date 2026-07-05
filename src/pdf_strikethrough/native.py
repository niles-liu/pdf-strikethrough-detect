"""Native-PDF strikethrough extraction — for born-digital PDFs, vector geometry is exact
ground truth (no OCR, no model, no guessing).

In born-digital documents a strikethrough is a DRAWING: a thin horizontal vector line ("l"
item) or a thin filled rectangle ("re"/"qu" item) painted over the text. A word is struck when
the merged stroke coverage through its MIDDLE BAND (0.22h..0.78h — excludes underlines and
overlines) reaches half its width; smaller mid-band coverage that still spans >= 2 characters is
a genuine partial strike ('semi-' of 'semi-monthly', '19' of '192012') with the char range
estimated proportionally from the covered x-span.

All output bbox_frac values are fractions of the ROTATED (as-rendered) page, so they map
directly onto ``page.get_pixmap()`` output; detection itself runs in MuPDF's unrotated text
space, where strikes stay horizontal regardless of /Rotate.
"""
import math
import re

import pymupdf  # (formerly imported as the deprecated `fitz` alias)

# word-level strike thresholds (fractions of word width covered by mid-band strokes)
FULL_COV = 0.70          # >= this: the whole word is struck
STRUCK_COV = 0.50        # >= this: struck (partial if < FULL_COV)
PARTIAL_COV = 0.25       # >= this AND >= 2 chars: partial strike; below = grazing stroke end
MIN_STROKE_LEN = 6.0     # pt; shorter "lines" are dashes/tick marks
MAX_STROKE_DY = 1.5      # pt; a strike stroke is horizontal
MAX_RECT_H = 3.5         # pt; a strike drawn as a filled rect is thin
MID_BAND = 0.22          # strokes within [y0 + f*h, y1 - f*h] count as through-text

FLAG_MIN_WCOV = 0.15     # flag path: a struck span must cover >= this of a word to count


def _bbox_frac(page, x0, y0, x1, y1):
    """Unrotated-space rect -> (x0, y0, x1, y1) fractions of the rendered (rotated) page."""
    r = pymupdf.Rect(x0, y0, x1, y1) * page.rotation_matrix
    r.normalize()
    pw = page.rect.width or 1.0
    ph = page.rect.height or 1.0
    return (r.x0 / pw, r.y0 / ph, r.x1 / pw, r.y1 / ph)


def _paint_invisible(color, opacity):
    """A stroke/fill leaves no ink — and so cannot strike anything — when it is fully transparent
    or painted in ~the page background (near-white). An unset (None) stroke color is PDF-default
    black, i.e. visible."""
    if opacity is not None and opacity <= 0.05:
        return True
    if color is None:
        return False
    return all(c >= 0.95 for c in color[:3])


def horiz_strokes(page):
    """All horizontal stroke intervals on the page: [(x0, x1, y), ...] in pt (unrotated space).
    Invisible strokes (transparent, or drawn in the page background color) leave no ink and are
    skipped — geometry alone would otherwise confirm a white / opacity-0 line as a strike."""
    out = []
    for d in page.get_drawings():
        stroke_col, stroke_op = d.get("color"), d.get("stroke_opacity", 1.0)
        for it in d["items"]:
            if it[0] == "l":
                if _paint_invisible(stroke_col, stroke_op):
                    continue
                p1, p2 = it[1], it[2]
                if abs(p1.y - p2.y) <= MAX_STROKE_DY and abs(p1.x - p2.x) >= MIN_STROKE_LEN:
                    out.append((min(p1.x, p2.x), max(p1.x, p2.x), (p1.y + p2.y) / 2))
            elif it[0] in ("re", "qu"):
                # a strike drawn as a thin bar is a FILLED rect — judge it by its fill paint,
                # falling back to the stroke paint when it is only stroked
                if d.get("fill") is not None:
                    col, op = d.get("fill"), d.get("fill_opacity", 1.0)
                else:
                    col, op = stroke_col, stroke_op
                if _paint_invisible(col, op):
                    continue
                r = it[1] if it[0] == "re" else it[1].rect
                if r.height <= MAX_RECT_H and r.width >= MIN_STROKE_LEN:
                    out.append((r.x0, r.x1, (r.y0 + r.y1) / 2))
    return out


def _merged_intervals(wx0, wx1, ivals):
    """Clip intervals to [wx0, wx1] and merge overlaps. Returns (merged, covered_length)."""
    ivals = sorted((max(a, wx0), min(b, wx1)) for a, b in ivals)
    merged, tot = [], 0.0
    for a, b in ivals:
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    tot = sum(b - a for a, b in merged)
    return merged, tot


def native_page_strikes(page, page_index):
    """Struck-word records for one native page, in reading order.

    Each record: {page, text, chars, char_span, partial, bbox_frac, coverage, tier='vector',
    verdict='struck', final=True}. Vector geometry is exact, so there is no unsure tier.
    """
    words = [w for w in page.get_text("words")
             if w[4].strip() and (w[3] - w[1]) >= 4 and (w[2] - w[0]) >= 3]
    if not words:
        return []
    strokes = horiz_strokes(page)
    if not strokes:
        return []

    out = []
    for (wx0, wy0, wx1, wy1, txt, *_r) in words:
        wh = wy1 - wy0
        mid = [(a, b) for (a, b, sy) in strokes
               if wy0 + MID_BAND * wh <= sy <= wy1 - MID_BAND * wh and min(b, wx1) > max(a, wx0)]
        if not mid:
            continue
        merged, covered = _merged_intervals(wx0, wx1, mid)
        cov = covered / max(wx1 - wx0, 1e-6)
        if cov < PARTIAL_COV:
            continue
        full = cov >= FULL_COV
        if full:
            c0, c1 = 0, len(txt)
        else:
            a, b = max(merged, key=lambda m: m[1] - m[0])
            f0 = (a - wx0) / max(wx1 - wx0, 1e-9)
            f1 = (b - wx0) / max(wx1 - wx0, 1e-9)
            c0 = int(f0 * len(txt))
            c1 = max(c0 + 1, int(round(f1 * len(txt))))
            if cov < STRUCK_COV and c1 - c0 < 2:
                continue                       # grazing stroke end, not a deletion
        out.append({
            "page": page_index, "text": txt, "chars": txt[c0:c1], "char_span": (c0, c1),
            "partial": not full,
            "bbox_frac": _bbox_frac(page, wx0, wy0, wx1, wy1),
            "coverage": round(cov, 3), "tier": "vector", "verdict": "struck", "final": True,
        })
    out.sort(key=lambda h: (round(h["bbox_frac"][1], 3), h["bbox_frac"][0]))
    return out


def native_flag_strikes(page, page_index):
    """Struck words from MuPDF's own strikeout detection — the FZ_STEXT_STRIKEOUT char flag,
    enabled by extracting with COLLECT_STYLES|COLLECT_VECTORS (base PyMuPDF >= 1.26, no
    pymupdf4llm needed; this is the same signal pymupdf4llm renders as ``~~``).

    Struck spans are snapped onto the page's ``get_text("words")`` boxes, so records carry the
    SAME exact word boxes and text as the vector detector — a span covering only part of a word
    ('Policy' of 'PolicyTo') becomes a partial strike with a proportional char range. Records:
    {page, text, chars, char_span, partial, bbox_frac, coverage, tier='flag',
    verdict='struck', final=True}.
    """
    flags = pymupdf.TEXTFLAGS_DICT | pymupdf.TEXT_COLLECT_STYLES | pymupdf.TEXT_COLLECT_VECTORS
    strike_bit = pymupdf.mupdf.FZ_STEXT_STRIKEOUT
    page_words = [w for w in page.get_text("words") if w[4].strip()]
    if not page_words:
        return []

    by_word = {}                               # (word tuple) -> [(c0, c1), ...]
    for block in page.get_text("dict", flags=flags).get("blocks", []):
        for line in block.get("lines", []):
            if line.get("dir") not in ((1, 0), (0, 1)):        # strikeout is axis-parallel only
                continue
            for span in line.get("spans", []):
                if not (span.get("char_flags", 0) & strike_bit):
                    continue
                if not span.get("text", "").strip():
                    continue
                sx0, sy0, sx1, sy1 = span["bbox"]
                for w in page_words:
                    wx0, wy0, wx1, wy1, txt = w[0], w[1], w[2], w[3], w[4]
                    ov = min(sx1, wx1) - max(sx0, wx0)
                    if ov <= 0:
                        continue
                    wyc = (wy0 + wy1) / 2                      # same text line as the span?
                    if not (sy0 - 1 <= wyc <= sy1 + 1):
                        continue
                    wcov = ov / max(wx1 - wx0, 1e-9)
                    if wcov < FLAG_MIN_WCOV:
                        continue
                    f0 = (max(sx0, wx0) - wx0) / max(wx1 - wx0, 1e-9)
                    f1 = (min(sx1, wx1) - wx0) / max(wx1 - wx0, 1e-9)
                    c0 = int(math.floor(f0 * len(txt)))
                    c1 = max(c0 + 1, min(len(txt), int(math.ceil(f1 * len(txt)))))
                    by_word.setdefault(w[:5], []).append((c0, c1))

    out = []
    for (wx0, wy0, wx1, wy1, txt), spans in by_word.items():
        spans.sort()
        merged = [list(spans[0])]
        for c0, c1 in spans[1:]:
            if c0 <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], c1)
            else:
                merged.append([c0, c1])
        covered = sum(m1 - m0 for m0, m1 in merged)
        cov = covered / max(len(txt), 1)
        if cov < PARTIAL_COV:
            continue                       # grazing overshoot into a neighbor word, not a deletion
        full = covered >= FULL_COV * max(len(txt), 1)
        if full:
            c0, c1 = 0, len(txt)
        else:
            c0, c1 = max(merged, key=lambda m: m[1] - m[0])
            if cov < STRUCK_COV and c1 - c0 < 2:
                continue                   # <2 struck chars on a grazed word (vector-path guard)
        out.append({
            "page": page_index, "text": txt, "chars": txt[c0:c1], "char_span": (c0, c1),
            "partial": not full,
            "bbox_frac": _bbox_frac(page, wx0, wy0, wx1, wy1),
            "coverage": round(cov, 3),
            "tier": "flag", "verdict": "struck", "final": True,
        })
    out.sort(key=lambda h: (round(h["bbox_frac"][1], 3), h["bbox_frac"][0]))
    return out


def page_strikes(page, page_index, method="vector"):
    """Struck-word records for one native page by `method`:
      'vector' — this module's stroke-geometry detector (precise partial-char spans; default)
      'flag'   — MuPDF's own FZ_STEXT_STRIKEOUT span flag (also catches font-attribute
                 strikethroughs the vector path can miss)
      'both'   — union: all vector records, plus flag records for words no vector record
                 covers (maximum recall)
    In validation on 12 public redline PDFs, 98%+ of vector detections are independently
    confirmed by the flag signal; the flag signal typically marks additional words on top —
    'both' captures them.
    """
    if method == "vector":
        return native_page_strikes(page, page_index)
    if method == "flag":
        return native_flag_strikes(page, page_index)
    if method != "both":
        raise ValueError(f"unknown native method {method!r} (use 'vector', 'flag', or 'both')")
    vec = native_page_strikes(page, page_index)
    out = list(vec)
    for f in native_flag_strikes(page, page_index):
        fx = (f["bbox_frac"][0] + f["bbox_frac"][2]) / 2
        fy = (f["bbox_frac"][1] + f["bbox_frac"][3]) / 2
        if not any(v["bbox_frac"][0] - 1e-3 <= fx <= v["bbox_frac"][2] + 1e-3
                   and v["bbox_frac"][1] - 2e-3 <= fy <= v["bbox_frac"][3] + 2e-3
                   for v in vec):
            out.append(f)
    out.sort(key=lambda h: (round(h["bbox_frac"][1], 3), h["bbox_frac"][0]))
    return out


def native_doc_strikes(doc, method="vector"):
    """Struck-word records across every page of an open fitz document, in reading order.
    See :func:`page_strikes` for the `method` options."""
    out = []
    for pno in range(doc.page_count):
        out.extend(page_strikes(doc[pno], pno, method))
    return out


def native_markdown(doc):
    """pymupdf4llm markdown for the whole document — struck spans arrive as ~~text~~. Requires
    the ``[markdown]`` extra; used only for its richer layout (headings, tables, columns). The
    strikeout signal itself is base-PyMuPDF (see :func:`native_flag_strikes`)."""
    try:
        import pymupdf4llm
    except ImportError as e:
        raise ImportError(
            "native_markdown/clean_markdown require the [markdown] extra: "
            'pip install "pdf-strikethrough-detect[markdown]"') from e
    return pymupdf4llm.to_markdown(doc, show_progress=False)


_STRIKE_SPAN = re.compile(r"~~(.*?)~~", re.S)


def strip_struck_markdown(md):
    """Markdown with ~~struck~~ spans removed -> the surviving (non-deleted) text.

    Caveat: '~~' is pymupdf4llm's encoding for struck spans; a document whose TEXT contains a
    literal '~~' is inherently ambiguous in that format and may strip incorrectly. detect_pdf's
    ``clean_text`` is assembled from word records instead and is immune."""
    clean = _STRIKE_SPAN.sub("", md)
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    return re.sub(r"\n{3,}", "\n\n", clean)
