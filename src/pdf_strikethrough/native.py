"""Native-PDF strikethrough extraction — for born-digital PDFs, vector geometry is exact
ground truth (no OCR, no model, no guessing).

In born-digital documents a strikethrough is a DRAWING: a thin horizontal vector line ("l"
item), a thin filled bar ("re"/"qu" item, or a thin fill-only path), or — dashed or curve-drawn — a
run of short segments / a flat cubic bezier ("c" item), all painted over the text. A word is struck
when the merged stroke coverage through its MIDDLE BAND (0.22h..0.78h — excludes underlines and
overlines) reaches half its width; smaller mid-band coverage that still spans >= 2 characters is
a genuine partial strike ('semi-' of 'semi-monthly', '19' of '192012'). The struck characters are
the ones the strike crosses, read from MuPDF's per-character boxes; a word is a FULL strike only
when every character is struck.

All output bbox_frac values are fractions of the ROTATED (as-rendered) page, so they map
directly onto ``page.get_pixmap()`` output; detection itself runs in MuPDF's unrotated text
space, where strikes over upright text stay horizontal regardless of /Rotate.

Scope — HORIZONTAL (left-to-right) text only. The vector path (``horiz_strokes``) matches only
near-horizontal strokes, so it skips words that run vertically in text space (vertical writing, or
content rotated by /Rotate); the flag and annotation paths read those. Vertical writing modes and
non-Latin scripts whose strikes run along a different axis are otherwise out of scope. Full support
is roadmap R-cjk (add a CJK redline test doc + document the validated scripts).
"""
import math
import re

import pymupdf  # (formerly imported as the deprecated `fitz` alias)

# word-level strike thresholds (fractions of word width covered by mid-band strokes)
STRUCK_COV = 0.50        # >= this: struck (full when every character is struck, else partial)
PARTIAL_COV = 0.25       # >= this AND >= 2 chars: partial strike; below = grazing stroke end
CHAR_STRUCK_COV = 0.50   # a character is struck when the strike covers this much of its width
MIN_STROKE_LEN = 6.0     # pt; a solid strike stroke is at least this long
MAX_STROKE_DY = 1.5      # pt; a strike stroke is horizontal...
MAX_STROKE_SLOPE = math.tan(math.radians(2.0))   # ...or leans at most this much (line-tool strikes)
MAX_RECT_H = 3.5         # pt; a strike drawn as a filled rect is thin
MAX_STROKE_FRAC = 0.30   # a line wider than this share of the word's height is a highlighter
MID_BAND = 0.22          # strokes within [y0 + f*h, y1 - f*h] count as through-text
# Dashes / flat-bezier pieces are chained before the length gate (see _chain_short).
DASH_MIN_SEG = 1.0       # pt; below this a segment is graphics noise, never a dash
DASH_MAX_GAP = 4.0       # pt; max x-gap between dashes of one chained strike

FLAG_MIN_WCOV = 0.15     # flag path: a struck span must cover >= this of a word to count

METHODS = ("vector", "flag", "annot", "both")    # the native detectors page_strikes selects

# The flag detector is the only path that extracts with TEXT_COLLECT_VECTORS, and PyMuPDF
# 1.26.3-1.26.5 SEGFAULT on it -- a native access violation inside JM_make_textpage_dict, on
# ordinary real-world PDFs. `pyproject.toml` pins the floor so a fresh resolve cannot land there;
# this guard is for an environment that already has one installed, where the alternative outcome is
# a process crash with no exception to catch. Bisected: 1.26.5 crashes, 1.26.6 does not.
FLAG_MIN_PYMUPDF = (1, 26, 6)


def _pymupdf_version():
    """(major, minor, patch) of the installed PyMuPDF binding, or None if it cannot be parsed.
    Unparseable is treated as unguarded on purpose: a version string this does not recognize must
    not disable a detector for someone whose build is otherwise fine."""
    raw = getattr(pymupdf, "VersionBind", None) or getattr(pymupdf, "__version__", "")
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", str(raw))
    return tuple(int(g) for g in m.groups()) if m else None


_PYMUPDF_VERSION = _pymupdf_version()


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
    return all(c >= 0.95 for c in _rgb(color))


def _rgb(color):
    """Normalize a PDF color to an RGB 3-tuple of rounded floats in [0, 1]. An unset (None) color
    is PDF-default black; gray and CMYK colors are converted."""
    if color is None:
        return (0.0, 0.0, 0.0)
    c = [float(v) for v in color]
    if len(c) == 1:
        c = c * 3
    elif len(c) == 4:
        cy, m, y, k = c
        c = [(1 - cy) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)]
    return tuple(round(v, 3) for v in c[:3])


def _chain_short(segs):
    """Merge collinear short strike segments (dashes, flat-bezier pieces) into runs. Segments join
    when they share a y-band (bucketed at ~1 pt) and sit no more than DASH_MAX_GAP pt apart in x;
    each run is emitted as one ``(x0, x1, y, color, width)`` interval carrying its longest member's
    paint. The caller applies the MIN_STROKE_LEN test to the run, so a dashed strike passes while a
    lone tick does not."""
    buckets = {}
    for seg in segs:
        buckets.setdefault(round(seg[2]), []).append(seg)
    out = []
    for group in buckets.values():
        group.sort()                                   # by x0
        cx0 = cx1 = None
        members = []
        for (x0, x1, y, col, wid) in group:
            if cx0 is None or x0 > cx1 + DASH_MAX_GAP:
                if cx0 is not None:
                    dom = max(members, key=lambda s: s[1] - s[0])
                    out.append((cx0, cx1, dom[2], dom[3], dom[4]))
                cx0, cx1, members = x0, x1, [(x0, x1, y, col, wid)]
            else:
                cx1 = max(cx1, x1)
                members.append((x0, x1, y, col, wid))
        if cx0 is not None:
            dom = max(members, key=lambda s: s[1] - s[0])
            out.append((cx0, cx1, dom[2], dom[3], dom[4]))
    return out


def horiz_strokes(page):
    """All horizontal stroke intervals on the page: ``[(x0, x1, y, color, width), ...]`` in pt
    (unrotated space). ``color`` is the paint that makes the mark (the stroke color for a line,
    the fill color for a filled bar) as an RGB 3-tuple in [0, 1]; ``width`` is the stroke line
    width for a line, or the bar height for a filled rect — the visual thickness of the strike.
    Invisible strokes (transparent, or drawn in the page background color) leave no ink and are
    skipped — geometry alone would otherwise confirm a white / opacity-0 line as a strike. A
    fill-only path paints its interior, not its outline, so each of its subpaths counts as a bar
    when it is thin enough.

    A strike may be dashed (many short line segments) or drawn as a flat cubic bezier; those
    sub-MIN_STROKE_LEN pieces are collected and chained (see :func:`_chain_short`) so they clear the
    length gate a solid stroke clears directly. A line leaning up to MAX_STROKE_SLOPE is emitted as
    level pieces, so each word is tested where the line crosses it."""
    out, shorts = [], []
    for d in page.get_drawings():
        stroke_col, stroke_op = d.get("color"), d.get("stroke_opacity", 1.0)
        stroked = "s" in (d.get("type") or "s")         # 'f' = fill-only, 's' / 'fs' = stroked
        line_w = d.get("width")
        width = round(float(line_w), 2) if line_w is not None else 1.0    # PDF default line width 1
        draws_line = stroked and not _paint_invisible(stroke_col, stroke_op)
        fill_col = d.get("fill")
        if not stroked and not _paint_invisible(fill_col, d.get("fill_opacity", 1.0)):
            # one box per subpath (an item not starting where the last ended starts a new one), so
            # a compound path holding two bars is not judged by a union box spanning the text
            groups, last = [], None
            for it in d["items"]:
                if it[0] not in ("l", "c"):
                    continue
                pts = list(it[1:])
                if not groups or abs(pts[0].x - last.x) > 0.01 or abs(pts[0].y - last.y) > 0.01:
                    groups.append([])
                groups[-1].extend(pts)
                last = pts[-1]
            for g in groups:
                r = pymupdf.Rect(min(p.x for p in g), min(p.y for p in g),
                                 max(p.x for p in g), max(p.y for p in g))
                if r.height <= MAX_RECT_H and r.width >= MIN_STROKE_LEN:
                    out.append((r.x0, r.x1, (r.y0 + r.y1) / 2, _rgb(fill_col),
                                round(float(r.height), 2)))
        for it in d["items"]:
            seg = None
            if it[0] == "l":
                if not draws_line:
                    continue
                p1, p2 = sorted((it[1], it[2]), key=lambda p: p.x)
                dx, dy = p2.x - p1.x, p2.y - p1.y
                if abs(dy) <= MAX_STROKE_DY:
                    seg = (p1.x, p2.x, (p1.y + p2.y) / 2, _rgb(stroke_col), width)
                elif abs(dy) <= MAX_STROKE_SLOPE * dx:
                    n = math.ceil(abs(dy))              # pieces rising <= 1 pt each
                    for i in range(n):
                        out.append((p1.x + dx * i / n, p1.x + dx * (i + 1) / n,
                                    p1.y + dy * (i + 0.5) / n, _rgb(stroke_col), width))
            elif it[0] == "c":
                # a strike drawn as a (near-)flat cubic bezier reads as horizontal when all four
                # control points share the stroke's y-band; genuinely curved beziers are skipped
                if not draws_line:
                    continue
                pts = it[1:5]
                ys = [p.y for p in pts]
                xs = [p.x for p in pts]
                if max(ys) - min(ys) <= MAX_STROKE_DY:
                    seg = (min(xs), max(xs), sum(ys) / len(ys), _rgb(stroke_col), width)
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
                if r.height <= MAX_RECT_H:
                    seg = (r.x0, r.x1, (r.y0 + r.y1) / 2, _rgb(col), round(float(r.height), 2))
            if seg is None:
                continue
            if seg[1] - seg[0] >= MIN_STROKE_LEN:
                out.append(seg)                        # solid stroke: emitted as-is (no chaining)
            elif seg[1] - seg[0] >= DASH_MIN_SEG:
                shorts.append(seg)                     # dash / bezier piece: chain it first
    for chain in _chain_short(shorts):
        if chain[1] - chain[0] >= MIN_STROKE_LEN:
            out.append(chain)
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


# PyMuPDF's word-split characters for get_text("words") (JM_is_word_delimiter): controls, space,
# no-break space and the bidi embedding marks. A thin or narrow no-break space does not split.
_WORD_DELIMITERS = frozenset(map(chr, [*range(33), 0xA0, *range(0x202A, 0x202F)]))


class _PageChars:
    """Per-character boxes for a page's ``get_text("words")`` entries, so a strike maps onto the
    characters it crosses. Built lazily from one rawdict pass and shared by the detectors in
    ``page_strikes(method='both')``."""

    def __init__(self, page):
        self.page = page
        self._by_text = None

    def _add(self, chars, vertical):
        if not chars:
            return
        boxes = [tuple(c["bbox"]) for c in chars]
        cx = (min(b[0] for b in boxes) + max(b[2] for b in boxes)) / 2
        cy = (min(b[1] for b in boxes) + max(b[3] for b in boxes)) / 2
        text = "".join(c["c"] for c in chars)
        self._by_text.setdefault(text, []).append((cx, cy, vertical, boxes))

    def lookup(self, word):
        """``(vertical, [char bbox, ...])`` for one ``get_text("words")`` entry, or None when no
        extracted line reproduces it (the caller then splits the word box evenly)."""
        if self._by_text is None:
            self._by_text = {}
            raw = self.page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_WORDS)
            for block in raw.get("blocks", []):
                for line in block.get("lines", []):
                    dx, dy = line.get("dir", (1.0, 0.0))
                    vertical = abs(dy) > abs(dx)
                    cur = []
                    for span in line.get("spans", []):
                        for ch in span.get("chars", []):
                            if ch["c"] in _WORD_DELIMITERS:
                                self._add(cur, vertical)
                                cur = []
                            else:
                                cur.append(ch)
                    self._add(cur, vertical)
        x0, y0, x1, y1, txt = word[:5]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        best = None
        for ecx, ecy, vertical, boxes in self._by_text.get(txt, ()):
            dist = abs(ecx - cx) + abs(ecy - cy)
            if dist <= 2.0 and len(boxes) == len(txt) and (best is None or dist < best[0]):
                best = (dist, vertical, boxes)
        return None if best is None else (best[1], best[2])


def _even_char_boxes(word):
    """Fallback character boxes: the word box split into equal-width slices, one per character."""
    x0, y0, x1, y1, txt = word[:5]
    step = (x1 - x0) / max(len(txt), 1)
    return [(x0 + i * step, y0, x0 + (i + 1) * step, y1) for i in range(len(txt))]


def _struck_span(txt, struck, cov, exact, enters=(False, False)):
    """``(c0, c1, full)`` from a word's per-character struck mask, or None when the marks are not a
    deletion. Full only when every character is struck; otherwise the longest struck run is the
    partial span. The grazing guards (`cov` below PARTIAL_COV, or under 2 characters below
    STRUCK_COV) drop a stroke end clipping a neighbor; they apply when the boxes are not `exact`
    or the run touches an edge the mark `enters` from outside ``(from_left, from_right)``. A mark
    inside a word with exact boxes is a strike however small ('(ed)' striking just the 'd')."""
    runs, start = [], None
    for i, s in enumerate([*struck, False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append((start, i))
            start = None
    if not runs:
        return None
    if runs == [(0, len(txt))]:
        return 0, len(txt), True
    c0, c1 = max(runs, key=lambda r: r[1] - r[0])
    overshoot = (c0 == 0 and enters[0]) or (c1 == len(txt) and enters[1])
    if (not exact or overshoot) and (cov < PARTIAL_COV or (cov < STRUCK_COV and c1 - c0 < 2)):
        return None
    return c0, c1, False


def native_page_strikes(page, page_index, words=None, _chars=None):
    """Struck-word records for one native page, top to bottom then left to right (``markdown`` and
    ``passages`` read multi-column pages column by column instead).

    Each record: {page, text, chars, char_span, partial, bbox_frac, coverage, stroke_color,
    stroke_width, tier='vector', verdict='struck', final=True}. Vector geometry is exact, so there
    is no unsure tier. ``stroke_color`` (RGB 3-tuple in [0, 1]) and ``stroke_width`` (pt) come from
    the dominant contributing stroke — pen-color conventions (red = opposing counsel) are evidence
    in legal review.

    ``words`` optionally supplies this page's ``get_text("words")`` output so a caller running
    several detectors on the same page extracts it once instead of per detector (default None =
    extract here).
    """
    if words is None:
        words = page.get_text("words")
    # below the 3 pt width floor, gridlines crossing glyphs outnumber real strikes
    words = [w for w in words
             if w[4].strip() and (w[3] - w[1]) >= 4 and (w[2] - w[0]) >= 3]
    if not words:
        return []
    strokes = horiz_strokes(page)
    if not strokes:
        return []
    chars = _chars if _chars is not None else _PageChars(page)

    out = []
    for word in words:
        wx0, wy0, wx1, wy1, txt = word[:5]
        wh = wy1 - wy0
        max_wid = max(MAX_RECT_H, MAX_STROKE_FRAC * wh)
        mid = [(a, b, col, wid) for (a, b, sy, col, wid) in strokes
               if wy0 + MID_BAND * wh <= sy <= wy1 - MID_BAND * wh and min(b, wx1) > max(a, wx0)
               and wid <= max_wid]
        if not mid:
            continue
        merged, covered = _merged_intervals(wx0, wx1, [(a, b) for (a, b, _c, _w) in mid])
        cov = covered / max(wx1 - wx0, 1e-6)
        found = chars.lookup(word)
        if found is not None and found[0]:
            continue        # vertical text: a horizontal line crosses it, it does not strike it
        boxes = found[1] if found is not None else _even_char_boxes(word)
        struck = []
        for bx0, _, bx1, _ in boxes:
            if bx1 - bx0 <= 1e-6:      # zero-width (a combining mark): its position decides
                struck.append(any(a <= bx0 <= b for a, b in merged))
            else:
                covered = sum(max(0.0, min(b, bx1) - max(a, bx0)) for a, b in merged)
                struck.append(covered >= CHAR_STRUCK_COV * (bx1 - bx0))
        if found is not None:
            # a sub/superscript next to a struck character is struck with it: redline tools strike
            # it at its own height with a stub too short to pass as a stroke
            tall = max(b[3] - b[1] for b in boxes)
            for order in (range(len(boxes)), range(len(boxes) - 1, -1, -1)):
                prev = None
                for i in order:
                    if (not struck[i] and boxes[i][3] - boxes[i][1] <= 0.8 * tall
                            and prev is not None and struck[prev]):
                        struck[i] = True
                    prev = i
        enters = (any(a < wx0 - 1.0 for a, *_ in mid), any(b > wx1 + 1.0 for _a, b, *_ in mid))
        span = _struck_span(txt, struck, cov, exact=found is not None, enters=enters)
        if span is None:
            continue
        c0, c1, full = span
        # forensics: report the paint of the stroke that covers the most of this word's width
        dom = max(mid, key=lambda s: min(s[1], wx1) - max(s[0], wx0))
        out.append({
            "page": page_index, "text": txt, "chars": txt[c0:c1], "char_span": (c0, c1),
            "partial": not full,
            "bbox_frac": _bbox_frac(page, wx0, wy0, wx1, wy1),
            "coverage": round(cov, 3), "stroke_color": dom[2], "stroke_width": dom[3],
            "tier": "vector", "verdict": "struck", "final": True,
        })
    out.sort(key=lambda h: (round(h["bbox_frac"][1], 3), h["bbox_frac"][0]))
    return out


def _snap_rects_to_words(page, page_index, page_words, rects, tier, extra=None, chars=None):
    """Snap strike rects (``(sx0, sy0, sx1, sy1)``, unrotated text space) onto the page words: a
    character is struck when its center lies inside a rect, in any of the four text directions. A
    word without character boxes falls back to an even split along x. ``extra`` is merged into every
    record (annotation forensics)."""
    if chars is None:
        chars = _PageChars(page)
    masks = {}                                 # (word tuple) -> per-character struck flags
    for (sx0, sy0, sx1, sy1) in rects:
        for w in page_words:
            wx0, wy0, wx1, wy1, txt = w[:5]
            if min(sx1, wx1) <= max(sx0, wx0) or min(sy1, wy1) + 1 <= max(sy0, wy0):
                continue                       # the rect does not touch this word
            key = w[:5]
            if key not in masks:
                found = chars.lookup(w)
                masks[key] = (found[1] if found is not None else None, [False] * len(txt))
            boxes, mask = masks[key]
            if boxes is None:
                # no character boxes: a same-line test plus an even split along x
                if not (sy0 - 1 <= (wy0 + wy1) / 2 <= sy1 + 1):
                    continue
                if (min(sx1, wx1) - max(sx0, wx0)) / max(wx1 - wx0, 1e-9) < FLAG_MIN_WCOV:
                    continue
                for i, b in enumerate(_even_char_boxes(w)):
                    mask[i] = mask[i] or sx0 <= (b[0] + b[2]) / 2 <= sx1
                continue
            for i, b in enumerate(boxes):
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                inside = sx0 - 0.5 <= cx <= sx1 + 0.5 and sy0 - 0.5 <= cy <= sy1 + 0.5
                mask[i] = mask[i] or inside

    out = []
    for (wx0, wy0, wx1, wy1, txt), (boxes, mask) in masks.items():
        if not any(mask):
            continue
        # a mark enters an edge when a rect on the same line runs up to it from outside (MuPDF
        # flags the letter after a struck word as a span of its own)
        gap = 0.35 * (wy1 - wy0)                        # about a word space
        enters = [False, False]
        for (sx0, sy0, sx1, sy1) in rects:
            if min(sy1, wy1) <= max(sy0, wy0):
                continue                                # another line
            enters[0] = enters[0] or (sx0 < wx0 - 1.0 and sx1 >= wx0 - gap)
            enters[1] = enters[1] or (sx1 > wx1 + 1.0 and sx0 <= wx1 + gap)
        cov = sum(mask) / max(len(txt), 1)
        span = _struck_span(txt, mask, cov, exact=boxes is not None, enters=enters)
        if span is None:
            continue
        c0, c1, full = span
        rec = {
            "page": page_index, "text": txt, "chars": txt[c0:c1], "char_span": (c0, c1),
            "partial": not full,
            "bbox_frac": _bbox_frac(page, wx0, wy0, wx1, wy1),
            "coverage": round(cov, 3),
            "tier": tier, "verdict": "struck", "final": True,
        }
        if extra:
            rec.update(extra)
        out.append(rec)
    out.sort(key=lambda h: (round(h["bbox_frac"][1], 3), h["bbox_frac"][0]))
    return out


def native_flag_strikes(page, page_index, words=None, _chars=None):
    """Struck words from MuPDF's own strikeout detection — the FZ_STEXT_STRIKEOUT char flag,
    enabled by extracting with COLLECT_STYLES|COLLECT_VECTORS (base PyMuPDF >= 1.26.6, no
    pymupdf4llm needed; this is the same signal pymupdf4llm renders as ``~~``).

    Raises ``RuntimeError`` on PyMuPDF 1.26.3–1.26.5, which segfault on that extraction rather
    than raising (see :data:`FLAG_MIN_PYMUPDF`). The ``'vector'`` and ``'annot'`` detectors never
    pass the flag and are unaffected.

    Struck spans are snapped onto the page's ``get_text("words")`` boxes, so records carry the
    SAME exact word boxes and text as the vector detector — a span covering only part of a word
    ('Policy' of 'PolicyTo') becomes a partial strike over exactly the flagged characters. All
    four axis directions are read. The character set is MuPDF's own and can include the character
    a strike ends against ('DecemberM'); 'vector' and 'both' stop at 'December'. Records: {page,
    text, chars, char_span, partial, bbox_frac, coverage, tier='flag', verdict='struck',
    final=True}.

    ``words`` optionally supplies this page's ``get_text("words")`` output (see
    :func:`native_page_strikes`); the ``get_text("dict", ...)`` styled-span pass this detector
    also needs is separate and always runs.
    """
    if _PYMUPDF_VERSION is not None and _PYMUPDF_VERSION < FLAG_MIN_PYMUPDF:
        have, want = (".".join(map(str, v)) for v in (_PYMUPDF_VERSION, FLAG_MIN_PYMUPDF))
        raise RuntimeError(
            f"the flag detector needs PyMuPDF >= {want}; {have} is installed and crashes the "
            "interpreter on this extraction (access violation in JM_make_textpage_dict, no "
            f"traceback). Run `pip install -U 'pymupdf>={want}'`, or use method='vector' or "
            "'annot', which do not collect vectors.")

    flags = pymupdf.TEXTFLAGS_DICT | pymupdf.TEXT_COLLECT_STYLES | pymupdf.TEXT_COLLECT_VECTORS
    strike_bit = pymupdf.mupdf.FZ_STEXT_STRIKEOUT
    if words is None:
        words = page.get_text("words")
    page_words = [w for w in words if w[4].strip()]
    if not page_words:
        return []

    rects = []
    for block in page.get_text("dict", flags=flags).get("blocks", []):
        for line in block.get("lines", []):
            dx, dy = line.get("dir", (1, 0))
            if (abs(dx), abs(dy)) not in ((1, 0), (0, 1)):     # strikeout is axis-parallel only
                continue
            for span in line.get("spans", []):
                if not (span.get("char_flags", 0) & strike_bit):
                    continue
                if not span.get("text", "").strip():
                    continue
                rects.append(tuple(span["bbox"]))
    if not rects:
        return []
    return _snap_rects_to_words(page, page_index, page_words, rects, "flag", chars=_chars)


def native_annot_strikes(page, page_index, words=None, _chars=None):
    """Struck words from explicit ``/StrikeOut`` markup annotations — the redlines Acrobat, Preview,
    and other editors write as annotation objects (QuadPoints over the struck text), distinct from
    the vector drawings the geometry path reads and from the font-attribute flag path.

    Each annotation's QuadPoints are snapped onto the page words (same word boxes/text as the other
    detectors; partial spans supported). Records carry tier='annot' plus forensic keys — the value
    no extractor exposes — when the annotation supplies them:
      ``annot_author`` (/T), ``annot_created`` (/CreationDate), ``annot_modified`` (/M),
      ``annot_color`` (RGB 3-tuple), ``annot_id`` (/NM). "Who struck this, and when."

    Hidden annotations (the /Hidden or /NoView flag) paint no ink, so they are skipped. ``words``
    optionally supplies this page's ``get_text("words")`` output (see :func:`native_page_strikes`).
    """
    if words is None:
        words = page.get_text("words")
    page_words = [w for w in words if w[4].strip()]
    if not page_words:
        return []
    chars = _chars if _chars is not None else _PageChars(page)

    records = {}                               # (bbox_frac, text) -> best record
    for annot in page.annots():
        try:
            if annot.type[1] != "StrikeOut":
                continue
        except (AttributeError, IndexError, TypeError):
            continue
        if annot.flags & (pymupdf.PDF_ANNOT_IS_HIDDEN | pymupdf.PDF_ANNOT_IS_NO_VIEW):
            continue
        rects = _annot_quad_rects(annot)
        if not rects:
            continue
        info = annot.info or {}
        stroke = (annot.colors or {}).get("stroke")
        extra = {"annot_author": info.get("title") or None,
                 "annot_created": info.get("creationDate") or None,
                 "annot_modified": info.get("modDate") or None,
                 "annot_color": _rgb(stroke) if stroke else None,
                 "annot_id": info.get("id") or None}
        extra = {k: v for k, v in extra.items() if v is not None}
        for rec in _snap_rects_to_words(page, page_index, page_words, rects, "annot", extra,
                                        chars=chars):
            key = (rec["bbox_frac"], rec["text"])
            if key not in records or rec["coverage"] > records[key]["coverage"]:
                records[key] = rec
    out = list(records.values())
    out.sort(key=lambda h: (round(h["bbox_frac"][1], 3), h["bbox_frac"][0]))
    return out


def _annot_quad_rects(annot):
    """A markup annotation's QuadPoints -> list of ``(x0, y0, x1, y1)`` rects (unrotated text
    space), one per struck quad (a multi-line strikeout has several). Vertices arrive as points in
    groups of four; fall back to the annotation /Rect when QuadPoints are absent."""
    verts = annot.vertices
    if verts and len(verts) >= 4 and len(verts) % 4 == 0:
        rects = []
        for i in range(0, len(verts), 4):
            quad = verts[i:i + 4]
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            rects.append((min(xs), min(ys), max(xs), max(ys)))
        return rects
    r = annot.rect
    return [(r.x0, r.y0, r.x1, r.y1)]


_ANNOT_KEYS = ("annot_author", "annot_created", "annot_modified", "annot_color", "annot_id")


def _covering_record(records, rec):
    """The record in `records` whose box contains `rec`'s center, or None (union dedup key)."""
    cx = (rec["bbox_frac"][0] + rec["bbox_frac"][2]) / 2
    cy = (rec["bbox_frac"][1] + rec["bbox_frac"][3]) / 2
    for r in records:
        if (r["bbox_frac"][0] - 1e-3 <= cx <= r["bbox_frac"][2] + 1e-3
                and r["bbox_frac"][1] - 2e-3 <= cy <= r["bbox_frac"][3] + 2e-3):
            return r
    return None


def page_strikes(page, page_index, method="vector", words=None):
    """Struck-word records for one native page by `method`:
      'vector' — this module's stroke-geometry detector (precise partial-char spans; default)
      'flag'   — MuPDF's own FZ_STEXT_STRIKEOUT span flag (also catches font-attribute
                 strikethroughs the vector path can miss)
      'annot'  — explicit /StrikeOut markup annotations (Acrobat/Preview redlines), with
                 author/date/color forensics (see :func:`native_annot_strikes`)
      'both'   — union of all three: vector records, plus flag/annot records for words no
                 earlier record covers (maximum recall)
    In validation on 10 public redline PDFs (55.2k struck words), 99.8% of vector detections
    are independently confirmed by the flag signal (99.6-100% per document); the flag signal
    typically marks additional words on top — 'both' captures them. Reproduce with
    ``benchmarks/confirmation_rate.py``.

    ``words`` optionally supplies this page's ``get_text("words")`` output; passing it means
    'both' extracts the word list once instead of once per detector (default None = extract here).
    """
    if method == "vector":
        return native_page_strikes(page, page_index, words)
    if method == "flag":
        return native_flag_strikes(page, page_index, words)
    if method == "annot":
        return native_annot_strikes(page, page_index, words)
    if method != "both":
        raise ValueError(f"unknown native method {method!r} (use one of {', '.join(METHODS)})")
    if words is None:
        words = page.get_text("words")
    chars = _PageChars(page)                   # one character extraction for all three detectors
    out = native_page_strikes(page, page_index, words, _chars=chars)
    for r in native_flag_strikes(page, page_index, words, _chars=chars):
        if _covering_record(out, r) is None:
            out.append(r)
    # annotations: add the record where nothing covers it, else GRAFT its forensics onto the
    # covering record — a PDF annotation's appearance stream is also caught by the vector/flag
    # paths, so a union that merely dropped it would lose the author/date/color evidence.
    for a in native_annot_strikes(page, page_index, words, _chars=chars):
        cov = _covering_record(out, a)
        if cov is None:
            out.append(a)
        else:
            for k in _ANNOT_KEYS:
                if k in a:
                    cov.setdefault(k, a[k])
    out.sort(key=lambda h: (round(h["bbox_frac"][1], 3), h["bbox_frac"][0]))
    return out


def native_doc_strikes(doc, method="vector"):
    """Struck-word records across every page of an open fitz document, page by page, each page
    top to bottom then left to right. See :func:`page_strikes` for the `method` options."""
    out = []
    for pno in range(doc.page_count):
        page = doc[pno]
        out.extend(page_strikes(page, pno, method, words=page.get_text("words")))
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
