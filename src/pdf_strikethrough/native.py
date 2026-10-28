"""Native-PDF strikethrough extraction — for born-digital PDFs, vector geometry is exact
ground truth (no OCR, no model, no guessing).

In born-digital documents a strikethrough is a DRAWING: a thin horizontal vector line ("l"
item), a thin filled bar ("re"/"qu" item, or a thin fill-only path), or — dashed or curve-drawn — a
run of short segments / a flat cubic bezier ("c" item), all painted over the text. A word is struck
where a stroke runs through its STRIKE BAND, fully or in part ('semi-' of 'semi-monthly', '19' of
'192012'); where its character boxes are estimated, or the stroke enters it from outside, the
grazing guards of :func:`_struck_span` drop a stroke end that only clips it. The band is measured
from the glyph baseline (the characters' origins) in ems of the height the glyphs are drawn at:
STRIKE_BAND_LO..STRIKE_BAND_HI above the baseline (to
STRIKE_BAND_HI_LEVEL for a level stroke), so an underline drawn at or under the baseline and an
overline or rule above the letters are not strikes, in every font. A word whose character origins
are missing, on more than one baseline, or at odds with its box uses the box-share band instead
(see :func:`_through_text`). The struck characters are the ones the strike crosses through their
own bands, read from MuPDF's per-character boxes and origins, so a fraction bar does not strike the
small numerator or denominator a word joins to a full-size character, nor a radical's bar its
radicand (see :func:`_fits`); a word is a FULL strike only when every character is struck. A line
behind the box a text is set on (a plot's gridline under its legend) is not a strike either (see
:func:`_through_strokes`).

Text can also be struck in the text layer itself, as online strikethrough generators write it: each
character followed by U+0336 COMBINING LONG STROKE OVERLAY (``o̶f̶``). The vector detector reads
that too: every character an overlay follows is struck. The overlay characters are strike marks,
not text, so its record drops those on the characters it reports (``text`` is ``of``, its
``char_span`` indexes that); 'flag' and 'annot' don't read them and leave the text as it is. Only
U+0336 is read: the other overlays (U+0334, the short U+0335 and U+0337, and U+0338, the slash
of a decomposed ≠) are left as they are.

All output bbox_frac values are fractions of the ROTATED (as-rendered) page, so they map
directly onto ``page.get_pixmap()`` output; detection itself runs in MuPDF's unrotated text
space, where strikes over upright text stay horizontal regardless of /Rotate.

Scope — HORIZONTAL (left-to-right) text only. The vector path (``horiz_strokes``) matches only
near-horizontal strokes, so it skips words that run vertically in text space (vertical writing, or
content rotated by /Rotate); the flag and annotation paths read those. Vertical writing modes and
non-Latin scripts whose strikes run along a different axis are otherwise out of scope. Horizontal
CJK is regression-tested; vertical writing modes remain on the roadmap.
"""
import bisect
import contextlib
import functools
import math
import re
import sys
import threading
import unicodedata
from collections import namedtuple

import pymupdf

# grazing guards (see _struck_span): where a word's character boxes are estimated, or the stroke
# enters it from outside, a struck run is kept only at these shares of the word (its `coverage`)
STRUCK_COV = 0.50        # >= this: kept, however few characters the run holds
PARTIAL_COV = 0.25       # >= this AND >= 2 chars: kept; below = a grazing stroke end
CHAR_STRUCK_COV = 0.50   # a character is struck when the strike covers this much of its width
MIN_STROKE_LEN = 3.0     # pt; a strike stroke (or a chained run of dashes) is at least this long
MAX_STROKE_SLOPE = math.tan(math.radians(12.0))  # a strike line leans at most this much...
MAX_STROKE_DY = 1.5      # pt; ...and one rising more is tested in level pieces (see horiz_strokes)
LEVEL_SLOPE = math.tan(math.radians(2.0))  # a line leaning more strikes only level lines of text
MAX_RECT_H = 3.5         # pt; a strike drawn as a filled rect is thin
MAX_STROKE_FRAC = 0.30   # a line wider than this share of the word's height is a highlighter
# A stroke runs through the text when its center is more than STRIKE_BAND_LO and at most
# STRIKE_BAND_HI ems (see _char_em) above the glyph baseline: an underline sits at or under the
# baseline, a strike crosses the x-height band (about 0.42-0.54 em tall in Latin fonts), and an
# overline or a ruled line clears the capitals (0.56-0.73 em). A level stroke stops at
# STRIKE_BAND_HI_LEVEL instead: a leaning one (over LEVEL_SLOPE) climbs across a long word, so its
# ends need the room, while a level one that high runs along the tops of the letters. LO and HI
# are each the middle of the range that scored best on synthetic redline pages (0-0.055 em,
# 0.45-0.85 em), with strokes allowed to lean 2, 10 and 12 degrees. Any HI_LEVEL from 0.30 to
# 0.85 em scored the same there; from 0.55 up a strike leaning 1.5 degrees across a sentence
# strikes every word, and up to 0.61 documents without strikes get the fewest false ones (a rule
# through the ascenders sits at 0.62 em). HI_LEVEL is the middle of 0.55-0.61.
STRIKE_BAND_LO = 0.03    # em; at or under this, the stroke is an underline
STRIKE_BAND_HI = 0.65    # em; over this, a leaning stroke is an overline or a rule
STRIKE_BAND_HI_LEVEL = 0.58  # em; over this, a level stroke is an overline or a rule
# Fallback for a word (or, for a short stroke, a character) without trustworthy character origins:
# strokes within [y0 + f*h, y1 - f*h] of its box count as through-text.
MID_BAND = 0.22
# Character origins are trusted only when the word box is BOX_EM_MIN..BOX_EM_MAX ems tall (the
# base-14 fonts are 1.0-1.4): outside that, the font's metrics do not describe its glyphs at this
# size (a Type3 font drawn at a scale of its own, broken metrics).
BOX_EM_MIN, BOX_EM_MAX = 0.5, 3.0
BASELINE_SPREAD = 0.1    # em; a word whose main characters' origins spread wider has no one baseline
CHAR_BUCKET = 4.0        # pt; character data is filed by text and this band of height
# Dashes / flat-bezier pieces are chained before the length gate (see _chain_short).
DASH_MIN_SEG = 1.0       # pt; below this a segment is graphics noise, never a dash
DASH_MAX_SEG = 6.0       # pt; a segment shorter than this may be a dash, so it is chained first
DASH_MAX_GAP = 4.0       # pt; max x-gap between dashes of one chained strike

FLAG_MIN_WCOV = 0.15     # a span on a word with no character boxes must cover >= this of it

# A fill of one subpath at least COVER_OPACITY opaque and taller than MAX_RECT_H, painted after a
# stroke and before the text over it, is that text's background (a plot's legend box over its
# gridlines): the stroke runs behind the text, not through it. The fills are filed by COVER_ROW-pt
# bands of height, so a stroke looks up only the fills at its own height.
COVER_OPACITY = 0.5
COVER_ROW = 8.0          # pt
RADICALS = frozenset("\N{SQUARE ROOT}\N{CUBE ROOT}\N{FOURTH ROOT}")

OVERLAY = "\N{COMBINING LONG STROKE OVERLAY}"    # U+0336: strikes the character before it

METHODS = ("vector", "flag", "annot", "both")    # the native detectors page_strikes selects

# The flag detector is the only path that extracts with TEXT_COLLECT_VECTORS, and PyMuPDF
# 1.26.3-1.26.5 SEGFAULT on it -- a native access violation inside JM_make_textpage_dict, on
# ordinary real-world PDFs. `pyproject.toml` pins the floor so a fresh resolve cannot land there;
# this guard is for an environment that already has one installed, where the alternative outcome is
# a process crash with no exception to catch. Bisected: 1.26.5 crashes, 1.26.6 does not.
FLAG_MIN_PYMUPDF = (1, 26, 6)


def _pymupdf_version():
    """(major, minor, patch) of the installed PyMuPDF binding, or None if it cannot be parsed. A
    missing patch reads as 0 (PyMuPDF 2.0 reports "2.0"); anything after the third number is
    ignored. Unparseable is treated as unguarded on purpose: a version string this does not
    recognize must not disable a detector for someone whose build is otherwise fine."""
    raw = getattr(pymupdf, "VersionBind", None) or getattr(pymupdf, "__version__", "")
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", str(raw))
    return tuple(int(g or 0) for g in m.groups()) if m else None


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
    each run of ``(x0, x1, y, color, width, seqno)`` segments is emitted as one such interval
    carrying its longest member's paint and its last-painted member's ``seqno``. The caller applies
    the MIN_STROKE_LEN test to the run, so a dashed strike passes while a lone fleck does not."""
    buckets = {}
    for seg in segs:
        buckets.setdefault(round(seg[2]), []).append(seg)
    out = []

    def emit(cx0, cx1, members):
        dom = max(members, key=lambda s: s[1] - s[0])
        out.append((cx0, cx1, dom[2], dom[3], dom[4], max(s[5] for s in members)))

    for group in buckets.values():
        group.sort()                                   # by x0
        cx0 = cx1 = None
        members = []
        for seg in group:
            if cx0 is None or seg[0] > cx1 + DASH_MAX_GAP:
                if cx0 is not None:
                    emit(cx0, cx1, members)
                cx0, cx1, members = seg[0], seg[1], [seg]
            else:
                cx1 = max(cx1, seg[1])
                members.append(seg)
        if cx0 is not None:
            emit(cx0, cx1, members)
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

    A strike may be dashed (many short line segments) or drawn as a flat cubic bezier; pieces
    shorter than DASH_MAX_SEG are collected and chained (see :func:`_chain_short`) before the
    MIN_STROKE_LEN gate, so the gaps of a dashed strike count as struck. A line or curve counts
    only when it leans at most MAX_STROKE_SLOPE end to end, and a bar narrower than DASH_MAX_SEG
    only when it is at most MAX_STROKE_DY tall, so a tick or a dot over one character is not a
    strike. A steeper short piece still chains into a run (the jitter of a hand-drawn strike), but a
    run with no level piece is dropped. A line rising more than MAX_STROKE_DY is emitted as level
    pieces, so each word is tested where the line crosses it."""
    return [s[:5] for s in _strokes(page)]


def _strokes(page, drawn=None):
    """:func:`horiz_strokes` with three more fields per interval: ``extent``, the length of the
    mark it comes from (a whole line, bar or chained run, not one level piece of a leaning line),
    ``leaning``, true for a line or curve leaning more than LEVEL_SLOPE, and ``seqno``, its
    drawing's ``seqno`` (see :class:`_PaintOrder`). Each drawing is appended to the list `drawn`,
    when one is given, as ``(seqno, type, rect, cover)``, ``cover`` true for a fill of one subpath
    (:func:`_one_subpath`) at least COVER_OPACITY opaque and taller than MAX_RECT_H."""
    out, level, steep = [], [], []
    for d in page.get_drawings():
        seqno = d.get("seqno", 0)
        if drawn is not None:
            fill_op = d.get("fill_opacity")
            drawn.append((seqno, d.get("type"), d["rect"],
                          d.get("fill") is not None and "f" in (d.get("type") or "")
                          and (1.0 if fill_op is None else fill_op) >= COVER_OPACITY
                          and d["rect"].height > MAX_RECT_H and _one_subpath(d["items"])))
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
                if (r.height <= MAX_RECT_H and r.width >= MIN_STROKE_LEN
                        and (r.width >= DASH_MAX_SEG or r.height <= MAX_STROKE_DY)):
                    out.append((r.x0, r.x1, (r.y0 + r.y1) / 2, _rgb(fill_col),
                                round(float(r.height), 2), r.width, False, seqno))
        for it in d["items"]:
            seg, flat, lean = None, True, False
            if it[0] == "l":
                if not draws_line:
                    continue
                p1, p2 = sorted((it[1], it[2]), key=lambda p: p.x)
                dx, dy = p2.x - p1.x, p2.y - p1.y
                flat = abs(dy) <= MAX_STROKE_SLOPE * dx     # else too steep: a tick, a slash
                lean = abs(dy) > LEVEL_SLOPE * dx
                if abs(dy) <= MAX_STROKE_DY:
                    seg = (p1.x, p2.x, (p1.y + p2.y) / 2, _rgb(stroke_col), width, seqno)
                elif flat:
                    n = math.ceil(abs(dy))              # pieces rising <= 1 pt each
                    for i in range(n):
                        out.append((p1.x + dx * i / n, p1.x + dx * (i + 1) / n,
                                    p1.y + dy * (i + 0.5) / n, _rgb(stroke_col), width, dx, lean,
                                    seqno))
            elif it[0] == "c":
                # a strike drawn as a (near-)flat cubic bezier reads as horizontal when all four
                # control points share the stroke's y-band; genuinely curved beziers are skipped
                if not draws_line:
                    continue
                pts = it[1:5]
                ys = [p.y for p in pts]
                xs = [p.x for p in pts]
                if max(ys) - min(ys) <= MAX_STROKE_DY:
                    seg = (min(xs), max(xs), sum(ys) / len(ys), _rgb(stroke_col), width, seqno)
                    dx, dy = abs(pts[3].x - pts[0].x), abs(pts[3].y - pts[0].y)
                    flat, lean = dy <= MAX_STROKE_SLOPE * dx, dy > LEVEL_SLOPE * dx
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
                    seg = (r.x0, r.x1, (r.y0 + r.y1) / 2, _rgb(col), round(float(r.height), 2),
                           seqno)
                    flat = r.height <= MAX_STROKE_DY or r.width >= DASH_MAX_SEG
            if seg is None:
                continue
            if seg[1] - seg[0] >= DASH_MAX_SEG:
                if flat:                               # solid stroke: emitted as-is (no chaining)
                    out.append((*seg[:5], seg[1] - seg[0], lean, seqno))
            elif seg[1] - seg[0] >= DASH_MIN_SEG:
                (level if flat else steep).append(seg)  # dash, curve piece or short stroke: chain
    runs = _chain_short(level)
    for chain in _chain_short(level + steep):
        # a run must hold a level piece: a lone tick never strikes, a jittery hand-drawn one does
        if chain[1] - chain[0] >= MIN_STROKE_LEN and any(
                round(f[2]) == round(chain[2]) and chain[0] <= f[0] and f[1] <= chain[1]
                for f in runs):
            out.append((*chain[:5], chain[1] - chain[0], False, chain[5]))
    return out


def _one_subpath(items):
    """Whether a path's `items` (``get_drawings``) make one subpath: a lone rectangle or quad, or
    lines and curves each starting where the last ended. A fill of more (a frame around a hole,
    two bars) leaves parts of its box unpainted."""
    if any(it[0] in ("re", "qu") for it in items):
        return len(items) == 1
    return bool(items) and all(abs(a[-1].x - b[1].x) <= 0.01 and abs(a[-1].y - b[1].y) <= 0.01
                               for a, b in zip(items, items[1:]))


def _merged_intervals(wx0, wx1, ivals):
    """Clip intervals to [wx0, wx1] and merge overlaps. Returns (merged, covered_length)."""
    ivals = sorted((max(a, wx0), min(b, wx1)) for a, b in ivals)
    merged = []
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


# Character data for one get_text("words") entry: its text direction, character boxes, whether its
# line leans more than LEVEL_SLOPE, whether every character is invisible, the word's baseline
# (:func:`_word_baseline`) and each character's own band (:func:`_char_band`)
_Found = namedtuple("_Found", "vertical boxes tilted hidden baseline bands")


class _PageChars:
    """Per-character boxes, fonts and the baseline of a page's ``get_text("words")`` entries, so a
    strike maps onto the characters it crosses and is measured from the glyph baseline, with each
    word's text direction and visibility. Built lazily from one rawdict pass and shared by the
    detectors in ``page_strikes(method='both')``. Entries are filed by text and a CHAR_BUCKET-pt
    band of height, so a lookup scans one line's copies of a word, not the page's."""

    def __init__(self, page):
        self.page = page
        self._by_text = None
        self._visible = None                   # (box, (char, metrics), line dir) per visible char

    def _add(self, chars, line_dir):
        if not chars:
            return
        boxes = [tuple(c[0]["bbox"]) for c in chars]
        x0s, y0s, x1s, y1s = zip(*boxes)
        cy = (min(y0s) + max(y1s)) / 2
        key = ("".join([c[0]["c"] for c in chars]), math.floor(cy / CHAR_BUCKET))
        entry = ((min(x0s) + max(x1s)) / 2, cy, boxes, chars, line_dir)
        self._by_text.setdefault(key, []).append(entry)

    def _build(self):
        self._by_text, self._visible = {}, []
        for block in _rawdict_corrected(self.page).get("blocks", []):
            for line in block.get("lines", []):
                line_dir = line.get("dir", (1.0, 0.0))
                cur = []
                for span in line.get("spans", []):
                    metrics = (span.get("ascender", 0.0), span.get("descender", 0.0))
                    hidden = span.get("alpha", 255) == 0    # render mode 3, or fully transparent
                    font = span.get("font", "")
                    for ch in span.get("chars", []):
                        if ch["c"] in _WORD_DELIMITERS:
                            self._add(cur, line_dir)
                            cur = []
                        else:
                            cur.append((ch, *metrics, hidden, font))
                            if not hidden:
                                self._visible.append((pymupdf.Rect(ch["bbox"]), (ch, *metrics),
                                                      line_dir))
                self._add(cur, line_dir)

    def _find(self, word):
        """The extracted run that reproduces one ``get_text("words")`` entry, or None."""
        if self._by_text is None:
            self._build()
        x0, y0, x1, y1, txt = word[:5]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        best = None
        k = math.floor(cy / CHAR_BUCKET)
        for entry in (e for kk in (k - 1, k, k + 1) for e in self._by_text.get((txt, kk), ())):
            dist = abs(entry[0] - cx) + abs(entry[1] - cy)
            if dist <= 2.0 and len(entry[2]) == len(txt) and (best is None or dist < best[0]):
                best = (dist, entry)
        return None if best is None else best[1]

    def lookup(self, word):
        """A :data:`_Found` for one ``get_text("words")`` entry, or None when no extracted line
        reproduces it (the caller then splits the word box evenly). ``tilted``: its line of text
        leans more than LEVEL_SLOPE; ``hidden``: every character is invisible (render mode 3, or
        fully transparent); ``baseline``: None for vertical text or a word with no one
        baseline."""
        found = self._find(word)
        if found is None:
            return None
        ecx, _ecy, boxes, chars, (dx, dy) = found
        vertical = abs(dy) > abs(dx)
        hidden = all(c[3] for c in chars)
        if vertical:
            return _Found(True, boxes, False, hidden, None, [None] * len(boxes))
        return _Found(False, boxes, abs(dy) > LEVEL_SLOPE * abs(dx), hidden,
                      _word_baseline(chars, ecx, dx, dy), [_char_band(c[:3], dx, dy) for c in chars])

    def fonts(self, word):
        """The font name of each character of one ``get_text("words")`` entry, or None."""
        found = self._find(word)
        return None if found is None else [c[4] for c in found[3]]

    def visible_under(self, word):
        """``[(box, band), ...]`` for the visible characters the word's box overlaps (touching edges
        do not count), each with its own band (:func:`_char_band`)."""
        if self._by_text is None:
            self._build()
        r = pymupdf.Rect(word[:4])
        return [(v, _char_band(ch, *line_dir)) for v, ch, line_dir in self._visible
                if r.intersects(v)]


class _PaintOrder:
    """Where in a page's paint order its text and its drawings come, read on first use.
    ``get_bboxlog`` lists the paint operations in order and ``get_texttrace`` numbers its text by
    the same count. ``get_drawings`` numbers them too but leaves out a path with no segments (a
    lone moveto, which PyMuPDF before 2.0 skips, and 2.0 too when it is closed), so each drawing
    (``drawn``: ``(seqno, type, rect, ...)`` per drawing, see :func:`_strokes`) is found in the
    log by its kind, its box and its ``seqno``, which counts the operations before it that are not
    skipped paths. A failed pass, or a drawing not found where its ``seqno`` puts it, leaves the
    order unknown."""

    def __init__(self, page, drawn):
        self.page, self.drawn, self._read = page, drawn, None

    def _load(self):
        if self._read is None:
            self._read = False
            try:
                ops = self.page.get_bboxlog()
                chars = [((c[3][0] + c[3][2]) / 2, (c[3][1] + c[3][3]) / 2, span["seqno"])
                         for span in self.page.get_texttrace() for c in span.get("chars", ())]
            except Exception:                  # a failed pass leaves the order unknown
                return None
            at = _align_drawings(ops, self.drawn)
            if at is not None:
                self._read = (ops, chars, at)
        return self._read or None

    def first_text(self, word):
        """The place in the paint order of the first text drawn inside the word's box, or None."""
        read = self._load()
        if read is None:
            return None
        ops, chars, _at = read
        x0, y0, x1, y1 = word[:4]
        first = min((s for cx, cy, s in chars if x0 <= cx <= x1 and y0 <= cy <= y1),
                    default=None)
        if first is None or not (0 <= first < len(ops) and ops[first][0].endswith("text")):
            return None
        return first

    def place(self, seqno):
        """The place in the paint order of the drawing with this ``seqno``, or None."""
        read = self._load()
        return None if read is None else read[2].get(seqno)


def _align_drawings(ops, drawn):
    """``{seqno: index in ops}`` for the drawings `drawn` (``(seqno, type, rect, ...)`` in order)
    in the paint log `ops` (``get_bboxlog``), or None when one is not where its ``seqno`` puts it.
    A path in the log that is not the next drawing is one ``get_drawings`` skips; a fill and
    stroke of one path (type ``fs``) take two operations."""
    at, k, n, i = {}, 0, 0, 0                  # next drawing, operations numbered, log index
    while i < len(ops):
        kind, box = ops[i]
        if not kind.endswith("path"):
            n += 1
        elif k < len(drawn) and drawn[k][0] == n and _logged_as(kind, box, *drawn[k][1:3]):
            at[n] = i
            n += 1
            if drawn[k][1] == "fs":
                if i + 1 >= len(ops) or ops[i + 1][0] != "stroke-path":
                    return None
                i += 1
                n += 1
            k += 1
        i += 1
    return at if k == len(drawn) else None


def _logged_as(kind, box, typ, rect):
    """Whether a log entry (`kind`, `box`) can be the drawing of type `typ` over `rect`: the log's
    box takes in a stroke's width and the start of every subpath, so it holds the drawing's."""
    if kind != ("stroke-path" if typ == "s" else "fill-path"):
        return False
    return (box[0] - 0.5 <= rect[0] and box[1] - 0.5 <= rect[1]
            and rect[2] <= box[2] + 0.5 and rect[3] <= box[3] + 0.5)


# One character pass at a time while PyMuPDF's process-wide quad-corrections switch is flipped.
_QUAD_LOCK = threading.Lock()


def _rawdict_corrected(page):
    """The page's rawdict with PyMuPDF's quad corrections on, whatever the caller's setting:
    importing pymupdf4llm turns them off for the whole process, and every span then reports
    MuPDF's default ascender and descender whatever its font. The caller's setting is restored
    after; calls here run one at a time, though other threads' PyMuPDF calls still see the switch
    while it is flipped (PyMuPDF is not thread-safe anyway)."""
    with _QUAD_LOCK:
        off = pymupdf.TOOLS.unset_quad_corrections()
        if not off:
            return page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_WORDS)
        pymupdf.TOOLS.unset_quad_corrections(False)
        try:
            return page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_WORDS)
        finally:
            pymupdf.TOOLS.unset_quad_corrections(off)


def _attaches(c):
    """A combining mark, or a format character (a joiner, a soft hyphen): part of the character
    before it, so it neither sets a word's em nor its baseline, and is never a base an overlay
    strikes on its own."""
    cat = unicodedata.category(c)
    return cat[0] == "M" or cat == "Cf"


def _char_em(ch, asc, desc, c, s, k, small):
    """The em one character's glyph is drawn at, its vertical scale: its box height over the
    box's height in ems. PyMuPDF makes the box ascender minus descender ems tall, or exactly 1 em
    when that is under 1 (LaTeX's Type1 fonts), when the ascender is about 0, or with small glyph
    heights on (``small``). The span's ``size`` is not this em when text is stretched (``Tz``, a
    text matrix scaled unevenly). ``c`` and ``s`` are the absolute cosine and sine of the line's
    direction, ``k`` is c*c - s*s."""
    b = ch["bbox"]
    # the glyph box's own height, undoing its axis-aligned box's growth on a sloped line
    h = (b[3] - b[1]) if not s else ((b[3] - b[1]) * c - (b[2] - b[0]) * s) / k
    span = asc - desc
    if small or asc < 1e-3 or span + 1.2e-7 < 1:      # PyMuPDF's own test, FLT_EPSILON included
        span = 1.0
    return h / span


def _word_baseline(chars, cx, dx, dy):
    """``(cx, y, slope, up, em)``: a horizontal word's baseline, as its y at the word's center
    ``cx``, its slope dy/dx, ``up`` (1.0 when the letters stand above the baseline in unrotated
    space, -1.0 when they hang below it, so ``(y - stroke y) * up`` is the height above it), and
    the em of its main characters. ``chars`` holds ``(rawdict char, ascender, descender, ...)``.

    Each character's em is :func:`_char_em`; combining marks and format characters are left out.
    The main characters are those within 10% of the word's largest em, so a sub- or superscript
    (always the smaller) neither moves the baseline nor sets the em; the baseline is their median
    origin, and ``up`` is the side of it their boxes lie on. None when no character has a usable
    em, when the main characters sit on baselines more than BASELINE_SPREAD ems apart (MuPDF can
    join an inline fraction's denominator to the symbols beside it into one word), or when their
    boxes are centered on the baseline."""
    c, s = abs(dx), abs(dy)                  # the line's direction is a unit vector
    if c * c - s * s < 0.25:                 # steeper than about 35 degrees: no baseline to use
        return None
    slope, k = dy / dx, c * c - s * s
    small = pymupdf.TOOLS.set_small_glyph_heights()
    ems, chs = [], []
    for ch, asc, desc, *_ in chars:
        em = _char_em(ch, asc, desc, c, s, k, small)
        if em > 0 and not _attaches(ch["c"]):
            ems.append(em)
            chs.append(ch)
    if not ems:
        return None
    em = max(ems)
    main = [ch for ch, e in zip(chs, ems) if e >= 0.9 * em]
    ys = sorted([ch["origin"][1] + (cx - ch["origin"][0]) * slope for ch in main])
    if ys[-1] - ys[0] > BASELINE_SPREAD * em:
        return None
    # each box's middle against its own origin, both taken at the origin's x
    sides = sorted([(b[1] + b[3]) / 2 + (o[0] - (b[0] + b[2]) / 2) * slope - o[1]
                    for b, o in [(ch["bbox"], ch["origin"]) for ch in main]])
    side = sides[len(sides) // 2]
    if abs(side) < 0.05 * em or not math.isfinite(em):
        return None
    return cx, ys[len(ys) // 2], slope, 1.0 if side < 0 else -1.0, em


def _char_band(char, dx, dy):
    """``(x, y, slope, up, em)``: one character's own baseline through its origin ``(x, y)``, read
    as :func:`_word_baseline` reads a word's, from ``char`` = ``(rawdict char, ascender,
    descender)`` on a line running ``(dx, dy)``. None, so its box-share band applies (see
    :func:`_in_char_band`), when its line is steep, it has no usable em, its box is centered on its
    origin, its origin falls outside its box, or its box is not BOX_EM_MIN..BOX_EM_MAX ems tall."""
    ch, asc, desc = char
    c, s = abs(dx), abs(dy)
    if c * c - s * s < 0.25:
        return None
    slope, k = dy / dx, c * c - s * s
    em = _char_em(ch, asc, desc, c, s, k, pymupdf.TOOLS.set_small_glyph_heights())
    if not (em > 0 and math.isfinite(em)):
        return None
    b, (ox, oy) = ch["bbox"], ch["origin"]
    h = b[3] - b[1]
    side = (b[1] + b[3]) / 2 + (ox - (b[0] + b[2]) / 2) * slope - oy
    if (abs(side) < 0.05 * em or not BOX_EM_MIN * em <= h <= BOX_EM_MAX * em
            or not b[1] - 0.05 * h <= oy <= b[3] + 0.05 * h):
        return None
    return ox, oy, slope, 1.0 if side < 0 else -1.0, em


def _in_char_band(band, box, a, b, y, leaning=False):
    """Whether a level stroke piece over ``[a, b]`` at height ``y`` runs through one character's
    own strike band: more than STRIKE_BAND_LO and at most STRIKE_BAND_HI_LEVEL of its own em above
    its own baseline (``band``, see :func:`_char_band`), or STRIKE_BAND_HI for a piece of a
    `leaning` stroke, judged where the piece crosses its box; with no trusted baseline (``band``
    None), inside ``[y0 + MID_BAND*h, y1 - MID_BAND*h]`` of its ``box``. A word's band
    (:func:`_through_text`) is the same test, taken from its main characters."""
    if band is None:
        h = box[3] - box[1]
        return box[1] + MID_BAND * h <= y <= box[3] - MID_BAND * h
    ox, oy, slope, up, em = band
    x = (max(a, box[0]) + min(b, box[2])) / 2
    height = (oy + (x - ox) * slope - y) * up
    hi = STRIKE_BAND_HI if leaning else STRIKE_BAND_HI_LEVEL
    return STRIKE_BAND_LO * em < height <= hi * em


def _through_text(word, base):
    """``f(x0, x1, y, leaning) -> (x0, x1) | None``: the part of a level stroke piece at height
    ``y`` that runs through ``word``'s text, clipped to the word, or None. ``leaning`` says the
    piece comes from a stroke leaning more than LEVEL_SLOPE.

    With a baseline from the word's character origins (``base``, see :func:`_word_baseline`), the
    stroke must lie more than STRIKE_BAND_LO and at most STRIKE_BAND_HI ems above the baseline
    (STRIKE_BAND_HI_LEVEL for a level stroke),
    judged at each x it crosses (on a sloped line the height changes along the word), and inside
    the word box (the font's ascent and descent bound its glyphs). The origins are not used, and
    the box-share band ``[y0 + MID_BAND*h, y1 - MID_BAND*h]`` is, when the word has no character
    data or no one baseline (``base`` None), when the baseline falls outside the word box, or when
    the box is not BOX_EM_MIN..BOX_EM_MAX ems tall (the font's metrics do not describe its glyphs
    at this size)."""
    wx0, wy0, wx1, wy1 = word[:4]
    wh = wy1 - wy0
    if base is not None:
        bx, by, slope, up, em = base
        tol = 0.05 * wh
        if BOX_EM_MIN * em <= wh <= BOX_EM_MAX * em and wy0 - tol <= by <= wy1 + tol:
            lo, k = STRIKE_BAND_LO * em, slope * up

            def through(a, b, y, leaning=False):
                a, b = max(a, wx0), min(b, wx1)
                if not (wy0 <= y <= wy1 and b > a):
                    return None
                hi = (STRIKE_BAND_HI if leaning else STRIKE_BAND_HI_LEVEL) * em
                h0 = (by - y) * up                     # height above the baseline at x = bx
                if not k:
                    return (a, b) if lo < h0 <= hi else None
                # the height h0 + k * (x - bx) is in (lo, hi] over one interval of x
                xa, xb = sorted((bx + (lo - h0) / k, bx + (hi - h0) / k))
                a, b = max(a, xa), min(b, xb)
                return (a, b) if b > a else None
            return through
    lo, hi = wy0 + MID_BAND * wh, wy1 - MID_BAND * wh

    def through_box(a, b, y, _leaning=False):
        a, b = max(a, wx0), min(b, wx1)
        return (a, b) if lo <= y <= hi and b > a else None
    return through_box


def _split_overlay(txt):
    """``(text, struck, keep)`` for text that may carry U+0336 overlays: the text without them, a
    per-character mask of the characters they strike, and the index in `txt` of each character
    kept. An overlay strikes the base character before it with everything :func:`_attaches` to
    that base, before or after the overlay (normalization puts U+0336, combining class 1, ahead of
    most accents: ``e`` U+0336 U+0301). One with no base character before it (at the start of a
    word, or after whitespace, where it struck a space) strikes nothing."""
    if OVERLAY not in txt:
        return txt, [False] * len(txt), range(len(txt))
    keep, struck, base = [], [], None
    for i, c in enumerate(txt):
        if c == OVERLAY:
            if base is not None:
                struck[base:] = [True] * (len(struck) - base)
            continue
        mark = _attaches(c)
        if c.isspace():
            base = None
        elif not mark:
            base = len(keep)
        keep.append(i)
        struck.append(mark and base is not None and struck[base])
    return "".join(txt[i] for i in keep), struck, keep


def _drop_overlaid(text):
    """`text` without the characters U+0336 overlays strike, or the overlays."""
    if OVERLAY not in text:
        return text
    bare, struck, _keep = _split_overlay(text)
    return "".join(c for c, s in zip(bare, struck) if not s)


def _overlay_text(raw, c0, c1):
    """``(text, t0, t1)``: a word's `raw` text with the U+0336 overlays that strike characters
    ``[c0, c1)`` of its overlay-free text dropped, along with those that strike nothing. Overlays on
    characters outside that span stay, so text a record can't report still reads as struck.
    ``(t0, t1)`` is the span in the returned text. Bases are found as :func:`_split_overlay` finds
    them."""
    out, n, base, t0, t1 = [], 0, None, None, None
    for c in raw:
        if c == OVERLAY:
            if base is not None and not c0 <= base < c1:
                out.append(c)
            continue
        if n == c0:
            t0 = len(out)
        if n == c1:
            t1 = len(out)
        if c.isspace():
            base = None
        elif not _attaches(c):
            base = n
        out.append(c)
        n += 1
    return "".join(out), t0, len(out) if t1 is None else t1


def _runs(mask):
    """``[(start, end), ...]``: the runs of True in a per-character mask."""
    runs, start = [], None
    for i, s in enumerate([*mask, False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append((start, i))
            start = None
    return runs


def _even_char_boxes(word):
    """Fallback character boxes: the word box split into equal-width slices, one per character."""
    x0, y0, x1, y1, txt = word[:5]
    step = (x1 - x0) / max(len(txt), 1)
    return [(x0 + i * step, y0, x0 + (i + 1) * step, y1) for i in range(len(txt))]


# Garbled text layers. A font whose glyph-to-Unicode map is wrong puts the wrong characters in the
# text layer. The common case is an equation Word exported in Cambria Math: some glyphs (stretched
# brackets, the minus sign) are mapped to their glyph numbers, which read as Ethiopic, Indic or
# unassigned code points ('⋅ሺ0.955ሻെ' where the page shows '·(0.955) −'). Those scripts are real
# scripts too, so a word is flagged only on evidence that real text in them never shows:
#   - a noncharacter (U+FFFF, say), or a code point left unassigned inside one of the _SCRIPTS
#     blocks (an unassigned one elsewhere may be a character newer than Python's Unicode tables,
#     such as U+20C1 SAUDI RIYAL SIGN, so it is not judged);
#   - a private-use code point in one word with a character of such a script or a mathematical
#     letter (a private-use character beside Latin or CJK text is a symbol, icon or bullet font);
#   - letters, marks or digits of two different scripts from _SCRIPTS in one word (Ethiopic beside
#     Malayalam);
#   - a character of such a script in one word with a mathematical letter (U+1D400-U+1D7FF);
#   - a combining mark of such a script right after a letter, digit or mark of another script from
#     _SCRIPTS, or right after any other character with no letter of its own script following
#     (text extracted in visual order puts a vowel sign before its consonant, as in '(िक' or
#     '2015िक', so that case is spared);
#   - a character of such a script drawn in a math font (Cambria, Cambria Math, or a name ending a
#     word in "Math"), which has no glyph for it.
# Real text in these scripts, digits and Latin abbreviations included ('በ2015', 'UNDP-এর'), shows
# none of them. A broken map that only yields letters of one script, next to digits or Latin, in
# a font that is not a math font, is not caught: it reads like real text. Neither is one that
# yields Latin, Greek or Cyrillic letters, or U+FFFD.
_SCRIPTS = sorted([
    (0x0590, 0x05FF, "Hebrew"), (0xFB1D, 0xFB4F, "Hebrew"), (0x0600, 0x06FF, "Arabic"),
    (0x0750, 0x077F, "Arabic"), (0x0870, 0x08FF, "Arabic"), (0xFB50, 0xFDFF, "Arabic"),
    (0xFE70, 0xFEFF, "Arabic"), (0x0700, 0x074F, "Syriac"), (0x0860, 0x086F, "Syriac"),
    (0x0780, 0x07BF, "Thaana"), (0x07C0, 0x07FF, "NKo"), (0x0800, 0x083F, "Samaritan"),
    (0x0840, 0x085F, "Mandaic"), (0x0900, 0x097F, "Devanagari"), (0xA8E0, 0xA8FF, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"), (0x0A00, 0x0A7F, "Gurmukhi"), (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"), (0x0B80, 0x0BFF, "Tamil"), (0x11FC0, 0x11FFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"), (0x0C80, 0x0CFF, "Kannada"), (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"), (0x0E00, 0x0E7F, "Thai"), (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"), (0x1000, 0x109F, "Myanmar"), (0xA9E0, 0xA9FF, "Myanmar"),
    (0xAA60, 0xAA7F, "Myanmar"), (0x10A0, 0x10FF, "Georgian"), (0x1C90, 0x1CBF, "Georgian"),
    (0x1100, 0x11FF, "Hangul"), (0x1200, 0x139F, "Ethiopic"), (0x2D80, 0x2DDF, "Ethiopic"),
    (0xAB00, 0xAB2F, "Ethiopic"), (0x1E7E0, 0x1E7FF, "Ethiopic"), (0x13A0, 0x13FF, "Cherokee"),
    (0x1400, 0x167F, "Canadian"), (0x18B0, 0x18FF, "Canadian"), (0x1680, 0x169F, "Ogham"),
    (0x16A0, 0x16FF, "Runic"), (0x1700, 0x177F, "Philippine"), (0x1780, 0x17FF, "Khmer"),
    (0x19E0, 0x19FF, "Khmer"), (0x1800, 0x18AF, "Mongolian"), (0x1900, 0x194F, "Limbu"),
    (0x1950, 0x19DF, "Tai"), (0x1A00, 0x1AAF, "Tai"), (0x1B00, 0x1B7F, "Balinese"),
    (0x1B80, 0x1BBF, "Sundanese"), (0x1CC0, 0x1CCF, "Sundanese"), (0x1BC0, 0x1BFF, "Batak"),
    (0x1C00, 0x1C4F, "Lepcha"), (0x1C50, 0x1C7F, "Ol Chiki"),
])
_SCRIPT_STARTS = [r[0] for r in _SCRIPTS]
# characters several of those scripts share, so they belong to none: the Arabic tatweel, harakat,
# digits and superscript alef (Syriac and Thaana write them too), the Devanagari stress signs and
# dandas (Bengali and Gurmukhi sentences end in a danda), and the Myanmar digits (Tai Le's too)
_SCRIPT_SHARED = frozenset([0x0640, *range(0x064B, 0x0656), *range(0x0660, 0x066A), 0x0670,
                            0x0951, 0x0952, 0x0953, 0x0954, 0x0964, 0x0965, *range(0x1040, 0x104A)])
# skipped when looking for the character a combining mark attaches to: the joiners, the combining
# grapheme joiner, the dotted circle that stands in for a base, and the Vedic signs
_MARK_SKIP = frozenset([0x200C, 0x200D, 0x034F, 0x25CC, *_SCRIPT_SHARED, *range(0x1CD0, 0x1D00)])
# assigned in Unicode 14 to 18 inside those blocks: an older Python's tables call them unassigned
_SCRIPT_RECENT = frozenset([
    0x061D, *range(0x0870, 0x0892), *range(0x0897, 0x08A0), 0x08B5,
    *range(0x08C8, 0x08D3), 0x0B53, 0x0B54, 0x0C3C, 0x0C5C, 0x0C5D, 0x0CDC, 0x0CDD, 0x0CF3, 0x0ECE,
    0x170D, 0x1715, 0x171F, 0x180F, 0x1B4C, 0x1B4E, 0x1B4F, 0x1B7D, 0x1B7E, 0x1B7F,
    *range(0xFBC2, 0xFBD3), *range(0xFD40, 0xFD50), 0xFD90, 0xFD91, *range(0xFDC8, 0xFDD0),
    0xFDFE, 0xFDFF])
# a math font has no glyph in those scripts; a subset font's name may carry a 'ABCDEF+' tag
_MATH_FONT = re.compile(r"(?i)^(?:[a-z]{6}\+)?cambria|math(?![a-z])")


def _script(c):
    """The _SCRIPTS name of a character, or None for any other character."""
    o = ord(c)
    if o < 0x0590 or o in _SCRIPT_SHARED:
        return None
    i = bisect.bisect_right(_SCRIPT_STARTS, o) - 1
    return _SCRIPTS[i][2] if i >= 0 and o <= _SCRIPTS[i][1] else None


def _unassigned(c, cat):
    """A noncharacter, or a code point unassigned inside a _SCRIPTS block of the BMP."""
    o = ord(c)
    if 0xFDD0 <= o <= 0xFDEF or o & 0xFFFE == 0xFFFE:
        return True
    return cat == "Cn" and o <= 0xFFFF and o not in _SCRIPT_RECENT and _script(c) is not None


@functools.lru_cache(maxsize=8192)
def _text_verdict(text):
    """1 when `text`, one extracted word, is garbled by its characters alone; 2 when it is not but
    holds a _SCRIPTS character, which a math font would garble; else 0."""
    if all(ord(c) < 0x0370 for c in text):
        return 0                            # Latin, IPA and spacing marks: nothing to judge
    cats = [unicodedata.category(c) for c in text]
    if any(_unassigned(c, k) for c, k in zip(text, cats)):
        return 1
    scripts = [_script(c) for c in text]
    math_letter = any(0x1D400 <= ord(c) <= 0x1D7FF for c in text)
    if "Co" in cats and (math_letter or any(scripts)):
        return 1
    if not any(scripts):
        return 0
    if math_letter or len({s for s, k in zip(scripts, cats) if s and k[0] in "LMN"}) >= 2:
        return 1
    for i, (s, k) in enumerate(zip(scripts, cats)):
        if s is None or k not in ("Mn", "Mc"):
            continue
        j = i - 1
        while j >= 0 and ord(text[j]) in _MARK_SKIP:
            j -= 1
        if j < 0 or scripts[j] == s:
            continue
        n = i + 1
        while n < len(text) and ord(text[n]) in _MARK_SKIP:
            n += 1
        own_letter_next = n < len(text) and scripts[n] == s and cats[n][0] == "L"
        if own_letter_next and scripts[j] is None:
            continue                        # visual order: '(िक', '2015िक', 'UNDPைய'
        if cats[j][0] in "LMN" or not own_letter_next:
            return 1                        # after a digit, or an Ethiopic letter: '20െ', 'ሻെx'
    return 2


def _garbled_text(text, fonts=None):
    """True when `text`, one extracted word, carries a broken character map's output (the signals
    are listed above _SCRIPTS). `fonts` is the font name of each character, or None to judge the
    text alone."""
    verdict = _text_verdict(text)
    if verdict != 2 or fonts is None:
        return verdict == 1
    return any(f and _MATH_FONT.search(f) and _script(c) for c, f in zip(text, fonts))


def _math_font_draws_scripts(page):
    """Whether a span in a math font on the page holds a _SCRIPTS character: a page-fonts check,
    then one span-level pass (much cheaper than the per-character one)."""
    if not any(_MATH_FONT.search(f[3]) for f in page.get_fonts()):
        return False
    for block in page.get_text("dict", flags=pymupdf.TEXTFLAGS_WORDS).get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if _MATH_FONT.search(span.get("font", "")) and any(
                        _script(c) for c in span.get("text", "")):
                    return True
    return False


def _garbled_words(page, words=None, chars=None):
    """The page's ``get_text("words")`` entries whose text a broken character map produced (see
    _garbled_text). Character fonts are read, from the rawdict pass `chars` (a _PageChars the
    detectors share) makes, only when a math font on the page draws a character of _SCRIPTS and a
    word holds one that nothing else flags."""
    if words is None:
        words = page.get_text("words")
    out, math_font = [], None
    for w in words:
        verdict = _text_verdict(w[4])
        if verdict == 1:
            out.append(w)
        elif verdict == 2:
            if math_font is None:
                math_font = _math_font_draws_scripts(page)
            if not math_font:
                continue
            if chars is None:
                chars = _PageChars(page)
            fonts = chars.fonts(w)
            if fonts is not None and _garbled_text(w[4], fonts):
                out.append(w)
    return out


def _mark_garbled(page, out, words, chars=None, garbled=None):
    """Set ``garbled: True`` on the records in `out` whose word a broken character map garbled.
    `garbled` optionally supplies the page's garbled ``get_text("words")`` entries; else they are
    found here, and only when there is a record to mark. Returns `out`."""
    if out:
        if garbled is None:
            garbled = _garbled_words(page, words, chars)
        # compared without U+0336: a vector record's text drops the overlays its span strikes
        keys = {(_bbox_frac(page, *w[:4]), w[4].replace(OVERLAY, "")) for w in garbled}
        for r in out:
            if (r["bbox_frac"], r["text"].replace(OVERLAY, "")) in keys:
                r["garbled"] = True
    return out


def _struck_span(txt, struck, cov, exact, enters=(False, False)):
    """``(c0, c1, full)`` from a word's per-character struck mask, or None when the marks are not a
    deletion. Full only when every character is struck; otherwise the longest struck run is the
    partial span. The grazing guards (`cov` below PARTIAL_COV, or under 2 characters below
    STRUCK_COV) drop a stroke end clipping a neighbor; they apply when the boxes are not `exact`
    or the run touches an edge the mark `enters` from outside ``(from_left, from_right)``. A mark
    inside a word with exact boxes is a strike however small ('(ed)' striking just the 'd')."""
    runs = _runs(struck)
    if not runs:
        return None
    if runs == [(0, len(txt))]:
        return 0, len(txt), True
    c0, c1 = max(runs, key=lambda r: r[1] - r[0])
    overshoot = (c0 == 0 and enters[0]) or (c1 == len(txt) and enters[1])
    if (not exact or overshoot) and (cov < PARTIAL_COV or (cov < STRUCK_COV and c1 - c0 < 2)):
        return None
    return c0, c1, False


# One stroke piece through a word's text: ``(a, b)`` the part of it in the word's band, clipped to
# the word, ``y`` its height, ``color`` and ``width`` its paint, ``(x0, x1)`` the whole stroke's
# extent (whether it enters the word from outside), ``extent`` the length of the mark it comes
# from, ``leaning`` whether that mark leans more than LEVEL_SLOPE and ``seqno`` its drawing's
# ``seqno`` (see :func:`_strokes`)
_Piece = namedtuple("_Piece", "a b y color width x0 x1 extent leaning seqno")


def _fits(m, box, band):
    """Whether stroke piece `m` counts on the character with this `box` and `band`: only through
    the character's own band (see :func:`_in_char_band`), since in a word mixing sizes the word's
    band also takes in rules above or below the small characters (a fraction's bar over its
    denominator, a radical's bar over its radicand). A stroke at least DASH_MAX_SEG long counts on
    a character without a band of its own wherever the word's band takes it."""
    if band is None and m.extent >= DASH_MAX_SEG:
        return True
    return _in_char_band(band, box, m.a, m.b, m.y, m.leaning)


def _covered(m, cover):
    """Whether the fill `cover` (``(seqno, rect)``) lies over stroke piece `m`, painted after it."""
    seqno, r = cover
    return seqno > m.seqno and r.x0 <= m.a and m.b <= r.x1 and r.y0 <= m.y <= r.y1


def _cover_rows(drawn, strokes):
    """The fills that may cover a stroke (``(seqno, rect)`` for each drawing in `drawn` that is a
    ``cover``, see :func:`_strokes`) filed by COVER_ROW-pt band of height, in the bands that hold
    one of the `strokes`."""
    rows = {}
    fills = [(seqno, r) for seqno, _typ, r, cover in drawn if cover]
    if not strokes or not fills:
        return rows
    lo = min(s[2] for s in strokes)
    hi = max(s[2] for s in strokes)
    keys = sorted({math.floor(s[2] / COVER_ROW) for s in strokes})
    for c in fills:
        r = c[1]
        if r.y0 <= hi and r.y1 >= lo:
            for k in keys[bisect.bisect_left(keys, math.floor(max(lo, r.y0) / COVER_ROW)):
                          bisect.bisect_right(keys, math.floor(min(hi, r.y1) / COVER_ROW))]:
                rows.setdefault(k, []).append(c)
    return rows


def _through_strokes(word, near, found, boxes, bands, chars, covers=None, order=None, signs=()):
    """The :data:`_Piece` of each stroke in `near` (``_strokes`` tuples crossing the word box) that
    runs through the word's text: a stroke at least DASH_MAX_SEG long through its band
    (:func:`_through_text`), and every stroke through some character's own band (:func:`_fits`)
    or across half of a radical sign's box in `signs` (see :func:`_stroke_span`); on an invisible
    word laid over visible text, a short stroke only through the band of a visible character
    beneath, not as its underline; and on a tilted line, no leaning stroke (a slanted line along
    slanted text crosses its upright box). `boxes` and `bands` are the word's characters'
    (overlays left out).

    A stroke under a fill from `covers` (:func:`_cover_rows`) painted after the stroke and before
    the word's text (`order`, a :class:`_PaintOrder`) is dropped as well, such as a gridline under
    the box of a plot's legend: it runs behind the text's background. When the paint order is
    unknown, the stroke is kept."""
    through = _through_text(word, found.baseline if found is not None else None)
    wx0, wx1 = word[0], word[2]
    mid = []
    for a, b, sy, col, wid, ext, lean, seqno in near:
        # a short stroke skips the word's band, as one through a lowered subscript runs under the
        # band of the word's main characters; every stroke must also run through the own band of
        # a character it strikes (_fits, below)
        part = through(a, b, sy, lean) if ext >= DASH_MAX_SEG else (max(a, wx0), min(b, wx1))
        if part is not None:
            mid.append(_Piece(*part, sy, col, wid, a, b, ext, lean, seqno))
    beneath = [(m, c) for m in mid for c in (covers or {}).get(math.floor(m.y / COVER_ROW), ())
               if _covered(m, c)]
    first = order.first_text(word) if beneath and order is not None else None
    if first is not None:
        behind = [m for m, c in beneath if (at := order.place(c[0])) is not None and at < first]
        mid = [m for m in mid if m not in behind]
    if found is not None and mid:
        under = chars.visible_under(word) if found.hidden else []
        if under:
            mid = [m for m in mid if m.extent >= DASH_MAX_SEG or any(
                min(m.x1, v.x1) > max(m.x0, v.x0)
                and _in_char_band(band, v, m.x0, m.x1, m.y, m.leaning) for v, band in under)]
        if found.tilted:
            mid = [m for m in mid if not m.leaning]
    return [m for m in mid if any(_fits(m, bx, bd) and min(m.b, bx[2]) > max(m.a, bx[0])
                                  for bx, bd in zip(boxes, bands))
            or any(min(m.b, bx[2]) - max(m.a, bx[0]) >= CHAR_STRUCK_COV * (bx[2] - bx[0])
                   for bx in signs)]


def _radicands(txt, boxes):
    """``{i: j}`` for each radical sign ``txt[i]`` (RADICALS) with a character after it: ``j``, the
    first of nonzero width in `boxes`, as a combining mark after the sign goes with the sign."""
    out = {}
    for i, ch in enumerate(txt):
        if ch in RADICALS:
            j = next((k for k in range(i + 1, len(boxes)) if boxes[k][2] - boxes[k][0] > 1e-6),
                     None)
            if j is not None:
                out[i] = j
    return out


def _stroke_span(word, txt, boxes, bands, exact, mid, roots=None):
    """``(span, cov)`` for the stroke pieces in `mid` (:data:`_Piece`, already through the word's
    text): `span` as :func:`_struck_span` gives it for `txt` (the word's text without overlays)
    from its character `boxes` and `bands` (`exact` when MuPDF's, not an even split), `cov` the
    share of the word's width the strokes cover. A character is struck by the pieces that
    :func:`_fits` it, except a radical sign (RADICALS): struck when the strokes cross it and its
    radicand, the character after it in `roots` (:func:`_radicands`), is struck."""
    wx0, wx1 = word[0], word[2]
    merged, covered = _merged_intervals(wx0, wx1, [(m.a, m.b) for m in mid])
    cov = covered / max(wx1 - wx0, 1e-6)
    struck = []
    for bx, bd in zip(boxes, bands):
        bx0, bx1 = bx[0], bx[2]
        own = merged
        if not all(_fits(m, bx, bd) for m in mid):
            own = _merged_intervals(wx0, wx1, [(m.a, m.b) for m in mid if _fits(m, bx, bd)])[0]
        if bx1 - bx0 <= 1e-6:
            # zero-width: the rest of a ligature ('fi' gives an 'f' spanning the glyph and an
            # 'i' at its right edge) or a combining mark, so it goes with the character before
            struck.append(struck[-1] if struck else any(a <= bx0 <= b for a, b in own))
        else:
            covered = sum(max(0.0, min(b, bx1) - max(a, bx0)) for a, b in own)
            struck.append(covered >= CHAR_STRUCK_COV * (bx1 - bx0))
    # a radical sign is struck when the strokes cross at least half its width and its radicand,
    # the character after it, is struck, whatever its own band says: TeX's math fonts hang the sign
    # from its origin, so the band read from there takes in a fraction bar above the sign and
    # misses a strike through its middle. A sign with nothing after it is not struck, as the
    # extractor often parts TeX's sign from its radicand under a fraction bar
    if roots is None:
        roots = _radicands(txt, boxes)
    for i in range(len(struck) - 1, -1, -1):
        if txt[i] in RADICALS:
            j = roots.get(i)
            bx0, bx1 = boxes[i][0], boxes[i][2]
            crossed = sum(max(0.0, min(b, bx1) - max(a, bx0)) for a, b in merged)
            struck[i] = (j is not None and struck[j]
                         and crossed >= CHAR_STRUCK_COV * (bx1 - bx0))
            for k in range(i + 1, len(struck) if j is None else j):
                struck[k] = struck[i]           # a zero-width mark after the sign goes with it
    if exact:
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
    enters = (any(m.x0 < wx0 - 1.0 for m in mid), any(m.x1 > wx1 + 1.0 for m in mid))
    return _struck_span(txt, struck, cov, exact=exact, enters=enters), cov


def native_page_strikes(page, page_index, words=None, _chars=None, _garbled=None):
    """Struck-word records for one native page, top to bottom then left to right (``markdown`` and
    ``passages`` read multi-column pages column by column instead).

    Each record: {page, text, chars, char_span, partial, bbox_frac, coverage, stroke_color,
    stroke_width, tier='vector', verdict='struck', final=True}. Vector geometry is exact, so there
    is no unsure tier. ``stroke_color`` (RGB 3-tuple in [0, 1]) and ``stroke_width`` (pt) come from
    the dominant contributing stroke — pen-color conventions (red = opposing counsel) are evidence
    in legal review.

    Characters followed by U+0336 overlays are struck too, and ``text`` leaves out the overlays on
    the characters ``char_span`` reports. A record holds one span, so where a word has more than
    one struck run it reports the drawn strike's span merged with any overlay run it touches, else
    the longest overlay run; overlays on the other runs stay in ``text`` and ``markdown`` so that
    text still reads as struck (``detect_pdf``'s ``clean_text`` drops it). A word struck by
    overlays alone has no stroke, so its record carries no ``stroke_color`` or ``stroke_width``.

    ``words`` optionally supplies this page's ``get_text("words")`` output so a caller running
    several detectors on the same page extracts it once instead of per detector (default None =
    extract here).

    A record whose word comes from a broken character map also carries ``garbled: True``: its
    ``text`` and ``chars`` are not what the page shows (see :func:`_garbled_text`). ``_garbled``
    optionally supplies the page's garbled ``get_text("words")`` entries, when the caller has them.
    """
    if words is None:
        words = page.get_text("words")
    items = []                                # (word, text, overlay mask, kept indices, sized)
    for w in words:
        txt, over, keep = _split_overlay(w[4])
        if not txt.strip():
            continue
        # below the 3 pt width floor, gridlines crossing glyphs outnumber real strikes
        sized = (w[3] - w[1]) >= 4 and (w[2] - w[0]) >= 3
        if sized or any(over):
            items.append((w, txt, over, keep, sized))
    drawn = []                                # (seqno, type, rect, cover) per drawing
    strokes = _strokes(page, drawn) if any(it[4] for it in items) else []
    covers, order = _cover_rows(drawn, strokes), _PaintOrder(page, drawn)
    if not strokes and not any(any(it[2]) for it in items):
        return []
    chars = _chars if _chars is not None else _PageChars(page)

    out = []
    for word, txt, over, keep, sized in items:
        wx0, wy0, wx1, wy1 = word[:4]
        wh = wy1 - wy0
        max_wid = max(MAX_RECT_H, MAX_STROKE_FRAC * wh)
        # strokes across the word box: either band lies inside it, and the character data is
        # read only for a word some stroke crosses or overlays strike
        near = [s for s in strokes
                if wy0 <= s[2] <= wy1 and min(s[1], wx1) > max(s[0], wx0)
                and s[4] <= max_wid] if sized else []
        overlaid = any(over)
        if not near and not overlaid:
            continue
        found = chars.lookup(word)
        vertical = found is not None and found.vertical
        if vertical and not overlaid:
            continue        # vertical text: a horizontal line crosses it, it does not strike it
        if found is not None:
            boxes, bands = [found.boxes[i] for i in keep], [found.bands[i] for i in keep]
        else:
            boxes = _even_char_boxes((wx0, wy0, wx1, wy1, txt))
            bands = [None] * len(boxes)
        roots = _radicands(txt, boxes) if near and not vertical else {}
        mid = (_through_strokes(word, near, found, boxes, bands, chars, covers, order,
                                [boxes[i] for i in roots])
               if near and not vertical else [])
        span, cov = (_stroke_span(word, txt, boxes, bands, found is not None, mid, roots)
                     if mid else (None, 0.0))
        stroked = span is not None
        text = txt
        if overlaid:
            # an overlay is written into the text, so no grazing guard applies to it. It joins the
            # stroke span in one mask, so a character both strike counts once. A record holds one
            # span: the drawn strike's merged with any overlay run it touches, so overlays never
            # displace it, else the longest overlaid run; overlays outside it stay in the text
            mask = [o or (stroked and span[0] <= i < span[1]) for i, o in enumerate(over)]
            runs = _runs(mask)
            c0, c1 = (next(r for r in runs if r[0] <= span[0] and span[1] <= r[1]) if stroked
                      else max(runs, key=lambda r: r[1] - r[0]))
            # still a width share: the struck characters' boxes over the word's width
            ivals = [(b[0], b[2]) for b, s in zip(boxes, mask) if s]
            cov = _merged_intervals(wx0, wx1, ivals)[1] / max(wx1 - wx0, 1e-6)
            text, t0, t1 = _overlay_text(word[4], c0, c1)
            span = (t0, t1, (c0, c1) == (0, len(txt)))
        if span is None:
            continue
        c0, c1, full = span
        rec = {
            "page": page_index, "text": text, "chars": text[c0:c1], "char_span": (c0, c1),
            "partial": not full,
            "bbox_frac": _bbox_frac(page, wx0, wy0, wx1, wy1),
            "coverage": round(cov, 3),
        }
        if stroked:
            # forensics: report the paint of the stroke that covers the most of this word's width
            dom = max(mid, key=lambda m: m.b - m.a)
            rec["stroke_color"], rec["stroke_width"] = dom.color, dom.width
        rec.update(tier="vector", verdict="struck", final=True)
        out.append(rec)
    out.sort(key=lambda h: (round(h["bbox_frac"][1], 3), h["bbox_frac"][0]))
    return _mark_garbled(page, out, words, chars, _garbled)


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


def native_flag_strikes(page, page_index, words=None, _chars=None, _garbled=None):
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
    a strike ends against ('DecemberM'), more often under PyMuPDF 2.0 (MuPDF 1.28.5), which also
    flags it when the line ends within half its stroke width of it; 'vector' and 'both' stop at
    'December'. U+0336 overlays are characters, not drawn lines, so this finds nothing on them.
    Records: {page, text, chars, char_span, partial, bbox_frac, coverage, tier='flag',
    verdict='struck', final=True}.

    ``words`` optionally supplies this page's ``get_text("words")`` output (see
    :func:`native_page_strikes`); the ``get_text("dict", ...)`` styled-span pass this detector
    also needs is separate and always runs.

    A record whose word comes from a broken character map also carries ``garbled: True``: its
    ``text`` and ``chars`` are not what the page shows (see :func:`_garbled_text`). ``_garbled``
    optionally supplies the page's garbled ``get_text("words")`` entries, when the caller has them.
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
    out = _snap_rects_to_words(page, page_index, page_words, rects, "flag", chars=_chars)
    return _mark_garbled(page, out, page_words, _chars, _garbled)


def native_annot_strikes(page, page_index, words=None, _chars=None, _garbled=None):
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

    A record whose word comes from a broken character map also carries ``garbled: True``: its
    ``text`` and ``chars`` are not what the page shows (see :func:`_garbled_text`). ``_garbled``
    optionally supplies the page's garbled ``get_text("words")`` entries, when the caller has them.
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
    return _mark_garbled(page, out, page_words, chars, _garbled)


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


def _bare_span(rec):
    """A record's ``char_span`` counted in characters other than U+0336, so records whose text
    keeps different overlays compare."""
    t, (c0, c1) = rec["text"], rec["char_span"]
    return len(t[:c0].replace(OVERLAY, "")), len(t[:c1].replace(OVERLAY, ""))


def _displaces(rec, cov):
    """Whether flag/annot record `rec` replaces `cov`, the record covering its word, in 'both':
    when `cov` is a vector record struck by overlays alone (no stroke) and `rec` strikes characters
    outside its span. One record per word can't hold both spans; `rec`'s text keeps every overlay,
    so the overlaid characters still read as struck."""
    if cov.get("tier") != "vector" or "stroke_color" in cov:
        return False
    (a0, a1), (b0, b1) = _bare_span(rec), _bare_span(cov)
    return not (b0 <= a0 and a1 <= b1)


def page_strikes(page, page_index, method="vector", words=None, _garbled=None, _chars=None):
    """Struck-word records for one native page by `method`:
      'vector' — this module's stroke-geometry detector (precise partial-char spans; default),
                 which also reads U+0336 overlay characters; the others don't
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

    A record whose word comes from a broken character map carries ``garbled: True``: its ``text``
    and ``chars`` are not what the page shows (see :func:`_garbled_text`).

    ``words`` optionally supplies this page's ``get_text("words")`` output; passing it means
    'both' extracts the word list once instead of once per detector (default None = extract here).
    ``_garbled`` and ``_chars`` optionally supply the page's garbled entries of it and its
    _PageChars, when the caller has them.
    """
    if words is None:
        words = page.get_text("words")
    chars = _chars if _chars is not None else _PageChars(page)   # one character extraction
    if method == "vector":
        return native_page_strikes(page, page_index, words, _chars=chars, _garbled=_garbled)
    if method == "flag":
        return native_flag_strikes(page, page_index, words, _chars=chars, _garbled=_garbled)
    if method == "annot":
        return native_annot_strikes(page, page_index, words, _chars=chars, _garbled=_garbled)
    if method != "both":
        raise ValueError(f"unknown native method {method!r} (use one of {', '.join(METHODS)})")
    if _garbled is None:
        _garbled = _garbled_words(page, words, chars)
    out = native_page_strikes(page, page_index, words, _chars=chars, _garbled=_garbled)
    for r in native_flag_strikes(page, page_index, words, _chars=chars, _garbled=_garbled):
        cov = _covering_record(out, r)
        if cov is None:
            out.append(r)
        elif _displaces(r, cov):
            out = [r if x is cov else x for x in out]
    # annotations: add the record where nothing covers it, replace an overlay-only record it
    # displaces (_displaces), else GRAFT its forensics onto the covering record: a PDF
    # annotation's appearance stream is also caught by the vector/flag paths, so a union that
    # merely dropped it would lose the author/date/color evidence.
    for a in native_annot_strikes(page, page_index, words, _chars=chars, _garbled=_garbled):
        cov = _covering_record(out, a)
        if cov is None:
            out.append(a)
        elif _displaces(a, cov):
            out = [a if x is cov else x for x in out]
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


# One pymupdf4llm call at a time: _stdout_to_stderr() switches process-wide streams, and two calls
# that overlapped would each restore what the other switched in, leaving stdout on stderr for good.
# PyMuPDF is not thread-safe either, so this costs no real parallelism.
_MARKDOWN_LOCK = threading.Lock()


@contextlib.contextmanager
def _stdout_to_stderr():
    """Send what pymupdf4llm prints to stderr instead of stdout while it runs. On PyMuPDF 2.0 its
    layout parser ends every call with a "=== Document parser messages ===" banner ("RapidOCR is
    missing" without RapidOCR, "Using RapidOCR" with it), written with print() and
    pymupdf.message(), and older releases print a hint on their first call; a library call must not
    write to its caller's stdout. pymupdf.message() writes to the stream PyMuPDF captured at import,
    so redirecting sys.stdout alone misses it; that stream is switched only while it is still the
    default, so a caller who routed PyMuPDF's messages elsewhere keeps that, and only to a stderr
    with the flush() set_messages() asserts on. Both switches are process-wide while they last:
    callers hold _MARKDOWN_LOCK."""
    prev = getattr(pymupdf, "_g_out_message", None)     # PyMuPDF has no public getter
    swap = (prev is not None and hasattr(sys.stderr, "flush")
            and (prev is sys.stdout or prev is sys.__stdout__))
    if swap:
        pymupdf.set_messages(stream=sys.stderr)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        if swap:
            pymupdf.set_messages(stream=prev)


def native_markdown(doc):
    """pymupdf4llm markdown for the whole document — struck spans arrive as ~~text~~. Requires
    the ``[markdown]`` extra; used only for its richer layout (headings, tables, columns). The
    strikeout signal itself is base-PyMuPDF (see :func:`native_flag_strikes`). Anything
    pymupdf4llm prints goes to stderr, and calls from several threads run one at a time."""
    with _MARKDOWN_LOCK, _stdout_to_stderr():
        try:
            import pymupdf4llm
        except ImportError as e:
            raise ImportError(
                "native_markdown/clean_markdown require the [markdown] extra: "
                'pip install "pdf-strikethrough-detect[markdown]"') from e
        return pymupdf4llm.to_markdown(doc, show_progress=False)


_STRIKE_SPAN = re.compile(r"~~(.*?)~~", re.S)
# A struck span's stand-in while overlays are read, so an overlay right after the span strikes it
# and not the text before it. A noncharacter: it never occurs in interchanged text.
_SPAN_MARK = chr(0xFDD0)
# An overlay right after emphasis markup ('**' or '_') sat on a space before it: it strikes nothing
_OVERLAY_ON_MARKUP = re.compile(f"(?<=[*_]){OVERLAY}+")


def strip_struck_markdown(md):
    """Markdown with ~~struck~~ spans removed -> the surviving (non-deleted) text. Characters
    struck with U+0336 overlays go too, with the overlays.

    Caveat: '~~' is pymupdf4llm's encoding for struck spans; a document whose TEXT contains a
    literal '~~' is inherently ambiguous in that format and may strip incorrectly. detect_pdf's
    ``clean_text`` is assembled from word records instead and is immune."""
    clean = _OVERLAY_ON_MARKUP.sub("", _STRIKE_SPAN.sub(_SPAN_MARK, md))
    clean = _drop_overlaid(clean).replace(_SPAN_MARK, "")
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    return re.sub(r"\n{3,}", "\n\n", clean)
