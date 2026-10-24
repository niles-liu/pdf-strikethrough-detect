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
import contextlib
import math
import re
import sys
import threading
import unicodedata
from collections import namedtuple

import pymupdf  # (formerly imported as the deprecated `fitz` alias)

# word-level strike thresholds (fractions of word width covered by mid-band strokes)
STRUCK_COV = 0.50        # >= this: struck (full when every character is struck, else partial)
PARTIAL_COV = 0.25       # >= this AND >= 2 chars: partial strike; below = grazing stroke end
CHAR_STRUCK_COV = 0.50   # a character is struck when the strike covers this much of its width
MIN_STROKE_LEN = 3.0     # pt; a strike stroke (or a chained run of dashes) is at least this long
MAX_STROKE_SLOPE = math.tan(math.radians(12.0))  # a strike line leans at most this much...
MAX_STROKE_DY = 1.5      # pt; ...and one rising more is tested in level pieces (see horiz_strokes)
LEVEL_SLOPE = math.tan(math.radians(2.0))  # a line leaning more strikes only level lines of text
MAX_RECT_H = 3.5         # pt; a strike drawn as a filled rect is thin
MAX_STROKE_FRAC = 0.30   # a line wider than this share of the word's height is a highlighter
MID_BAND = 0.22          # strokes within [y0 + f*h, y1 - f*h] count as through-text
# Dashes / flat-bezier pieces are chained before the length gate (see _chain_short).
DASH_MIN_SEG = 1.0       # pt; below this a segment is graphics noise, never a dash
DASH_MAX_SEG = 6.0       # pt; a segment shorter than this may be a dash, so it is chained first
DASH_MAX_GAP = 4.0       # pt; max x-gap between dashes of one chained strike

FLAG_MIN_WCOV = 0.15     # a span on a word with no character boxes must cover >= this of it

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
    each run is emitted as one ``(x0, x1, y, color, width)`` interval carrying its longest member's
    paint. The caller applies the MIN_STROKE_LEN test to the run, so a dashed strike passes while a
    lone fleck does not."""
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

    A strike may be dashed (many short line segments) or drawn as a flat cubic bezier; pieces
    shorter than DASH_MAX_SEG are collected and chained (see :func:`_chain_short`) before the
    MIN_STROKE_LEN gate, so the gaps of a dashed strike count as struck. A line or curve counts
    only when it leans at most MAX_STROKE_SLOPE end to end, and a bar narrower than DASH_MAX_SEG
    only when it is at most MAX_STROKE_DY tall, so a tick or a dot over one character is not a
    strike. A steeper short piece still chains into a run (the jitter of a hand-drawn strike), but a
    run with no level piece is dropped. A line rising more than MAX_STROKE_DY is emitted as level
    pieces, so each word is tested where the line crosses it."""
    return [s[:5] for s in _strokes(page)]


def _strokes(page):
    """:func:`horiz_strokes` with two more fields per interval: ``extent``, the length of the
    mark it comes from (a whole line, bar or chained run, not one level piece of a leaning line),
    and ``leaning``, true for a line or curve leaning more than LEVEL_SLOPE."""
    out, level, steep = [], [], []
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
                if (r.height <= MAX_RECT_H and r.width >= MIN_STROKE_LEN
                        and (r.width >= DASH_MAX_SEG or r.height <= MAX_STROKE_DY)):
                    out.append((r.x0, r.x1, (r.y0 + r.y1) / 2, _rgb(fill_col),
                                round(float(r.height), 2), r.width, False))
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
                    seg = (p1.x, p2.x, (p1.y + p2.y) / 2, _rgb(stroke_col), width)
                elif flat:
                    n = math.ceil(abs(dy))              # pieces rising <= 1 pt each
                    for i in range(n):
                        out.append((p1.x + dx * i / n, p1.x + dx * (i + 1) / n,
                                    p1.y + dy * (i + 0.5) / n, _rgb(stroke_col), width, dx, lean))
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
                    seg = (r.x0, r.x1, (r.y0 + r.y1) / 2, _rgb(col), round(float(r.height), 2))
                    flat = r.height <= MAX_STROKE_DY or r.width >= DASH_MAX_SEG
            if seg is None:
                continue
            if seg[1] - seg[0] >= DASH_MAX_SEG:
                if flat:                               # solid stroke: emitted as-is (no chaining)
                    out.append((*seg, seg[1] - seg[0], lean))
            elif seg[1] - seg[0] >= DASH_MIN_SEG:
                (level if flat else steep).append(seg)  # dash, curve piece or short stroke: chain
    runs = _chain_short(level)
    for chain in _chain_short(level + steep):
        # a run must hold a level piece: a lone tick never strikes, a jittery hand-drawn one does
        if chain[1] - chain[0] >= MIN_STROKE_LEN and any(
                round(f[2]) == round(chain[2]) and chain[0] <= f[0] and f[1] <= chain[1]
                for f in runs):
            out.append((*chain, chain[1] - chain[0], False))
    return out


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


_Found = namedtuple("_Found", "vertical boxes tilted hidden")


class _PageChars:
    """Per-character boxes for a page's ``get_text("words")`` entries, so a strike maps onto the
    characters it crosses, with each word's text direction and visibility. Built lazily from one
    rawdict pass and shared by the detectors in ``page_strikes(method='both')``."""

    def __init__(self, page):
        self.page = page
        self._by_text = None
        self._visible = None                   # the boxes of the page's visible characters

    def _add(self, chars, vertical, tilted):
        if not chars:
            return
        boxes = [tuple(c["bbox"]) for c, _ in chars]
        cx = (min(b[0] for b in boxes) + max(b[2] for b in boxes)) / 2
        cy = (min(b[1] for b in boxes) + max(b[3] for b in boxes)) / 2
        text = "".join(c["c"] for c, _ in chars)
        found = _Found(vertical, boxes, tilted, all(hidden for _, hidden in chars))
        self._by_text.setdefault(text, []).append((cx, cy, found))

    def _build(self):
        self._by_text, self._visible = {}, []
        raw = self.page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_WORDS)
        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                dx, dy = line.get("dir", (1.0, 0.0))
                vertical = abs(dy) > abs(dx)
                tilted = not vertical and abs(dy) > LEVEL_SLOPE * abs(dx)
                cur = []
                for span in line.get("spans", []):
                    hidden = span.get("alpha", 255) == 0    # render mode 3, or fully transparent
                    for ch in span.get("chars", []):
                        if ch["c"] in _WORD_DELIMITERS:
                            self._add(cur, vertical, tilted)
                            cur = []
                        else:
                            cur.append((ch, hidden))
                            if not hidden:
                                self._visible.append(pymupdf.Rect(ch["bbox"]))
                self._add(cur, vertical, tilted)

    def lookup(self, word):
        """``(vertical, [char bbox, ...], tilted, hidden)`` for one ``get_text("words")`` entry, or
        None when no extracted line reproduces it (the caller then splits the word box evenly).
        ``tilted``: its line of text leans more than LEVEL_SLOPE; ``hidden``: every character is
        invisible (render mode 3, or fully transparent)."""
        if self._by_text is None:
            self._build()
        x0, y0, x1, y1, txt = word[:5]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        best = None
        for ecx, ecy, found in self._by_text.get(txt, ()):
            dist = abs(ecx - cx) + abs(ecy - cy)
            if dist <= 2.0 and len(found.boxes) == len(txt) and (best is None or dist < best[0]):
                best = (dist, found)
        return None if best is None else best[1]

    def visible_under(self, word):
        """The boxes of the visible characters the word's box overlaps (touching edges do not
        count)."""
        if self._by_text is None:
            self._build()
        r = pymupdf.Rect(word[:4])
        return [v for v in self._visible if r.intersects(v)]


def _attaches(c):
    """A combining mark, or a format character (a joiner, a soft hyphen): part of the character
    before it, never a base an overlay strikes on its own."""
    cat = unicodedata.category(c)
    return cat[0] == "M" or cat == "Cf"


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


def _in_char_band(box, y):
    """Whether height `y` lies in one character's own middle band, ``[y0 + MID_BAND*h,
    y1 - MID_BAND*h]`` of its `box`."""
    h = box[3] - box[1]
    return box[1] + MID_BAND * h <= y <= box[3] - MID_BAND * h


def _fits(s, box):
    """Whether stroke `s` (a :func:`_strokes` tuple) counts on the character with this `box`: a
    stroke at least DASH_MAX_SEG long counts wherever the word's middle band takes it; a shorter
    one (a fraction bar, a root's bar) only through the character's own middle band, since in a
    word mixing sizes the word's band also takes in rules above or below the small characters."""
    return s[5] >= DASH_MAX_SEG or _in_char_band(box, s[2])


def _through_strokes(word, mid, found, boxes, chars):
    """The strokes in `mid` (:func:`_strokes` tuples through the word's middle band) that strike
    its text: on an invisible word laid over visible text, a short stroke only through the middle
    of a visible character beneath, not as its underline; on a tilted line, no leaning stroke (a
    slanted line along slanted text crosses its upright box); and a short stroke only through some
    character's own middle band (:func:`_fits`). `boxes` are the word's characters' (overlays left
    out)."""
    if found is not None:
        under = chars.visible_under(word) if found.hidden else []
        if under:
            mid = [s for s in mid if s[5] >= DASH_MAX_SEG or any(
                min(s[1], v.x1) > max(s[0], v.x0) and _in_char_band(v, s[2]) for v in under)]
        if found.tilted:
            mid = [s for s in mid if not s[6]]
    return [s for s in mid if s[5] >= DASH_MAX_SEG or any(
        _fits(s, b) and min(s[1], b[2]) > max(s[0], b[0]) for b in boxes)]


def _stroke_span(word, txt, boxes, exact, mid):
    """``(span, cov)`` for the strokes in `mid` (:func:`_strokes` tuples, already through the
    word's text): `span` as :func:`_struck_span` gives it for `txt` (the word's text without
    overlays) from its character `boxes` (`exact` when MuPDF's, not an even split), `cov` the
    share of the word's width the strokes cover. A character is struck by the strokes that
    :func:`_fits` it."""
    wx0, wx1 = word[0], word[2]
    merged, covered = _merged_intervals(wx0, wx1, [s[:2] for s in mid])
    cov = covered / max(wx1 - wx0, 1e-6)
    struck = []
    for bx in boxes:
        bx0, bx1 = bx[0], bx[2]
        own = merged
        if not all(_fits(s, bx) for s in mid):
            own = _merged_intervals(wx0, wx1, [s[:2] for s in mid if _fits(s, bx)])[0]
        if bx1 - bx0 <= 1e-6:
            # zero-width: the rest of a ligature ('fi' gives an 'f' spanning the glyph and an
            # 'i' at its right edge) or a combining mark, so it goes with the character before
            struck.append(struck[-1] if struck else any(a <= bx0 <= b for a, b in own))
        else:
            covered = sum(max(0.0, min(b, bx1) - max(a, bx0)) for a, b in own)
            struck.append(covered >= CHAR_STRUCK_COV * (bx1 - bx0))
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
    enters = (any(s[0] < wx0 - 1.0 for s in mid), any(s[1] > wx1 + 1.0 for s in mid))
    return _struck_span(txt, struck, cov, exact=exact, enters=enters), cov


def native_page_strikes(page, page_index, words=None, _chars=None):
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
    strokes = _strokes(page) if any(it[4] for it in items) else []
    if not strokes and not any(any(it[2]) for it in items):
        return []
    chars = _chars if _chars is not None else _PageChars(page)

    out = []
    for word, txt, over, keep, sized in items:
        wx0, wy0, wx1, wy1 = word[:4]
        wh = wy1 - wy0
        max_wid = max(MAX_RECT_H, MAX_STROKE_FRAC * wh)
        mid = [s for s in strokes
               if wy0 + MID_BAND * wh <= s[2] <= wy1 - MID_BAND * wh
               and min(s[1], wx1) > max(s[0], wx0) and s[4] <= max_wid] if sized else []
        overlaid = any(over)
        if not mid and not overlaid:
            continue
        found = chars.lookup(word)
        vertical = found is not None and found.vertical
        if vertical and not overlaid:
            continue        # vertical text: a horizontal line crosses it, it does not strike it
        boxes = ([found.boxes[i] for i in keep] if found is not None
                 else _even_char_boxes((wx0, wy0, wx1, wy1, txt)))
        mid = _through_strokes(word, mid, found, boxes, chars) if mid and not vertical else []
        span, cov = (_stroke_span(word, txt, boxes, found is not None, mid)
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
            dom = max(mid, key=lambda s: min(s[1], wx1) - max(s[0], wx0))
            rec["stroke_color"], rec["stroke_width"] = dom[3], dom[4]
        rec.update(tier="vector", verdict="struck", final=True)
        out.append(rec)
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


def page_strikes(page, page_index, method="vector", words=None):
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
        cov = _covering_record(out, r)
        if cov is None:
            out.append(r)
        elif _displaces(r, cov):
            out = [r if x is cov else x for x in out]
    # annotations: add the record where nothing covers it, else GRAFT its forensics onto the
    # covering record — a PDF annotation's appearance stream is also caught by the vector/flag
    # paths, so a union that merely dropped it would lose the author/date/color evidence.
    for a in native_annot_strikes(page, page_index, words, _chars=chars):
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
