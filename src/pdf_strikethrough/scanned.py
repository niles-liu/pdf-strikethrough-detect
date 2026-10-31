"""Scanned word-level strike classification: attach detected strokes to OCR words.

Given the strokes from :func:`pdf_strikethrough.lines.strike_lines` and a list of
:class:`pdf_strikethrough.ocr.Word` (from any OCR backend), decide which words are struck, with
char-level spans and full/partial resolution. This is layer 1 (geometry + OCR); the CNN verdict
(layer 2) is applied by the orchestrator in :mod:`pdf_strikethrough.detect`.

Engine-independent: the only piece specific to Azure Document Intelligence is the confidence
calibration, captured in :class:`ScanConfig`.
"""
from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np

from .lines import (RENDER_DPI, STEM_MIN_RUNS, THIN_STROKE_RUN_PX, ink_mask, rule_mask,
                    strike_lines, to_gray_u8)

# --- per-word classification tunables (geometry; engine-independent) ---
STRIKE_TOL          = 0.16   # half-width of the strike band (fraction of word-box height)
MIN_WORD_XOVER      = 0.45   # full-word strike: line covers >= this fraction of the word width
PARTIAL_MIN_LCOV    = 0.70   # partial strike: >= this fraction of the LINE lies inside the word
PARTIAL_MIN_CHARS   = 2      # partial strike must span >= this many estimated characters
PARTIAL_MIN_WCOV    = 0.12   # ...and >= this fraction of the word width
PARTIAL_ISO_MAX_OFF = 0.08   # isolated partial (line strikes nothing else): must sit dead-center
INK_MAX_OFF         = 0.40   # pixel test still considered up to this |off| (box-lies rescue)
INK_MIN_FRAC        = 0.35   # min fraction of covered columns with glyph ink above resp. below
INK_ONESIDE_MAX     = 0.15   # a side with less ink than this is "empty"; an underline or table
                             # rule has ink on ONE side and ~none on the other
INK_GAP_PX          = 2      # the ink above and below is read from this many px past the stroke
STROKE_ROW_FILL     = 0.90   # on the thick retry, a row just past the line's band inked across
                             # this share of the columns is the stroke's own and is skipped
INK_GAP_THICK_PX    = 0      # a word the ink tests drop under a thick line (run_px above
                             # THIN_STROKE_RUN_PX scaled with the dpi, and above 2 px) is tried
                             # again from this many px past it: a 2-3 pt bar covers most of the
                             # x-height and leaves 2-4 rows of glyph ink beside it, inside
                             # INK_GAP_PX. INK_GAP_PX or more turns it off
INK_GAP_THICK_MAX_LEN_IN = 4.0  # ...for a line up to this long: a thick bar across the page is the
                             # ruling of lined paper, and words written on it reach down into it
INK_GAP_THICK_STEM_RATIO = 1.5  # ...and for a line at least this many times as thick as the
                             # median ink run along it in the rows the retry adds (the stem test):
                             # under a printed bar they are the struck letters' stems, beside a pen
                             # stroke its own uneven edge and strokes of the same pen. A test
                             # like lines.STEM_RATIO's, read closer in; here too few runs to
                             # measure means no retry. 0 turns it off
INK_GAP_THICK_STEM_WOBBLE = 1.2  # ...the stem test, applied only to a line whose straightness
                             # (wobble, px at RENDER_DPI) is at least this or unmeasured: a pen
                             # stroke rarely runs straighter, and under a straight printed bar the
                             # letters' stems can be nearly as thick as the bar (bold text, low
                             # dpi). 0 applies it to every line
GLYPH_INK_RULE_LEN_IN = 0.25 # the INK_* tests read glyph ink: horizontal runs at least this long
                             # (50 px @200 dpi) are rules or underlines and are left out, so the
                             # rule under an underlined word is not its "ink below". TRIED 1/6 in
                             # on scanned ruled redlines: fewer over-marks (-18 on the confirming
                             # half) but 2 struck words lost, as it also cuts the stair-step pieces
                             # of a long, slightly sloped strike, which skewed scans make worse
INK_SHORT_LEN_IN    = 0.50   # pixel test REQUIRED only for lines shorter than this (+ rescues)
STROKE_SNAP_PX      = 4      # a word the ink tests drop is tried again at the stroke's own row,
                             # looked for this many px (at RENDER_DPI, scaled with dpi) either side
                             # of the line: a stitched line can run a few px off a thin or light
                             # stroke, so its "ink below" falls under the glyphs. 0 turns it off
STROKE_SNAP_MIN_RUN = 0.80   # ...the row band whose unbroken ink run covers the most of the columns
                             # the word shares with the line, and at least this share of them
PAIR_REACH          = 0.30   # a word the ink tests drop passes them when a second unbroken
                             # stroke runs beside its line, within this share of the box height,
                             # with glyph ink between the two and their middle in the strike band:
                             # a double strike, each of whose lines sits off the middle of the
                             # x-height with a few rows of it outside. 0 turns it off
PAIR_MAX_LEN_IN     = 4.0    # ...under a line at most this long: across the page it is the ruling
                             # of lined paper; a pen stroke above it makes a pair. 0 turns it off
PAIR_MAX_RUN_PX     = 4.0    # ...and at most this thick (px at RENDER_DPI, scaled with the dpi,
                             # never under 2 px; lines.THIN_STROKE_RUN_PX): a double strike's lines
                             # are thin; in handwriting a pen stroke or a ruling beside another
                             # makes a pair, and a thick line has the thick retry. 0 turns it off
PAIR_LEN_RATIO      = 0.5    # ...and only when the two strokes run about as long as each other, the
                             # shorter at least this share of the longer (_stroke_length): a double
                             # strike's lines are drawn together, while beside a pen stroke the
                             # other is a ruling or a stroke of its own. 0 turns it off
PAIR_MAX_WOBBLE     = 1.3    # ...and under a line straighter than this (wobble, px at RENDER_DPI;
                             # unmeasured counts as not straight): a printed line runs straight, a
                             # pen stroke wobbles. 0 turns it off
RESCUE_RETRIES      = True   # a word the OCR-confidence rescue passes (under a short line, without
                             # ink on both sides) still goes through the retries above, so one
                             # that gives it strike geometry (in the band with ink on both sides,
                             # or a pair) does so for the DI-confidence veto. False turns it off
FILL_STRONG         = 0.87   # fill >= this: line accepted on geometry alone
FILL_UNMEASURED     = 0.80   # score_struck's fill for a caller-built line without one
TABLE_RULE_MIN_LEN_IN = 0.75 # in-band solid line >= this + fill>=FILL_STRONG: a table rule, not a
                             # strike, unless it shows through-glyph ink on BOTH sides
# --- printed-rule veto (issue #7), opt-in via ScanConfig.ruled_forms(): a degraded-scan table
# rule / form line rides a dead-straight, near-solid path, while a pen or typed strike crossing
# raised glyphs SHATTERS (fill drops) and WANDERS (wobble rises). Geometry-only, so it needs no
# OCR confidence. PROVISIONAL and thin: calibrated on ONE positive document (3 real strikes, all
# the same morphology, clearing both bars). Do not tighten these without more labeled positives.
PRINTED_RULE_FILL_MAX     = 0.88  # fill above this = a solid drawn rule, not a shattered strike
PRINTED_RULE_STRAIGHT_MAX = 1.80  # wobble px @RENDER_DPI below this = a drawn rule, not a strike
TWIN_MIN_LEN_IN     = 0.60   # substantial line: >=2 fully-struck words, or one word + long line
FULL_CHAR_COVER     = 0.70   # unioned char coverage >= this -> whole word counts as struck, when
                             # the strokes also reach its first and last character (below)
# --- char spans against glyph ink: an OCR box is looser than its glyphs, and a phrase-grained
# engine (RapidOCR) boxes several words as one, so a stroke's char span is measured against the
# x-extent of the glyph ink inside the box, not the box. A detected line's ends are no better:
# stitching runs a line on through crossbars and arches past where the pen stopped, so the stroke's
# reach is taken from its own ink, its long runs near the line. An end character counts as struck
# when the stroke covers at least CHAR_STRUCK_FRAC of its estimated width (the ink extent over the
# character count), and a word is promoted to a whole-word strike only when its strokes reach both
# its first and its last character, so a stroke over most but not all of a word keeps its partial
# span.
CHAR_STRUCK_FRAC    = 0.50
STROKE_RUN_MIN_CHARS = 1.5   # an ink run near the line at least this many character widths long
                             # is the stroke itself; a glyph's crossbar or arch is shorter
STROKE_SEARCH_BAND  = 0.25   # ...looked for this share of the word box's height either side of
                             # the line: a stitched line can sit several px off a pen stroke
STROKE_PIECE_MIN_CHARS = 1.0  # a shorter run on the stroke's rows that crosses a gap between
                             # glyphs is the stroke too (a short word struck on its own)
STROKE_LINK_MIN_CHARS = 0.5  # ...and so is a run this long within STROKE_BRIDGE_CHARS of the
STROKE_BRIDGE_CHARS = 0.25   # stroke on its row: a pen break, a dashed stroke, typed hyphens
GLYPH_BAND          = 0.30   # glyph ink is read this share of the box's height either side of the
                             # line, so a speck, rule or underline far above or below can't widen
GLYPH_MIN_COLUMN_PX = 2      # ...in columns holding at least this many px of it
GLYPH_ROW_MAX_FILL  = 0.90   # a box row inked across more than this share of its width is a rule
                             # or underline, not glyphs, and is left out of the glyph-ink extent
WORD_GAP            = 0.20   # a gap in the glyph ink this share of the box's height is a word space;
STRAY_MAX_WIDTH     = 0.25   # an outer cluster past one, narrower than this share, is not a letter

AUTO_SCORE = 0.55            # score >= this: tier 'auto'; >= REVIEW_SCORE: 'review'; else 'weak'
REVIEW_SCORE = 0.20


@dataclass(frozen=True)
class ScanConfig:
    """Confidence calibration for the scanned classifier. The defaults are the Azure Document
    Intelligence calibration, built on degraded ruled forms, where DI reads struck words at
    0.43-0.94 and clean text at 0.976-1.0; on a crisp printed scan it reads struck words as clean.
    So ``ScanConfig(...)`` keeps confidence gating on whatever else it sets: for another engine
    start from ``confidence_free(...)``, which takes the same fields. The functions that take a
    config default to ``confidence_free()``, except ``detect_pdf`` on the pages a ``di_result``
    covers.

    ``confidence_free()`` disables EVERY confidence-dependent decision: the chain-rejection
    gate, the ink-fail rescue, the DI-confidence veto, and the
    confidence term in the evidence score (the remaining geometry terms are renormalized so the
    auto/review/weak thresholds keep their meaning).
    Use it for engines whose confidence doesn't separate struck from clean text (e.g. RapidOCR);
    detection then rests on geometry + the CNN alone.

    ``recall_first()`` / ``precision_first()`` pick the CNN *operating point* — the strike/clean
    decision threshold applied to StrikeNet's probability. ``cnn_p_hi``/``cnn_p_lo`` (None = the
    model's shipped thresholds) override it; calibrate them from labeled data with
    :mod:`pdf_strikethrough.calibration`.

    ``ruled_forms()`` is a **provisional** precision bias for degraded, heavily-ruled scans; it is
    outside the v1.0 stability contract and may be removed. See its docstring.

    ``rescue_clean_chains`` controls the chain gate's escape hatch (see :func:`classify_lines`). Set
    it False on degraded ruled forms, where it is a dominant false-positive source. Deliberately
    NOT bundled into ``ruled_forms()``: the two switches overlap little and each costs recall on
    its own terms. ⚠ Unlike the veto it needs calibrated confidences, so it is **inert when
    ``confidence_gating`` is off**. Rationale and per-switch
    measurements: the ``rescue_clean_chains`` entry in ``CHANGELOG.md``, reproducible with
    ``benchmarks/confidence_veto.py --switches``.

    ``glyph_ink`` (on by default) measures the ink-on-both-sides test on glyph ink only: the
    page's ink without its horizontal runs of ``GLYPH_INK_RULE_LEN_IN`` or more (table rules,
    underlines). Without it, an underlined word sitting on a table rule reads as struck, because
    the rule is ink below the underline. Ruled text is then judged like unruled text. A long solid
    line through a word's middle band with no glyph ink on either side is judged on all ink, so
    one of a double rule stays a rule. ``ScanConfig.confidence_free(glyph_ink=False)`` (or
    ``dataclasses.replace(preset, glyph_ink=False)``) measures on all ink, with no thick-bar
    retry and no double-strike pair test, which read glyph ink.

    ``line_support`` (on by default; off in ``precision_first()``) keeps a word of three characters
    or fewer, not counting surrounding punctuation, that StrikeNet leaves unconfirmed when the line
    through it strikes confirmed words on both sides of it and the word has a stroke of its own.
    Such a crop is mostly padding at the CNN's input width, so a struck "of" or "a" scores low
    inside a confirmed struck run. Kept records carry ``line_support: True``. See
    :func:`pdf_strikethrough.detect._line_support`."""
    confidence_gating: bool = True
    max_clean_conf: float = 0.955     # a struck word OCRs at or below this: the chain gate
                                      # (fill<FILL_STRONG) and the DI-confidence veto
    inkfail_max_conf: float = 0.974   # a pixel-failing in-band hit is rescued at OCR conf <= this
    page_edited_min: float = 0.03     # frac words conf<=0.90 >= this -> page has pen edits
    cnn_p_hi: float | None = None     # override the CNN struck threshold; None = the model's
    cnn_p_lo: float | None = None     # override the CNN clean threshold; None = the model's
    veto_printed_rules: bool = False  # drop solid/dead-straight lines as drawn rules (issue #7)
    rescue_clean_chains: bool = True  # let an EDITED page's clean-OCR glyph chain reach the CNN
    glyph_ink: bool = True            # ink tests ignore long horizontal runs (rules, underlines)
    line_support: bool = True         # keep a short unconfirmed word inside a confirmed struck run
                                      # (detect._line_support)

    def __post_init__(self):
        # every number here is a confidence, a page fraction or a probability: a float in [0, 1]
        for name in ("max_clean_conf", "inkfail_max_conf", "page_edited_min", "cnn_p_hi",
                     "cnn_p_lo"):
            v = getattr(self, name)
            if v is None and name.startswith("cnn_"):
                continue
            if isinstance(v, bool) or not isinstance(v, numbers.Real) or not 0.0 <= v <= 1.0:
                raise ValueError(f"ScanConfig.{name} must be a number in [0, 1], got {v!r}")
            object.__setattr__(self, name, float(v))
        p_lo, p_hi = self.cnn_p_lo, self.cnn_p_hi
        if p_lo is not None and p_hi is not None and p_lo > p_hi:
            raise ValueError(f"ScanConfig.cnn_p_lo ({p_lo}) is above cnn_p_hi ({p_hi})")

    @classmethod
    def azure_di(cls, **kw):
        """Azure Document Intelligence calibration: the defaults, with confidence gating on. `kw`
        sets other fields: ``azure_di(glyph_ink=False)``."""
        return cls(**{**kw, "confidence_gating": True})

    @classmethod
    def ruled_forms(cls, veto_printed_rules=True, **kw):
        """**PROVISIONAL**: a precision bias for DEGRADED, HEAVILY-RULED scans (e.g.
        printed forms or tables) where a faint table/form rule crossing text mimics a pen strike and
        the CNN over-fires on it. Turns on the printed-rule veto (:func:`_is_printed_rule`): any
        detected line that is solid and/or dead-straight is treated as printed furniture rather than
        a strike. In practice most of what it removes is short solid/straight *fragments* — glyph
        strokes and pieces of rules thrown off by a degraded scan — not just full-width rules. Off
        by default because on a CLEAN scan a real strike is ALSO solid and straight — enable this
        only when inputs are known-degraded ruled forms and precision matters more than catching a
        pristine strike. See issue #7.

        This is a stopgap for one input class, and its thresholds are calibrated on a small labeled
        set. It is **not covered by the v1.0 stability contract** and is expected to be removed once
        StrikeNet handles ruled forms natively (issue #4·C); pin exactly if you depend on it.

        Controls the printed-rule veto ONLY. On degraded ruled forms you usually also want
        ``ruled_forms(rescue_clean_chains=False)``, the best-measured configuration; the two are
        opted into separately so a caller who has validated one does not silently acquire the other.
        Confidence gating stays on, which assumes Azure DI confidences; add
        ``confidence_gating=False`` for other engines.
        """
        return cls(veto_printed_rules=veto_printed_rules, **kw)

    @classmethod
    def confidence_free(cls, **kw):
        """For engines whose confidence doesn't separate struck from clean (e.g. RapidOCR). `kw`
        sets other fields: ``confidence_free(line_support=False)``."""
        return cls(**{**kw, "confidence_gating": False})

    @classmethod
    def recall_first(cls, cnn_p_hi=0.50, **kw):
        """Recall-first operating point (legal / audit review): bias toward catching every
        deletion. Lowers the CNN's struck-confirmation bar so a borderline strike still counts as
        struck — more false positives and more manual review, but a deleted word is not silently
        missed. Combine with ``confidence_gating=False`` for a confidence-free OCR engine."""
        return cls(cnn_p_hi=cnn_p_hi, **kw)

    @classmethod
    def precision_first(cls, cnn_p_hi=0.97, line_support=False, **kw):
        """Precision-first operating point (RAG / indexing): bias toward never dropping live text
        as if it were deleted. Only high-confidence CNN strikes are reported; an uncertain strike
        falls back to 'unsure'/clean rather than being removed, so ``line_support`` is off.
        Confidence gating stays on, which assumes Azure DI confidences; add
        ``confidence_gating=False`` for other engines."""
        return cls(cnn_p_hi=cnn_p_hi, line_support=line_support, **kw)


def _glyph_ink(ink, dpi=RENDER_DPI):
    """`ink` without its horizontal runs of GLYPH_INK_RULE_LEN_IN or more at `dpi`: the glyph ink
    the two-sided ink test reads."""
    ink = np.asarray(ink, dtype=bool)
    return ink & ~rule_mask(ink, max(1, int(round(GLYPH_INK_RULE_LEN_IN * dpi))))


def _ink_above_below(ink, line_ends_px, line_run_px, word_bbox_px, gap=None, skip_full=False):
    """Fractions of covered columns with glyph ink above resp. below the stroke, read from `gap`
    px past its edges (INK_GAP_PX when None). A strike runs THROUGH glyphs -> ink both sides; an
    underline has ink above only; a glyph chain has ~none above. The stroke y is interpolated from
    the LINE ENDPOINTS at the center of the word's overlap with the line. With `skip_full`, rows
    just past the band inked across STROKE_ROW_FILL of the columns, as many as the gap is short of
    INK_GAP_PX on each side, are the stroke's own and are skipped too: read from close in, the
    stroke's last row would count as glyph ink. `ink` is the mask to measure: glyph ink
    (:func:`_glyph_ink`) unless ``ScanConfig.glyph_ink`` is off."""
    (sx, sy), (ex, ey) = line_ends_px
    wx0, wy0, wx1, wy1 = word_bbox_px
    x0, x1 = max(sx, wx0), min(ex, wx1)
    if x1 <= x0:
        return 0.0, 0.0
    xm = (x0 + x1) / 2.0
    t = 0.0 if ex == sx else (xm - sx) / (ex - sx)
    ly = sy + t * (ey - sy)
    half = line_run_px / 2.0 + (INK_GAP_PX if gap is None else gap)
    H, W = ink.shape
    top0, top1 = max(0, int(wy0)), max(0, int(ly - half))
    bot0, bot1 = min(H, int(ly + half) + 1), min(H, int(wy1) + 1)
    cols = slice(max(0, int(x0)), min(W, int(x1) + 1))
    if skip_full:
        for _ in range(max(0, INK_GAP_PX - (INK_GAP_PX if gap is None else int(gap)))):
            if top1 > top0 and ink[top1 - 1, cols].mean() >= STROKE_ROW_FILL:
                top1 -= 1
            if bot1 > bot0 and ink[bot0, cols].mean() >= STROKE_ROW_FILL:
                bot0 += 1
    above = ink[top0:top1, cols].any(axis=0).mean() if top1 > top0 else 0.0
    below = ink[bot0:bot1, cols].any(axis=0).mean() if bot1 > bot0 else 0.0
    return float(above), float(below)


def _edge_runs(ink, line_ends_px, line_run_px, word_bbox_px, near, far):
    """Median length of the ink runs along the line, over the columns the word shares with it, in
    the rows `near` + 1 to `far` px past its band on each side: the rows a reading of the ink beside
    it from `near` px adds to one from `far` px (:func:`_ink_above_below`, which reads at the
    line's height at the word's middle; these rows follow the line column by column). None with
    fewer than lines.STEM_MIN_RUNS runs, or no rows to read."""
    (sx, sy), (ex, ey) = line_ends_px
    H, W = ink.shape
    c0 = max(0, int(max(word_bbox_px[0], min(sx, ex))))
    c1 = min(W, int(min(word_bbox_px[2], max(sx, ex))) + 1)
    k = np.arange(int(near) + 1, int(far) + 1)[:, None]
    if c1 - c0 < 2 or k.size == 0:
        return None
    xs = np.arange(c0, c1)
    ly = np.full(xs.shape, (sy + ey) / 2.0) if ex == sx else sy + (xs - sx) * (ey - sy) / (ex - sx)
    half = line_run_px / 2.0
    runs = []
    for yy in (np.floor(ly - half).astype(int)[None, :] - k,
               np.floor(ly + half).astype(int)[None, :] + k):
        ok = (yy >= 0) & (yy < H)
        sel = np.zeros(yy.shape, dtype=bool)
        sel[ok] = ink[yy[ok], np.broadcast_to(xs, yy.shape)[ok]]
        d = np.diff(np.pad(sel, ((0, 0), (1, 1))).astype(np.int8), axis=1).ravel()
        runs.append(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))
    runs = np.concatenate(runs)
    return float(np.median(runs)) if runs.size >= STEM_MIN_RUNS else None


def _one_sided(above, below):
    """The underline / table-rule signature: well inked on one side, about empty on the other."""
    return max(above, below) >= INK_MIN_FRAC and min(above, below) < INK_ONESIDE_MAX


def _glyph_ink_extent(ink, line_ends_px, line_run_px, word_bbox_px, gap=2):
    """(x0, x1) in px: the columns of the word box holding glyph ink, or None when it holds none.
    Glyph ink is read only on rows near the line (GLYPH_BAND, interpolated from the line
    endpoints so a sloped stroke is followed), leaving out the stroke's own rows and any row inked
    across nearly the whole box (a rule or underline); a column needs GLYPH_MIN_COLUMN_PX of it.
    An outer cluster past a word-space gap that is too narrow to be a letter (a speck, or a sliver
    of the next word in a loose box) is dropped. So a stroke running past the glyphs, or anything
    else a loose box holds, does not widen the word."""
    wx0, wy0, wx1, wy1 = word_bbox_px
    H, W = ink.shape
    c0, c1 = max(0, int(wx0)), min(W, int(np.ceil(wx1)))
    r0, r1 = max(0, int(wy0)), min(H, int(np.ceil(wy1)))
    if c1 <= c0 or r1 <= r0:
        return None
    (sx, sy), (ex, ey) = line_ends_px
    xs = np.arange(c0, c1, dtype=np.float64)
    ly = np.full(xs.shape, (sy + ey) / 2.0) if ex == sx else sy + (xs - sx) * (ey - sy) / (ex - sx)
    rows = np.arange(r0, r1, dtype=np.float64)[:, None]
    box_h = wy1 - wy0
    dist = np.abs(rows - ly[None, :])
    glyph = (ink[r0:r1, c0:c1] & (dist > line_run_px / 2.0 + gap)
             & (dist <= line_run_px / 2.0 + GLYPH_BAND * box_h))
    glyph &= (glyph.mean(axis=1) <= GLYPH_ROW_MAX_FILL)[:, None]
    cols = np.flatnonzero(glyph.sum(axis=0) >= GLYPH_MIN_COLUMN_PX)
    if cols.size == 0:
        return None
    clusters = np.split(cols, np.flatnonzero(np.diff(cols) > max(3.0, WORD_GAP * box_h)) + 1)
    while len(clusters) > 1 and clusters[-1][-1] - clusters[-1][0] + 1 < STRAY_MAX_WIDTH * box_h:
        clusters.pop()
    while len(clusters) > 1 and clusters[0][-1] - clusters[0][0] + 1 < STRAY_MAX_WIDTH * box_h:
        clusters.pop(0)
    return float(c0 + clusters[0][0]), float(c0 + clusters[-1][-1] + 1)


def _stroke_reach(ink, line_ends_px, line_run_px, line_x_px, word_bbox_px, min_run_px,
                  min_piece_px, min_link_px, bridge_px):
    """(x0, x1) in px: how far the stroke's own ink reaches inside the word box, or None when none
    of it runs at least `min_run_px` long. Looked for in a band around the line (a stitched line's
    height and slope can be off by several px), thickened by 2 px up and down so that a leaning
    stroke still reads as long horizontal runs; only the line's own x-range counts. A run at least
    `min_piece_px` long counts too when it lies on a long run's rows and crosses a gap between
    glyphs, and so does one at least `min_link_px` long within `bridge_px` of a counted run on the
    same row (a pen break, a dashed stroke, typed hyphens). A glyph's crossbar or arch is too short
    to count."""
    wx0, wy0, wx1, wy1 = word_bbox_px
    H, W = ink.shape
    lx0, lx1 = line_x_px
    c0, c1 = max(0, int(wx0), int(lx0) - 2), min(W, int(np.ceil(wx1)), int(np.ceil(lx1)) + 2)
    if c1 - c0 < max(3, min_run_px):
        return None
    (sx, sy), (ex, ey) = line_ends_px
    xs = np.arange(c0, c1, dtype=np.float64)
    ly = np.full(xs.shape, (sy + ey) / 2.0) if ex == sx else sy + (xs - sx) * (ey - sy) / (ex - sx)
    half = line_run_px / 2.0 + max(3.0, STROKE_SEARCH_BAND * (wy1 - wy0))
    r0 = max(0, int(np.floor(ly.min() - half)))
    r1 = min(H, int(np.ceil(ly.max() + half)) + 1)
    if r1 - r0 < 1:
        return None
    rows = np.arange(r0, r1, dtype=np.float64)[:, None]
    near = ink[r0:r1, c0:c1] & (np.abs(rows - ly[None, :]) <= half)
    thick = near.copy()
    for d in (1, 2):                                    # thicken vertically by 2 px each way
        thick[d:] |= near[:-d]
        thick[:-d] |= near[d:]
    edges = np.diff(np.pad(thick, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    run_rows, starts = np.nonzero(edges == 1)
    ends = np.nonzero(edges == -1)[1]                   # row-major order: paired with starts
    long = ends - starts >= min_run_px
    if not long.any():
        return None
    # A shorter piece is the same stroke past a gap (a short word struck on its own) when it lies
    # on the long runs' rows and crosses a gap between two glyphs: a column inside it whose only
    # ink in the box is on the piece's rows. A crossbar or a bowl on those rows has the rest of
    # its glyph above or below it there.
    keep = long.copy()
    pieces = np.flatnonzero(~long & np.isin(run_rows, run_rows[long])
                            & (ends - starts >= min_piece_px))
    if pieces.size:
        b0, b1 = max(0, int(wy0)), min(H, int(np.ceil(wy1)))
        box = ink[b0:b1, c0:c1]
        column_ink = box.sum(axis=0)
        for i in pieces:
            top, bot = max(0, r0 + run_rows[i] - 2 - b0), max(0, r0 + run_rows[i] + 3 - b0)
            inside = slice(starts[i] + 1, ends[i] - 1)
            on_rows = box[top:bot, inside].sum(axis=0)
            keep[i] = bool(((on_rows > 0) & (on_rows == column_ink[inside])).any())
    # A broken stroke: chain runs across short breaks outward from the counted runs, row by row
    # (runs are in column order within a row; only the outermost ones move the reach).
    lo, hi = starts[keep].min(), ends[keep].max()
    link = ends - starts >= min_link_px
    for row in np.unique(run_rows[keep]):
        idx = np.flatnonzero(run_rows == row)
        s, e, ok = starts[idx], ends[idx], link[idx]
        kept = np.flatnonzero(keep[idx])
        j = kept[-1]
        while j + 1 < idx.size and ok[j + 1] and s[j + 1] - e[j] <= bridge_px:
            j += 1
        hi = max(hi, e[j])
        j = kept[0]
        while j > 0 and ok[j - 1] and s[j] - e[j - 1] <= bridge_px:
            j -= 1
        lo = min(lo, s[j])
    return float(c0 + lo), float(c0 + hi)


def _band_ink(ink, line_ends_px, line_run_px, x0_px, x1_px, dy=0.0, pad=1):
    """Per column from x0_px to x1_px (inside the line's x-extent): does the stroke's band hold
    ink? The band is the stroke's thickness plus `pad` px each side, its y interpolated from the
    LINE ENDPOINTS per column and moved down by `dy` px. Empty when no column is left."""
    (sx, sy), (ex, ey) = line_ends_px
    H, W = ink.shape
    x0 = max(int(x0_px), int(min(sx, ex)), 0)
    x1 = min(int(x1_px), int(max(sx, ex)), W - 1)
    if x1 <= x0:
        return np.zeros(0, bool)
    half = line_run_px / 2.0 + pad
    xs = np.arange(x0, x1 + 1)
    t = (xs - sx) / (ex - sx) if ex != sx else np.zeros(len(xs))
    ys = sy + t * (ey - sy) + dy
    top = np.clip((ys - half).astype(int), 0, H)
    bot = np.clip((ys + half).astype(int) + 1, 0, H)
    return np.array([ink[r0:r1, x].any() for x, r0, r1 in zip(xs, top, bot)], bool)


def _longest_run(flags):
    """Length of the longest stretch of True in a 1-D bool array."""
    d = np.diff(np.concatenate(([0], np.asarray(flags, dtype=np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int((ends - starts).max()) if starts.size else 0


def _stroke_row_dy(ink, line_ends_px, line_run_px, x0_px, x1_px, max_dy):
    """How far (px, + = down) to move the line, at most `max_dy` either way, so its band sits on
    the stroke: the band whose longest unbroken run of ink covers the most of columns x0_px to
    x1_px, the nearest one on a tie. 0 when no band's run covers STROKE_SNAP_MIN_RUN of them, or
    when the line's own band covers as much as any. Bands are the line's thickness, without the
    pad the ink tests use, so a stroke a few px off is told from one under the line."""
    (sx, sy), (ex, ey) = line_ends_px
    H, W = ink.shape
    if max_dy < 1 or not np.isfinite([sx, sy, ex, ey, line_run_px]).all():
        return 0
    x0 = max(int(x0_px), int(min(sx, ex)), 0)
    x1 = min(int(x1_px), int(max(sx, ex)), W - 1)
    if x1 <= x0:
        return 0
    xs = np.arange(x0, x1 + 1)
    ys = sy + ((xs - sx) / (ex - sx) if ex != sx else np.zeros(len(xs))) * (ey - sy)
    half = line_run_px / 2.0
    dys = np.arange(-max_dy, max_dy + 1)
    tops = (ys[None, :] + dys[:, None] - half).astype(int)    # (band, column): rows top..bot-1
    bots = (ys[None, :] + dys[:, None] + half).astype(int) + 1
    # Each column's reach only, so a steep line costs no more than a level one: the window starts
    # at its first band's top and is as tall as the tallest stack of bands.
    first = tops[0]
    rows = first[None, :] + np.arange(int((bots - first).max()))[:, None]
    inside = (rows >= 0) & (rows < H)
    win = np.zeros(rows.shape, bool)
    win[inside] = ink[rows[inside], np.broadcast_to(xs, rows.shape)[inside]] != 0
    need = STROKE_SNAP_MIN_RUN * len(xs)
    if _longest_run(win.any(axis=0)) < need:      # no band runs further than all its rows together
        return 0
    cum = np.vstack([np.zeros((1, len(xs)), np.int32), np.cumsum(win, axis=0, dtype=np.int32)])
    cols = np.arange(len(xs))
    runs = [_longest_run(band) for band in (cum[bots - first, cols] - cum[tops - first, cols]) > 0]
    best = max(range(len(dys)), key=lambda i: (runs[i], -abs(dys[i])))   # 0 if the line's band ties
    return int(dys[best]) if runs[best] >= need else 0


def _reach(flags, gap):
    """How far a stroke starting at flags[0] runs: the index of its last True before the first
    `gap` + 1 False in a row (breaks of up to `gap` are bridged), 0 when it has none."""
    off = ~np.asarray(flags, bool)
    stop = off.size
    if off.size > gap:
        hole = np.flatnonzero(np.convolve(off.astype(int), np.ones(gap + 1, int), "valid") > gap)
        stop = int(hole[0]) if hole.size else off.size
    on = np.flatnonzero(~off[:stop])
    return int(on[-1]) if on.size else 0


def _stroke_length(ink, line_ends_px, line_run_px, dy, x_mid, scale=1.0):
    """Length (px) of the stroke through the row `dy` px off the line (+ = down), followed along
    the line's slope both ways from column `x_mid` and past the line's ends (a ruling runs on
    across the page): a column is on it when ink lies within the line's half thickness plus 2 px
    of that row, and breaks of up to 3 px are bridged, both scaled by `scale` (the dpi over
    RENDER_DPI)."""
    (sx, sy), (ex, ey) = line_ends_px
    H, W = ink.shape
    xs = np.arange(W)
    ys = sy + (xs - sx) * ((ey - sy) / (ex - sx) if ex != sx else 0.0) + dy
    half = line_run_px / 2.0 + 2.0 * scale
    top = np.clip((ys - half).astype(int), 0, H)
    bot = np.clip((ys + half).astype(int) + 1, 0, H)
    rows = top[None, :] + np.arange(max(int((bot - top).max()), 0))[:, None]
    on = ((ink[np.minimum(rows, H - 1), xs] != 0) & (rows < bot)).any(axis=0)
    gap, xm = max(1, int(round(3 * scale))), int(min(max(x_mid, 0), W - 1))
    return _reach(on[xm:], gap) + _reach(on[xm::-1], gap)


def _partner_stroke(ink, glyph, line_ends_px, line_run_px, x0_px, x1_px, dys, min_run_px=0.0,
                    len_ratio=0.0, scale=1.0):
    """The offset (px, + = down) of a second stroke beside the line, or 0: the nearest of `dys`
    whose band (the line's thickness) holds an unbroken run of ink over STROKE_SNAP_MIN_RUN of
    columns x0_px to x1_px and at least `min_run_px` long, the line's own band holding one too,
    while the rows between the two bands hold `glyph` ink in at least INK_MIN_FRAC of the columns
    and the row midway is not a stroke too. That band is the first on the stroke to pass: usually
    the one reaching its edge nearest the line, so the offset falls short of the stroke's middle by
    up to about the line's thickness; when nearer bands are not in `dys` or fail the test for glyph
    ink between, it lies further onto the stroke, as far as its far edge. Two strokes with glyph
    ink between them run through the glyphs: a double strike. A double rule has
    nothing between its lines, one thick stroke has no gap, and the bars of one glyph (P, B, E, =)
    are shorter than `min_run_px`. With `len_ratio`, the two strokes must also run about as long as
    each other, the shorter at least that share of the longer (:func:`_stroke_length` through the
    line and through that band, from the middle of x0_px to x1_px, `scale` the dpi over
    RENDER_DPI): when the nearest stroke that passes the rest does not, there is no partner."""
    (sx, sy), (ex, ey) = line_ends_px
    H, W = ink.shape
    if not len(dys) or not np.isfinite([sx, sy, ex, ey, line_run_px]).all():
        return 0
    x0 = max(int(x0_px), int(min(sx, ex)), 0)
    x1 = min(int(x1_px), int(max(sx, ex)), W - 1)
    if x1 <= x0:
        return 0
    xs = np.arange(x0, x1 + 1)
    ys = sy + ((xs - sx) / (ex - sx) if ex != sx else np.zeros(len(xs))) * (ey - sy)
    half = line_run_px / 2.0
    bands = np.array(sorted({0, *(int(d) for d in dys)}))
    tops = (ys[None, :] + bands[:, None] - half).astype(int)    # (band, column): rows top..bot-1
    bots = (ys[None, :] + bands[:, None] + half).astype(int) + 1
    first = tops[0]
    rows = first[None, :] + np.arange(int((bots - first).max()))[:, None]
    inside = (rows >= 0) & (rows < H)
    at = (rows[inside], np.broadcast_to(xs, rows.shape)[inside])
    cols = np.arange(len(xs))
    cums = []
    for mask in (ink, glyph):
        win = np.zeros(rows.shape, bool)
        win[inside] = mask[at] != 0
        cums.append(np.vstack([np.zeros((1, len(xs)), np.int32),
                               np.cumsum(win, axis=0, dtype=np.int32)]))
    cum, gcum = cums
    need = max(STROKE_SNAP_MIN_RUN * len(xs), min_run_px)

    def stroke(i):
        return _longest_run((cum[bots[i] - first, cols] - cum[tops[i] - first, cols]) > 0) >= need

    own = int(np.flatnonzero(bands == 0)[0])
    if not stroke(own):
        return 0
    for d in sorted(dys, key=abs):
        i = int(np.flatnonzero(bands == int(d))[0])
        if not stroke(i):
            continue
        lo, hi = (bots[own], tops[i]) if d > 0 else (bots[i], tops[own])
        between = (gcum[np.maximum(hi, lo) - first, cols] - gcum[lo - first, cols]) > 0
        mid = np.clip((ys + d / 2.0).astype(int), 0, H - 1)    # one thick stroke: no gap
        if between.mean() >= INK_MIN_FRAC and _longest_run(ink[mid, xs] != 0) < need:
            if len_ratio > 0:
                a, b = (_stroke_length(ink, line_ends_px, line_run_px, dy, (x0 + x1) / 2.0, scale)
                        for dy in (0, d))
                if min(a, b) < len_ratio * max(a, b):
                    return 0
            return int(d)
    return 0


def _own_stroke(ink, line_ends_px, line_run_px, x0_px, x1_px, beside_ink=None):
    """(run, excess) for a word under a line. `run` is the longest unbroken stretch of band ink as
    a fraction of the word's columns: a stroke through the word is one stretch, while glyph ink at
    that height breaks between letters. `excess` is the band's ink fraction minus the larger of the
    same band one stroke height plus 3 px above and below: a stroke adds ink at its own height, a
    glyph's crossbar or bowl has as much just beside it. The bands beside it are read on
    `beside_ink` when given (glyph ink, :func:`_glyph_ink`, so a rule or underline just above or
    below is not taken for a glyph's ink), else on `ink`."""
    band = _band_ink(ink, line_ends_px, line_run_px, x0_px, x1_px)
    if not len(band):
        return 0.0, 0.0
    best = cur = 0
    for v in band:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    d = line_run_px + 3
    beside_ink = ink if beside_ink is None else beside_ink
    beside = max(_band_ink(beside_ink, line_ends_px, line_run_px, x0_px, x1_px, dy=s * d).mean()
                 for s in (-1, 1))
    return best / len(band), float(band.mean() - beside)


def _is_printed_rule(ln):
    """A detected line whose appearance is a drawn rule (solid and/or dead-straight), not a strike.
    Keyed on stored line geometry only (no raster access). See the PRINTED_RULE_* constants.

    A True answer DROPS the detection, so a missing or None metric must read as "not a rule" — both
    directions here fail safe. ``classify_lines`` accepts caller-built line dicts, where reading an
    absent `fill` as a high number would discard every detection.
    """
    fill, wobble = ln.get("fill"), ln.get("straightness")
    return ((fill is not None and fill > PRINTED_RULE_FILL_MAX)
            or (wobble is not None and wobble < PRINTED_RULE_STRAIGHT_MAX))


def classify_lines(lines, words, gray, ink=None, config=ScanConfig.confidence_free(),
                   dpi=RENDER_DPI, glyph_ink=None):
    """(tagged_lines, struck_words) from detected strokes + OCR words.
    `words`: list of ocr.Word (bbox in [0,1] fractions). `gray`: the page raster the lines were
    detected on (uint8 HxW), at `dpi`. Returns per-WORD struck records with tier in
    auto/review/weak. With ``config.glyph_ink`` on, the ink tests read `glyph_ink`, a bool mask;
    None builds it from `ink` at `dpi`. `config` defaults to ``ScanConfig.confidence_free()``;
    pass ``ScanConfig.azure_di()`` for words from Azure Document Intelligence.

    A line dict without a `fill` (a caller-built line) never loses a word for it: it is not taken
    for a solid table rule or put through the chain gate, and it scores as FILL_UNMEASURED.

    A word the ink tests drop is tried again at its stroke's own row, within STROKE_SNAP_PX of the
    line (scaled with `dpi`): a stitched line can run a few px off a thin or light stroke. Under a
    thick line it is then tried with the ink beside the line read from INK_GAP_THICK_PX px past it,
    not INK_GAP_PX: a thick bar leaves only a few rows of the x-height beside it. Under a line
    that wobbles (INK_GAP_THICK_STEM_WOBBLE), or whose wobble is not measured, only when the line
    is at least INK_GAP_THICK_STEM_RATIO times as thick as the median ink run in the rows that
    reading adds: beside a pen stroke they are its own uneven edge and strokes of the same pen.
    Last, it passes when a second stroke about as long runs beside the line with glyph ink between
    the two and their middle in the band (PAIR_REACH, :func:`_partner_stroke`): a double strike,
    one of whose lines the finder gave. Only under a line at most PAIR_MAX_LEN_IN long and
    PAIR_MAX_RUN_PX thick whose wobble is measured and under PAIR_MAX_WOBBLE: in handwriting the
    ruling of lined paper or a pen stroke would make a pair with another stroke beside it.

    A word the OCR-confidence rescue passes goes through the same retries (RESCUE_RETRIES),
    whether the rescue first passes it at the line or at the stroke's row, and one is taken only
    when it gives strike geometry; otherwise the rescue stands where it first passed the word. A
    partial word that a retry moves off the middle, past PARTIAL_ISO_MAX_OFF, keeps the retry
    beside a strong hit on its line and its rescued pass on a line with none."""
    pix_h, pix_w = gray.shape
    if ink is None:
        ink = ink_mask(gray)
    if config.glyph_ink:
        side_ink = _glyph_ink(ink, dpi) if glyph_ink is None else glyph_ink
    else:
        side_ink = ink
    ws = [(w.bbox, w.text, w.confidence) for w in words if (w.text or "").strip()]
    confs = [c for (_, _, c) in ws if c is not None]
    edit_prior = (sum(1 for c in confs if c <= 0.90) / len(confs)) if confs else 0.0
    gating = config.confidence_gating and bool(confs)
    snap_px = int(round(STROKE_SNAP_PX * dpi / RENDER_DPI))

    tagged, struck_words = [], []
    for li, ln in enumerate(lines):
        # Decided before any attribution work: a drawn rule gets no words, so there is nothing to
        # attribute. NOTE `tagged` must stay index-aligned with `lines` — score_struck() looks lines
        # up as tagged[line_idx] — so every branch appends exactly one entry per input line.
        if config.veto_printed_rules and _is_printed_rule(ln):
            tagged.append({**ln, "label": "rule", "struck": [], "rel": None, "words": []})
            continue

        x0, y0, x1, y1 = ln["bbox_px"]
        lx0, lx1 = x0 / pix_w, x1 / pix_w
        llen = max(lx1 - lx0, 1e-9)
        # endpoints for interpolating the stroke-y over each word, at the center of its overlap with
        # the line (a sloped strike sits at a different height over each word — a single global
        # center mis-attributes them all).
        ends = ln.get("ends_px") or ((x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2))
        (sx, sy), (ex, ey) = ends
        short_line = ln.get("len_in", 0.0) < INK_SHORT_LEN_IN
        # both fill tests can drop words, so an unmeasured fill triggers neither
        # (see _is_printed_rule)
        line_fill = ln.get("fill")
        long_solid = (line_fill is not None and line_fill >= FILL_STRONG
                      and ln.get("len_in", 0.0) >= TABLE_RULE_MIN_LEN_IN)

        def make_hit(wbox, txt, off, wcov, strong, ink_ok=False, conf=None, strike_geom=False,
                     wends=None):
            wends = ends if wends is None else wends      # the line, or its stroke's row
            wx0, wy0, wx1, wy1 = wbox
            n = len(txt)
            # the box's share of the line, rounded outward: what PARTIAL_MIN_CHARS gates on, and
            # the span when the box holds no glyph ink to measure against
            f0 = (max(lx0, wx0) - wx0) / max(wx1 - wx0, 1e-9)
            f1 = (min(lx1, wx1) - wx0) / max(wx1 - wx0, 1e-9)
            c0 = int(np.floor(f0 * n))
            c1 = max(c0 + 1, int(np.ceil(f1 * n)))
            box_chars = c1 - c0
            wpx = (wx0 * pix_w, wy0 * pix_h, wx1 * pix_w, wy1 * pix_h)
            run = ln.get("run_px", 3.0)
            glyphs = _glyph_ink_extent(ink, wends, min(run, THIN_STROKE_RUN_PX * dpi / RENDER_DPI),
                                       wpx)
            if glyphs is not None:  # the glyph ink: an end character needs CHAR_STRUCK_FRAC of it
                gx0, gx1 = glyphs
                cw = max(gx1 - gx0, 1.0) / n                 # estimated character width
                reach = _stroke_reach(ink, wends, run, (x0, x1), wpx, STROKE_RUN_MIN_CHARS * cw,
                                      STROKE_PIECE_MIN_CHARS * cw, STROKE_LINK_MIN_CHARS * cw,
                                      STROKE_BRIDGE_CHARS * cw)
                rx0, rx1 = reach if reach is not None else (x0, x1)
                f0 = min(1.0, max(0.0, (rx0 - gx0) / (cw * n)))
                f1 = min(1.0, max(0.0, (rx1 - gx0) / (cw * n)))
                c0 = min(n - 1, max(0, int(np.ceil(f0 * n - (1.0 - CHAR_STRUCK_FRAC) - 1e-9))))
                c1 = min(n, max(c0 + 1, int(np.floor(f1 * n - CHAR_STRUCK_FRAC + 1e-9)) + 1))
            return {"text": txt, "chars": txt[c0:c1], "char_span": (c0, c1), "strong": strong,
                    "bbox_frac": wbox, "off": round(off, 2), "wcov": round(wcov, 2),
                    "cover_frac": (round(f0, 2), round(f1, 2)), "line_idx": li, "ink_ok": ink_ok,
                    "conf": conf, "strike_geom": strike_geom, "glyph_span": glyphs is not None,
                    "box_chars": box_chars,
                    # the stroke's row, where a retry moved off the line (line support reads it)
                    **({"stroke_ends": wends} if wends != ends else {})}

        run_px = ln.get("run_px", 3.0)
        wobble = ln.get("straightness")

        def ink_gate(off, wends, wpx, conf, gap=None, skip_full=False):
            """(passes, in_band, both_ink) for the line at `wends` through the word at `wpx`, the
            ink beside it read from `gap` px past its edges."""
            in_band = abs(off) <= STRIKE_TOL
            # Through-glyph ink at this word (see _ink_above_below), for EVERY hit: if in-band
            # lines skipped it, a full-width table rule would ride high fill straight to 'auto'.
            # `both_ink` requires substantial ink on both sides; `one_sided` is the underline/rule
            # signature (well on one side, ~none on the other) and is kept distinct so a real
            # strike over SPARSE text still reads as two-sided, not one.
            above, below = _ink_above_below(side_ink, wends, run_px, wpx, gap, skip_full)
            both_ink = above >= INK_MIN_FRAC and below >= INK_MIN_FRAC
            one_sided = _one_sided(above, below)
            if long_solid and config.glyph_ink and max(above, below) < INK_ONESIDE_MAX:
                # No glyph ink on either side: judge it on all ink, where the partner of a double
                # rule, which glyph ink leaves out, is the ink on its one side. On the thick retry
                # too this reads from INK_GAP_PX: from the bar's edges, a rule just past it would
                # read as ink beside it.
                one_sided = _one_sided(*_ink_above_below(ink, wends, run_px, wpx))
            if (not in_band) or short_line:
                # OCR damage is the tiebreaker — only meaningful when confidences are calibrated,
                # so the rescue is off under confidence_free()
                rescue = (config.confidence_gating and in_band
                          and conf is not None and conf <= config.inkfail_max_conf)
                return both_ink or rescue, in_band, both_ink
            # A long, solid (high-fill) in-band line with ink on only ONE side is a table rule or
            # underline, not a strike. Real strikes SHATTER on the glyphs (fill < FILL_STRONG) and
            # keep ink on BOTH sides, so they are spared.
            return not (long_solid and one_sided), in_band, both_ink

        hits, weak_band = [], []
        best = None
        for (wx0, wy0, wx1, wy1), txt, conf in ws:
            ov = min(lx1, wx1) - max(lx0, wx0)
            if ov <= 0:
                continue
            wcov = ov / max(wx1 - wx0, 1e-9)
            lcov = ov / llen
            xm = (max(x0, wx0 * pix_w) + min(x1, wx1 * pix_w)) / 2.0   # word/line x-overlap center
            t = 0.0 if ex == sx else min(1.0, max(0.0, (xm - sx) / (ex - sx)))
            lcy = (sy + t * (ey - sy)) / pix_h                         # stroke-y here, interpolated
            rel = (lcy - wy0) / max(wy1 - wy0, 1e-9)
            off = rel - 0.5
            if best is None or abs(off) < best[0]:
                best = (abs(off), off, txt, round(rel, 2), wcov)
            if abs(off) > INK_MAX_OFF or wcov < PARTIAL_MIN_WCOV:
                continue
            wpx = (wx0 * pix_w, wy0 * pix_h, wx1 * pix_w, wy1 * pix_h)
            wends = ends
            passes, in_band, both_ink = ink_gate(off, ends, wpx, conf)
            # A pass that rests on the confidence rescue alone gives the word no strike geometry,
            # so the DI-confidence veto can still drop it. The retries run for it too, and one is
            # taken only when it gives strike geometry (in the band with ink on both sides, or a
            # pair), and then sets `off` and the row the span is read at as for any word a retry
            # passes; a partial word's stroke-row retry is not taken when it would move the
            # word past PARTIAL_ISO_MAX_OFF, where an isolated partial strike is dropped.
            # Otherwise the rescue stands as it was, at the line (a retry can move the word out of
            # the band where the rescue applies). A rescue that first passes the word at the
            # stroke's row is held the same way, so the retries after it still run.
            held = None                      # the rescue's pass, while the retries run
            alt = None                       # a held partial word's pass, for a line with no
                                             # strong hit
            if RESCUE_RETRIES and passes and not both_ink and short_line:
                held, passes = (in_band, both_ink, off, wends), False
            if not passes and snap_px:
                # The line may run a few px off the stroke (a thin or light one, stitched to glyph
                # pieces), so the ink "below" it is under the glyphs. Try again at the stroke's row.
                dy = _stroke_row_dy(ink, ends, run_px, max(x0, wpx[0]), min(x1, wpx[2]), snap_px)
                if dy:
                    wends = ((sx, sy + dy), (ex, ey + dy))
                    off += dy / max(wpx[3] - wpx[1], 1e-9)
                    passes, in_band, both_ink = ink_gate(off, wends, wpx, conf)
                    if RESCUE_RETRIES and passes and not both_ink and short_line:
                        held, passes = held or (in_band, both_ink, off, wends), False
                    elif held:
                        passes = passes and in_band and both_ink
                        # a partial word is kept off the middle only beside a strong hit on its
                        # line, so where none is found the rescue's pass stands instead (`off`
                        # compared rounded, as a hit stores it and that filter reads it)
                        if (passes and wcov < MIN_WORD_XOVER
                                and abs(round(off, 2)) > PARTIAL_ISO_MAX_OFF
                                >= abs(round(held[2], 2))):
                            alt = held
            if (not passes and in_band and wcov >= MIN_WORD_XOVER and config.glyph_ink
                    and INK_GAP_THICK_PX < INK_GAP_PX
                    and ln.get("len_in", 0.0) <= INK_GAP_THICK_MAX_LEN_IN
                    and run_px > max(2.0, THIN_STROKE_RUN_PX * dpi / RENDER_DPI)
                    and (INK_GAP_THICK_STEM_RATIO <= 0
                         or (wobble is not None and wobble < INK_GAP_THICK_STEM_WOBBLE)
                         or run_px >= INK_GAP_THICK_STEM_RATIO
                         * (_edge_runs(side_ink, wends, run_px, wpx, INK_GAP_THICK_PX, INK_GAP_PX)
                            or float("inf")))):
                # A thick bar through the middle of the x-height covers most of it: the few rows of
                # glyph ink beside it lie within the thin stroke's gap. Try again with the ink read
                # from closer in, for a line across the word: an underline or a rule over the text
                # is out of the band, and a glyph chain in bold text crosses a letter or two. Rows
                # just past the band inked across the word are the stroke's own (the line's y can be
                # a fraction of a px off) and are skipped. Glyph ink only: on all ink, a rule beside
                # the bar would read as ink beside it. And for a line that wobbles, or whose wobble
                # is not measured, only when it is clearly thicker than the ink runs in the rows the
                # closer reading adds: under a printed bar they are the stems of the struck letters,
                # beside a pen stroke its own uneven edge and strokes of the same pen, which would
                # read as glyph ink on both sides.
                passes, in_band, both_ink = ink_gate(off, wends, wpx, conf, INK_GAP_THICK_PX,
                                                     skip_full=True)
                if RESCUE_RETRIES and passes and not both_ink and short_line:
                    held, passes = held or (in_band, both_ink, off, wends), False
                elif held:
                    passes = passes and in_band and both_ink
            if (not passes and PAIR_REACH > 0 and wcov >= MIN_WORD_XOVER and config.glyph_ink
                    and (PAIR_MAX_LEN_IN <= 0 or ln.get("len_in", 0.0) <= PAIR_MAX_LEN_IN)
                    and (PAIR_MAX_RUN_PX <= 0
                         or run_px <= max(2.0, PAIR_MAX_RUN_PX * dpi / RENDER_DPI))
                    and (PAIR_MAX_WOBBLE <= 0
                         or (wobble is not None and wobble < PAIR_MAX_WOBBLE))):
                # A double strike: the finder gives one of its lines, off the middle of the
                # x-height, and the rows of glyph ink outside it lie within the gap. Passed when
                # a second stroke about as long runs beside the line with glyph ink between the
                # two, and their middle is in the band, the second stroke taken at the nearest
                # band on it that puts the middle there and passes the rest: usually its edge
                # nearest the line, or, when the line lies out of the band on the side away from
                # it, further onto it, up to its far edge. `off` moves to that middle; spans are
                # still read at the line, or at the stroke's row where the retry above moved it.
                # The line's own band must be a stroke, so a line stitched beside one is paired
                # only after that move. Glyph ink only: on all ink, a thick bar between two rules
                # has its own rows between them. Not under a line across the page, the ruling of
                # lined paper, nor a thick or wobbly one, a pen stroke: in handwriting either one
                # with another stroke beside it would make a pair.
                box_h = max(wpx[3] - wpx[1], 1e-9)
                reach, near = int(PAIR_REACH * box_h), int(np.ceil(run_px)) + 2
                dys = [d for d in range(-reach, reach + 1)
                       if abs(d) >= near and abs(off + d / (2 * box_h)) <= STRIKE_TOL]
                pair = _partner_stroke(ink, side_ink, wends, run_px, max(x0, wpx[0]),
                                       min(x1, wpx[2]), dys, STROKE_RUN_MIN_CHARS
                                       * (wpx[2] - wpx[0]) / max(len(txt), 1),
                                       PAIR_LEN_RATIO, dpi / RENDER_DPI)
                if pair:
                    off += pair / (2 * box_h)
                    passes = in_band = both_ink = True
            if held and not passes:
                (in_band, both_ink, off, wends), passes = held, True
            if not passes:
                continue
            ink_ok = not in_band
            # Strike-geometry corroboration: a genuine strike crosses the x-height (in-band)
            # and keeps substantial ink on BOTH sides, solid or broken up, or is a double strike's
            # pair. The DI-confidence veto (detect.apply_cnn_verdict) spares a high-OCR-confidence
            # word only with it.
            strike_geom = in_band and both_ink
            wbox = (wx0, wy0, wx1, wy1)
            if wcov >= MIN_WORD_XOVER:
                hits.append(make_hit(wbox, txt, off, wcov, strong=True, ink_ok=ink_ok, conf=conf,
                                     strike_geom=strike_geom, wends=wends))
            elif lcov >= PARTIAL_MIN_LCOV and len(txt) >= 2:
                h = make_hit(wbox, txt, off, wcov, strong=False, ink_ok=ink_ok, conf=conf,
                             strike_geom=strike_geom, wends=wends)
                if h["box_chars"] >= PARTIAL_MIN_CHARS:
                    if alt:
                        a_in, a_both, a_off, a_wends = alt
                        h["_alt"] = make_hit(wbox, txt, a_off, wcov, strong=False, ink_ok=not a_in,
                                             conf=conf, strike_geom=a_in and a_both,
                                             wends=a_wends)
                    hits.append(h)
            else:
                weak_band.append((wbox, txt, off, wcov, ink_ok, conf, strike_geom, wends))
        if any(h["strong"] for h in hits):
            for h in hits:
                h.pop("_alt", None)
            for wbox, txt, off, wcov, ink_ok, conf, strike_geom, wends in weak_band:
                h = make_hit(wbox, txt, off, wcov, strong=False, ink_ok=ink_ok, conf=conf,
                             strike_geom=strike_geom, wends=wends)
                if h["box_chars"] >= PARTIAL_MIN_CHARS:
                    hits.append(h)
        else:
            hits = [h.pop("_alt", None) or h for h in hits]
            hits = [h for h in hits if abs(h["off"]) <= PARTIAL_ISO_MAX_OFF or h["ink_ok"]]
        chain = False
        if hits and line_fill is not None and line_fill < FILL_STRONG and gating:
            # marginal spine fill: real strike or serif glyph chain? A real strike corrupts what
            # it crosses -> at least one struck word must read with degraded OCR confidence.
            min_conf = min((h["conf"] if h["conf"] is not None else 1.0) for h in hits)
            if min_conf > config.max_clean_conf:
                n_strong = sum(1 for h in hits if h["strong"])
                substantial = n_strong >= 2 or (
                    n_strong >= 1 and ln.get("len_in", 0) >= TWIN_MIN_LEN_IN)
                # The escape: on a page that looks pen-edited, a substantial clean-OCR line might
                # still be a real strike, so let the CNN call it. Turn it OFF on degraded ruled
                # forms: there `edit_prior` reads scan quality rather than edits (it is
                # anti-correlated with real edits on the corpus) and the saturated CNN confirms
                # whatever it is given.
                if (config.rescue_clean_chains and substantial
                        and edit_prior >= config.page_edited_min):
                    for h in hits:
                        h["twin"] = True       # edited-page pixel twin: the CNN decides (review)
                else:
                    hits, chain = [], True
        if hits:
            label, rel = "strike", None
        elif chain:
            label, rel = "chain", None
        elif best is None:
            label, rel = "rule", None
        else:
            _, off, txt, rel, _ = best
            label = "underline" if off > STRIKE_TOL else ("over" if off < -STRIKE_TOL else "strike")
        tagged.append({**ln, "label": label, "struck": hits, "rel": rel,
                       "words": [h["chars"] for h in hits]})
        struck_words.extend(hits)
    # scored with the confidence term only when the chain gate could use confidences: with none on
    # the page, the term would cap every score at 0.60
    return tagged, consolidate_struck(struck_words, tagged, use_conf=gating)


def consolidate_struck(hits, tagged, use_conf=True):
    """One record per WORD: union the char spans of all hits for that word (the detector often
    returns several overlapping fragments of one physical strike). Full vs partial is decided
    from the UNIONED char coverage: FULL_CHAR_COVER of the characters, and, when a hit's span was
    measured against glyph ink, the first and last character too. `use_conf=False` scores
    without the OCR-confidence term."""
    by_word = {}
    for h in hits:
        by_word.setdefault((h["bbox_frac"], h["text"]), []).append(h)
    out = []
    for (bbox, txt), hs in by_word.items():
        spans = sorted(h["char_span"] for h in hs)
        merged = [list(spans[0])]
        for c0, c1 in spans[1:]:
            if c0 <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], c1)
            else:
                merged.append([c0, c1])
        best = max(merged, key=lambda m: m[1] - m[0])
        covered = sum(m1 - m0 for m0, m1 in merged)
        full = covered >= FULL_CHAR_COVER * max(len(txt), 1)
        if full and any(h.get("glyph_span") for h in hs):     # the strokes reach both ends
            full = merged[0][0] == 0 and merged[-1][1] >= len(txt)
        c0, c1 = (0, len(txt)) if full else (best[0], best[1])
        conf = min((h["conf"] for h in hs if h["conf"] is not None), default=None)
        rec = {
            "text": txt, "chars": txt[c0:c1], "char_span": (c0, c1), "partial": not full,
            "bbox_frac": bbox, "conf": conf,
            "off": round(float(np.median([h["off"] for h in hs])), 2),
            "wcov": round(max(h["wcov"] for h in hs), 2),
            "line_idx": sorted({h["line_idx"] for h in hs}),
        }
        rec["twin"] = any(h.get("twin") for h in hs)
        # True when any fragment carries genuine strike geometry (in-band with ink on both sides,
        # or a double strike's pair): the DI-confidence veto keeps it even at high OCR confidence.
        rec["geom_corroborated"] = any(h.get("strike_geom") for h in hs)
        rec["score"] = score_struck(rec, tagged, use_conf=use_conf)
        rec["tier"] = ("auto" if rec["score"] >= AUTO_SCORE
                       else ("review" if rec["score"] >= REVIEW_SCORE else "weak"))
        if rec["twin"] and rec["tier"] == "auto":
            rec["tier"] = "review"
        out.append(rec)
    out.sort(key=lambda h: (round(h["bbox_frac"][1], 3), h["bbox_frac"][0]))
    return out


def score_struck(rec, tagged, use_conf=True):
    """Evidence score in [0,1]: OCR damage + line length + spine fill + row support. Words
    geometry can't decide (short + clean OCR) land in 0.20-0.55 -> tier 'review' for the CNN.
    With `use_conf=False` (confidence-free engines) the OCR-damage term is dropped and the
    geometry terms renormalize to keep the same [0,1] range and tier thresholds."""
    lines = [tagged[li] for li in rec["line_idx"]]
    length = max(l.get("len_in", 0.0) for l in lines)   # optional on caller-built lines
    fill = max(FILL_UNMEASURED if l.get("fill") is None else l["fill"] for l in lines)
    row_support = max(sum(1 for h in l["struck"] if h["strong"]) for l in lines)
    s = 0.25 * min(1.0, length / 0.75)
    s += 0.15 * min(1.0, max(0.0, (fill - 0.70) / 0.28))
    s += 0.20 * min(1.0, row_support / 3.0)
    if use_conf:
        conf = rec["conf"] if rec["conf"] is not None else 1.0
        s += 0.40 * min(1.0, max(0.0, (0.97 - conf) / 0.12))
    else:
        s /= 0.60
    return round(s, 2)


def promote_context_orphans(struck, words):
    """Visual-row consistency pass: when nearly everything in a row is struck, promote the few
    survivor words to tier='review' (orphan=True) so the CNN inspects them — they carry no line
    evidence of their own and never go 'auto'."""
    MIN_GROUP, ORPHAN_MIN_FRAC, ORPHAN_EDGE_FRAC, ORPHAN_MAX_RUN = 5, 0.65, 0.80, 4
    ws = [(w.bbox, w.text or "", w.confidence) for w in words if (w.text or "").strip()]
    if not ws:
        return struck
    struck_keys = {(h["bbox_frac"], h["text"]) for h in struck}
    med_h = float(np.median([b[3] - b[1] for b, _, _ in ws]))

    groups, row, row_y = [], [], None      # cluster words into visual rows by y-center
    for it in sorted(ws, key=lambda w: ((w[0][1] + w[0][3]) / 2, w[0][0])):
        yc = (it[0][1] + it[0][3]) / 2
        if row_y is None or abs(yc - row_y) <= 0.6 * med_h:
            row.append(it)
            row_y = yc if row_y is None else 0.8 * row_y + 0.2 * yc
        else:
            groups.append(sorted(row, key=lambda w: w[0][0]))
            row, row_y = [it], yc
    if row:
        groups.append(sorted(row, key=lambda w: w[0][0]))

    promoted, seen = [], set(struck_keys)
    for grp in groups:
        n = len(grp)
        if n < MIN_GROUP:
            continue
        flags = [(w[0], w[1]) in struck_keys for w in grp]
        frac = sum(flags) / n
        if frac < ORPHAN_MIN_FRAC:
            continue
        max_run = min(ORPHAN_MAX_RUN, max(2, int(np.ceil(0.2 * n))))
        i = 0
        while i < n:
            if flags[i]:
                i += 1
                continue
            j = i
            while j < n and not flags[j]:
                j += 1
            run, interior = j - i, (i > 0 and j < n)
            if (interior and run <= max_run) or (frac >= ORPHAN_EDGE_FRAC and run <= 2):
                for k in range(i, j):
                    wbox, txt, conf = grp[k]
                    key = (wbox, txt)
                    if key in seen or not txt.strip():
                        continue
                    seen.add(key)
                    promoted.append({
                        "text": txt, "chars": txt, "char_span": (0, len(txt)), "partial": False,
                        "bbox_frac": wbox, "conf": conf, "off": None, "wcov": 0.0,
                        "line_idx": [], "twin": False, "orphan": True, "geom_corroborated": False,
                        "score": round(0.22 + 0.18 * frac, 2), "tier": "review",
                    })
            i = j
    out = struck + promoted
    out.sort(key=lambda h: (round(h["bbox_frac"][1], 3), h["bbox_frac"][0]))
    return out


def analyze_scanned_page(gray, words, ink=None, config=ScanConfig.confidence_free(),
                         dpi=RENDER_DPI):
    """Full layer-1 pass on one scanned page: detect strokes, classify per word, promote orphans.
    `gray`: page raster (uint8 HxW; RGB/float inputs are coerced). `words`: list of ocr.Word.
    `dpi` must be the resolution the raster was rendered at. `config` defaults to
    ``ScanConfig.confidence_free()``; pass ``ScanConfig.azure_di()`` for words from Azure Document
    Intelligence. Returns (tagged_lines, struck)."""
    gray = to_gray_u8(gray)
    if ink is None:
        ink = ink_mask(gray)
    lines = strike_lines(gray, dpi=dpi, ink=ink)
    tagged, struck = classify_lines(lines, words, gray, ink=ink, config=config, dpi=dpi)
    struck = promote_context_orphans(struck, words)
    return tagged, struck
