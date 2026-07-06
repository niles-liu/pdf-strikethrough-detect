"""Scanned word-level strike classification: attach detected strokes to OCR words.

Given the strokes from :func:`pdf_strikethrough.lines.strike_lines` and a list of
:class:`pdf_strikethrough.ocr.Word` (from any OCR backend), decide which words are struck, with
char-level spans and full/partial resolution. This is layer 1 (geometry + OCR); the CNN verdict
(layer 2) is applied by the orchestrator in :mod:`pdf_strikethrough.detect`.

Ported from a corpus-validated pipeline; decoupled from Azure Document Intelligence — the only
DI-specific piece was the confidence calibration, now captured in :class:`ScanConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lines import RENDER_DPI, ink_mask, strike_lines, to_gray_u8

# --- per-word classification tunables (geometry; engine-independent) ---
STRIKE_TOL          = 0.16   # half-width of the strike band (fraction of word-box height)
MIN_WORD_XOVER      = 0.45   # full-word strike: line covers >= this fraction of the word width
PARTIAL_MIN_LCOV    = 0.70   # partial strike: >= this fraction of the LINE lies inside the word
PARTIAL_MIN_CHARS   = 2      # partial strike must span >= this many estimated characters
PARTIAL_MIN_WCOV    = 0.12   # ...and >= this fraction of the word width
PARTIAL_ISO_MAX_OFF = 0.08   # isolated partial (line strikes nothing else): must sit dead-center
INK_MAX_OFF         = 0.40   # pixel test still considered up to this |off| (box-lies rescue)
INK_MIN_FRAC        = 0.35   # min fraction of covered columns with glyph ink above resp. below
INK_SHORT_LEN_IN    = 0.50   # pixel test REQUIRED only for lines shorter than this (+ rescues)
FILL_STRONG         = 0.87   # fill >= this: line accepted on geometry alone
TWIN_MIN_LEN_IN     = 0.60   # substantial line: >=2 fully-struck words, or one word + long line
FULL_CHAR_COVER     = 0.70   # unioned char coverage >= this -> whole word counts as struck

AUTO_SCORE = 0.55            # score >= this -> tier 'auto'; >= REVIEW_SCORE -> 'review'; else 'weak'
REVIEW_SCORE = 0.20


@dataclass(frozen=True)
class ScanConfig:
    """Confidence calibration for the scanned classifier. Defaults match Azure Document
    Intelligence (struck words OCR at 0.43-0.94, clean text at 0.976-1.0).

    ``confidence_free()`` disables EVERY confidence-dependent decision: the chain-rejection
    gate, the ink-fail rescue, and the confidence term in the evidence score (the remaining
    geometry terms are renormalized so the auto/review/weak thresholds keep their meaning).
    Use it for engines whose confidence doesn't separate struck from clean text (e.g. RapidOCR);
    detection then rests on geometry + the CNN alone.

    ``recall_first()`` / ``precision_first()`` pick the CNN *operating point* — the strike/clean
    decision threshold applied to StrikeNet's probability. ``cnn_p_hi``/``cnn_p_lo`` (None = the
    model's shipped thresholds) override it; calibrate them from labeled data with
    :mod:`pdf_strikethrough.calibration`."""
    confidence_gating: bool = True
    max_clean_conf: float = 0.955     # fill<FILL_STRONG: some struck word must OCR at or below this
    inkfail_max_conf: float = 0.974   # a pixel-failing in-band hit is rescued if OCR is this damaged
    page_edited_min: float = 0.03     # frac words conf<=0.90 >= this -> page has pen edits
    cnn_p_hi: float | None = None     # override the CNN struck threshold (operating point); None = model default
    cnn_p_lo: float | None = None     # override the CNN clean threshold; None = model default

    @classmethod
    def azure_di(cls):
        return cls()

    @classmethod
    def confidence_free(cls):
        """For engines whose confidence doesn't separate struck from clean (e.g. RapidOCR)."""
        return cls(confidence_gating=False)

    @classmethod
    def recall_first(cls, cnn_p_hi=0.50, **kw):
        """Recall-first operating point (legal / audit review): bias toward catching every
        deletion. Lowers the CNN's struck-confirmation bar so a borderline strike still counts as
        struck — more false positives and more manual review, but a deleted word is not silently
        missed. Combine with ``confidence_gating=False`` for a confidence-free OCR engine."""
        return cls(cnn_p_hi=cnn_p_hi, **kw)

    @classmethod
    def precision_first(cls, cnn_p_hi=0.97, **kw):
        """Precision-first operating point (RAG / indexing): bias toward never dropping live text
        as if it were deleted. Only high-confidence CNN strikes are reported; an uncertain strike
        falls back to 'unsure'/clean rather than being removed."""
        return cls(cnn_p_hi=cnn_p_hi, **kw)


def _ink_above_below(ink, line_ends_px, line_run_px, word_bbox_px, gap=2):
    """Fractions of covered columns with glyph ink above resp. below the stroke. A strike runs
    THROUGH glyphs -> ink both sides; an underline has ink above only; a glyph chain has ~none
    above. The stroke y is interpolated from the LINE ENDPOINTS at the word's x-midpoint."""
    (sx, sy), (ex, ey) = line_ends_px
    wx0, wy0, wx1, wy1 = word_bbox_px
    x0, x1 = max(sx, wx0), min(ex, wx1)
    if x1 <= x0:
        return 0.0, 0.0
    xm = (x0 + x1) / 2.0
    t = 0.0 if ex == sx else (xm - sx) / (ex - sx)
    ly = sy + t * (ey - sy)
    half = line_run_px / 2.0 + gap
    H, W = ink.shape
    top0, top1 = max(0, int(wy0)), max(0, int(ly - half))
    bot0, bot1 = min(H, int(ly + half) + 1), min(H, int(wy1) + 1)
    cols = slice(max(0, int(x0)), min(W, int(x1) + 1))
    above = ink[top0:top1, cols].any(axis=0).mean() if top1 > top0 else 0.0
    below = ink[bot0:bot1, cols].any(axis=0).mean() if bot1 > bot0 else 0.0
    return float(above), float(below)


def classify_lines(lines, words, gray, ink=None, config=ScanConfig()):
    """(tagged_lines, struck_words) from detected strokes + OCR words.
    `words`: list of ocr.Word (bbox in [0,1] fractions). `gray`: the page raster the lines were
    detected on (uint8 HxW). Returns per-WORD struck records with tier in auto/review/weak."""
    pix_h, pix_w = gray.shape
    if ink is None:
        ink = ink_mask(gray)
    ws = [(w.bbox, w.text, w.confidence) for w in words if (w.text or "").strip()]
    confs = [c for (_, _, c) in ws if c is not None]
    edit_prior = (sum(1 for c in confs if c <= 0.90) / len(confs)) if confs else 0.0
    gating = config.confidence_gating and bool(confs)

    tagged, struck_words = [], []
    for li, ln in enumerate(lines):
        x0, y0, x1, y1 = ln["bbox_px"]
        lx0, lx1 = x0 / pix_w, x1 / pix_w
        llen = max(lx1 - lx0, 1e-9)
        # endpoints for interpolating the stroke-y at each word's x-midpoint (a sloped strike sits
        # at a different height over each word — a single global center mis-attributes them all).
        (sx, sy), (ex, ey) = ln.get("ends_px") or ((x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2))
        short_line = ln.get("len_in", 0.0) < INK_SHORT_LEN_IN

        def make_hit(wbox, txt, off, wcov, strong, ink_ok=False, conf=None):
            wx0, wy0, wx1, wy1 = wbox
            f0 = (max(lx0, wx0) - wx0) / max(wx1 - wx0, 1e-9)
            f1 = (min(lx1, wx1) - wx0) / max(wx1 - wx0, 1e-9)
            c0 = int(np.floor(f0 * len(txt)))
            c1 = max(c0 + 1, int(np.ceil(f1 * len(txt))))
            return {"text": txt, "chars": txt[c0:c1], "char_span": (c0, c1), "strong": strong,
                    "bbox_frac": wbox, "off": round(off, 2), "wcov": round(wcov, 2),
                    "cover_frac": (round(f0, 2), round(f1, 2)), "line_idx": li, "ink_ok": ink_ok,
                    "conf": conf}

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
            in_band = abs(off) <= STRIKE_TOL
            if (not in_band) or short_line:
                wpx = (wx0 * pix_w, wy0 * pix_h, wx1 * pix_w, wy1 * pix_h)
                ends = ln.get("ends_px") or ((x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2))
                above, below = _ink_above_below(ink, ends, ln.get("run_px", 3.0), wpx)
                if above < INK_MIN_FRAC or below < INK_MIN_FRAC:
                    # OCR damage is the tiebreaker — only meaningful when confidences are
                    # calibrated, so the rescue is off under confidence_free()
                    rescue = (config.confidence_gating and in_band
                              and conf is not None and conf <= config.inkfail_max_conf)
                    if not rescue:
                        continue
            ink_ok = not in_band
            wbox = (wx0, wy0, wx1, wy1)
            if wcov >= MIN_WORD_XOVER:
                hits.append(make_hit(wbox, txt, off, wcov, strong=True, ink_ok=ink_ok, conf=conf))
            elif lcov >= PARTIAL_MIN_LCOV and len(txt) >= 2:
                h = make_hit(wbox, txt, off, wcov, strong=False, ink_ok=ink_ok, conf=conf)
                if h["char_span"][1] - h["char_span"][0] >= PARTIAL_MIN_CHARS:
                    hits.append(h)
            else:
                weak_band.append((wbox, txt, off, wcov, ink_ok, conf))
        if any(h["strong"] for h in hits):
            for wbox, txt, off, wcov, ink_ok, conf in weak_band:
                h = make_hit(wbox, txt, off, wcov, strong=False, ink_ok=ink_ok, conf=conf)
                if h["char_span"][1] - h["char_span"][0] >= PARTIAL_MIN_CHARS:
                    hits.append(h)
        else:
            hits = [h for h in hits if abs(h["off"]) <= PARTIAL_ISO_MAX_OFF or h["ink_ok"]]
        chain = False
        if hits and ln.get("fill", 1.0) < FILL_STRONG and gating:
            # marginal spine fill: real strike or serif glyph chain? A real strike corrupts what
            # it crosses -> at least one struck word must read with degraded OCR confidence.
            min_conf = min((h["conf"] if h["conf"] is not None else 1.0) for h in hits)
            if min_conf > config.max_clean_conf:
                n_strong = sum(1 for h in hits if h["strong"])
                substantial = n_strong >= 2 or (n_strong >= 1 and ln.get("len_in", 0) >= TWIN_MIN_LEN_IN)
                if edit_prior >= config.page_edited_min and substantial:
                    for h in hits:
                        h["twin"] = True       # pixel-twin on an edited page: CNN decides (->review)
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
    return tagged, consolidate_struck(struck_words, tagged, use_conf=config.confidence_gating)


def consolidate_struck(hits, tagged, use_conf=True):
    """One record per WORD: union the char spans of all hits for that word (the detector often
    returns several overlapping fragments of one physical strike). Full vs partial is decided
    from the UNIONED char coverage. `use_conf=False` scores without the OCR-confidence term."""
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
    length = max(l["len_in"] for l in lines)
    fill = max(l.get("fill", 0.8) for l in lines)
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
                        "line_idx": [], "twin": False, "orphan": True,
                        "score": round(0.22 + 0.18 * frac, 2), "tier": "review",
                    })
            i = j
    out = struck + promoted
    out.sort(key=lambda h: (round(h["bbox_frac"][1], 3), h["bbox_frac"][0]))
    return out


def analyze_scanned_page(gray, words, ink=None, config=ScanConfig(), dpi=RENDER_DPI):
    """Full layer-1 pass on one scanned page: detect strokes, classify per word, promote orphans.
    `gray`: page raster (uint8 HxW; RGB/float inputs are coerced). `words`: list of ocr.Word.
    `dpi` must be the resolution the raster was rendered at. Returns (tagged_lines, struck)."""
    gray = to_gray_u8(gray)
    if ink is None:
        ink = ink_mask(gray)
    lines = strike_lines(gray, dpi=dpi, ink=ink)
    tagged, struck = classify_lines(lines, words, gray, ink=ink, config=config)
    struck = promote_context_orphans(struck, words)
    return tagged, struck
