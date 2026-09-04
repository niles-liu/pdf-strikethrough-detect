"""OCR-free geometric strike-line detection on rasterized (scanned) pages.

This is the engine that finds the physical strokes — it needs no OCR and no PDF; give it a
grayscale image and it returns the near-horizontal straight-ish lines (strikes, underlines,
rules) with per-line geometry (fill, stroke run-thickness, angle, length).

Strikes over typed text SHATTER: hyphen-overtype strikes break at every char gap, and slightly
sloped pen strokes straddle two angle bins. So the pipeline extracts fragments PERMISSIVELY,
stitches collinear ones, and only then applies the strict filters to the stitched line:
  spine fill >= MIN_FILL              kills chains stitched from sparse glyph bits
  run-thickness p25 <= MAX_STROKE_RUN kills bold display-font crossbars
  MIN_LINE_LEN / angle                geometric sanity on the stitched line

Attributing lines to words (which word was struck, char spans, partial strikes) needs word
boxes from an OCR engine and lives in a separate layer.
"""
import numpy as np
from scipy import ndimage

# --- raster / detector tunables ---
RENDER_DPI        = 200           # reference raster resolution (px-per-inch the tunables assume)
MIN_LINE_LEN_IN   = 0.08          # min STITCHED line length (~16px@200dpi; '19'-class strikes ~0.12in)
SEG_LEN_IN        = 0.05          # straight-segment length for the orientation filter
GAP_BRIDGE_PX     = 8             # morphological closing along the angle (pen gaps within a fragment)
ANGLE_STEP_DEG    = 15            # sweep 0..180 in these steps
MAX_ANGLE_DEG     = 25            # near-horizontal only; drops vertical letter-stems (l, I, f, 1)
MAX_LINE_THICK_PX = 10            # pre-stitch PCA thickness cap
DEDUP_IOU         = 0.30          # merge near-duplicate detections from adjacent angles, keep longest
OTSU_OFFSET       = 15            # recall knob: ink if gray < otsu()+this (recovers faint strokes)
MIN_FILL          = 0.65          # fraction of the stitched spine that must be RAW ink
MAX_STROKE_RUN_PX = 4             # p25 of vertical ink-run lengths along the spine = stroke thickness
STITCH_GAP_PX     = 26            # max x-gap between fragments to stitch (~a word space + slack)
STITCH_DY_PX      = 6.0           # max |y| between endpoints at the junction
STITCH_DY_TIGHT   = 2.5           # fallback tolerance for re-stitching poisoned groups
PRE_LEN_PX        = 8             # permissive PRE-stitch minimums; strict tests run post-stitch
PRE_ASPECT        = 2.5


def otsu_threshold(gray):
    """Otsu's method: the gray level that maximizes between-class variance."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    levels = np.arange(256)
    sum_all = np.dot(levels, hist)
    wB = np.cumsum(hist)
    wF = total - wB
    sumB = np.cumsum(levels * hist)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = wB * wF * ((sumB / wB) - ((sum_all - sumB) / wF)) ** 2
    between[~np.isfinite(between)] = 0
    return int(np.argmax(between))


def to_gray_u8(image):
    """Coerce input to the uint8 grayscale (H, W) array the detectors expect. Accepts (H, W)
    grayscale or (H, W, 3|4) RGB(A) arrays; float images in [0, 1] are rescaled to 0..255; wide
    integer scans (16-bit and up) are rescaled from their dtype range instead of saturating to
    all-white; out-of-range values are clipped (no mod-256 wraparound)."""
    a = np.asarray(image)
    if a.ndim == 3 and a.shape[2] in (3, 4):
        # sRGB -> linear -> Rec.709 luminance -> sRGB, which is what PyMuPDF's csGRAY does on the
        # PDF path. A channel MEAN puts a yellow highlighter at 170 where csGRAY puts it at 248, so
        # the same highlighted page used to be readable through detect_pdf and solid ink through an
        # RGB array.
        c = a[..., :3].astype(np.float64)
        if np.issubdtype(a.dtype, np.floating) and c.size and c.max() <= 1.0:
            c = c * 255.0
        elif np.issubdtype(a.dtype, np.integer) and np.iinfo(a.dtype).max > 255:
            c = c * (255.0 / np.iinfo(a.dtype).max)
        c = np.clip(c, 0, 255) / 255.0
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        y = lin @ np.array([0.2126, 0.7152, 0.0722])
        a = np.where(y <= 0.0031308, y * 12.92, 1.055 * y ** (1 / 2.4) - 0.055) * 255.0
    if a.ndim != 2:
        raise ValueError(f"expected a (H, W) grayscale or (H, W, 3) RGB image, got shape {a.shape}")
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            if a.size and float(a.max()) <= 1.0:
                a = a * 255.0
        elif np.issubdtype(a.dtype, np.integer) and np.iinfo(a.dtype).max > 255:
            a = a.astype(np.float64) * (255.0 / np.iinfo(a.dtype).max)   # 16-bit -> 8-bit
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


BG_BLOCK_PX     = 64     # background-estimate grid for the shaded-page fallback below
BG_INK_MAX      = 0.35   # a global-Otsu mask inkier than this has split SHADING from paper


def ink_mask(gray):
    """Binarized ink mask: global Otsu + OTSU_OFFSET (clean on dense text, keeps faint strokes).

    One global threshold cannot serve a page holding both white and shaded regions: once a highlight
    block's ground is dark enough, Otsu splits page-from-block instead of ink-from-paper and the
    whole block comes back as ink (measured: ink fraction 0.07 at ground 211, 1.00 at 195, and every
    downstream stage then returns nothing). So when the mask comes back implausibly inky, flatten the
    paper block-wise and threshold that instead. The fallback is gated, not unconditional, because
    every geometry filter downstream is calibrated against the plain global mask."""
    gray = to_gray_u8(gray)
    mask = gray < (otsu_threshold(gray) + OTSU_OFFSET)
    if mask.mean() <= BG_INK_MAX:
        return mask
    H, W = gray.shape
    by, bx = max(1, H // BG_BLOCK_PX), max(1, W // BG_BLOCK_PX)
    ys = np.linspace(0, H, by + 1).astype(int)
    xs = np.linspace(0, W, bx + 1).astype(int)
    coarse = np.array([[np.percentile(gray[ys[i]:ys[i+1], xs[j]:xs[j+1]], 90) or 255.0
                        for j in range(bx)] for i in range(by)], dtype=np.float64)
    bg = ndimage.zoom(np.clip(coarse, 1, 255), (H / by, W / bx), order=1)[:H, :W]
    flat = np.clip(gray.astype(np.float64) * (255.0 / bg), 0, 255).astype(np.uint8)
    return flat < (otsu_threshold(flat) + OTSU_OFFSET)


def line_kernel(length, angle_deg):
    """A 1-px-wide straight structuring element of `length` px at `angle_deg`."""
    a = np.deg2rad(angle_deg)
    t = np.arange(max(1, length))
    xs = np.round(t * np.cos(a)).astype(int)
    ys = np.round(t * np.sin(a)).astype(int)
    xs -= xs.min(); ys -= ys.min()
    k = np.zeros((ys.max() + 1, xs.max() + 1), dtype=bool)
    k[ys, xs] = True
    return k


def _spine_fill(ink, center, u, length, halfwidth=2):
    """Fraction of steps along the major axis with raw ink within +/-halfwidth px perpendicular."""
    n = max(int(length), 2)
    ts = np.linspace(-length / 2.0, length / 2.0, n)
    v = np.array([-u[1], u[0]])
    H, W = ink.shape
    offs = np.arange(-halfwidth, halfwidth + 1)          # all perpendicular offsets in one pass
    xs = np.clip(np.round(center[0] + ts * u[0] + offs[:, None] * v[0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(center[1] + ts * u[1] + offs[:, None] * v[1]).astype(int), 0, H - 1)
    return float(ink[ys, xs].any(axis=0).mean())


def _spine_run_thickness(ink, center, u, length, max_k=20):
    """p25 of vertical ink-run lengths sampled along the major axis, in the RAW ink mask.
       Real strikes give short runs (2-3px) in inter-glyph gaps; bold-title crossbars are >=5px
       thick everywhere, so their p25 is high."""
    n = max(int(length), 2)
    ts = np.linspace(-length / 2.0, length / 2.0, n)
    H, W = ink.shape
    xs = np.clip(np.round(center[0] + ts * u[0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(center[1] + ts * u[1]).astype(int), 0, H - 1)
    base = ink[ys, xs].copy()
    for dy in (-1, 1):                      # the spine can sit 1px off a thin stroke: snap to ink
        miss = ~base
        yy = np.clip(ys + dy, 0, H - 1)
        snap = miss & ink[yy, xs]
        ys = np.where(snap, yy, ys)
        base |= snap
    if not base.any():
        return np.inf
    runs = np.ones(n)
    for sign in (-1, 1):
        alive = base.copy()
        for k in range(1, max_k + 1):
            yy = np.clip(ys + sign * k, 0, H - 1)
            alive = alive & ink[yy, xs]
            if not alive.any():
                break
            runs += alive
    return float(np.percentile(runs[base], 25))


STRAIGHTNESS_MIN_SAMPLES = 3     # fewer ink-bearing samples than this: wobble is not measurable


def _spine_straightness(ink, center, u, length, run_px):
    """RMS perpendicular wobble (px) of the local ink centroid about the straight spine, sampled
    along the major axis. A printed rule rides a dead-straight path (~0-1 px); a pen or typed
    strike crossing raised glyphs on a degraded scan wanders more. Returned in raw px at the
    caller's dpi — normalize to RENDER_DPI outside so a threshold is resolution-independent.

    Returns ``inf`` when the wobble is not measurable (non-finite `run_px`, or fewer than
    ``STRAIGHTNESS_MIN_SAMPLES`` ink-bearing samples). ``inf`` is the fail-SAFE direction: the
    scanned printed-rule veto reads a LOW value as "drawn rule, drop the detection", so an
    unmeasurable line must never come back low or a real strike is silently lost.

    TRIED AND REJECTED (2026-07-29): **detrending** the deviations — residual RMS about a fitted line
    instead of scatter about their mean — to stop a spine/ink angular mismatch (a linear ramp) from
    inflating the wobble of dotted pre-printed rules. Measured, it missed them anyway (their wobble is
    jitter across sparse dots, not a ramp) and halved the real strikes' margin above the veto bar.
    Dotted fill-in rules are underlines below the baseline; they need a positional discriminator.
    """
    if not np.isfinite(run_px):
        return float("inf")
    n = max(int(length), 2)
    ts = np.linspace(-length / 2.0, length / 2.0, n)
    v = np.array([-u[1], u[0]])                       # unit perpendicular
    H, W = ink.shape
    win = max(3, int(round(run_px * 2)))
    offs = np.arange(-win, win + 1)
    # Sample the whole (n x 2*win+1) perpendicular neighbourhood in one shot. This runs on the
    # DEFAULT path for every detected line — a per-sample Python loop cost ~37% of strike_lines.
    cx = center[0] + ts * u[0]
    cy = center[1] + ts * u[1]
    xs = np.clip(np.round(cx[:, None] + offs[None, :] * v[0]), 0, W - 1).astype(np.intp)
    ys = np.clip(np.round(cy[:, None] + offs[None, :] * v[1]), 0, H - 1).astype(np.intp)
    hit = ink[ys, xs]
    cnt = hit.sum(axis=1)
    inked = cnt > 0
    if int(inked.sum()) < STRAIGHTNESS_MIN_SAMPLES:
        return float("inf")
    devs = (hit * offs[None, :]).sum(axis=1)[inked] / cnt[inked]
    return float(np.std(devs))


def _collect_fragments(ink, dpi, scale=1.0):
    """Per-angle opening + gap-bridging with PERMISSIVE per-fragment filters.
       Returns fragments as (start_xy, end_xy) endpoint pairs along the major axis.
       `scale` = dpi / RENDER_DPI rescales the pixel-space tunables (calibrated at 200 dpi)."""
    seg = max(3, int(SEG_LEN_IN * dpi))
    gap_bridge = int(round(GAP_BRIDGE_PX * scale))
    frags = []
    for ang in range(0, 180, ANGLE_STEP_DEG):
        if min(ang, 180 - ang) > MAX_ANGLE_DEG:
            continue
        mask = ndimage.binary_opening(ink, structure=line_kernel(seg, ang))
        if gap_bridge:
            mask = ndimage.binary_closing(mask, structure=line_kernel(2 * gap_bridge + 1, ang))
        lbl, _ = ndimage.label(mask, structure=np.ones((3, 3)))
        for i, sl in enumerate(ndimage.find_objects(lbl), start=1):
            if sl is None:
                continue
            ys, xs = np.nonzero(lbl[sl] == i)
            ys = ys + sl[0].start; xs = xs + sl[1].start
            pts = np.column_stack([xs, ys]).astype(np.float32)
            c = pts.mean(0)
            d = pts - c
            evals, evecs = np.linalg.eigh((d.T @ d) / len(d))
            major, minor = evecs[:, 1], evecs[:, 0]
            if major[0] < 0:                                   # orient +x so endpoints sort by x
                major = -major
            length = float(np.ptp(d @ major))
            thick = float(np.ptp(d @ minor))
            if (length < PRE_LEN_PX * scale or thick > MAX_LINE_THICK_PX * scale
                    or length / max(thick, 0.5) < PRE_ASPECT):
                continue
            angle = abs(np.degrees(np.arctan2(major[1], major[0])))
            if min(angle, 180 - angle) > MAX_ANGLE_DEG:
                continue
            frags.append((c - major * length / 2, c + major * length / 2))
    return frags


def _stitch_fragments(frags, dy=None, scale=1.0):
    """Union-find merge of collinear fragments. Returns [(seg, member_fragments), ...] so failed
       groups can be re-stitched tighter. `scale` rescales the pixel-space stitch tolerances."""
    # floor the pixel-space stitch tolerances so they don't collapse below a couple of pixels at
    # low dpi (at 72 dpi a raw *scale would leave sub-pixel gaps/dy that never stitch)
    dy = max(2.0, STITCH_DY_PX * scale) if dy is None else dy
    max_gap = max(6.0, STITCH_GAP_PX * scale)
    n = len(frags)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if n:
        # Endpoint arrays in start-x order. This is the pipeline's hottest loop (n runs to thousands
        # on a dense scan), so the pair tests are vectorized per fragment; native dtype is preserved
        # so the comparisons match the scalar ones exactly.
        pts = np.asarray(frags)                       # (n, 2, 2): [frag][start|end][x|y]
        order = np.argsort(pts[:, 0, 0], kind="stable")
        sx, sy = pts[order, 0, 0], pts[order, 0, 1]
        ex, ey = pts[order, 1, 0], pts[order, 1, 1]
        idx = order.tolist()
        for oi in range(n - 1):
            # Only later fragments whose START lies within max_gap of i's END can stitch; start-x is
            # sorted, so the rest are unreachable (what the scalar loop's `break` relied on). The
            # bound is slack by a hair and the exact gap test is reapplied below.
            hi = int(np.searchsorted(sx, ex[oi] + max_gap + 1e-3, side="right"))
            lo = oi + 1
            if hi <= lo:
                continue
            ok = (sx[lo:hi] - ex[oi]) <= max_gap
            ok &= np.abs(ey[oi] - sy[lo:hi]) <= dy
            ok &= ~((sx[lo:hi] > sx[oi]) & (ex[lo:hi] < ex[oi]))  # j inside i: dedup handles overlap
            for j in np.flatnonzero(ok):
                ra, rb = find(idx[oi]), find(idx[lo + int(j)])
                if ra != rb:
                    parent[rb] = ra
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    merged = []
    for idxs in groups.values():
        pts = np.array([p for i in idxs for p in frags[i]])
        left = pts[np.argmin(pts[:, 0])]
        right = pts[np.argmax(pts[:, 0])]
        merged.append(((left, right), [frags[i] for i in idxs]))
    return merged


def strike_lines(gray, dpi=RENDER_DPI, ink=None):
    """Detect near-horizontal straight-ish lines (strikes / underlines / rules) on a grayscale
    page raster (uint8 HxW ndarray). Returns dicts:
        {bbox_px, ends_px, len_in, angle_deg, fill, run_px, straightness}
    `straightness` is perpendicular ink wobble in px normalized to RENDER_DPI, or **None** when it
    is not measurable — treat None as "unknown", never as "straight" (see `_spine_straightness`).
    `dpi` is the raster's px-per-inch — pass the DPI the image was rendered at; both the length
    tunables and the pixel-space tunables (calibrated at 200 dpi) rescale from it. Pass `ink`
    (a bool mask) to reuse a precomputed binarization.

    This finds strokes, not struck words: labelling a line as strike-vs-underline-vs-rule and
    attaching it to words requires OCR word boxes (a separate layer).
    """
    gray = to_gray_u8(gray)
    if ink is None:
        ink = ink_mask(gray)
    scale = dpi / float(RENDER_DPI)
    min_len = int(MIN_LINE_LEN_IN * dpi)
    fill_halfwidth = max(2, int(round(2 * scale)))

    groups = _stitch_fragments(_collect_fragments(ink, dpi, scale), scale=scale)

    def evaluate(start, end):
        vec = end - start
        length = float(np.hypot(*vec))
        if length < min_len:
            return None, False
        u = vec / max(length, 1e-9)
        angle = float(min(a := abs(np.degrees(np.arctan2(u[1], u[0]))), 180 - a))
        if angle > MAX_ANGLE_DEG:
            return None, True
        center = (start + end) / 2
        fill = _spine_fill(ink, center, u, length, fill_halfwidth)
        if fill < MIN_FILL:
            return None, True
        run_px = _spine_run_thickness(ink, center, u, length)
        # floored: unfloored, a 2-px strike at 72 dpi faces a "> 1.44 px" gate and is rejected
        if run_px > max(2.0, MAX_STROKE_RUN_PX * scale):
            return None, True
        # perpendicular wobble, normalized to RENDER_DPI so the scanned-path printed-rule veto's
        # threshold is dpi-independent (issue #7). Reported as None, never a number, when it is not
        # measurable: the veto reads LOW as "drawn rule", and None keeps the field JSON-safe.
        wobble = _spine_straightness(ink, center, u, length, run_px) / max(scale, 1e-9)
        x0, y0 = int(min(start[0], end[0])), int(min(start[1], end[1]))
        x1, y1 = int(max(start[0], end[0])), int(max(start[1], end[1]))
        x1 = max(x1, x0 + 1)                   # never a zero-area box: near-horizontal lines
        y1 = max(y1, y0 + 1)                   # would defeat the IoU dedup below
        return {
            "bbox_px": (x0, y0, x1, y1),
            "ends_px": ((float(start[0]), float(start[1])), (float(end[0]), float(end[1]))),
            "len_in": round(length / dpi, 2),
            "angle_deg": round(angle, 1),
            "fill": round(fill, 2),
            "run_px": round(run_px, 1),
            "straightness": round(wobble, 2) if np.isfinite(wobble) else None,
            "_len": length,
        }, False

    cands = []
    for (start, end), members in groups:
        cand, salvage = evaluate(start, end)
        if cand is not None:
            cands.append(cand)
        elif salvage and len(members) > 2:
            for (s2, e2), _ in _stitch_fragments(members, dy=STITCH_DY_TIGHT * scale, scale=scale):
                sub, _ = evaluate(s2, e2)
                if sub is not None:
                    cands.append(sub)

    cands.sort(key=lambda c: -c["_len"])                       # longest first, then drop overlaps
    # Vectorized equivalent of "keep c unless it overlaps anything already kept": one IoU pass per
    # candidate against every kept box at once, instead of a scalar call per pair.
    kept = []
    boxes = np.empty((len(cands), 4), dtype=np.float64)
    for c in cands:
        x0, y0, x1, y1 = c["bbox_px"]
        if kept:
            k = boxes[:len(kept)]
            ix = np.minimum(k[:, 2], x1) - np.maximum(k[:, 0], x0)
            iy = np.minimum(k[:, 3], y1) - np.maximum(k[:, 1], y0)
            inter = np.maximum(ix, 0) * np.maximum(iy, 0)
            union = (k[:, 2] - k[:, 0]) * (k[:, 3] - k[:, 1]) + (x1 - x0) * (y1 - y0) - inter
            iou = np.where(inter > 0, inter / np.maximum(union, 1), 0.0)
            if (iou >= DEDUP_IOU).any():
                continue
        boxes[len(kept)] = (x0, y0, x1, y1)
        kept.append(c)
    for c in kept:
        del c["_len"]
    return kept
