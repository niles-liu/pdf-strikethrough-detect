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
    grayscale or (H, W, 3|4) RGB(A) arrays; float images in [0, 1] are rescaled to 0..255;
    out-of-range values are clipped (no mod-256 wraparound)."""
    a = np.asarray(image)
    if a.ndim == 3 and a.shape[2] in (3, 4):
        a = a[..., :3].mean(axis=2)
    if a.ndim != 2:
        raise ValueError(f"expected a (H, W) grayscale or (H, W, 3) RGB image, got shape {a.shape}")
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating) and a.size and float(a.max()) <= 1.0:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def ink_mask(gray):
    """Binarized ink mask: global Otsu + OTSU_OFFSET (clean on dense text, keeps faint strokes)."""
    gray = to_gray_u8(gray)
    return gray < (otsu_threshold(gray) + OTSU_OFFSET)


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


def _bbox_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / max(union, 1)


def _spine_fill(ink, center, u, length, halfwidth=2):
    """Fraction of steps along the major axis with raw ink within +/-halfwidth px perpendicular."""
    n = max(int(length), 2)
    ts = np.linspace(-length / 2.0, length / 2.0, n)
    v = np.array([-u[1], u[0]])
    hit = np.zeros(n, dtype=bool)
    H, W = ink.shape
    for off in range(-halfwidth, halfwidth + 1):
        xs = np.clip(np.round(center[0] + ts * u[0] + off * v[0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(center[1] + ts * u[1] + off * v[1]).astype(int), 0, H - 1)
        hit |= ink[ys, xs]
    return float(hit.mean())


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
    dy = STITCH_DY_PX * scale if dy is None else dy
    max_gap = STITCH_GAP_PX * scale
    n = len(frags)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    order = sorted(range(n), key=lambda i: frags[i][0][0])
    for oi, i in enumerate(order):
        si, ei = frags[i]
        for j in order[oi + 1:]:
            sj, ej = frags[j]
            gap = sj[0] - ei[0]
            if gap > max_gap:
                break
            if sj[0] - si[0] > 0 and ej[0] - ei[0] < 0:
                continue                      # j fully inside i's x-range: dedup handles overlap
            if abs(ei[1] - sj[1]) > dy:
                continue
            ra, rb = find(i), find(j)
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
        {bbox_px, ends_px, len_in, angle_deg, fill, run_px}
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
        if run_px > MAX_STROKE_RUN_PX * scale:
            return None, True
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
    kept = []
    for c in cands:
        if all(_bbox_iou(c["bbox_px"], k["bbox_px"]) < DEDUP_IOU for k in kept):
            kept.append(c)
    for c in kept:
        del c["_len"]
    return kept
