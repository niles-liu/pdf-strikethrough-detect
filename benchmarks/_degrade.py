"""Scan-degradation ladder — turn a born-digital page into a *plausibly bad* scan.

``_scanned.py`` rasterizes a born-digital page at a fixed 200 dpi and calls the result a "scan".
That raster is **pristine**: sharp glyphs, full contrast, no skew, no compression. It exercises the
scanned code path but not the regime that actually breaks the detector — the SOF corpus that
produced 113 false positives is faint, resampled, skewed and JPEG-mangled
(`benchmarks/private/ruled-tables/CORPUS_README.md`). A 95–97% recovery figure measured on a clean
raster therefore says nothing about a photocopied form, which is why this module exists.

The point is **free labels in the failing regime**: the native vector detector gives the exact
strike set on the born-digital original (``_scanned.struck_pages``), and degradation does not move
the ink, so those labels stay valid through the ladder. That yields labeled positives at scale
without hand-labeling and without an Azure DI call per document.

⚠ **A synthetic photocopy is not a real one.** This ladder models the *sensor* chain (faintness,
blur, resampling, skew, noise, compression) and models nothing about a real one: toner speckle,
staple shadows, bleed-through, dog-eared corners, a densely-ruled form underneath the text. Treat
"degraded positives" as a hypothesis about transfer, testable only by holding out genuinely scanned
positives — of which there are **3, all on one page of one document**. That holdout is too thin to
settle the question, so do not report ladder numbers as if they were corpus numbers.

Every level is deterministic: noise is drawn from a seeded generator keyed on the level name and
page index, so a rebuild reproduces the same bytes and a DI result captured against one stays valid.
"""
from __future__ import annotations

import dataclasses
import io
import zlib

import fitz
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SCAN_DPI = 200          # matches _scanned.SCAN_DPI — the detector's calibration point


@dataclasses.dataclass(frozen=True)
class DegradeLevel:
    """One rung of the ladder. Applied in physical order: fade -> blur -> skew -> resample ->
    sensor noise -> compression, which is the order a real print/copy/scan chain applies them."""

    name: str
    contrast: float = 1.0        # <1 fades ink toward the page (faint-photocopy look)
    blur_px: float = 0.0         # gaussian blur radius, pixels at SCAN_DPI
    rotate_deg: float = 0.0      # platen skew; positive = counter-clockwise as displayed
    scale: float = 1.0           # resample down to this fraction, then back up (resolution loss)
    noise_sigma: float = 0.0     # additive gaussian sensor noise, 0-255 units
    jpeg_quality: int = 0        # 0 = no JPEG round-trip; else PIL quality
    seed_name: str = ""          # noise RNG key; "" means use `name`

    @property
    def effective_dpi(self) -> float:
        return SCAN_DPI * self.scale

    @property
    def noise_key(self) -> str:
        """What the sensor-noise generator is keyed on.

        Defaults to the level name, so every named rung keeps the bytes it has always produced. The
        ablation probes below override it to share their base rung's key: otherwise changing one
        factor would *also* redraw the noise, and a one-factor comparison would carry two changes.
        """
        return self.seed_name or self.name


# Named rungs, roughly matched to what the SOF corpus looks like. L3 is the closest analogue to the
# documents that over-flag; L0 reproduces the existing clean-raster behaviour as a control.
LADDER: tuple[DegradeLevel, ...] = (
    DegradeLevel("L0_clean"),
    DegradeLevel("L1_light", contrast=0.95, blur_px=0.3, rotate_deg=0.15, scale=0.85,
                 noise_sigma=1.5, jpeg_quality=88),
    DegradeLevel("L2_photocopy", contrast=0.82, blur_px=0.6, rotate_deg=-0.4, scale=0.72,
                 noise_sigma=3.5, jpeg_quality=72),
    DegradeLevel("L3_fax", contrast=0.68, blur_px=1.0, rotate_deg=0.7, scale=0.55,
                 noise_sigma=6.0, jpeg_quality=55),
    DegradeLevel("L4_thrice_copied", contrast=0.55, blur_px=1.5, rotate_deg=-1.1, scale=0.42,
                 noise_sigma=9.0, jpeg_quality=38),
)

BY_NAME = {lv.name: lv for lv in LADDER}

# --- L2 -> L3 ablation probes ------------------------------------------------------------------
# The gate's curve collapses between L2_photocopy and L3_fax (66% recall / 1163 predicted -> 0.4% /
# 5), but that single step moves SIX parameters at once, so the collapse attributes to nothing. Each
# probe below takes L2 and advances exactly ONE factor to its L3 value, which is the only way to name
# the cause before redesigning the fade model.
#
# Deliberately NOT in LADDER: they are reachable through `--levels` but do not change the default
# gate, so the recorded ladder numbers stay comparable. They also inherit L2's `noise_key`, so the
# noise realization is held fixed and the one factor under test is the only thing that moves.
_L2, _L3 = BY_NAME["L2_photocopy"], BY_NAME["L3_fax"]


def _probe(suffix: str, **field) -> DegradeLevel:
    return dataclasses.replace(_L2, name=f"A_{suffix}", seed_name=_L2.name, **field)


ABLATION: tuple[DegradeLevel, ...] = (
    _probe("contrast", contrast=_L3.contrast),        # 0.82 -> 0.68
    _probe("blur", blur_px=_L3.blur_px),              # 0.6  -> 1.0
    _probe("scale", scale=_L3.scale),                 # 0.72 -> 0.55  (144 -> 110 effective dpi)
    _probe("skew", rotate_deg=_L3.rotate_deg),        # -0.4 -> 0.7
    _probe("noise", noise_sigma=_L3.noise_sigma),     # 3.5  -> 6.0
    _probe("jpeg", jpeg_quality=_L3.jpeg_quality),    # 72   -> 55
)

BY_NAME.update({lv.name: lv for lv in ABLATION})


def _rng(level_name: str, page_index: int):
    """Deterministic generator keyed on (level, page) — reproducible across processes, unlike
    ``hash()``, whose string salt varies per interpreter run."""
    seed = zlib.crc32(f"{level_name}:{page_index}".encode()) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def degrade_image(img: Image.Image, level: DegradeLevel, page_index: int = 0) -> Image.Image:
    """Apply one ladder rung to a rendered page image. Canvas size is preserved, so page-fraction
    boxes still map 1:1 (except for skew — see ``degrade_frac_boxes``)."""
    if img.mode != "L":
        img = img.convert("L")
    w, h = img.size

    if level.contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(level.contrast)
    if level.blur_px > 0:
        img = img.filter(ImageFilter.GaussianBlur(level.blur_px))
    if level.rotate_deg:
        # expand=False keeps the canvas (and therefore the page mapping); corners fill with paper
        # white rather than black, which is what a skewed sheet on a platen actually looks like.
        img = img.rotate(level.rotate_deg, resample=Image.BICUBIC, expand=False, fillcolor=255)
    if level.scale != 1.0:
        small = (max(1, int(w * level.scale)), max(1, int(h * level.scale)))
        img = img.resize(small, Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if level.noise_sigma > 0:
        arr = np.asarray(img, dtype=np.float32)
        arr += _rng(level.noise_key, page_index).normal(0.0, level.noise_sigma, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))  # 2-D uint8 -> mode "L"
    if level.jpeg_quality:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=level.jpeg_quality)
        buf.seek(0)
        img = Image.open(buf)
        img.load()
    return img


def rotate_frac_bbox(bbox, deg: float, aspect: float):
    """Map a page-fraction bbox through the same skew ``degrade_image`` applies, returning the
    axis-aligned envelope of the rotated rectangle.

    Fraction coords normalize x and y independently, so a rotation is only a rotation once x is
    rescaled by the page aspect (width/height); this converts, rotates about the page centre in
    y-down display coords, and converts back. The result is an **envelope**, marginally larger than
    the true rotated box — at the ladder's sub-degree angles that is a fraction of a pixel on a word
    box, and it errs toward accepting a detection rather than rejecting it.
    """
    if not deg:
        return tuple(bbox)
    x0, y0, x1, y1 = bbox
    a = np.radians(deg)
    ca, sa = np.cos(a), np.sin(a)
    cx, cy = 0.5 * aspect, 0.5
    xs, ys = [], []
    for fx, fy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        dx, dy = fx * aspect - cx, fy - cy
        xs.append((cx + dx * ca + dy * sa) / aspect)
        ys.append(cy - dx * sa + dy * ca)
    return (min(xs), min(ys), max(xs), max(ys))


def degrade_frac_boxes(boxes, level: DegradeLevel, aspect: float):
    """Ground-truth boxes carried through a ladder rung. Only skew moves them — fading, blur,
    resampling, noise and compression all preserve the canvas and so preserve the mapping."""
    return [rotate_frac_bbox(b, level.rotate_deg, aspect) for b in boxes]


def render_pages(orig_path, page_indices, dpi: float = SCAN_DPI):
    """Rasterize ``page_indices`` of ``orig_path`` once, as ``[(gray image, width_pt, height_pt)]``.

    Split out from ``build_degraded_pdf`` because rendering dominates a ladder walk and every rung
    starts from the *same* raster: an N-level run should render once and degrade N times, not
    re-render per rung. The page rects come back with the images so callers need not reopen the
    source for geometry (``rotate_frac_bbox`` needs the aspect).
    """
    src = fitz.open(str(orig_path))
    try:
        out = []
        for pno in page_indices:
            page = src[pno]
            pix = page.get_pixmap(dpi=dpi)
            # via PNG rather than frombytes: makes no assumption about the pixmap's channel count
            # (matches how _scanned.build_scanned_pdf hands the raster over)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img.load()                           # detach from the BytesIO before it goes out of use
            out.append((img.convert("L"), page.rect.width, page.rect.height))
        return out
    finally:
        src.close()


def build_degraded_pdf(rendered, level: DegradeLevel) -> bytes:
    """``render_pages`` output, degraded through ``level``, as an image-only PDF.

    Mirrors ``_scanned.build_scanned_pdf`` (no text layer, so every page classifies as ``scanned``;
    original page geometry preserved) with the ladder applied to each rendered page.
    """
    out = fitz.open()
    try:
        for i, (img, width, height) in enumerate(rendered):
            buf = io.BytesIO()
            # PNG container; any JPEG loss is already baked into the pixels by degrade_image
            degrade_image(img, level, page_index=i).save(buf, format="PNG")
            npage = out.new_page(width=width, height=height)
            npage.insert_image(npage.rect, stream=buf.getvalue())
        return out.tobytes()
    finally:
        out.close()
