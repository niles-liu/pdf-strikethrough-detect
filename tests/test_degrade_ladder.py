"""The scan-degradation ladder (`benchmarks/_degrade.py`) — free labels in the failing regime.

The degrade-and-relabel route only yields usable training data if two properties hold: the ladder
must be **deterministic** (a rebuilt PDF has to stay aligned with labels, and with any DI result
captured against it) and it must **preserve the label mapping** (page geometry unchanged, with skew
the one operation that moves boxes and therefore the one the ground truth must be carried through).
These tests pin both, plus the direction of the skew transform, which is easy to get backwards in
y-down image coordinates, and the non-mutation `render_pages` depends on: a ladder walk renders each
document once and hands the *same* image object to every rung, so a `degrade_image` that modified
its input in place would silently compound the rungs on top of each other.

The corpus itself is git-ignored, so everything here runs against a synthetic page built in memory.
"""
import pathlib
import sys

import fitz
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))

from _degrade import (BY_NAME, LADDER, DegradeLevel, build_degraded_pdf,  # noqa: E402
                      degrade_frac_boxes, degrade_image, render_pages, rotate_frac_bbox)


def _page_pdf(width=400, height=300, text="keep deleted text here"):
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.insert_text((40, 150), text, fontsize=14)
    out = doc.tobytes()
    doc.close()
    return out


def _page_image(width=400, height=300):
    """A white page with black bars standing in for text rows."""
    arr = np.full((height, width), 255, dtype=np.uint8)
    for y in range(100, 160, 20):
        arr[y:y + 8, 40:360] = 0
    return Image.fromarray(arr)  # 2-D uint8 -> mode "L"


def test_ladder_shape():
    assert [lv.name for lv in LADDER][0] == "L0_clean"
    assert BY_NAME["L3_fax"].effective_dpi == pytest.approx(200 * 0.55)
    # the rungs get monotonically worse on every axis that has an ordering
    fades = [lv.contrast for lv in LADDER]
    blurs = [lv.blur_px for lv in LADDER]
    skews = [abs(lv.rotate_deg) for lv in LADDER]
    assert fades == sorted(fades, reverse=True)
    assert blurs == sorted(blurs)
    assert skews == sorted(skews)


def test_l0_is_identity():
    img = _page_image()
    out = degrade_image(img, BY_NAME["L0_clean"])
    assert np.array_equal(np.asarray(img), np.asarray(out))


def test_degradation_is_deterministic():
    img = _page_image()
    lv = BY_NAME["L3_fax"]
    a = np.asarray(degrade_image(img, lv, page_index=2))
    b = np.asarray(degrade_image(img, lv, page_index=2))
    assert np.array_equal(a, b)
    # ...and keyed on the page, so different pages don't share a noise field
    c = np.asarray(degrade_image(img, lv, page_index=3))
    assert not np.array_equal(a, c)


def test_each_rung_diverges_further_from_clean():
    img = _page_image()
    clean = np.asarray(degrade_image(img, LADDER[0]), dtype=np.float32)
    mad = [float(np.abs(np.asarray(degrade_image(img, lv), dtype=np.float32) - clean).mean())
           for lv in LADDER]
    assert mad[0] == 0.0
    assert mad == sorted(mad), mad


def test_ink_fades_and_page_contrast_collapses():
    """The photocopy look: ink lifts toward the paper and the ink/paper separation narrows.

    Note it is the *separation* that collapses, not the page mean — PIL's contrast operator blends
    toward the image mean, so faint pages darken their paper as well as lightening their ink, and a
    mostly-white page's mean is dominated by the paper.
    """
    img = _page_image()
    ink, sep = [], []
    for lv in LADDER:
        a = np.asarray(degrade_image(img, lv), dtype=np.float32)
        paper = float(np.percentile(a, 90))
        band = float(a[101:107, 60:340].mean())      # the core of the top synthetic text row
        ink.append(band)
        sep.append(paper - band)
    assert ink == sorted(ink), ink
    assert sep == sorted(sep, reverse=True), sep


def test_canvas_size_is_preserved():
    img = _page_image()
    for lv in LADDER:
        assert degrade_image(img, lv).size == img.size


def test_rotate_frac_bbox_no_op_without_skew():
    box = (0.2, 0.3, 0.4, 0.35)
    assert rotate_frac_bbox(box, 0.0, 1.3) == box


def test_rotate_frac_bbox_direction_is_ccw_in_display_coords():
    """+deg is counter-clockwise as displayed: a box right of centre moves *up* (y decreases)."""
    box = (0.79, 0.49, 0.81, 0.51)
    x0, y0, x1, y1 = rotate_frac_bbox(box, 90.0, 1.0)
    assert (x0 + x1) / 2 == pytest.approx(0.5, abs=1e-9)
    assert (y0 + y1) / 2 == pytest.approx(0.2, abs=1e-9)
    assert y1 < 0.5


def test_rotate_frac_bbox_envelope_contains_the_true_box():
    """Round-tripping grows the box (it is an axis-aligned envelope) but never loses the original."""
    box = (0.30, 0.40, 0.55, 0.44)
    there = rotate_frac_bbox(box, 1.1, 1.29)
    back = rotate_frac_bbox(there, -1.1, 1.29)
    assert back[0] <= box[0] and back[1] <= box[1]
    assert back[2] >= box[2] and back[3] >= box[3]
    # sub-degree skew must not bloat a word box by anything like its own size
    assert (back[2] - back[0]) < 1.15 * (box[2] - box[0])


def _mark_image(width, height, cx_frac, cy_frac, half=6):
    """White page with one small black square centred on the given page fraction."""
    arr = np.full((height, width), 255, dtype=np.uint8)
    cx, cy = int(cx_frac * width), int(cy_frac * height)
    arr[cy - half:cy + half, cx - half:cx + half] = 0
    return Image.fromarray(arr)  # 2-D uint8 -> mode "L"


def _dark_centroid_frac(img):
    arr = np.asarray(img)
    ys, xs = np.nonzero(arr < 128)
    assert xs.size, "the mark vanished"
    return (float(xs.mean()) / img.width, float(ys.mean()) / img.height)


@pytest.mark.parametrize("size,deg,tol", [
    ((400, 400), 90.0, 0.005),      # square: aspect drops out, so a right angle is exact
    ((400, 300), 1.1, 0.004),       # non-square at ladder-scale skew: pins the aspect handling
    ((400, 300), -1.1, 0.004),
])
def test_skew_transform_agrees_with_the_pixels_pil_produces(size, deg, tol):
    """`rotate_frac_bbox` must move ground truth the way `degrade_image` actually moves ink.

    The other tests check the transform against itself; this one checks it against PIL. A sign or
    aspect error here would silently mislabel every positive at a skewed rung, which is the one
    mistake in this module that no downstream number would reveal.
    """
    width, height = size
    aspect = width / height
    src_frac = (0.78, 0.34)
    img = _mark_image(width, height, *src_frac)

    level = DegradeLevel("skew_only", rotate_deg=deg)
    got = _dark_centroid_frac(degrade_image(img, level))

    # predict where the mark's box lands, and take the envelope's centre
    half_w, half_h = 6 / width, 6 / height
    box = (src_frac[0] - half_w, src_frac[1] - half_h, src_frac[0] + half_w, src_frac[1] + half_h)
    px0, py0, px1, py1 = rotate_frac_bbox(box, deg, aspect)
    want = ((px0 + px1) / 2, (py0 + py1) / 2)

    assert got[0] == pytest.approx(want[0], abs=tol), f"x: pixels {got[0]} vs transform {want[0]}"
    assert got[1] == pytest.approx(want[1], abs=tol), f"y: pixels {got[1]} vs transform {want[1]}"


def test_degrade_frac_boxes_moves_boxes_only_for_skewed_rungs():
    boxes = [(0.2, 0.3, 0.4, 0.34)]
    assert degrade_frac_boxes(boxes, BY_NAME["L0_clean"], 1.33) == [tuple(boxes[0])]
    moved = degrade_frac_boxes(boxes, BY_NAME["L4_thrice_copied"], 1.33)
    assert moved != [tuple(boxes[0])]


def test_degrade_image_does_not_mutate_its_input():
    """`render_pages` rasterizes once and every rung degrades that same image object. If any step
    worked in place the rungs would compound, and the ladder would silently stop being a ladder."""
    img = _page_image()
    before = np.asarray(img).copy()
    for lv in LADDER:
        degrade_image(img, lv)
    assert np.array_equal(np.asarray(img), before)


def test_render_pages_returns_gray_images_and_page_geometry(tmp_path):
    src = tmp_path / "src.pdf"
    src.write_bytes(_page_pdf(width=400, height=300))
    rendered = render_pages(src, [0])
    assert len(rendered) == 1
    img, width, height = rendered[0]
    assert img.mode == "L"                            # degrade_image's working space
    assert (width, height) == pytest.approx((400, 300), abs=0.5)
    assert img.size[0] / img.size[1] == pytest.approx(400 / 300, abs=0.01)


def test_build_degraded_pdf_is_imageonly_and_keeps_page_geometry(tmp_path):
    src = tmp_path / "src.pdf"
    src.write_bytes(_page_pdf(width=400, height=300))
    out = build_degraded_pdf(render_pages(src, [0]), BY_NAME["L2_photocopy"])

    doc = fitz.open(stream=out, filetype="pdf")
    try:
        assert doc.page_count == 1
        assert doc[0].rect.width == pytest.approx(400, abs=0.5)
        assert doc[0].rect.height == pytest.approx(300, abs=0.5)
        assert doc[0].get_text().strip() == ""        # no text layer -> classifies as scanned
        assert doc[0].get_images()                    # ...because the page is an image
    finally:
        doc.close()


def test_build_degraded_pdf_is_reproducible(tmp_path):
    """Same input, same rung, same page: identical pixels. This is what lets a DI result captured
    against one build stay valid against a rebuild — the reason the noise is seeded at all."""
    src = tmp_path / "src.pdf"
    src.write_bytes(_page_pdf())
    lv = BY_NAME["L3_fax"]
    pages = [_rendered_page(build_degraded_pdf(render_pages(src, [0]), lv)) for _ in range(2)]
    assert np.array_equal(*pages)


def _rendered_page(pdf_bytes):
    """The first page of a built PDF, as pixels — compares the raster rather than the container,
    which carries an /ID that varies per write."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pix = doc[0].get_pixmap(dpi=72)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    finally:
        doc.close()
