"""Issue #15 — colour handling on the raster path.

A channel mean disagreed with the PDF path's csGRAY on every saturated highlight, and a
single global Otsu splits page-from-block instead of ink-from-paper once a block is dark.
"""
import numpy as np
import pytest


def _page_with_band(ground_rgb, h=400, w=600):
    """A white page of text-like bars with a horizontal band washed to `ground_rgb`."""
    page = np.full((h, w, 3), 255.0, dtype=np.float32)
    for y in range(20, h - 20, 20):
        for x in range(20, w - 20, 14):
            page[y:y + 8, x:x + 9] = 30.0
    page[int(0.3*h):int(0.7*h)] *= np.asarray(ground_rgb, np.float32) / 255.0
    return np.clip(page, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("rgb, csgray", [
    ((255, 255, 0), 248), ((0, 255, 0), 220), ((255, 0, 255), 144),
    ((0, 255, 255), 228), ((211, 211, 211), 211),
])
def test_to_gray_u8_matches_pymupdf_csgray(rgb, csgray):
    """The image path must agree with the PDF path, which renders through PyMuPDF's csGRAY. A
    channel mean put a yellow highlighter at 170 against csGRAY's 248, so the same highlighted page
    was readable through detect_pdf and solid ink through an RGB array."""
    from pdf_strikethrough.lines import to_gray_u8
    got = int(to_gray_u8(np.array([[rgb]], dtype=np.uint8))[0, 0])
    assert abs(got - csgray) <= 7


def test_to_gray_u8_grayscale_input_untouched():
    from pdf_strikethrough.lines import to_gray_u8
    a = np.array([[0, 77, 128, 255]], dtype=np.uint8)
    assert np.array_equal(to_gray_u8(a), a)


def test_ink_mask_survives_a_shaded_band():
    """One global Otsu splits page-from-block instead of ink-from-paper once a band is dark enough,
    and returns the whole band as ink. The gated fallback re-thresholds on a flattened background."""
    from pdf_strikethrough.lines import ink_mask
    for ground in (255, 211, 195, 175, 150):
        page = _page_with_band((ground, ground, ground))
        m = ink_mask(page)
        band = m[int(0.3*page.shape[0]):int(0.7*page.shape[0])]
        assert 0.01 < band.mean() < 0.35, f"ground {ground}: ink fraction {band.mean():.3f}"


def test_ink_mask_unchanged_on_an_ordinary_page():
    """The fallback is gated: a normal page must take the plain global-Otsu path."""
    from pdf_strikethrough.lines import ink_mask, otsu_threshold, to_gray_u8, OTSU_OFFSET
    page = _page_with_band((255, 255, 255))
    g = to_gray_u8(page)
    assert np.array_equal(ink_mask(page), g < (otsu_threshold(g) + OTSU_OFFSET))
