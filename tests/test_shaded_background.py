"""Issue #15 — a word on a shaded or highlighted block must not read as struck.

Three separate causes, one per section below. `std_crop`'s `1 - gray/255` turned a shaded paper
ground into a uniform ink pedestal across the net input, which the model reads as a wash over the
word and scores at p=1.0; `to_gray_u8` collapsed RGB with a channel mean that disagreed with the
PDF path's csGRAY on every saturated highlight; and one global Otsu split page-from-block instead
of ink-from-paper once a block's ground was dark enough.
"""
import numpy as np
import pytest

from pdf_strikethrough import cnn


def _word_on(ground, h=40, w=120):
    """Crude 'glyphs' (dark bars) on a paper ground of the given grey level."""
    crop = np.full((h, w), float(ground), dtype=np.float32)
    for x in range(8, w - 8, 14):
        crop[12:28, x:x + 5] = 20.0
    return crop


def _bg_level(std):
    """The ink-positive level the net sees where the paper is — the pedestal, if any."""
    return float(np.median(std[:4]))


def test_ordinary_paper_is_untouched():
    """The whole no-regression argument: a non-shaded crop reaches the net exactly as in 0.10.0."""
    for ground in (255, 252, 245):
        assert _bg_level(cnn.std_crop(_word_on(ground))) == pytest.approx(0.0, abs=0.04)


def test_shaded_ground_loses_its_pedestal():
    shaded, white = cnn.std_crop(_word_on(211)), cnn.std_crop(_word_on(255))
    assert _bg_level(shaded) == pytest.approx(_bg_level(white), abs=0.02)


def test_shaded_crop_keeps_its_glyphs():
    std = cnn.std_crop(_word_on(211))
    assert std.max() > 0.8                        # the dark bars survive as strong ink
    assert 0.02 < float((std > 0.5).mean()) < 0.5


def test_ground_is_the_mode_not_a_percentile():
    """A crop whose PAD_Y margin runs off the highlight band into white paper is still shaded: a
    percentile reads the margin (255) and would skip the flatten, the mode reads the band."""
    crop = _word_on(211)
    crop[:7] = 255.0
    crop[-7:] = 255.0
    assert np.percentile(crop, 90) == 255.0
    assert _bg_level(cnn.std_crop(crop)) == pytest.approx(0.0, abs=0.04)


def test_dark_scan_is_untouched():
    """Below the guard this is not a highlight block; rescaling would amplify noise into ink."""
    std = cnn.std_crop(_word_on(100))
    assert _bg_level(std) > 0.4


def test_all_ink_crop_does_not_blow_up():
    std = cnn.std_crop(np.full((20, 20), 10.0, dtype=np.float32))
    assert std.shape == (cnn.CROP_H, cnn.CROP_W)
    assert np.isfinite(std).all()


def test_shaded_word_scores_clean_like_its_white_twin():
    """End to end through the shipped model: an unstruck word must not score struck merely for
    sitting on a grey block."""
    p_white = float(cnn.score_crops([cnn.std_crop(_word_on(255))])[0])
    p_shaded = float(cnn.score_crops([cnn.std_crop(_word_on(211))])[0])
    assert p_shaded < 0.5
    assert abs(p_shaded - p_white) < 0.15


# --- grey conversion + shaded-page binarization (lines.py) -------------------------------------------------

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
