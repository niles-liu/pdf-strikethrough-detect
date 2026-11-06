"""Issue #15 — a word on a shaded or highlighted block must not read as struck.

Three separate causes, one per section below. `std_crop`'s `1 - gray/255` turned a shaded paper
ground into a uniform ink pedestal across the net input, which the model reads as a wash over the
word and scores at p=1.0; `to_gray_u8` collapsed RGB with a channel mean that disagreed with the
PDF path's csGRAY on every saturated highlight; and one global Otsu split page-from-block instead
of ink-from-paper once a block's ground was dark enough. The near-white section is a fourth cause:
through 0.14.0, `std_crop` left paper at 240-254, and white read from a colour image (254), as a
faint grey ground.
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


def test_white_and_near_white_paper_reach_the_net_white():
    """A crop on white paper, or on near-white paper (245, 252), reaches the net with no grey
    ground."""
    for ground in (255, 252, 245):
        assert _bg_level(cnn.std_crop(_word_on(ground))) == pytest.approx(0.0, abs=1e-6)


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
    # the paper is the commonest pixel of 128 and up, so a ground at 127 is left as it is
    assert _bg_level(cnn.std_crop(_word_on(127))) == pytest.approx(1 - 127 / 255, abs=1e-6)


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


# --- near-white paper (240-254) -------------------------------------------------------------------
# 0.14.0 flattened a ground below 240 and passed one at 240-254 through as a pedestal of up to 0.06
# on which the shipped model scored a clean word on paper at 240 at 1.0.

def test_every_grey_ground_loses_its_pedestal():
    for ground in (*range(128, 255), 239.6, 247.6, 254.6):      # and between levels (float crops)
        assert _bg_level(cnn.std_crop(_word_on(ground))) == pytest.approx(0.0, abs=1e-6), ground
    assert _bg_level(cnn.std_crop(_word_on(247.6) / 255)) == pytest.approx(0.0, abs=1e-6)  # [0, 1]


def test_near_white_paper_changes_the_input_smoothly():
    """From 240 up to white paper, one grey level of paper moves the net's input by at most 3
    levels. (From 239 to 240 the step is larger: below 240 the white point, at SHADE_WHITE_FRAC of
    the paper, lightens the ink too.)"""
    for ground in range(240, 255):
        step = np.abs(cnn.std_crop(_word_on(ground + 1)) - cnn.std_crop(_word_on(ground)))
        assert step.max() <= 3 / 255, (ground, step.max() * 255)


def test_white_paper_and_shaded_grounds_are_as_before(monkeypatch):
    """Only grounds at 240-254 changed: PAPER_WHITE_FROM = 240 gives 0.14.0's handling,
    and the two settings agree exactly on white paper and on every ground from 128 to 239."""
    crops = [_word_on(g) for g in (*range(128, 240), 255)]
    noisy = _word_on(255)
    noisy[::3, ::7] = 250.0                       # mode still 255: white paper with speckle
    crops.append(noisy)
    new = [cnn.std_crop(c) for c in crops]
    monkeypatch.setattr(cnn, "PAPER_WHITE_FROM", 240)
    for c, n in zip(crops, new):
        assert np.array_equal(cnn.std_crop(c), n)
    for ground in range(240, 255):                # 240 keeps 0.14.0's pedestal
        assert _bg_level(cnn.std_crop(_word_on(ground))) == pytest.approx(1 - ground / 255,
                                                                         abs=1 / 255)


def test_a_colour_image_of_white_paper_reaches_the_net_as_its_grey_twin():
    """White read through to_gray_u8's RGB conversion is 254, inside the near-white band, so this
    word from a colour image reaches the net exactly as its greyscale twin does (0.14.0 left it a
    one-level grey ground)."""
    from pdf_strikethrough.lines import to_gray_u8
    grey = _word_on(255)
    rgb = to_gray_u8(np.dstack([grey.astype(np.uint8)] * 3))
    assert rgb.max() == 254
    assert np.array_equal(cnn.std_crop(rgb), cnn.std_crop(grey))


def test_near_white_word_scores_clean_like_its_white_twin():
    p_white = float(cnn.score_crops([cnn.std_crop(_word_on(255))])[0])
    for ground in range(240, 255):
        p = float(cnn.score_crops([cnn.std_crop(_word_on(ground))])[0])
        assert p < 0.5 and abs(p - p_white) < 0.15, (ground, p)


@pytest.mark.parametrize("ground, cut", [
    (255, None), (254, 254.0), (247, 247.0), (240, 240.0), (239, cnn.SHADE_WHITE_FRAC * 239),
    (211, cnn.SHADE_WHITE_FRAC * 211), (128, cnn.SHADE_WHITE_FRAC * 128)])
def test_the_white_point_by_ground(ground, cut):
    """The paper step exactly: pixels at or above the white point go to 255 and the rest are
    scaled by 255 over it; on white paper there is no step. The crop is 200 px wide, so
    std_crop's resize fills the net's 160 px and nothing is cropped or padded."""
    from PIL import Image
    crop = _word_on(ground, w=200)
    crop[30:33, 10:190] = 160.0                   # a mid-grey stroke, where the scaling shows
    crop[::5, ::9] = min(ground + 3, 255)         # speckle lighter than the paper
    crop[2::5, 4::9] = ground - 1                 # and a level darker
    want = crop if cut is None else np.where(crop >= cut, 255.0, crop * (255.0 / cut))
    want = Image.fromarray(want.astype(np.uint8)).resize((cnn.CROP_W, cnn.CROP_H), Image.LANCZOS)
    assert np.array_equal(cnn.std_crop(crop), 1.0 - np.asarray(want, dtype=np.float32) / 255.0)


# --- grey conversion + shaded-page binarization (lines.py) ---------------------------------------

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
    and returns the whole band as ink. The gated fallback re-thresholds on a flattened
    background."""
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
