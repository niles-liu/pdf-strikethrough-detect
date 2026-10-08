"""Raster input hardening: image files as cameras and screenshot tools write them (placeholder
dpi, EXIF rotation, transparency, color), the array conversions behind every raster, and the
scanned-path records and settings that fail fast instead of silently detecting nothing."""
import io
import os
import warnings

import numpy as np
import pymupdf
import pytest
from PIL import Image

import pdf_strikethrough as st
from pdf_strikethrough import detect
from pdf_strikethrough.lines import to_gray_u8
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import classify_lines


def _page(highlight=None):
    """A rendered 200-dpi page with 'struck' struck through: (gray, rgb, exact Word boxes)."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=120)
    page.insert_text((20, 60), "keep struck", fontsize=20)
    r = {w[4]: pymupdf.Rect(w[:4]) for w in page.get_text("words")}["struck"]
    if highlight is not None:
        page.draw_rect(pymupdf.Rect(r.x0 - 2, r.y0, r.x1 + 2, r.y1), color=None, fill=highlight,
                       overlay=False)
    y = (r.y0 + r.y1) / 2
    page.draw_line((r.x0, y), (r.x1, y), width=1.5)
    W, H = page.rect.width, page.rect.height
    words = [Word(w[4], (w[0] / W, w[1] / H, w[2] / W, w[3] / H)) for w in page.get_text("words")]
    rgb = page.get_pixmap(dpi=200, colorspace=pymupdf.csRGB)
    rgb = np.frombuffer(rgb.samples, np.uint8).reshape(rgb.height, rgb.width, 3).copy()
    gray = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(gray.samples, np.uint8).reshape(gray.height, gray.width).copy()
    return gray, rgb, words


def _struck(res):
    return [w["chars"] for w in res["words"] if w["final"]]


def _png(arr, **save):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", **save)
    return buf.getvalue()


# ----------------------------------------------------------------------- image files

@pytest.mark.parametrize("meta_dpi", [72, 96])
def test_placeholder_metadata_dpi_is_not_trusted(meta_dpi):
    gray, _, words = _page()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.detect_image_file(_png(gray, dpi=(meta_dpi, meta_dpi)), words=words)
    assert _struck(res) == ["struck"]
    assert any(f"{meta_dpi} dpi" in w and "placeholder" in w for w in res["warnings"])


def test_real_metadata_dpi_is_trusted_silently():
    gray, _, words = _page()
    res = st.detect_image_file(_png(gray, dpi=(200, 200)), words=words)
    assert _struck(res) == ["struck"] and res["warnings"] == []


def test_transparent_png_reads_as_ink_on_white():
    gray, _, words = _page()
    rgba = np.zeros(gray.shape + (4,), np.uint8)       # black ink, alpha = how much ink
    rgba[..., 3] = 255 - gray
    res = st.detect_image_file(_png(rgba, dpi=(200, 200)), words=words)
    assert _struck(res) == ["struck"]


def test_exif_rotated_photo_is_turned_upright_for_ocr():
    gray, _, words = _page()
    exif = Image.Exif()
    exif[0x0112] = 6                                    # stored rotated; displays upright
    stored = Image.fromarray(gray).rotate(90, expand=True)
    buf = io.BytesIO()
    stored.save(buf, format="PNG", exif=exif.tobytes(), dpi=(200, 200))
    seen = []

    def ocr(image):                                     # an engine reading what it is handed
        seen.append(image.shape[:2])
        return words if image.shape[:2] == gray.shape else []
    res = st.detect_image_file(buf.getvalue(), ocr=ocr)
    assert seen == [gray.shape] and _struck(res) == ["struck"]


def test_exif_rotation_with_supplied_words_warns():
    gray, _, words = _page()
    exif = Image.Exif()
    exif[0x0112] = 3
    buf = io.BytesIO()
    Image.fromarray(gray).save(buf, format="PNG", exif=exif.tobytes(), dpi=(200, 200))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.detect_image_file(buf.getvalue(), words=words)
    assert any("EXIF rotation" in w for w in res["warnings"])


def test_color_image_file_matches_the_array_and_pdf_conversion():
    _, rgb, words = _page(highlight=(0.6, 0.9, 0.6))                # a green highlighter
    frame = detect._image_frames(_png(rgb))[0][0]
    assert np.array_equal(frame, to_gray_u8(rgb))
    res = st.detect_image_file(_png(rgb, dpi=(200, 200)), words=words)
    assert _struck(res) == _struck({"words": st.detect_scanned_image(rgb, words)})


def test_open_binary_file_is_accepted(tmp_path):
    gray, _, words = _page()
    path = tmp_path / "page.png"
    path.write_bytes(_png(gray, dpi=(200, 200)))
    with open(path, "rb") as f:
        res = st.detect_image_file(f, words=words)
    assert _struck(res) == ["struck"] and res["source"] == str(path)


def test_file_is_released_after_a_decode_error(tmp_path):
    frames = [Image.new("L", (64, 64), 255) for _ in range(3)]
    buf = io.BytesIO()
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
    path = tmp_path / "truncated.tif"
    path.write_bytes(buf.getvalue()[: len(buf.getvalue()) // 2])
    with pytest.raises((OSError, ValueError)):
        st.detect_image_file(str(path), words=[])
    os.remove(path)                                     # an open handle would lock it on Windows


# ----------------------------------------------------------------------- array conversion

@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint16, np.uint32])
def test_wide_integer_arrays_holding_8_bit_data(dtype):
    gray, _, _ = _page()
    assert np.array_equal(to_gray_u8(gray.astype(dtype)), gray)


def test_16_bit_data_scales_by_257():
    gray, _, _ = _page()
    assert np.array_equal(to_gray_u8(gray.astype(np.int32) * 257), gray)


def test_rgba_and_gray_alpha_arrays_composite_over_white():
    clear = np.zeros((4, 4, 4), np.uint8)
    assert to_gray_u8(clear).min() >= 254                # fully transparent -> paper, not ink
    la = np.zeros((4, 4, 2), np.uint8)
    la[..., 1] = 255                                     # opaque black
    assert to_gray_u8(la).max() == 0


def test_score_word_on_a_16_bit_page_matches_8_bit():
    gray, _, words = _page()
    box = {w.text: w.bbox for w in words}["struck"]
    assert st.score_word(gray.astype(np.uint16) * 257, box) == st.score_word(gray, box)


# ----------------------------------------------------------------------- records and settings

def test_unscored_scanned_record_carries_cnn_prob_none():
    (rec,) = detect.apply_cnn_verdict(
        [{"tier": "auto", "bbox_frac": (0.0, 0.0, 0.5, 0.5), "text": "x", "conf": None}],
        np.full((8, 8), 255, np.uint8), meta={"p_hi": 0.85, "p_lo": 0.15})
    assert "cnn_prob" in rec and rec["cnn_prob"] is None


def test_caller_built_line_without_len_in_is_scored():
    gray, _, words = _page()
    H, W = gray.shape
    b = {w.text: w.bbox for w in words}["struck"]
    y = int((b[1] + b[3]) / 2 * H)
    line = {"bbox_px": (int(b[0] * W), y - 1, int(b[2] * W), y + 1)}
    classify_lines([line], words, gray, config=st.ScanConfig.confidence_free())   # no KeyError


@pytest.mark.parametrize("kw", [{"cnn_p_hi": 85}, {"cnn_p_hi": float("nan")},
                                {"cnn_p_hi": "0.9"}, {"max_clean_conf": 1.5},
                                {"cnn_p_hi": 0.2, "cnn_p_lo": 0.6}])
def test_scan_config_rejects_out_of_range_settings(kw):
    with pytest.raises(ValueError):
        st.ScanConfig(**kw)


def test_scan_config_presets_still_build():
    for cfg in (st.ScanConfig(), st.ScanConfig.confidence_free(), st.ScanConfig.recall_first(),
                st.ScanConfig.precision_first(), st.ScanConfig.ruled_forms()):
        assert isinstance(cfg, st.ScanConfig)


@pytest.mark.parametrize("dpi", [0, -200])
def test_non_positive_dpi_is_rejected(dpi):
    gray, _, words = _page()
    with pytest.raises(ValueError, match="dpi"):
        st.detect_scanned_image(gray, words, dpi=dpi)
    with pytest.raises(ValueError, match="dpi"):
        st.detect_image_file(_png(gray), words=words, dpi=dpi)


def test_word_rejects_nan_and_orders_corners():
    with pytest.raises(ValueError, match="finite"):
        Word("x", (float("nan"),) * 4)
    assert Word("x", (0.5, 0.4, 0.1, 0.2)).bbox == (0.1, 0.2, 0.5, 0.4)
    conf = Word("x", (0.1, 0.1, 0.2, 0.2), np.float32(0.5)).confidence
    assert type(conf) is float


# ----------------------------------------------------------------------- review follow-ups

def test_malformed_exif_block_is_read_as_no_rotation():
    gray, _, words = _page()
    buf = io.BytesIO()
    Image.fromarray(gray).save(buf, format="PNG", exif=b"Exif\x00\x00not-a-tiff", dpi=(200, 200))
    assert _struck(st.detect_image_file(buf.getvalue(), words=words)) == ["struck"]


def test_tiff_orientation_is_applied_on_load_without_a_warning():
    gray, _, words = _page()
    buf = io.BytesIO()
    Image.fromarray(gray).rotate(90, expand=True).save(
        buf, format="TIFF", tiffinfo={274: 6}, dpi=(200, 200))     # Pillow turns it upright on load
    frame, upright, _dpi = detect._image_frames(buf.getvalue())[0]
    assert frame.shape == gray.shape and upright is None
    res = st.detect_image_file(buf.getvalue(), words=words)
    assert _struck(res) == ["struck"] and not any("EXIF" in w for w in res["warnings"])


def test_scan_config_takes_numpy_numbers_and_stores_floats():
    cfg = st.ScanConfig(cnn_p_hi=np.float32(0.9), max_clean_conf=np.float64(0.95))
    assert type(cfg.cnn_p_hi) is float and type(cfg.max_clean_conf) is float
