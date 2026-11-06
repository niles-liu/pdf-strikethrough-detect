"""What `std_crop`, `verdict_of` and the model loader do with what a direct caller hands them.

The detector converts each page to uint8 grey and crops it as float32 before `std_crop`, so these
paths reach only code that calls the CNN helpers itself (a notebook, the training script, an
app): an integer crop holding 0-255 in a wider type read as solid ink, one NaN in a [0, 1] crop
turned it black, empty and non-2-D crops failed with unrelated errors or a solid block,
`verdict_of` compared a raw score where the detector compares it rounded to 3 decimals, and a
model trained on crops made with other `std_crop` settings scored with no word of it."""
import json
import pathlib
import warnings

import numpy as np
import pytest

from pdf_strikethrough import cnn, detect

HERE = pathlib.Path(cnn.__file__).parent


def _word(ground=255.0):
    """A grey crop of glyph ticks with a strike through them, on paper at `ground`."""
    crop = np.full((40, 200), ground, np.float64)
    for x in range(20, 180, 9):
        crop[10:30, x:x + 3] = 30
    crop[19:21, 15:185] = 30
    return crop


# ----------------------------------------------------------------------- std_crop's inputs

@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.uint16, np.int16, np.uint32])
def test_8_bit_values_in_a_wider_integer_type_read_as_themselves(dtype):
    want = cnn.std_crop(_word().astype(np.uint8))
    assert np.array_equal(cnn.std_crop(_word().astype(dtype)), want)
    assert cnn.std_crop(np.full((40, 200), 255)).max() == 0.0          # int64 white is white


def test_16_bit_data_still_scales_to_8_bits():
    word = _word().astype(np.uint8)
    assert np.array_equal(cnn.std_crop(word.astype(np.uint16) * 257), cnn.std_crop(word))


@pytest.mark.parametrize("bad, level", [(np.nan, 0.0), (np.inf, 0.0), (-np.inf, 1.0)])
def test_a_nan_or_inf_pixel_in_a_0_1_crop_leaves_the_rest_as_it_was(bad, level):
    crop = _word() / 255.0
    want = cnn.std_crop(crop)
    crop[0, 0] = bad                                   # NaN and +inf read as white, -inf as black
    got = cnn.std_crop(crop)
    if level == 0.0:                                   # on white paper: the crop it was
        assert np.array_equal(got, want)
    else:                                              # one black pixel in a corner
        assert np.isfinite(got).all() and np.abs(got - want).mean() < 1e-3
    whole = np.full((40, 200), bad)
    assert np.array_equal(cnn.std_crop(whole), np.full_like(want, level))
    crop = _word()                                     # and on the 0-255 scale
    want = cnn.std_crop(crop)
    crop[0, 0] = bad
    got = cnn.std_crop(crop)
    assert np.array_equal(got, want) if level == 0.0 else np.abs(got - want).mean() < 1e-3


@pytest.mark.parametrize("shape", [(0, 50), (50, 0), (0, 0), (50,), (40, 120, 3), (40, 120, 1),
                                   ()])
def test_a_crop_that_is_not_a_2d_grey_image_is_refused(shape):
    with pytest.raises(ValueError, match="std_crop needs a 2-D grey crop"):
        cnn.std_crop(np.zeros(shape, np.float32))


def test_float32_copies_of_uint8_pixels_standardize_as_the_pixels_do():
    """The detector's crops are float32 copies of uint8 pixels."""
    rng = np.random.default_rng(3)
    for ground in (255, 250, 241, 230, 200, 120, 60):
        crop = np.clip(_word(ground) + rng.normal(0, 4, (40, 200)), 0, 255).astype(np.uint8)
        got = cnn.std_crop(crop.astype(np.float32))
        assert np.array_equal(got, cnn.std_crop(crop))


def test_a_float_crop_no_brighter_than_1_reads_as_0_to_1_even_when_copied_from_uint8():
    """A float crop's [0, 1] reading is decided on its pixels, not where they came from: a float32
    copy of uint8 pixels that are all 1 reads as white paper, where the uint8 pixels read as
    ink. The detector's crops are such copies, so this stays as it is."""
    ones = np.ones((30, 90), np.uint8)
    assert cnn.std_crop(ones.astype(np.float32)).max() == 0.0
    assert cnn.std_crop(ones).max() > 0.9


def test_16_bit_data_in_int16_reads_with_32767_as_white():
    word = _word().astype(np.uint8)
    assert np.array_equal(cnn.std_crop((word.astype(np.int32) * 128).astype(np.int16)),
                          cnn.std_crop(word))


def test_32_bit_data_reads_with_the_type_maximum_as_white():
    word = _word().astype(np.uint8)
    assert np.array_equal(cnn.std_crop(word.astype(np.uint32) * 16843009), cnn.std_crop(word))


# ----------------------------------------------------------------------- verdict_of

@pytest.mark.parametrize("p, want, meta", [
    (0.8496, "struck", None), (0.8494, "unsure", None), (0.1504, "clean", None),
    (0.1506, "unsure", None), (np.float32(0.8496), "struck", None),
    (0.0085, "struck", {"p_hi": 0.009, "p_lo": 0.001})])     # round(), not np.round: 0.008
def test_verdict_of_reads_a_score_as_the_detector_does(monkeypatch, p, want, meta):
    meta = meta or {"p_hi": 0.85, "p_lo": 0.15}
    assert cnn.verdict_of(p, meta) == want
    rec = {"tier": "review", "bbox_frac": (0.1, 0.1, 0.3, 0.2)}
    monkeypatch.setattr(cnn, "score_crops", lambda crops, **kw: np.array([float(p)]))
    detect.apply_cnn_verdict([rec], np.full((200, 400), 255, np.uint8), meta=meta)
    assert rec["verdict"] == want


def test_verdict_of_takes_the_array_score_crops_gives_for_one_crop():
    assert cnn.verdict_of(np.array([0.8496]), {"p_hi": 0.85, "p_lo": 0.15}) == "struck"
    with pytest.raises(ValueError):
        cnn.verdict_of(np.array([0.9, 0.1]), {"p_hi": 0.85, "p_lo": 0.15})


# ----------------------------------------------------------------------- the crops a model saw

def _model(tmp_path, training):
    """The packaged model in a folder of its own, its meta carrying `training` when given."""
    meta = json.loads((HERE / "strike_verdict_cnn.meta.json").read_text(encoding="utf-8"))
    if training is not None:
        meta["training"] = training
    (tmp_path / "strike_verdict_cnn.onnx").write_bytes((HERE / "strike_verdict_cnn.onnx")
                                                       .read_bytes())
    (tmp_path / "strike_verdict_cnn.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    cnn.set_model_dir(str(tmp_path))


@pytest.fixture
def scoring(monkeypatch):
    """Score one crop with the model set up, and return the crop-settings warnings it gave."""
    monkeypatch.setattr(cnn, "_warned_settings", set())

    def score():
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            cnn.score_crops([cnn.std_crop(_word())])
        return [str(w.message) for w in seen if "std_crop settings" in str(w.message)]
    yield score
    cnn.set_model_dir(None)


SCRIPT_NOW = {"dataset_sha256": "0" * 64, "package_version": "0.14.2", "paper_white_from": 240,
              "shade_white_frac": 0.85}
SCRIPT_0140 = {"dataset_sha256": "0" * 64, "package_version": "0.14.0", "torch_version": "2.9"}
OTHER_LOOP_BLOCK = {"package_version": "0.14.0", "torch_version": "2.9"}  # no dataset_sha256


def test_a_model_trained_on_crops_made_otherwise_warns_when_it_scores(tmp_path, monkeypatch,
                                                                      scoring):
    _model(tmp_path, SCRIPT_NOW)
    said = scoring()
    assert len(said) == 1 and "paper_white_from 240" in said[0] and "255" in said[0]
    assert "shade_white_frac" not in said[0] and "cnn.PAPER_WHITE_FROM = 240" in said[0]
    assert scoring() == []                                           # once per model and settings
    monkeypatch.setattr(cnn, "PAPER_WHITE_FROM", 240)                # as it was trained
    assert scoring() == []
    monkeypatch.setattr(cnn, "SHADE_WHITE_FRAC", 0.8)                # changed after the load
    assert len(scoring()) == 1


def test_an_error_filter_raises_on_every_score(tmp_path, monkeypatch):
    monkeypatch.setattr(cnn, "_warned_settings", set())
    _model(tmp_path, SCRIPT_NOW)
    try:
        for _ in range(2):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with pytest.raises(UserWarning, match="std_crop settings"):
                    cnn.score_crops([cnn.std_crop(_word())])
    finally:
        cnn.set_model_dir(None)


def test_a_model_from_the_older_script_is_taken_as_made_at_240(tmp_path, monkeypatch, scoring):
    _model(tmp_path, SCRIPT_0140)
    said = scoring()
    assert len(said) == 1 and "paper_white_from 240" in said[0] and "0.14.0" in said[0]
    monkeypatch.setattr(cnn, "_warned_settings", set())
    monkeypatch.setattr(cnn, "PAPER_WHITE_FROM", 240)
    assert scoring() == []


@pytest.mark.parametrize("training", [None, OTHER_LOOP_BLOCK,
                                      {**SCRIPT_NOW, "paper_white_from": 255},
                                      "not a block", ["paper_white_from", 240]])
def test_a_model_that_says_nothing_else_scores_quietly(tmp_path, scoring, training):
    _model(tmp_path, training)
    assert scoring() == []


def test_the_packaged_model_scores_quietly(scoring):
    cnn.set_model_dir(None)
    assert scoring() == []
