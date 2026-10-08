"""StrikeNet loading and threshold calibration: digests that verify what they promise, thresholds a
model cannot be loaded without, and calibration helpers that read the labels and ties a
``dump_crops`` manifest actually holds."""
import hashlib
import json
import pathlib

import numpy as np
import pytest

import pdf_strikethrough as st
from pdf_strikethrough import calibration as C, cnn


def _packaged():
    here = pathlib.Path(cnn.__file__).parent
    onnx = here / "strike_verdict_cnn.onnx"
    meta = here / "strike_verdict_cnn.meta.json"
    return onnx, meta, hashlib.sha256(onnx.read_bytes()).hexdigest()


# ----------------------------------------------------------------------- ensure_model

def test_uppercase_digest_is_accepted(tmp_path):
    onnx, meta, sha = _packaged()
    try:
        st.ensure_model(onnx.as_uri(), sha.upper(), meta=json.loads(meta.read_text()),
                        cache_dir=str(tmp_path / "c"))
    finally:
        cnn.set_model_dir(None)


@pytest.mark.parametrize("digest", ["", None, "sha256:" + "0" * 64, "abc"])
def test_malformed_digest_is_rejected_before_any_download(tmp_path, digest):
    onnx, meta, _ = _packaged()
    with pytest.raises(ValueError, match="64-character"):
        st.ensure_model(onnx.as_uri(), digest, meta=json.loads(meta.read_text()),
                        cache_dir=str(tmp_path / "c"))
    assert not (tmp_path / "c").exists()


def test_meta_url_needs_its_digest(tmp_path):
    onnx, meta, sha = _packaged()
    with pytest.raises(ValueError, match="meta_sha256"):
        st.ensure_model(onnx.as_uri(), sha, meta_url=meta.as_uri(), cache_dir=str(tmp_path))


def test_tampered_meta_is_rejected(tmp_path):
    onnx, meta, sha = _packaged()
    evil = tmp_path / "evil.meta.json"
    evil.write_text(json.dumps({"p_hi": 0.0, "p_lo": 0.0}))
    good = hashlib.sha256(meta.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="sha256 mismatch"):
        st.ensure_model(onnx.as_uri(), sha, meta_url=evil.as_uri(), meta_sha256=good,
                        cache_dir=str(tmp_path / "c"))


def test_meta_url_with_digest_loads(tmp_path):
    onnx, meta, sha = _packaged()
    try:
        st.ensure_model(onnx.as_uri(), sha, meta_url=meta.as_uri(),
                        meta_sha256=hashlib.sha256(meta.read_bytes()).hexdigest(),
                        cache_dir=str(tmp_path / "c"))
        assert st.get_model_meta()["p_hi"] == json.loads(meta.read_text())["p_hi"]
    finally:
        cnn.set_model_dir(None)


@pytest.mark.parametrize("thresholds", [{"p_hi": 0.2, "p_lo": 0.6}, {"p_hi": 85, "p_lo": 15},
                                        {"p_lo": 0.15}])
def test_model_with_impossible_thresholds_is_refused(tmp_path, thresholds):
    onnx, meta, _ = _packaged()
    (tmp_path / "strike_verdict_cnn.onnx").write_bytes(onnx.read_bytes())
    bad = {**json.loads(meta.read_text()), "p_hi": None, **thresholds}
    bad = {k: v for k, v in bad.items() if v is not None}
    (tmp_path / "strike_verdict_cnn.meta.json").write_text(json.dumps(bad))
    try:
        cnn.set_model_dir(str(tmp_path))
        with pytest.raises(ValueError, match="p_lo <= p_hi"):
            st.get_model_meta()
    finally:
        cnn.set_model_dir(None)


def test_loader_hands_back_the_model_it_loaded():
    cnn.set_model_dir(None)                              # the global is reset to None here
    score, meta = cnn._ensure_loaded()                   # callers use the tuple, not the global
    assert callable(score) and "p_hi" in meta


# ----------------------------------------------------------------------- calibration

def test_manifest_string_labels_are_read_as_labels():
    probs = [0.9, 0.8, 0.3, 0.2]
    assert (C.threshold_for_precision(probs, ["struck", "struck", "clean", "clean"], 1.0)
            == C.threshold_for_precision(probs, [1, 1, 0, 0], 1.0) == 0.8)


@pytest.mark.parametrize("bad", [None, "unsure", 2])
def test_unfilled_or_unknown_label_raises(bad):
    with pytest.raises(ValueError, match="label 1"):
        C.pr_curve([0.9, 0.5], ["struck", bad])


def test_precision_threshold_respects_ties():
    probs, labels = [1, 1, 1, 1, 0.4], [1, 1, 1, 0, 1]   # at 1.0 the precision is 3/4, not 1
    with pytest.raises(ValueError, match="no threshold"):
        C.threshold_for_precision(probs, labels, 1.0)
    # 0.5 admits both tied words (one clean): precision 2/3, so only 0.9 reaches 1.0
    assert C.threshold_for_precision([0.9, 0.5, 0.5, 0.1], [1, 1, 0, 0], 1.0) == 0.9


def test_recall_threshold_is_the_highest_that_qualifies():
    pos = list(np.round(np.linspace(0.05, 0.95, 10), 2))
    t = C.threshold_for_recall(pos, [1] * 10, 0.9)
    assert t == pos[1]                                    # 9 of 10 stay at or above it
    assert sum(p >= t for p in pos) / 10 >= 0.9
