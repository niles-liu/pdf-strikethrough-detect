"""v0.9.0 "prove it" surface: CNN operating points, threshold calibration, active-learning crop
export, and the digest-verified model loader."""
import copy
import hashlib
import json
import pathlib

import fitz
import numpy as np
import pytest

import pdf_strikethrough as st
from pdf_strikethrough import calibration
from pdf_strikethrough.ocr import Word


# ------------------------------------------------------------------------- fixtures

def _redline_image_png(dpi=200):
    """PNG bytes of a rendered page with a vector strike through 'struck', + its Word boxes."""
    doc = fitz.open()
    page = doc.new_page(width=300, height=120)
    page.insert_text((20, 60), "keep struck", fontsize=20)
    b = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}
    r = b["struck"]
    ymid = (r.y0 + r.y1) / 2
    page.draw_line(fitz.Point(r.x0, ymid), fitz.Point(r.x1, ymid), width=1.5)
    png = page.get_pixmap(dpi=dpi).tobytes("png")
    W, H = page.rect.width, page.rect.height
    words = [Word(w[4], (w[0] / W, w[1] / H, w[2] / W, w[3] / H)) for w in page.get_text("words")]
    doc.close()
    return png, words


# ------------------------------------------------------------------------- R-cal calibration

def test_threshold_for_recall_meets_floor():
    probs = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    labels = np.array([0, 0, 1, 1, 1, 1])                 # 4 struck at 0.4/0.6/0.8/0.95
    t = calibration.threshold_for_recall(probs, labels, 0.75)
    recall = np.mean(probs[labels.astype(bool)] >= t)
    assert recall >= 0.75
    # a higher recall demand lowers the threshold (keeps more struck words)
    assert calibration.threshold_for_recall(probs, labels, 1.0) <= t


def test_threshold_for_precision_meets_floor():
    probs = np.array([0.1, 0.3, 0.55, 0.7, 0.9, 0.99])
    labels = np.array([0, 1, 0, 1, 1, 1])
    t = calibration.threshold_for_precision(probs, labels, 1.0)
    pred = probs >= t
    assert pred.sum() > 0
    assert np.all(labels[pred] == 1)                      # perfect precision above t
    with pytest.raises(ValueError):
        calibration.threshold_for_precision(probs, labels, 1.0 + 1e-9)  # out of range


def test_conformal_threshold_recall_guarantee():
    # 100 struck-word probabilities; the alpha-quantile threshold should keep >= 1-alpha of them
    rng = np.random.default_rng(0)
    pos = np.clip(rng.beta(6, 2, size=100), 0, 1)
    t = calibration.conformal_threshold(pos, alpha=0.1)
    assert np.mean(pos >= t) >= 0.9 - 1e-9
    assert calibration.conformal_threshold([0.9], alpha=0.4) == 0.0    # too few points -> accept all
    with pytest.raises(ValueError):
        calibration.conformal_threshold([], alpha=0.1)


def test_pr_curve_monotone_recall_and_endpoints():
    probs = np.array([0.1, 0.4, 0.6, 0.8, 0.95])
    labels = np.array([0, 1, 0, 1, 1])
    thr, prec, rec = calibration.pr_curve(probs, labels)
    assert np.all(np.diff(thr) < 0)                       # thresholds descending
    assert np.all(np.diff(rec) >= 0)                      # recall non-decreasing as t drops
    assert rec[-1] == pytest.approx(1.0)                  # lowest threshold recalls everything
    assert np.all((0 <= prec) & (prec <= 1))


def test_calibration_length_mismatch_raises():
    with pytest.raises(ValueError):
        calibration.threshold_for_recall([0.1, 0.2], [1], 0.5)


# ------------------------------------------------------------------------- R-cal operating points

def test_operating_points_carry_thresholds():
    assert st.ScanConfig.recall_first().cnn_p_hi == 0.50
    assert st.ScanConfig.precision_first().cnn_p_hi == 0.97
    assert st.ScanConfig().cnn_p_hi is None                # default = model thresholds
    rf = st.ScanConfig.recall_first(confidence_gating=False)
    assert rf.cnn_p_hi == 0.50 and rf.confidence_gating is False


def test_operating_point_changes_review_verdict(monkeypatch):
    """A review word scoring 0.7: struck under recall_first (p_hi 0.5), not under the 0.85 default."""
    from pdf_strikethrough import cnn, detect
    monkeypatch.setattr(cnn, "get_model_meta", lambda: {"p_hi": 0.85, "p_lo": 0.15})
    monkeypatch.setattr(cnn, "score_crops", lambda crops: np.full(len(crops), 0.7))
    gray = np.full((100, 400), 255, np.uint8)
    rec = {"tier": "review", "bbox_frac": (0.10, 0.40, 0.50, 0.50), "text": "x", "cnn_prob": None}

    r_def = detect.apply_cnn_verdict([copy.deepcopy(rec)], gray, config=st.ScanConfig())
    r_rec = detect.apply_cnn_verdict([copy.deepcopy(rec)], gray, config=st.ScanConfig.recall_first())
    assert r_def[0]["final"] is False and r_def[0]["verdict"] == "unsure"
    assert r_rec[0]["final"] is True and r_rec[0]["verdict"] == "struck"


# ------------------------------------------------------------------------- R-active dump_crops

def test_dump_crops_writes_manifest_and_pngs(tmp_path):
    png, words = _redline_image_png()
    out = tmp_path / "crops_out"
    summary = st.dump_crops(png, str(out), words_by_page={0: words}, image=True)
    assert summary["n_crops"] >= 1
    manifest = pathlib.Path(summary["manifest"])
    rows = [json.loads(ln) for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == summary["n_crops"]
    row = rows[0]
    assert row["label"] is None                            # left for the human to fill
    assert "cnn_prob" in row and "tier" in row and "bbox_frac" in row
    assert (out / row["crop"]).exists()                    # the referenced PNG was written
    # the struck word 'struck' should be among the exported crops
    assert any(r.get("chars") == "struck" or r.get("text") == "struck" for r in rows)


def test_dump_crops_cli(tmp_path):
    from pdf_strikethrough import __main__ as cli
    from PIL import Image
    import io
    png, words = _redline_image_png()
    img = tmp_path / "s.png"
    Image.open(io.BytesIO(png)).save(img, dpi=(200, 200))
    # supply words via a Textract result so no OCR backend is needed
    blocks = [{"BlockType": "WORD", "Text": w.text, "Confidence": 99.0, "Page": 1,
               "Geometry": {"BoundingBox": {"Left": w.bbox[0], "Top": w.bbox[1],
                            "Width": w.bbox[2] - w.bbox[0], "Height": w.bbox[3] - w.bbox[1]}}}
              for w in words]
    tj = tmp_path / "t.json"
    tj.write_text(json.dumps({"Blocks": blocks}), encoding="utf-8")
    out = tmp_path / "dump"
    rc = cli.main(["detect", str(img), "--textract-result", str(tj), "--dump-crops", str(out)])
    assert rc == 0
    assert (out / "crops.jsonl").exists()


# ------------------------------------------------------------------------- R-hf ensure_model

def _packaged_model_paths():
    d = pathlib.Path(st.cnn.__file__).parent
    return d / "strike_verdict_cnn.onnx", d / "strike_verdict_cnn.meta.json"


def test_ensure_model_verifies_and_loads(tmp_path):
    onnx, meta_json = _packaged_model_paths()
    onnx_sha = hashlib.sha256(onnx.read_bytes()).hexdigest()
    meta = json.loads(meta_json.read_text())
    cache = tmp_path / "cache"
    try:
        got = st.ensure_model(onnx.as_uri(), onnx_sha, meta=meta, cache_dir=str(cache))
        assert pathlib.Path(got) == cache
        assert (cache / "strike_verdict_cnn.onnx").exists()
        assert hashlib.sha256((cache / "strike_verdict_cnn.onnx").read_bytes()).hexdigest() == onnx_sha
        assert st.get_model_meta()["p_hi"] == meta["p_hi"]     # loader picked up the cached model
    finally:
        st.cnn.set_model_dir(None)                             # restore the packaged model


def test_ensure_model_rejects_bad_digest(tmp_path):
    onnx, _ = _packaged_model_paths()
    bad = "0" * 64
    with pytest.raises(ValueError, match="sha256 mismatch"):
        st.ensure_model(onnx.as_uri(), bad, meta={"p_hi": 0.85, "p_lo": 0.15},
                        cache_dir=str(tmp_path / "c"))
    assert not (tmp_path / "c" / "strike_verdict_cnn.onnx").exists()  # nothing written on mismatch


def test_ensure_model_needs_meta(tmp_path):
    onnx, _ = _packaged_model_paths()
    sha = hashlib.sha256(onnx.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="no model meta"):
        st.ensure_model(onnx.as_uri(), sha, cache_dir=str(tmp_path / "c"))
