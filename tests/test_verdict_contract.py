"""A record's verdict, as the README documents it: "struck", "clean" or "unsure", and final is True
exactly when it is "struck". A review queue routes on it."""
import itertools

import numpy as np
import pytest

import pdf_strikethrough as st
from pdf_strikethrough import cnn, detect
from _synth import _redline_image_png, _synthetic_native_pdf

META = {"p_hi": 0.85, "p_lo": 0.15}
PROBS = [0.0, 0.1, 0.15, 0.5, 0.85, 0.9, 1.0, None]          # None: a crop too small to score


def _records():
    """A scanned candidate for every tier, score, OCR confidence and geometry, each on its own
    box, plus a native record."""
    recs = []
    for i, (tier, p, conf, geom) in enumerate(itertools.product(
            ("auto", "review"), PROBS, (None, 0.5, 0.99), (False, True))):
        x = i / 200
        recs.append({"text": "word", "bbox_frac": (x, 0.1, x + 0.004, 0.2), "tier": tier, "p": p,
                     "conf": conf, "geom_corroborated": geom, "line_idx": [i % 3]})
    recs.append({"text": "word", "bbox_frac": (0.9, 0.5, 0.95, 0.6), "tier": "vector", "p": None})
    return recs


@pytest.mark.parametrize("config", [st.ScanConfig.confidence_free(), st.ScanConfig.azure_di()],
                         ids=["confidence-free", "DI calibration"])
def test_final_is_true_exactly_when_the_verdict_is_struck(monkeypatch, config):
    recs = _records()
    by_box = {r["bbox_frac"]: i for i, r in enumerate(recs)}
    # each scoreable record's crop carries its index; the model returns that record's score
    monkeypatch.setattr(cnn, "word_crop_px", lambda gray, bbox: (
        None if recs[by_box[bbox]]["p"] is None else np.full((8, 8), by_box[bbox], np.uint8)))
    monkeypatch.setattr(cnn, "std_crop", lambda crop: crop)
    monkeypatch.setattr(cnn, "score_crops", lambda crops: [recs[int(c[0, 0])]["p"] for c in crops])
    kept = detect.apply_cnn_verdict(recs, np.zeros((100, 100), np.uint8), META, config=config)
    assert len(kept) == len(recs)
    for r in kept:
        assert r["verdict"] in ("struck", "clean", "unsure")
        assert r["final"] == (r["verdict"] == "struck"), r
    assert {r["verdict"] for r in kept} == {"struck", "clean", "unsure"}


def test_detected_records_keep_it():
    png, words = _redline_image_png()
    recs = (st.detect_image_file(png, words=words)["words"]
            + st.detect_pdf(_synthetic_native_pdf())["words"])
    assert recs and all(r["final"] == (r["verdict"] == "struck") for r in recs)
