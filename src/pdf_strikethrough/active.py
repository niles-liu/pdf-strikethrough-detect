"""Active-learning crop export (R-active): turn a detection run into a labeling set.

The scanned pipeline scores each candidate word with StrikeNet; the borderline ones (tier
'review') and the geometry-strong ones it double-checks (tier 'auto') are exactly the crops worth
labeling to improve the model. :func:`dump_crops` runs a normal detection and writes, per scored
word, the standardized CNN input crop as a PNG plus a row of the model's verdict and the geometry
evidence to a JSONL manifest. Hand-fill each row's ``label`` ("struck" / "clean") and the set feeds
retraining (``training/train_strikenet.py``) and threshold calibration
(:mod:`pdf_strikethrough.calibration`). A "contribute a failing page" bug report becomes training
data the same way.

Only words the pipeline actually scores are exported (scanned tiers 'auto' / 'review'). Geometry-
rejected 'weak' words and exact native/vector strikes are never cropped by the detector, so they
are out of scope here.
"""
from __future__ import annotations

import json
import os

from .detect import RENDER_DPI, detect_image_file, detect_pdf

# Fields copied from each scored record into its manifest row (present-if-set).
_CROP_EVIDENCE = ("page", "text", "chars", "tier", "verdict", "final", "partial",
                  "score", "cnn_prob", "cnn_agrees", "conf", "off", "wcov", "bbox_frac")


def _save_crop_png(std, path):
    """Write a standardized crop (CROP_H x CROP_W float32, ink-positive in [0, 1]) as a normal
    dark-ink-on-white PNG a human can read."""
    import numpy as np
    from PIL import Image
    arr = np.clip(1.0 - np.asarray(std, dtype="float32"), 0.0, 1.0)
    Image.fromarray((arr * 255.0).astype("uint8")).save(path)


def dump_crops(source, out_dir, *, ocr=None, scan_config=None, dpi=None, di_result=None,
               words_by_page=None, pages=None, image=False, tiers=("auto", "review")):
    """Run detection on `source` and export every scored word crop under `out_dir` for labeling.

    Writes ``out_dir/crops/*.png`` (one standardized CNN input per scored word) and
    ``out_dir/crops.jsonl`` (one row per crop: the crop filename, the model's verdict, the
    geometry evidence, and a ``label`` field left null for you to fill). Returns a summary dict
    ``{out_dir, manifest, n_crops, source}``.

    `image=True` treats `source` as a raster image file (routes through
    :func:`detect_image_file`); otherwise it is a PDF (:func:`detect_pdf`). `tiers` narrows which
    scored tiers are exported (default: the CNN-adjudicated 'auto' and 'review'; None = all scored).
    The remaining arguments mirror :func:`detect_pdf`.
    """
    sink: list = []
    if image:
        res = detect_image_file(source, ocr=ocr, words_by_page=words_by_page,
                                scan_config=scan_config, dpi=dpi, _crop_sink=sink)
    else:
        res = detect_pdf(source, ocr=ocr, scan_config=scan_config,
                         dpi=dpi if dpi is not None else RENDER_DPI, di_result=di_result,
                         words_by_page=words_by_page, pages=pages, _crop_sink=sink)

    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    manifest = os.path.join(out_dir, "crops.jsonl")
    want = set(tiers) if tiers is not None else None
    n = 0
    with open(manifest, "w", encoding="utf-8") as f:
        for i, (std, rec) in enumerate(sink):
            if want is not None and rec.get("tier") not in want:
                continue
            fname = f"crops/p{int(rec.get('page', 0)):04d}_{i:05d}.png"
            _save_crop_png(std, os.path.join(out_dir, fname))
            row = {"crop": fname, "label": None,
                   **{k: rec[k] for k in _CROP_EVIDENCE if k in rec}}
            json.dump(row, f, default=list, ensure_ascii=False)
            f.write("\n")
            n += 1
    return {"out_dir": out_dir, "manifest": manifest, "n_crops": n, "source": res.get("source")}
