"""StrikeNet — a tiny (79k-param) CNN that answers the physical question
"is there a strike through this word crop?" from pixels alone.

It is OCR- and domain-independent: the input is a grayscale word crop, the output is a
strike probability in [0, 1]. It exists to resolve cases geometry alone cannot decide
(a real thin strike over an ascender-less word looks, pixel-for-pixel, like a serif glyph
chain) and as an agreement check on geometric detections.

Inference prefers ONNX Runtime (small dependency, CPU-fast); PyTorch is an optional fallback
for dev environments that supply a ``.pt`` checkpoint. The package ships ONNX only:
  strike_verdict_cnn.onnx + strike_verdict_cnn.meta.json
Override the model location with the PDF_STRIKETHROUGH_MODEL_DIR environment variable.

Thresholds live in the checkpoint metadata: p >= p_hi -> 'struck', p <= p_lo -> 'clean',
between -> 'unsure' (guardrailed 0.85 / 0.15).
"""
import json
import os
import threading

import numpy as np
from PIL import Image

CROP_H, CROP_W = 32, 160          # net input: ink-positive [0,1], height-normalized isotropically
PAD_X, PAD_Y = 5, 7               # crop margin around the word box, in PIXELS

# --- Calibration note (see the F6 TODO / roadmap R-highdpi) -----------------------------------
# The crop geometry above is calibrated for pages rendered at RENDER_DPI = 200 (detect.py). PAD_X/
# PAD_Y are FIXED PIXEL margins, so their share of a word box changes with resolution — the pad is
# a large fraction of a small 72-dpi glyph box and a tiny one at 600 dpi, which drifts the CNN input
# off its training distribution at non-200 dpi. `detect_pdf` renders scanned pages at 200 by
# default, so the shipped path is on-distribution; if you score crops from images rendered at a very
# different dpi, expect degraded CNN confidence. The proper fix (dpi-proportional pads, or box-height-
# relative pads, followed by a model RE-EXPORT so meta.json matches) is deferred and should be done
# together with the high-DPI downsampling work (roadmap R-highdpi) so it isn't re-tuned twice.

_MODEL_DIR_OVERRIDE = None        # set via set_model_dir(); wins over the env var
_lock = threading.Lock()
_model = None                     # lazy singleton: (score_fn, meta dict)


def _current_model_dir():
    """Resolve the model directory at LOAD time (not import time), so setting
    PDF_STRIKETHROUGH_MODEL_DIR — or calling set_model_dir() — before the first score actually
    takes effect (the documented 'set it then import' override used to silently no-op because the
    env var was read once at import)."""
    return (_MODEL_DIR_OVERRIDE
            or os.environ.get("PDF_STRIKETHROUGH_MODEL_DIR")
            or os.path.dirname(os.path.abspath(__file__)))


def set_model_dir(path):
    """Point the loader at a directory holding strike_verdict_cnn.onnx + .meta.json (or a .pt),
    clearing any already-loaded model so the next score reloads from `path`. Pass None to revert
    to PDF_STRIKETHROUGH_MODEL_DIR / the packaged model."""
    global _MODEL_DIR_OVERRIDE, _model
    with _lock:
        _MODEL_DIR_OVERRIDE = path
        _model = None


def word_crop_px(gray, bbox_frac, pad_x=PAD_X, pad_y=PAD_Y):
    """Grey page raster (0=black..255=white; uint8 or float — [0,1] floats are handled) +
    normalized word box (x0,y0,x1,y1 as PAGE FRACTIONS in [0,1], origin top-left) -> padded
    word crop (float32 HxW), or None if the box is too small. Raises ValueError when the box
    looks like pixel coordinates rather than fractions."""
    H, W = gray.shape
    wx0, wy0, wx1, wy1 = bbox_frac
    if max(abs(wx0), abs(wy0), abs(wx1), abs(wy1)) > 1.5:
        raise ValueError(
            f"bbox_frac must be normalized page fractions in [0,1], got {bbox_frac!r} "
            "(these look like pixel coordinates — divide by the image width/height)")
    x0, x1 = max(0, int(wx0 * W) - pad_x), min(W, int(wx1 * W) + pad_x)
    y0, y1 = max(0, int(wy0 * H) - pad_y), min(H, int(wy1 * H) + pad_y)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return gray[y0:y1, x0:x1].astype(np.float32)


def std_crop(crop):
    """Raw grey crop (0=black..255=white; [0,1] floats are rescaled, values clipped — no mod-256
       wraparound) -> (CROP_H, CROP_W) float32, ink-positive, height-normalized isotropically;
       width center-cropped/padded (a strike spans the word, so any window still shows it)."""
    crop = np.asarray(crop)
    if np.issubdtype(crop.dtype, np.floating):
        if crop.size and float(crop.max()) <= 1.0:
            crop = crop * 255.0
        crop = np.clip(crop, 0.0, 255.0)
    elif crop.dtype != np.uint8 and np.issubdtype(crop.dtype, np.integer):
        # rescale wide integer crops (16-bit and up); a bare .astype(uint8) below would wrap mod-256
        if np.iinfo(crop.dtype).max > 255:
            crop = crop.astype(np.float64) * (255.0 / np.iinfo(crop.dtype).max)
        crop = np.clip(crop, 0.0, 255.0)
    ch, cw = crop.shape
    nw = max(12, int(round(cw * CROP_H / ch)))
    arr = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize((nw, CROP_H), Image.LANCZOS),
                     dtype=np.float32) / 255.0
    arr = 1.0 - arr
    if nw >= CROP_W:
        o = (nw - CROP_W) // 2
        return arr[:, o:o + CROP_W].copy()
    out = np.zeros((CROP_H, CROP_W), np.float32)
    out[:, (CROP_W - nw) // 2:(CROP_W - nw) // 2 + nw] = arr
    return out


def _build_torch_net():
    import torch.nn as nn

    class StrikeNet(nn.Module):
        def __init__(self):
            super().__init__()
            def blk(ci, co):
                return [nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                        nn.MaxPool2d(2)]
            self.net = nn.Sequential(*blk(1, 16), *blk(16, 32), *blk(32, 64),
                                     nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return StrikeNet()


def _check_geometry(meta):
    """The shipped meta records the crop/pad geometry the model was trained with. If a
    retrained model ships different values than the code constants, preprocessing would silently
    feed off-distribution crops — fail loudly instead."""
    for key, const in (("crop_h", CROP_H), ("crop_w", CROP_W), ("pad_x", PAD_X), ("pad_y", PAD_Y)):
        if key in meta and meta[key] != const:
            raise ValueError(
                f"model meta {key}={meta[key]} disagrees with the code constant {const}: the "
                "model was trained with different preprocessing geometry. Re-export the model or "
                "align the constants in cnn.py before using it.")


def _load_model():
    """Try ONNX Runtime first, then a torch checkpoint. Returns (score_fn, meta)."""
    model_dir = _current_model_dir()
    onnx_path = os.path.join(model_dir, "strike_verdict_cnn.onnx")
    meta_path = os.path.join(model_dir, "strike_verdict_cnn.meta.json")
    pt_path = os.path.join(model_dir, "strike_verdict_cnn.pt")

    if os.path.exists(onnx_path) and os.path.exists(meta_path):
        import onnxruntime as ort
        with open(meta_path) as f:
            meta = json.load(f)
        _check_geometry(meta)
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name

        def score(batch):
            logits = sess.run(None, {iname: batch})[0].reshape(-1)
            return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

        meta["runtime"] = "onnx"
        return score, meta

    if os.path.exists(pt_path):
        import torch
        # weights_only=True: never execute arbitrary pickle from a user-pointed model dir
        # (the checkpoint holds only tensors + scalar thresholds, which load fine under it).
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
        net = _build_torch_net()
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        meta = {"p_hi": ckpt["p_hi"], "p_lo": ckpt["p_lo"],
                "version": ckpt.get("version", "unknown"), "runtime": "torch"}
        _check_geometry({k: ckpt[k] for k in ("crop_h", "crop_w", "pad_x", "pad_y") if k in ckpt})

        def score(batch):
            with torch.no_grad():
                return torch.sigmoid(net(torch.from_numpy(batch))).numpy().astype(np.float64)

        return score, meta

    raise FileNotFoundError(
        f"no strike-verdict model in {model_dir} "
        "(need strike_verdict_cnn.onnx + .meta.json, or strike_verdict_cnn.pt with torch)")


def get_model_meta():
    """Model metadata: {version, p_hi, p_lo, runtime}. Loads the model on first call."""
    _ensure_loaded()
    return dict(_model[1])


def _ensure_loaded():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = _load_model()


def score_crops(std_crops, batch_size=512):
    """Standardized crops (list/array of (CROP_H, CROP_W) float32) -> strike probabilities."""
    if not len(std_crops):
        return np.zeros(0)
    _ensure_loaded()
    score, _ = _model
    x = np.stack(std_crops).astype(np.float32)[:, None, :, :]
    return np.concatenate([score(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def score_word(gray, bbox_frac):
    """Convenience: grayscale page (0=black..255=white) + word box in [0,1] PAGE FRACTIONS ->
    strike probability (or None if the box is too small to crop). Raises ValueError if the box
    looks like pixel coordinates."""
    crop = word_crop_px(gray, bbox_frac)
    if crop is None:
        return None
    return float(score_crops([std_crop(crop)])[0])


def verdict_of(p, meta=None):
    """CNN probability -> 'struck' / 'clean' / 'unsure' using the checkpoint thresholds."""
    if meta is None:
        _ensure_loaded()
        meta = _model[1]
    return "struck" if p >= meta["p_hi"] else ("clean" if p <= meta["p_lo"] else "unsure")
