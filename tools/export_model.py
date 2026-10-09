"""Export the trained strike-verdict checkpoint to ONNX for torch-free serving.

Reads a StrikeNet checkpoint, a ``torch.save``'d dict with ``state_dict``, ``p_hi`` and ``p_lo``
and optionally ``version``, ``crop_h``, ``crop_w``, ``pad_x`` and ``pad_y`` (the format of the
training notebook that produced the shipped weights, which is not in this repository;
``training/train_strikenet.py`` trains and exports ONNX in one step), and writes the two files the
package ships and serves via onnxruntime (no PyTorch at runtime):
  src/pdf_strikethrough/strike_verdict_cnn.onnx        the network (dynamic batch, logits output)
  src/pdf_strikethrough/strike_verdict_cnn.meta.json   version, thresholds and crop geometry

The graph goes to a temporary file and replaces the existing model only once ONNX Runtime scores it
like the checkpoint, so a failed export leaves the previous model and its meta as they were.

Run (requires torch, the package's [torch] extra, and onnx, which torch's ONNX exporter imports;
onnxruntime is a base dependency):
    python tools/export_model.py --ckpt /path/to/strike_verdict_cnn.pt

Re-run whenever you have a new checkpoint in this format, then rebuild the wheel so the new ONNX
ships.
"""
import argparse
import inspect
import json
import os

import numpy as np
import torch

from pdf_strikethrough.cnn import _build_torch_net

# default output: the package's model dir (this file lives in <repo>/tools/)
PKG_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src", "pdf_strikethrough"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to the trained strike_verdict_cnn.pt")
    ap.add_argument("--out", default=PKG_DIR, help="output dir (default: the package model dir)")
    args = ap.parse_args()

    # your own checkpoint: the package's loader, which reads user-pointed model dirs, is the one
    # that insists on weights_only=True
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = _build_torch_net()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    crop_h = ckpt.get("crop_h", 32)
    crop_w = ckpt.get("crop_w", 160)
    meta = {"version": ckpt.get("version", "unknown"),
            "p_hi": ckpt["p_hi"], "p_lo": ckpt["p_lo"],
            "crop_h": crop_h, "crop_w": crop_w,
            "pad_x": ckpt.get("pad_x", 5), "pad_y": ckpt.get("pad_y", 7)}

    os.makedirs(args.out, exist_ok=True)
    onnx_path = os.path.join(args.out, "strike_verdict_cnn.onnx")
    meta_path = os.path.join(args.out, "strike_verdict_cnn.meta.json")
    tmp = onnx_path + ".tmp"
    kw = {"input_names": ["crops"], "output_names": ["logits"], "opset_version": 17,
          "dynamic_axes": {"crops": {0: "batch"}, "logits": {0: "batch"}}}
    # the TorchScript exporter: torch >= 2.9 defaults to the dynamo one, which needs onnxscript
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False
    try:
        torch.onnx.export(net, torch.randn(4, 1, crop_h, crop_w), tmp, **kw)
        import onnxruntime as ort
        sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
        x = np.random.randn(11, 1, crop_h, crop_w).astype(np.float32)
        with torch.no_grad():
            torch_logits = net(torch.from_numpy(x)).numpy().reshape(-1)
        onnx_logits = sess.run(None, {"crops": x})[0].reshape(-1)
        max_abs = float(np.abs(torch_logits - onnx_logits).max())
        # float32 accumulation differs between torch and ORT (BN folding); 1e-3 in logit space is
        # ~2e-4 in probability, far below anything that could flip a verdict at the 0.85/0.15
        # gates. A comparison, not an assert, so `python -O` cannot skip it; NaN fails too.
        if not max_abs < 1e-3:
            raise RuntimeError(f"ONNX/torch mismatch: max |dlogit| {max_abs:.2e}")
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    if os.path.exists(meta_path):
        os.remove(meta_path)                  # the new graph never sits beside an old meta
    os.replace(tmp, onnx_path)
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:   # LF, as git stores it
        json.dump(meta, f, indent=1)

    print(f"exported {meta['version']}  p_hi={meta['p_hi']} p_lo={meta['p_lo']}  "
          f"onnx/torch max |dlogit| {max_abs:.2e}")       # ASCII: a cp1252 console can't print Δ
    print(f"  -> {onnx_path}")
    print(f"  -> {meta_path}")


if __name__ == "__main__":
    main()
