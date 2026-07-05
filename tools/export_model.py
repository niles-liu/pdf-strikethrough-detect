"""Export the trained strike-verdict checkpoint to ONNX for torch-free serving.

Reads a StrikeNet checkpoint (as produced by the training notebook) and writes the two files
the package ships and serves via onnxruntime (no PyTorch at runtime):
  src/pdf_strikethrough/strike_verdict_cnn.onnx        the network (dynamic batch, logits output)
  src/pdf_strikethrough/strike_verdict_cnn.meta.json   {version, p_hi, p_lo, crop_h, crop_w, pad_x, pad_y}

Run (requires torch + onnxruntime — the package's [dev]/[torch] extras):
    python tools/export_model.py --ckpt /path/to/strike_verdict_cnn.pt

Re-run whenever the model is retrained, then rebuild the wheel so the new ONNX ships.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

# default output: the package's model dir (this file lives in <repo>/tools/)
PKG_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src", "pdf_strikethrough"))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to the trained strike_verdict_cnn.pt")
    ap.add_argument("--out", default=PKG_DIR, help="output dir (default: the package model dir)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = StrikeNet()
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
    dummy = torch.randn(4, 1, crop_h, crop_w)
    try:
        torch.onnx.export(
            net, dummy, onnx_path,
            input_names=["crops"], output_names=["logits"],
            dynamic_axes={"crops": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
    except TypeError:  # older torch without the dynamo kwarg
        torch.onnx.export(
            net, dummy, onnx_path,
            input_names=["crops"], output_names=["logits"],
            dynamic_axes={"crops": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )

    # numerical parity check against onnxruntime
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(11, 1, crop_h, crop_w).astype(np.float32)
    with torch.no_grad():
        torch_logits = net(torch.from_numpy(x)).numpy().reshape(-1)
    onnx_logits = sess.run(None, {"crops": x})[0].reshape(-1)
    max_abs = float(np.abs(torch_logits - onnx_logits).max())
    # float32 accumulation differs between torch and ORT (BN folding); 1e-3 in logit space is
    # ~2e-4 in probability — far below anything that could flip a verdict at the 0.85/0.15 gates.
    assert max_abs < 1e-3, f"ONNX/torch mismatch {max_abs}"

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)

    print(f"exported {meta['version']}  p_hi={meta['p_hi']} p_lo={meta['p_lo']}  "
          f"onnx/torch max|Δlogit|={max_abs:.2e}")
    print(f"  -> {onnx_path}")
    print(f"  -> {meta_path}")


if __name__ == "__main__":
    main()
