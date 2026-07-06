"""Train and export StrikeNet — the 79k-param strike/clean CNN the package ships as ONNX.

This closes the reproducibility loop: the shipped ``strike_verdict_cnn.onnx`` can be regenerated
from a labeled crop set, and the loop from failing pages back to a better model is one command each
step:

    pdf-strikethrough detect scan.pdf --ocr rapidocr --dump-crops crops_out/   # 1. export crops
    # 2. label: edit crops_out/crops.jsonl, set each row's "label" to "struck" or "clean"
    python training/train_strikenet.py crops_out/ -o model_out/                # 3. train + export
    # 4. use it: st.cnn.ensure_model(...) or PDF_STRIKETHROUGH_MODEL_DIR=model_out/

The dataset directory is a :func:`pdf_strikethrough.active.dump_crops` output: a ``crops.jsonl``
whose rows carry a ``crop`` path (relative to the directory) and a filled-in ``label``. The crop
PNGs are re-standardized here through the exact same ``std_crop`` the detector uses at inference,
so training and inference preprocessing cannot drift. Requires ``torch`` (dev only — the package
runs on ONNX Runtime alone).

    python training/train_strikenet.py DATASET_DIR [-o OUT_DIR] [--epochs N] [--val-frac F]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from pdf_strikethrough import calibration
from pdf_strikethrough.cnn import CROP_H, CROP_W, PAD_X, PAD_Y, std_crop

LABELS = {"struck": 1.0, "clean": 0.0}


def load_dataset(dataset_dir):
    """Read a dump_crops directory -> (X: (N, CROP_H, CROP_W) float32, y: (N,) float32). Rows with
    no/unknown ``label`` are skipped with a count (they still need labeling)."""
    from PIL import Image
    dataset_dir = pathlib.Path(dataset_dir)
    manifest = dataset_dir / "crops.jsonl"
    if not manifest.exists():
        sys.exit(f"no crops.jsonl in {dataset_dir} (point at a `dump_crops` output directory)")
    xs, ys, skipped = [], [], 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        label = (row.get("label") or "").strip().lower()
        if label not in LABELS:
            skipped += 1
            continue
        gray = np.asarray(Image.open(dataset_dir / row["crop"]).convert("L"), dtype=np.uint8)
        xs.append(std_crop(gray))                 # same preprocessing as inference
        ys.append(LABELS[label])
    if not xs:
        sys.exit(f"no labeled rows in {manifest} (set each row's \"label\" to struck/clean first)")
    if skipped:
        print(f"note: skipped {skipped} unlabeled row(s)")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def train(x, y, *, epochs=40, val_frac=0.2, batch=64, lr=1e-3, seed=0):
    """Train StrikeNet on standardized crops. Returns (net, val_probs, val_labels)."""
    import torch
    from torch import nn

    from pdf_strikethrough.cnn import _build_torch_net
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx = rng.permutation(len(x))
    n_val = max(1, int(len(x) * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    xt = torch.from_numpy(x[tr_idx][:, None, :, :])
    yt = torch.from_numpy(y[tr_idx])
    xv = torch.from_numpy(x[val_idx][:, None, :, :])

    net = _build_torch_net()
    pos = float(y[tr_idx].sum())
    pos_weight = torch.tensor([(len(tr_idx) - pos) / max(pos, 1.0)])  # rebalance rare struck class
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        net.train()
        order = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            sel = order[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(net(xt[sel]), yt[sel])
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"  epoch {ep + 1}/{epochs}: train loss {float(loss):.4f}")
    net.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(net(xv)).numpy()
    return net, val_probs, y[val_idx]


def export(net, out_dir, *, p_hi, p_lo, version):
    """Export the trained net to ONNX + meta.json in the layout the loader expects, recording the
    crop geometry so a preprocessing mismatch is caught at load time (cnn._check_geometry)."""
    import torch
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "strike_verdict_cnn.onnx"
    dummy = torch.zeros(1, 1, CROP_H, CROP_W)
    torch.onnx.export(net, dummy, str(onnx_path), input_names=["crop"], output_names=["logit"],
                      dynamic_axes={"crop": {0: "batch"}, "logit": {0: "batch"}}, opset_version=17)
    meta = {"version": version, "p_hi": round(float(p_hi), 4), "p_lo": round(float(p_lo), 4),
            "crop_h": CROP_H, "crop_w": CROP_W, "pad_x": PAD_X, "pad_y": PAD_Y}
    (out_dir / "strike_verdict_cnn.meta.json").write_text(json.dumps(meta, indent=2))
    return onnx_path, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("dataset_dir", help="a `dump_crops` output directory (crops.jsonl + crops/)")
    ap.add_argument("-o", "--out", default="model_out", help="output directory for the ONNX + meta")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="conformal miss rate for p_hi: p_hi guarantees a >= 1-alpha recall floor")
    ap.add_argument("--version", default="retrained", help="version string stamped into the meta")
    args = ap.parse_args(argv)

    x, y = load_dataset(args.dataset_dir)
    print(f"loaded {len(x)} labeled crops ({int(y.sum())} struck, {int((1 - y).sum())} clean)")
    net, val_probs, val_y = train(x, y, epochs=args.epochs, val_frac=args.val_frac)

    # p_hi: split-conformal threshold on validation struck-word probabilities (guaranteed recall
    # floor of 1-alpha). p_lo: the precision-oriented clean boundary, mirrored below.
    struck_probs = val_probs[val_y > 0.5]
    p_hi = calibration.conformal_threshold(struck_probs, alpha=args.alpha) if struck_probs.size \
        else 0.85
    clean_probs = val_probs[val_y <= 0.5]
    p_lo = float(np.quantile(clean_probs, 1 - args.alpha)) if clean_probs.size else 0.15
    p_lo = min(p_lo, p_hi)                          # keep the 'unsure' band well-formed
    print(f"calibrated thresholds: p_hi={p_hi:.3f}, p_lo={p_lo:.3f} (alpha={args.alpha})")

    onnx_path, meta = export(net, args.out, p_hi=p_hi, p_lo=p_lo, version=args.version)
    print(f"exported {onnx_path} + meta.json: {meta}")


if __name__ == "__main__":
    main()
