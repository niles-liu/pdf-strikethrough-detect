"""Train and export StrikeNet — the 79k-param strike/clean CNN the package ships as ONNX.

The shipped ``strike_verdict_cnn.onnx`` predates this script and was not trained by it (see
``training/MODEL_CARD.md#provenance-and-training``). A model this script trains records what it
was trained on in its meta (``training``: a digest of the labeled crops, the class counts, the
validation size and the hyperparameters), and the loop from failing pages back to a better model
is one command each step:

    pdf-strikethrough detect scan.pdf --ocr rapidocr --dump-crops crops_out/   # 1. export crops
    # 2. label: edit crops_out/crops.jsonl, set each row's "label" to "struck" or "clean"
    python training/train_strikenet.py crops_out/ -o model_out/                # 3. train + export
    # 4. use it: st.cnn.ensure_model(...) or PDF_STRIKETHROUGH_MODEL_DIR=model_out/

The dataset directory is a :func:`pdf_strikethrough.active.dump_crops` output: a ``crops.jsonl``
whose rows carry a ``crop`` path (relative to the directory) and a filled-in ``label``. The crop
PNGs are re-standardized here through the exact same ``std_crop`` the detector uses at inference,
so training and inference preprocessing cannot drift. Requires ``torch`` and ``onnx``, which
torch's ONNX exporter imports (dev only — the package runs on ONNX Runtime alone).

    python training/train_strikenet.py DATASET_DIR [-o OUT_DIR] [--epochs N] [--val-frac F]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import io
import json
import pathlib
import sys

import numpy as np

import pdf_strikethrough
from pdf_strikethrough import calibration
from pdf_strikethrough.cnn import CROP_H, CROP_W, PAD_X, PAD_Y, std_crop

LABELS = {"struck": 1.0, "clean": 0.0}


def load_dataset(dataset_dir):
    """Read a dump_crops directory -> (X: (N, CROP_H, CROP_W) float32, y: (N,) float32, digest).
    Rows with no/unknown ``label`` are skipped with a count (they still need labeling). `digest` is
    the sha256 of one ``"<crop sha256> <label>"`` line per labeled row, in manifest order: it
    changes with any crop file, label or row order, and not with file names or unlabeled rows."""
    from PIL import Image
    dataset_dir = pathlib.Path(dataset_dir)
    manifest = dataset_dir / "crops.jsonl"
    if not manifest.exists():
        sys.exit(f"no crops.jsonl in {dataset_dir} (point at a `dump_crops` output directory)")
    xs, ys, digest_lines, skipped = [], [], [], 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        label = (row.get("label") or "").strip().lower()
        if label not in LABELS:
            skipped += 1
            continue
        data = (dataset_dir / row["crop"]).read_bytes()
        gray = np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.uint8)
        xs.append(std_crop(gray))                 # same preprocessing as inference
        ys.append(LABELS[label])
        digest_lines.append(f"{hashlib.sha256(data).hexdigest()} {label}\n")
    if not xs:
        sys.exit(f"no labeled rows in {manifest} (set each row's \"label\" to struck/clean first)")
    if skipped:
        print(f"note: skipped {skipped} unlabeled row(s)")
    digest = hashlib.sha256("".join(digest_lines).encode("utf-8")).hexdigest()
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32), digest


def split(n, val_frac, seed):
    """The (validation, training) indices `train` uses for `n` crops: a seeded permutation whose
    first max(1, n * val_frac) entries are held out."""
    idx = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(n * val_frac))
    return idx[:n_val], idx[n_val:]


def train(x, y, *, epochs=40, val_frac=0.2, batch=64, lr=1e-3, seed=0):
    """Train StrikeNet on standardized crops, holding out :func:`split`'s validation crops.
    Returns (net, val_probs, val_labels)."""
    import torch
    from torch import nn

    from pdf_strikethrough.cnn import _build_torch_net
    torch.manual_seed(seed)
    val_idx, tr_idx = split(len(x), val_frac, seed)
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
            print(f"  epoch {ep + 1}/{epochs}: train loss {loss.item():.4f}")
    net.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(net(xv)).numpy()
    return net, val_probs, y[val_idx]


def export(net, out_dir, *, p_hi, p_lo, version, training=None):
    """Export the trained net to ONNX + meta.json in the layout the loader expects, recording the
    crop geometry so a preprocessing mismatch is caught at load time (cnn._check_geometry), and
    `training` (what the model was trained on) when given. The graph goes to a temporary file and
    must score like `net` before it replaces anything: otherwise this raises RuntimeError and
    leaves `out_dir`'s previous model as it was."""
    import onnxruntime as ort
    import torch
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "strike_verdict_cnn.onnx"
    meta_path = out_dir / "strike_verdict_cnn.meta.json"
    tmp = out_dir / "strike_verdict_cnn.onnx.tmp"
    kw = {"input_names": ["crop"], "output_names": ["logit"], "opset_version": 17,
          "dynamic_axes": {"crop": {0: "batch"}, "logit": {0: "batch"}}}
    # the TorchScript exporter: torch >= 2.9 defaults to the dynamo one, which needs onnxscript
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False
    try:
        torch.onnx.export(net, torch.zeros(1, 1, CROP_H, CROP_W), str(tmp), **kw)
        x = np.random.default_rng(0).standard_normal((8, 1, CROP_H, CROP_W)).astype(np.float32)
        with torch.no_grad():
            want = net(torch.from_numpy(x)).numpy().reshape(-1)
        sess = ort.InferenceSession(tmp.read_bytes(), providers=["CPUExecutionProvider"])
        drift = float(np.abs(sess.run(None, {"crop": x})[0].reshape(-1) - want).max())
        if not drift < 1e-3:                        # ~2e-4 in probability; NaN fails too
            raise RuntimeError(f"the ONNX export disagrees with the trained net: max |dlogit| "
                               f"{drift:.2e}")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    meta = {"version": version, "p_hi": round(float(p_hi), 4), "p_lo": round(float(p_lo), 4),
            "crop_h": CROP_H, "crop_w": CROP_W, "pad_x": PAD_X, "pad_y": PAD_Y}
    if training is not None:
        meta["training"] = training
    meta_path.unlink(missing_ok=True)               # the new graph never sits beside an old meta
    tmp.replace(onnx_path)
    meta_path.write_text(json.dumps(meta, indent=2))
    return onnx_path, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("dataset_dir", help="a `dump_crops` output directory (crops.jsonl + crops/)")
    ap.add_argument("-o", "--out", default="model_out", help="output directory for the ONNX + meta")
    ap.add_argument("--epochs", type=int, default=40, help="passes over the training crops")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of labeled crops held out to set p_hi / p_lo")
    ap.add_argument("--batch", type=int, default=64, help="minibatch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    ap.add_argument("--seed", type=int, default=0, help="seeds the train/val split and torch")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="conformal miss rate for p_hi: a >= 1-alpha recall floor on crops like "
                         "the validation set")
    ap.add_argument("--version", default="retrained", help="version string stamped into the meta")
    args = ap.parse_args(argv)
    missing = [m for m in ("torch", "onnx") if importlib.util.find_spec(m) is None]
    if missing:                                     # now, not after the epochs: export needs both
        sys.exit(f"training needs {' and '.join(missing)}: pip install {' '.join(missing)}")

    x, y, digest = load_dataset(args.dataset_dir)
    print(f"loaded {len(x)} labeled crops ({int(y.sum())} struck, {int((1 - y).sum())} clean)")
    # Checked on the split train() draws, before the epochs: too few struck validation crops and
    # conformal_threshold returns 0.0, a model that calls every candidate struck.
    val_idx, tr_idx = split(len(x), args.val_frac, args.seed)
    n_struck_val = int(y[val_idx].sum())
    n_clean_val = len(val_idx) - n_struck_val
    need = int(np.ceil(1 / args.alpha)) - 1
    if not len(tr_idx) or n_struck_val < need or not n_clean_val:
        sys.exit(f"the validation split holds {n_struck_val} struck and {n_clean_val} clean crops "
                 f"and the training split {len(tr_idx)}; --alpha {args.alpha} needs at least "
                 f"{need} struck and 1 clean to set the thresholds, and training needs a crop "
                 f"(label more crops, or adjust --val-frac or --alpha)")
    hparams = {"epochs": args.epochs, "val_frac": args.val_frac, "batch": args.batch,
               "lr": args.lr, "seed": args.seed}
    net, val_probs, val_y = train(x, y, **hparams)

    # p_hi: split-conformal threshold on validation struck-word probabilities (a 1-alpha recall
    # floor on crops like them). p_lo: the precision-oriented clean boundary, mirrored below.
    struck_probs, clean_probs = val_probs[val_y > 0.5], val_probs[val_y <= 0.5]
    p_hi = calibration.conformal_threshold(struck_probs, alpha=args.alpha)
    p_lo = float(np.quantile(clean_probs, 1 - args.alpha))
    p_lo = min(p_lo, p_hi)                          # keep the 'unsure' band well-formed
    print(f"calibrated thresholds: p_hi={p_hi:.3f}, p_lo={p_lo:.3f} (alpha={args.alpha})")

    import torch
    training = {"dataset_sha256": digest, "n_struck": int(y.sum()),
                "n_clean": int((1 - y).sum()), "n_val": len(val_y), **hparams,
                "alpha": args.alpha, "package_version": pdf_strikethrough.__version__,
                "torch_version": torch.__version__}
    onnx_path, meta = export(net, args.out, p_hi=p_hi, p_lo=p_lo, version=args.version,
                             training=training)
    print(f"exported {onnx_path} + meta.json: {meta}")


if __name__ == "__main__":
    main()
