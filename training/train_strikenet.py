"""Train and export StrikeNet, the 79k-param strike/clean CNN the package ships as ONNX.

The shipped ``strike_verdict_cnn.onnx`` predates this script and was not trained by it (see
``training/MODEL_CARD.md#provenance-and-training``). A model this script trains records what it
was trained on in its meta (``training``: a digest of the labeled crops, the class counts, the
validation size, the hyperparameters and how the split grouped the crops), and the loop from
failing pages back to a better model is one command each step:

    pdf-strikethrough detect scan.pdf --ocr rapidocr --dump-crops crops_out/   # 1. export crops
    # 2. label: edit crops_out/crops.jsonl, set each row's "label" to "struck" or "clean"
    python training/train_strikenet.py crops_out/ -o model_out/                # 3. train + export
    # 4. use it: st.cnn.ensure_model(...) or PDF_STRIKETHROUGH_MODEL_DIR=model_out/

The dataset directory is a :func:`pdf_strikethrough.active.dump_crops` output: a ``crops.jsonl``
whose rows carry a ``crop`` path (relative to the directory) and a filled-in ``label``, and may
carry a ``group``, whose crops the split keeps on the same side (:func:`group_key`). The crop
PNGs are re-standardized here through the exact same ``std_crop`` the detector uses at
inference, so training and inference preprocessing cannot drift. Requires ``torch`` and ``onnx``,
which torch's ONNX exporter imports (dev only: the package runs on ONNX Runtime alone).

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
import posixpath
import sys

import numpy as np

import pdf_strikethrough
from pdf_strikethrough import calibration
from pdf_strikethrough.cnn import CROP_H, CROP_W, PAD_X, PAD_Y, std_crop

LABELS = {"struck": 1.0, "clean": 0.0}


def _rows(dataset_dir):
    """Each row of a dump_crops directory's crops.jsonl with its label, lowercased and stripped
    ("" when it has none), in manifest order."""
    dataset_dir = pathlib.Path(dataset_dir)
    manifest = dataset_dir / "crops.jsonl"
    if not manifest.exists():
        sys.exit(f"no crops.jsonl in {dataset_dir} (point at a `dump_crops` output directory)")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        yield row, (row.get("label") or "").strip().lower()


def load_dataset(dataset_dir):
    """Read a dump_crops directory -> (X: (N, CROP_H, CROP_W) float32, y: (N,) float32, digest).
    Rows with no/unknown ``label`` are skipped with a count (they still need labeling). `digest` is
    the sha256 of one ``"<crop sha256> <label>"`` line per labeled row, in manifest order: it
    changes with any crop file, label or row order, and not with file names or unlabeled rows."""
    return _load(dataset_dir)[:3]


def _load(dataset_dir):
    """:func:`load_dataset`'s (X, y, digest), and the labeled rows they came from, in order."""
    from PIL import Image
    dataset_dir = pathlib.Path(dataset_dir)
    manifest = dataset_dir / "crops.jsonl"
    xs, ys, digest_lines, rows, skipped = [], [], [], [], 0
    for row, label in _rows(dataset_dir):
        if label not in LABELS:
            skipped += 1
            continue
        data = (dataset_dir / row["crop"]).read_bytes()
        gray = np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.uint8)
        xs.append(std_crop(gray))                 # same preprocessing as inference
        ys.append(LABELS[label])
        digest_lines.append(f"{hashlib.sha256(data).hexdigest()} {label}\n")
        rows.append(row)
    if not xs:
        sys.exit(f"no labeled rows in {manifest} (set each row's \"label\" to struck/clean first)")
    if skipped:
        print(f"note: skipped {skipped} unlabeled row(s)")
    digest = hashlib.sha256("".join(digest_lines).encode("utf-8")).hexdigest()
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32), digest, rows


def _given(value):
    """Whether a row's group or page holds a value: null, NaN, blank text, [] and {} count as no
    value (an empty spreadsheet cell exports as one of them), while 0 and false are values."""
    if value is None or (isinstance(value, float) and value != value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return not isinstance(value, (list, dict)) or bool(value)


def _page(row):
    """A row's page within its crop's folder, as JSON text, or None when it names no page."""
    if not _given(row.get("page")):
        return None
    folder = posixpath.dirname(posixpath.normpath(str(row["crop"]).replace("\\", "/")))
    return json.dumps([folder, row["page"]], sort_keys=True)


def group_key(row, i):
    """What keeps labeled row `i`'s crop on the same side of the split as the rest of its group:
    the row's ``group`` if it has one (any JSON value, compared as JSON text, such as a document
    name or a writer's ID), else its ``page`` within its crop's folder (dump_crops rows carry the
    page, and the folder keeps apart dumps merged into one directory, each in a folder of its
    own), else the row alone (:func:`_given` says what counts as a value)."""
    if _given(row.get("group")):
        return "group", json.dumps(row["group"], sort_keys=True)
    page = _page(row)
    return ("page", page) if page is not None else ("row", str(i))


def groups_of(rows):
    """Each row's group (:func:`group_key`), numbered as they first appear; how many rows each kind
    of key grouped (``{"group": ..., "page": ..., "row": ...}``); and how many pages have rows both
    with and without a ``group``, which the split can put on both sides."""
    keys = [group_key(row, i) for i, row in enumerate(rows)]
    number, grouped = {}, {}
    groups = np.array([number.setdefault(k, len(number)) for k in keys], dtype=np.intp)
    for row, (kind, _) in zip(rows, keys):
        page = _page(row)
        if page is not None:
            grouped.setdefault(page, set()).add(kind == "group")
    split_by = {kind: sum(k[0] == kind for k in keys) for kind in ("group", "page", "row")}
    return groups, split_by, sum(len(v) == 2 for v in grouped.values())


def split(n, val_frac, seed, groups=None):
    """The (validation, training) indices `train` uses for `n` crops: a seeded permutation of
    their groups (``groups[i]`` is crop i's; without them each crop is its own) whose first
    max(1, int(n_groups * val_frac)) are held out, so no group falls on both sides. Groups count
    in order of first appearance and keep their crops' order, so when each crop is its own group,
    the split is the same permutation of the crops as with no groups."""
    groups = range(n) if groups is None else groups
    if len(groups) != n:
        raise ValueError(f"{len(groups)} groups for {n} crops")
    members = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    order = list(members.values())
    perm = np.random.default_rng(seed).permutation(len(order))
    n_val = max(1, int(len(order) * val_frac))

    def crops(gs):
        return np.array([i for g in gs for i in order[g]], dtype=np.intp)
    return crops(perm[:n_val]), crops(perm[n_val:])


def fewest(alpha):
    """The fewest validation crops of one class from which conformal_threshold sets a threshold at
    `alpha`: the smallest n with floor(alpha * (n + 1)) >= 1, computed as it computes k."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), not {alpha}")
    need = int(np.ceil(1 / alpha)) - 1
    while int(np.floor(alpha * (need + 1))) < 1:     # 1/alpha can round down to a whole number
        need += 1                                     # (161.0 at alpha = 1/161): ceil one short
    return need


def split_shortfall(y, val_frac, seed, alpha, groups=None):
    """Why :func:`split` cannot train and calibrate a model from labels `y` (and crop `groups`), or
    None if it can: the validation split needs enough struck crops for `alpha`'s conformal
    threshold and a clean one, and the training split a struck and a clean crop. Too few struck
    validation crops and conformal_threshold returns 0.0, a model that calls every candidate
    struck; a training split without both classes trains a net that cannot learn to tell struck
    from clean. With groups (not one per crop), it names three cases where no seed can help."""
    val_idx, tr_idx = split(len(y), val_frac, seed, groups)
    n_struck = int(y[val_idx].sum())
    n_clean = len(val_idx) - n_struck
    tr_struck = int(y[tr_idx].sum())
    tr_clean = len(tr_idx) - tr_struck
    need = fewest(alpha)
    if tr_struck and tr_clean and n_struck >= need and n_clean:
        return None
    problem = (f"The validation split holds {n_struck} struck and {n_clean} clean crops and the "
               f"training split {tr_struck} struck and {tr_clean} clean; --alpha {alpha} needs at "
               f"least {need} struck and 1 clean validation crop to set the thresholds, and "
               f"training needs at least 1 struck and 1 clean crop")
    if groups is None or len(set(groups)) == len(groups):        # each crop its own group
        return problem + " (label more crops, or adjust --val-frac or raise --alpha)"
    by_group = {}
    for g, label in zip(groups, y):
        by_group.setdefault(g, []).append(float(label))
    struck = sorted((int(sum(v)) for v in by_group.values() if any(v)), reverse=True)
    with_clean = sum(0.0 in v for v in by_group.values())
    held = len({groups[i] for i in val_idx})
    if len(struck) < 2 or with_clean < 2:                       # each class needs both sides
        return (f"{problem}. The split keeps groups whole, so each class must be in 2 or more "
                f"groups, but struck crops are in {len(struck)} of the {len(by_group)} group(s) "
                f"and clean crops in {with_clean}, so no split can both train and calibrate: "
                f"label crops from more pages or groups, or give each row a \"group\" of its own, "
                f"such as its line number, to split by crop (crops of one page can then land on "
                f"both sides)")
    if held == len(by_group):
        return (f"{problem}. --val-frac {val_frac} holds out all {held} groups, leaving none to "
                f"train on: use a --val-frac below 1")
    best = sum(struck[:min(held, len(struck) - 1)])    # leaving a struck group to train on
    if best < need:
        more = ", raise --val-frac" if sum(struck) - struck[-1] >= need else ""
        return (f"{problem}. Holding out {held} of the {len(by_group)} groups and leaving struck "
                f"crops to train on keeps at most {best} struck crops for validation, so no seed "
                f"can help: label crops from more pages{more}, or raise --alpha")
    return (f"{problem} ({held} of {len(by_group)} groups held out: another --seed holds out "
            f"others and may pass; or label crops from more pages, adjust --val-frac or raise "
            f"--alpha)")


def as_called(p):
    """Each probability as the detector compares it with p_hi and p_lo: rounded to 3 decimals by
    Python's round(), as detect.apply_cnn_verdict sets a record's cnn_prob. Not np.round, which
    differs at some .0005 edges (0.0085 gives 0.008 there, 0.009 here)."""
    return np.array([round(float(v), 3) for v in np.asarray(p, dtype=float).reshape(-1)])


def conformal_pair(struck, clean, alpha):
    """(p_hi, p_lo) from validation scores, rounded as the detector rounds them
    (:func:`as_called`). p_hi is calibration.conformal_threshold of the struck scores at `alpha`:
    the k-th lowest, k = floor(alpha * (n + 1)). p_lo mirrors it on the clean scores: the k-th
    highest, with k from the clean count, or 1.0 when that k is 0. p_lo is then capped at p_hi.
    The detector calls a score at or above p_hi struck before it checks p_lo, so p_lo == p_hi
    leaves no word unsure."""
    struck, clean = as_called(struck), as_called(clean)
    if not (np.isfinite(struck).all() and np.isfinite(clean).all()):
        raise ValueError("validation scores must be finite")
    p_hi = calibration.conformal_threshold(struck, alpha=alpha)
    k = int(np.floor(alpha * (clean.size + 1)))      # conformal_threshold's order statistic
    p_lo = float(np.sort(clean)[clean.size - k]) if k >= 1 else 1.0
    return p_hi, min(p_lo, p_hi)


def train(x, y, *, groups=None, epochs=40, val_frac=0.2, batch=64, lr=1e-3, seed=0):
    """Train StrikeNet on standardized crops, holding out :func:`split`'s validation groups.
    Returns the trained net, in eval mode."""
    import torch
    from torch import nn

    from pdf_strikethrough.cnn import _build_torch_net
    torch.manual_seed(seed)
    _, tr_idx = split(len(x), val_frac, seed, groups)
    xt = torch.from_numpy(x[tr_idx][:, None, :, :])
    yt = torch.from_numpy(y[tr_idx])

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
    return net


def score_graph(graph, crops, batch=512):
    """StrikeNet probabilities for standardized crops from the ONNX graph `graph` (bytes),
    computed as cnn.score_crops computes them."""
    import onnxruntime as ort
    if not len(crops):
        return np.zeros(0)
    sess = ort.InferenceSession(graph, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.stack(crops).astype(np.float32)[:, None, :, :]
    logits = [sess.run(None, {name: x[i:i + batch]})[0].reshape(-1)
              for i in range(0, len(x), batch)]
    return 1.0 / (1.0 + np.exp(-np.concatenate(logits).astype(np.float64)))


def export(net, out_dir, *, thresholds, version, training=None):
    """Export the trained net to ONNX + meta.json in the layout the loader expects, recording the
    crop geometry so a preprocessing mismatch is caught at load time (cnn._check_geometry), and
    `training` (what the model was trained on) when given. The graph goes to a temporary file and
    must score like `net`. Then export calls ``thresholds(score)``, where ``score(crops)`` scores
    standardized crops through that graph as cnn.score_crops will (:func:`score_graph`), and
    writes the (p_hi, p_lo) it returns as given: they must already be rounded to 3 decimals
    (:func:`as_called`) and satisfy the loader's check, 0 <= p_lo <= p_hi <= 1. Nothing is
    replaced until all that succeeds: a failure raises and leaves `out_dir`'s previous model as
    it was. If the replace itself fails, the old model and its meta stay."""
    import onnxruntime as ort
    import torch

    from pdf_strikethrough.cnn import _check_thresholds
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
        graph = tmp.read_bytes()
        sess = ort.InferenceSession(graph, providers=["CPUExecutionProvider"])
        got = sess.run(None, {sess.get_inputs()[0].name: x})[0].reshape(-1)
        drift = float(np.abs(got - want).max())
        if not drift < 1e-3:                        # ~2e-4 in probability; NaN fails too
            raise RuntimeError(f"the ONNX export disagrees with the trained net: max |dlogit| "
                               f"{drift:.2e}")
        p_hi, p_lo = (float(t) for t in thresholds(lambda crops: score_graph(graph, crops)))
        meta = {"version": version, "p_hi": p_hi, "p_lo": p_lo,
                "crop_h": CROP_H, "crop_w": CROP_W, "pad_x": PAD_X, "pad_y": PAD_Y}
        if training is not None:
            meta["training"] = training
        _check_thresholds(meta)
        if round(p_hi, 3) != p_hi or round(p_lo, 3) != p_lo:
            raise ValueError(f"p_hi {p_hi!r} and p_lo {p_lo!r} must be rounded to 3 decimals, as "
                             f"the detector rounds the scores it compares with them (see "
                             f"as_called)")
        text = json.dumps(meta, indent=2, allow_nan=False)
        old = meta_path.read_bytes() if meta_path.exists() else None
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    try:
        meta_path.unlink(missing_ok=True)           # the new graph never sits beside an old meta
        tmp.replace(onnx_path)
    except BaseException:
        if old is not None and tmp.exists() and not meta_path.exists():
            meta_path.write_bytes(old)              # the replace failed: the old pair stands
        tmp.unlink(missing_ok=True)
        raise
    meta_path.write_text(text, encoding="utf-8", newline="\n")
    return onnx_path, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("dataset_dir", help="a `dump_crops` output directory (crops.jsonl + crops/)")
    ap.add_argument("-o", "--out", default="model_out", help="output directory for the ONNX + meta")
    ap.add_argument("--epochs", type=int, default=40, help="passes over the training crops")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of groups held out to set p_hi / p_lo; a row's group is its "
                         "\"group\" field, else its page, else the row alone")
    ap.add_argument("--batch", type=int, default=64, help="minibatch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    ap.add_argument("--seed", type=int, default=0, help="seeds the train/val split and torch")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="p_hi's miss rate: on average, at most alpha of struck crops like the "
                         "validation ones score below p_hi; p_lo mirrors p_hi on the clean crops, "
                         "capped at p_hi, and promises no rate")
    ap.add_argument("--version", default="retrained", help="version string stamped into the meta")
    args = ap.parse_args(argv)
    if not 0.0 < args.alpha < 1.0:
        ap.error(f"--alpha must be between 0 and 1, not {args.alpha}")
    missing = [m for m in ("torch", "onnx") if importlib.util.find_spec(m) is None]
    if missing:                                     # now, not after the epochs: export needs both
        sys.exit(f"training needs {' and '.join(missing)}: pip install {' '.join(missing)}")

    x, y, digest, rows = _load(args.dataset_dir)
    groups, split_by, mixed = groups_of(rows)
    n_groups = len(set(groups.tolist()))
    print(f"loaded {len(x)} labeled crops ({int(y.sum())} struck, {int((1 - y).sum())} clean) "
          f"in {n_groups} group{'s' * (n_groups != 1)} ({split_by['group']} row(s) by \"group\", "
          f"{split_by['page']} by page, {split_by['row']} alone)")
    if mixed:
        print(f"note: {mixed} page(s) have rows both with and without a \"group\", so the split "
              f"can put such a page on both sides; to keep a page whole, give all its rows the "
              f"same \"group\", or none")
    problem = split_shortfall(y, args.val_frac, args.seed, args.alpha, groups)  # before the epochs
    if problem:
        sys.exit(problem)
    val_idx, tr_idx = split(len(y), args.val_frac, args.seed, groups)   # as train() splits
    n_val_groups, val_struck = len(set(groups[val_idx].tolist())), int(y[val_idx].sum())
    print(f"validation: {n_val_groups} of {n_groups} groups, {len(val_idx)} of {len(y)} crops "
          f"({val_struck} struck, {len(val_idx) - val_struck} clean); training: "
          f"{len(tr_idx)} crops ({int(y[tr_idx].sum())} struck, "
          f"{len(tr_idx) - int(y[tr_idx].sum())} clean)")
    hparams = {"epochs": args.epochs, "val_frac": args.val_frac, "batch": args.batch,
               "lr": args.lr, "seed": args.seed}
    net = train(x, y, groups=groups, **hparams)

    def thresholds(score):
        # p_hi: the split-conformal threshold on the validation strikes; p_lo: its mirror on the
        # clean crops. Both from the exported graph's scores, as the detector reads them.
        p, val_y = score(x[val_idx]), y[val_idx]
        p_hi, p_lo = conformal_pair(p[val_y > 0.5], p[val_y <= 0.5], args.alpha)
        print(f"calibrated thresholds: p_hi={p_hi:.3f}, p_lo={p_lo:.3f} (alpha={args.alpha})")
        if p_lo == p_hi:
            clean = as_called(p[val_y <= 0.5])
            k = int(np.floor(args.alpha * (len(clean) + 1)))   # conformal_pair's clean k
            if k < 1:
                why = (f"--alpha {args.alpha} needs {fewest(args.alpha)} clean validation crops "
                       f"for p_lo to sit below p_hi and the validation split holds {len(clean)} "
                       f"(label more clean crops, or raise --val-frac or --alpha)")
            else:
                why = (f"at --alpha {args.alpha}, {k} or more clean validation crops at or above "
                       f"p_hi is too many for p_lo to sit below p_hi")
            print(f"note: p_lo equals p_hi, so the model calls no word unsure: {why}. Clean "
                  f"validation crops at or above p_hi, so called struck: "
                  f"{int((clean >= p_hi).sum())} of {len(clean)}")
        return p_hi, p_lo

    import torch
    groups_digest = hashlib.sha256("".join(f"{g}\n" for g in groups).encode("utf-8")).hexdigest()
    training = {"dataset_sha256": digest, "n_struck": int(y.sum()),
                "n_clean": int((1 - y).sum()), "n_val": len(val_idx), "split_by": split_by,
                "n_groups": n_groups, "n_val_groups": n_val_groups,
                "groups_sha256": groups_digest, **hparams,
                "alpha": args.alpha, "package_version": pdf_strikethrough.__version__,
                "numpy_version": np.__version__, "torch_version": torch.__version__}
    onnx_path, meta = export(net, args.out, thresholds=thresholds, version=args.version,
                             training=training)
    print(f"exported {onnx_path} + meta.json: {meta}")


if __name__ == "__main__":
    main()
