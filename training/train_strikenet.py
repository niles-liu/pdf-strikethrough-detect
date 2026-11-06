"""Train and export StrikeNet, the 79k-param strike/clean CNN the package ships as ONNX.

The shipped ``strike_verdict_cnn.onnx`` predates this script and was not trained by it (see
``training/MODEL_CARD.md#provenance-and-training``). A model this script trains records what it
was trained on in its meta (``training``: a digest of the labeled crops, the class counts, the
validation size, the hyperparameters, how the split grouped the crops and ``std_crop``'s grey-paper
settings), and the loop from failing pages back to a better model is one command each step:

    pdf-strikethrough detect scan.pdf --ocr rapidocr --dump-crops crops_out/   # 1. export crops
    # 2. label: edit crops_out/crops.jsonl, set each row's "label" to "struck" or "clean"
    python training/train_strikenet.py crops_out/ -o model_out/                # 3. train + export
    # 4. use it: st.cnn.ensure_model(...) or PDF_STRIKETHROUGH_MODEL_DIR=model_out/

The dataset directory is a :func:`pdf_strikethrough.active.dump_crops` output: a ``crops.jsonl``
whose rows carry a ``crop`` path (relative to the directory) and a filled-in ``label``, and may
carry a ``group``, whose crops the split keeps on the same side (:func:`group_key`), and a
``source``, the kind of crop p_hi is matched in (:func:`sources_of`). The crop PNGs are
re-standardized here through the exact same ``std_crop`` the detector uses at inference, so
training and inference crops share their geometry. A dump holds each crop as standardized by the
release that dumped it, so dump scans again with the release you train with (releases after
0.14.0 take paper at 240 to 254 to white). The new model's p_hi and p_lo are matched, on
the validation crops, to the calls of the model the package loads (:func:`matched_pair`).
Requires ``torch`` and ``onnx``, which torch's ONNX exporter imports (dev only: the package runs
on ONNX Runtime alone).

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
from pdf_strikethrough import calibration, cnn
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
    """Whether a row's group, page or source holds a value: null, NaN, blank text, [] and {} count
    as no value (an empty spreadsheet cell exports as one of them), while 0 and false are values."""
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


def sources_of(rows):
    """Each row's ``source`` as JSON text, or None when it has none (:func:`_given`): the kinds of
    crop, such as print and handwriting, that :func:`matched_pair` matches p_hi in one by one. Any
    other NaN or Infinity in a source stops the script, since the meta records each source."""
    def text(row):
        try:
            return json.dumps(row["source"], sort_keys=True, allow_nan=False)
        except ValueError:
            sys.exit(f"The \"source\" of {row['crop']} holds NaN or Infinity, which the model's "
                     f"meta cannot record: give it text, such as \"print\"")
    return np.array([text(row) if _given(row.get("source")) else None for row in rows],
                    dtype=object)


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
    None if it can: the validation split needs at least 19 struck crops for the matched p_hi (more
    when `alpha` is below 0.05, for its conformal threshold) and a clean one for p_lo, and the
    training split a struck and a clean crop. With fewer struck validation crops the matched p_hi
    rests on a few, and below `alpha`'s count conformal_threshold returns 0.0; a training split
    without both classes trains a net that cannot learn to tell struck from clean. With groups (not
    one per crop), it names three cases where no seed can help."""
    val_idx, tr_idx = split(len(y), val_frac, seed, groups)
    n_struck = int(y[val_idx].sum())
    n_clean = len(val_idx) - n_struck
    tr_struck = int(y[tr_idx].sum())
    tr_clean = len(tr_idx) - tr_struck
    need = max(fewest(alpha), fewest(0.05))          # 19 for the match, whatever --alpha
    if tr_struck and tr_clean and n_struck >= need and n_clean:
        return None
    match = need == fewest(0.05)                     # else the conformal pair asks for more
    who = "the thresholds need" if match else f"the split-conformal pair at --alpha {alpha} needs"
    floor = "" if match else " (19 at --alpha 0.05 or above)"
    problem = (f"The validation split holds {n_struck} struck and {n_clean} clean crops and the "
               f"training split {tr_struck} struck and {tr_clean} clean; {who} at least {need} "
               f"struck{floor} and 1 clean validation crop, and training at least 1 struck and 1 "
               f"clean crop")
    if groups is None or len(set(groups)) == len(groups):        # each crop its own group
        return problem + " (label more crops, or adjust --val-frac)"
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
        more = " or raise --val-frac" if sum(struck) - struck[-1] >= need else ""
        return (f"{problem}. Holding out {held} of the {len(by_group)} groups and leaving struck "
                f"crops to train on keeps at most {best} struck crops for validation, so no seed "
                f"can help: label crops from more pages{more}")
    return (f"{problem} ({held} of {len(by_group)} groups held out: another --seed holds out "
            f"others and may pass; or label crops from more pages or adjust --val-frac)")


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
    return _conformal(struck, clean, alpha)[:2]


def _conformal(struck, clean, alpha):
    """:func:`conformal_pair`'s (p_hi, p_lo), and p_lo's mirror before the cap: the k-th highest
    clean score, or None when k is 0."""
    struck, clean = as_called(struck), as_called(clean)
    if not (np.isfinite(struck).all() and np.isfinite(clean).all()):
        raise ValueError("validation scores must be finite")
    p_hi = calibration.conformal_threshold(struck, alpha=alpha)
    k = int(np.floor(alpha * (clean.size + 1)))      # conformal_threshold's order statistic
    mirror = float(np.sort(clean)[clean.size - k]) if k >= 1 else None
    return p_hi, min(1.0 if mirror is None else mirror, p_hi), mirror


def verdicts(p, p_hi, p_lo):
    """Each score's verdict at thresholds (p_hi, p_lo) as the detector gives it: cnn.verdict_of on
    the score as read (:func:`as_called`), so a score at p_hi is struck even when p_lo == p_hi."""
    meta = {"p_hi": p_hi, "p_lo": p_lo}
    return np.array([cnn.verdict_of(v, meta) for v in as_called(p)], dtype=object)


def matched_high(new, ref, reference):
    """The highest p_hi at which the new model calls at least as many of these crops struck as
    the reference model does at its thresholds `reference`, (p_hi, p_lo): the K-th highest of the
    new model's scores as read (:func:`as_called`), K the reference's count (:func:`verdicts`), so
    crops tied with it are called too. None when the reference calls none of them struck."""
    new, k = as_called(new), int((verdicts(ref, *reference) == "struck").sum())
    return float(np.sort(new)[new.size - k]) if k else None


def matched_low(new, ref, reference):
    """The lowest p_lo at which the new model calls at least as many of these crops clean as the
    reference model does at its thresholds `reference`, as long as p_hi sits above it: the K-th
    lowest of the new model's scores as read, K the reference's count, or 0.0 when the reference
    calls none of them clean."""
    new, k = as_called(new), int((verdicts(ref, *reference) == "clean").sum())
    return float(np.sort(new)[k - 1]) if k else 0.0


def matched_pair(new, ref, struck, sources, reference, groups=None, every=()):
    """(p_hi, p_lo, how) from a new model's validation scores `new`, matched to the reference
    model's scores `ref` for the same crops at its thresholds `reference`, (p_hi, p_lo). The
    struck crops of each source (``sources[i]`` is crop i's, from :func:`sources_of`) set a bound,
    :func:`matched_high`, unless the reference calls none of them struck. p_hi is the lowest
    bound, so in no source does the new model call fewer struck crops struck than the reference;
    with no bound at all, ValueError. p_lo is :func:`matched_low` on all the clean crops at once,
    capped at p_hi. `how` records, for p_hi, each source's struck crops (and how many `groups`
    they come from, when given), the reference's calls and the bound, the source that set it, and
    the bound for all the struck crops at once, reported only; for p_lo, the clean crops, the
    reference's calls and the match before the cap; and how many struck and clean crops each
    model calls struck at its thresholds. Sources in `every` are listed even with no struck crop
    here (they set no bound). Scores that are not finite raise ValueError."""
    new, ref = as_called(new), as_called(ref)
    if not (np.isfinite(new).all() and np.isfinite(ref).all()):
        raise ValueError("validation scores must be finite")
    struck, sources = np.asarray(struck, dtype=bool), np.asarray(sources, dtype=object)
    by_source = []
    for key in sorted(set(sources[struck].tolist()) | set(every),
                      key=lambda s: (s is not None, s or "")):
        here = struck & np.array([s == key for s in sources], dtype=bool)
        entry = {"source": None if key is None else json.loads(key), "struck": int(here.sum())}
        if groups is not None:
            entry["groups"] = len(set(np.asarray(groups)[here].tolist()))
        entry["reference_calls"] = int((verdicts(ref[here], *reference) == "struck").sum())
        entry["matched"] = matched_high(new[here], ref[here], reference)
        by_source.append(entry)
    bounds = [entry for entry in by_source if entry["matched"] is not None]
    if not bounds:
        raise ValueError(f"the reference model calls none of the {int(struck.sum())} struck "
                         f"crops struck: no recall to match")
    set_by = min(bounds, key=lambda entry: entry["matched"])          # the first, on a tie
    p_hi, clean = set_by["matched"], ~struck
    low = matched_low(new[clean], ref[clean], reference)
    p_lo = min(low, p_hi)
    old, now = verdicts(ref, *reference), verdicts(new, p_hi, p_lo)
    how = {"p_hi": {"by_source": by_source, "set_by": set_by["source"],
                    "pooled": matched_high(new[struck], ref[struck], reference)},
           "p_lo": {"clean": int(clean.sum()),
                    "reference_calls": int((old[clean] == "clean").sum()), "matched": low},
           "called_struck": {name: {"reference": int((old[mask] == "struck").sum()),
                                    "new": int((now[mask] == "struck").sum())}
                             for name, mask in (("struck", struck), ("clean", clean))}}
    return p_hi, p_lo, how


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
    try:            # a log redirected on Windows is cp1252: escape what it cannot encode
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass        # a stream that cannot be reconfigured (captured in tests, say)
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
                    help="miss rate of the split-conformal pair the meta reports beside the "
                         "matched thresholds (alpha does not change those): on average, at most "
                         "alpha of struck crops like the validation ones score below the pair's "
                         "p_hi. The validation split needs at least 19 struck crops, more when "
                         "alpha is below 0.05")
    ap.add_argument("--version", default="retrained", help="version string stamped into the meta")
    args = ap.parse_args(argv)
    if not 0.0 < args.alpha < 1.0:
        ap.error(f"--alpha must be between 0 and 1, not {args.alpha}")
    missing = [m for m in ("torch", "onnx") if importlib.util.find_spec(m) is None]
    if missing:                                     # now, not after the epochs: export needs both
        sys.exit(f"training needs {' and '.join(missing)}: pip install {' '.join(missing)}")

    x, y, digest, rows = _load(args.dataset_dir)
    groups, split_by, mixed = groups_of(rows)
    sources = sources_of(rows)
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
    # The reference is the model the package loads: its calls on the validation crops are what
    # the thresholds match, so whatever keeps them from matching stops here, before the epochs
    struck = y[val_idx] > 0.5
    try:
        reference = cnn.get_model_meta()
        ref = cnn.score_crops(x[val_idx])
    except Exception as e:                          # whatever the loader raises: nothing to match
        sys.exit(f"The thresholds are matched to the model the package loads, but no model loads "
                 f"from {cnn._current_model_dir()}: {type(e).__name__}: {e}. Set "
                 f"PDF_STRIKETHROUGH_MODEL_DIR to a folder with a model that loads, or unset it to "
                 f"match the packaged model")
    at = (reference["p_hi"], reference["p_lo"])
    recorded = {k: reference[k] for k in ("version", "sha256", "external_data", "p_hi", "p_lo")
                if k in reference}
    json.dumps(recorded, allow_nan=False)           # as export writes it: fails now, not later
    if not np.isfinite(ref).all():
        sys.exit(f"The reference model ({reference.get('version')}, sha256 "
                 f"{reference['sha256'][:12]}) gives {int((~np.isfinite(ref)).sum())} of the "
                 f"{len(ref)} validation crops no finite score, so there is nothing to match: set "
                 f"PDF_STRIKETHROUGH_MODEL_DIR to another model, or unset it to match the packaged "
                 f"model")
    caught = int((verdicts(ref[struck], *at) == "struck").sum())
    print(f"reference: {reference.get('version')} (sha256 {reference['sha256'][:12]}, p_hi "
          f"{at[0]}, p_lo {at[1]}), the model the package loads, calls {caught} of the "
          f"{val_struck} struck validation crops struck")
    if not caught:
        sys.exit(f"The reference model calls none of the {val_struck} struck validation crops "
                 f"struck, so p_hi has no recall to match: label more struck crops, try another "
                 f"--seed, or set PDF_STRIKETHROUGH_MODEL_DIR to another reference model")
    unheld = sorted(set(sources[y > 0.5].tolist()) - set(sources[val_idx][struck].tolist()),
                    key=lambda s: (s is not None, s or ""))
    for s in unheld:
        name = "the rows without a source have" if s is None else f"source {s} has"
        print(f"note: {name} struck crops only in the training split, so p_hi is not matched on "
              f"them and the new model can miss more of them: another --seed holds out other "
              f"groups")
    hparams = {"epochs": args.epochs, "val_frac": args.val_frac, "batch": args.batch,
               "lr": args.lr, "seed": args.seed}
    net = train(x, y, groups=groups, **hparams)

    def thresholds(score):
        # Matched to the reference's calls on the validation crops, both models' scores read as
        # the detector reads them; the split-conformal pair goes in the meta beside them
        p = score(x[val_idx])
        p_hi, p_lo, how = matched_pair(p, ref, struck, sources[val_idx], at, groups[val_idx],
                                       every=unheld)
        c_hi, c_lo, mirror = _conformal(p[struck], p[~struck], args.alpha)
        training["calibration"] = {
            "rule": "matched to the reference model's calls on the validation crops",
            "reference": recorded, **how,
            "conformal": {"alpha": args.alpha, "p_hi": c_hi, "p_lo": c_lo, "mirror": mirror}}
        entries, setter = how["p_hi"]["by_source"], how["p_hi"]["set_by"]
        named = "the rows without a source" if setter is None else f"source {json.dumps(setter)}"
        by = f" (set by {named})" if len(entries) > 1 else ""
        print(f"matched thresholds: p_hi={p_hi:.3f}{by}, p_lo={p_lo:.3f}")
        for e in entries:
            name = ("rows without a source" if e["source"] is None
                    else f"source {json.dumps(e['source'])}")
            if not e["struck"]:
                print(f"  {name}: no struck validation crops, so no bound")
                continue
            bound = "no bound" if e["matched"] is None else f"matched at {e['matched']:.3f}"
            print(f"  {name}: {e['struck']} struck validation crops from {e['groups']} group(s), "
                  f"{e['reference_calls']} called struck by the reference: {bound}")
        lo, called = how["p_lo"], how["called_struck"]
        cap = f", capped at p_hi {p_hi:.3f}" if p_lo < lo["matched"] else ""
        print(f"  {lo['clean']} clean validation crops, {lo['reference_calls']} called clean by "
              f"the reference: matched at {lo['matched']:.3f}{cap}")
        print(f"called struck by the reference and the new model: {called['struck']['reference']} "
              f"and {called['struck']['new']} of the {val_struck} struck validation crops, "
              f"{called['clean']['reference']} and {called['clean']['new']} of the {lo['clean']} "
              f"clean")
        print(f"split-conformal pair at --alpha {args.alpha}, reported only: p_hi={c_hi:.3f}, "
              f"p_lo={c_lo:.3f}")
        if p_lo == p_hi:
            print(f"note: p_lo equals p_hi, so the model calls no word unsure: to call as many "
                  f"clean validation crops clean as the reference ({lo['reference_calls']} of "
                  f"{lo['clean']}), p_lo would sit at {lo['matched']:.3f}, at or above p_hi. "
                  f"Clean validation crops at or above p_hi, so called struck: "
                  f"{called['clean']['new']} of {lo['clean']}")
        return p_hi, p_lo

    import torch
    groups_digest = hashlib.sha256("".join(f"{g}\n" for g in groups).encode("utf-8")).hexdigest()
    training = {"dataset_sha256": digest, "n_struck": int(y.sum()),
                "n_clean": int((1 - y).sum()), "n_val": len(val_idx), "split_by": split_by,
                "n_groups": n_groups, "n_val_groups": n_val_groups,
                "groups_sha256": groups_digest, **hparams,
                "alpha": args.alpha, "package_version": pdf_strikethrough.__version__,
                "numpy_version": np.__version__, "torch_version": torch.__version__,
                "paper_white_from": cnn.PAPER_WHITE_FROM, "shade_white_frac": cnn.SHADE_WHITE_FRAC}
    onnx_path, meta = export(net, args.out, thresholds=thresholds, version=args.version,
                             training=training)
    print(f"exported {onnx_path} + meta.json: {meta}")


if __name__ == "__main__":
    main()
