"""The training script's dataset digest and split: what a retrained StrikeNet records about its
data, and the check that stops a run before training when the split cannot calibrate it.

`training/train_strikenet.py` stamps a sha256 of its labeled set into the exported meta, so a model
can be traced back to the crops and labels it learned from (the shipped weights cannot be: their
training set is not recorded). That is only worth recording if the digest has the documented form
and moves with the crops, their labels and their order, not with file names or unlabeled rows. The
script is not part of the package, so this runs from a source checkout and skips against an
installed wheel."""
import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pytest
from PIL import Image

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "training" / "train_strikenet.py"
if not SCRIPT.is_file():
    pytest.skip("running against an installed package, not a source checkout",
                allow_module_level=True)

_spec = importlib.util.spec_from_file_location("train_strikenet", SCRIPT)
train_strikenet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_strikenet)


def _write(root, rows):
    (root / "crops.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                      encoding="utf-8")


def _dataset(root, labels):
    """A `dump_crops`-shaped directory: one distinct crop per row, labeled as given."""
    (root / "crops").mkdir()
    rows = []
    for i, label in enumerate(labels):
        arr = np.full((24, 96), 255, np.uint8)
        arr[8:16, 4 + 8 * i:10 + 8 * i] = 0
        name = f"crops/c{i}.png"
        Image.fromarray(arr).save(root / name)
        rows.append({"crop": name, "label": label})
    _write(root, rows)
    return rows


def _digest(root):
    return train_strikenet.load_dataset(root)[2]


def _darken_corner(path):
    arr = np.asarray(Image.open(path)).copy()
    arr[0, 0] = 0
    Image.fromarray(arr).save(path)


def test_digest_is_the_documented_hash(tmp_path):
    rows = _dataset(tmp_path, ["struck", "Clean ", None])
    lines = "".join(f"{hashlib.sha256((tmp_path / r['crop']).read_bytes()).hexdigest()} {lab}\n"
                    for r, lab in zip(rows[:2], ["struck", "clean"]))   # normalized labels
    assert _digest(tmp_path) == hashlib.sha256(lines.encode("utf-8")).hexdigest()


def test_digest_tracks_crops_and_labels_not_file_names(tmp_path):
    rows = _dataset(tmp_path, ["struck", "clean", "struck", None])
    x, y, base = train_strikenet.load_dataset(tmp_path)
    assert x.shape[0] == 3 and y.tolist() == [1.0, 0.0, 1.0]     # the unlabeled row is skipped
    assert len(base) == 64 and _digest(tmp_path) == base

    _darken_corner(tmp_path / "crops" / "c3.png")                  # an unlabeled crop is not data
    assert _digest(tmp_path) == base
    (tmp_path / "crops" / "c1.png").rename(tmp_path / "crops" / "renamed.png")
    rows[1]["crop"] = "crops/renamed.png"
    _write(tmp_path, rows)
    assert _digest(tmp_path) == base                               # nor is a rename

    rows[3]["label"] = "clean"                                     # labeling a row is
    _write(tmp_path, rows)
    assert _digest(tmp_path) != base
    rows[3]["label"], rows[0]["label"] = None, "clean"             # and so is a relabel
    _write(tmp_path, rows)
    assert _digest(tmp_path) != base
    rows[0]["label"] = "struck"
    _write(tmp_path, rows)
    assert _digest(tmp_path) == base

    _write(tmp_path, [rows[2], rows[1], rows[0], rows[3]])          # and the order, which moves
    assert _digest(tmp_path) != base                               # the seeded split
    _write(tmp_path, rows)
    _darken_corner(tmp_path / "crops" / "c0.png")                  # and one pixel
    assert _digest(tmp_path) != base


def test_split_holds_out_a_seeded_fraction():
    val, tr = train_strikenet.split(50, 0.2, seed=3)
    assert len(val) == 10 and len(tr) == 40
    assert sorted(np.concatenate([val, tr]).tolist()) == list(range(50))
    again = train_strikenet.split(50, 0.2, seed=3)
    assert (val == again[0]).all() and (tr == again[1]).all()        # the seed fixes it
    assert train_strikenet.split(1, 0.2, seed=0)[1].size == 0        # one crop: none to train on


def test_split_shortfall_names_what_the_split_lacks():
    shortfall = train_strikenet.split_shortfall
    labels = np.array([1.0] * 300 + [0.0] * 300, dtype=np.float32)
    assert shortfall(labels, 0.2, 0, 0.05) is None
    few_struck = np.array([1.0] * 10 + [0.0] * 300, dtype=np.float32)
    assert "at least 19 struck" in shortfall(few_struck, 0.2, 0, 0.05)
    no_clean = np.ones(300, dtype=np.float32)
    assert "0 clean crops" in shortfall(no_clean, 0.2, 0, 0.05)
    assert "training split 0" in shortfall(np.ones(1, dtype=np.float32), 0.2, 0, 0.05)
    assert "training split 0" in shortfall(labels, 1.0, 0, 0.05)    # only the training clause
