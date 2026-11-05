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
import sys
import types

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


def _at(root, rows):
    """A crop at each row's own `crop` path (folders made as needed), then the manifest."""
    for i, row in enumerate(rows):
        arr = np.full((24, 96), 255, np.uint8)
        arr[8:16, i % 90:i % 90 + 6] = 0
        path = root / row["crop"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(path)
    _write(root, rows)


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


def test_split_holds_out_whole_groups():
    pages = [i // 5 for i in range(50)]                              # 10 pages of 5 crops
    val, tr = train_strikenet.split(50, 0.2, 3, pages)
    held = {pages[i] for i in val}
    assert len(held) == 2 and len(val) == 10                         # max(1, int(10 * 0.2)) pages
    assert not held & {pages[i] for i in tr}                          # no page on both sides
    assert sorted(np.concatenate([val, tr]).tolist()) == list(range(50))
    again = train_strikenet.split(50, 0.2, 3, pages)
    assert (val == again[0]).all() and (tr == again[1]).all()        # the seed fixes it
    named = [f"p{p}" for p in pages]                                  # only the grouping counts,
    assert (train_strikenet.split(50, 0.2, 3, named)[0] == val).all()   # not the names
    two = train_strikenet.split(50, 0.2, 3, [p % 2 for p in pages])
    assert len(two[0]) == 25 and len(two[1]) == 25                   # one of two groups held out
    with pytest.raises(ValueError, match="49 groups for 50 crops"):
        train_strikenet.split(50, 0.2, 3, pages[:49])


def test_split_takes_groups_in_permuted_order_and_rounds_down():
    groups = ["b", "a", "b", "c", "a", "d", "d", "b"]
    members = [[0, 2, 7], [1, 4], [3], [5, 6]]                       # groups as they first appear
    perm = np.random.default_rng(5).permutation(4)
    val, tr = train_strikenet.split(8, 0.5, 5, groups)
    assert val.tolist() == members[perm[0]] + members[perm[1]]        # a group's crops in order
    assert tr.tolist() == members[perm[2]] + members[perm[3]]
    pages = [i // 5 for i in range(50)]
    assert len({pages[i] for i in train_strikenet.split(50, 0.29, 0, pages)[0]}) == 2  # 2.9 pages
    assert len(train_strikenet.split(50, 0.29, 0)[0]) == 14                            # 14.5 crops


def test_split_without_groups_is_the_crop_split_it_was():
    perm = np.random.default_rng(3).permutation(50)                  # 0.14.0's split, in order
    for groups in (None, list(range(50)), [f"c{i}" for i in range(50)]):
        val, tr = train_strikenet.split(50, 0.2, 3, groups)
        assert val.tolist() == perm[:10].tolist() and tr.tolist() == perm[10:].tolist()


def test_groups_are_the_group_else_the_page_in_its_folder_else_the_row():
    rows = [{"crop": "a/crops/c0.png", "page": 1},
            {"crop": "./a/crops//c1.png", "page": 1},                 # a's page 1, however written
            {"crop": "b\\crops\\c2.png", "page": 1},                  # b's page 1
            {"crop": "b/crops/c3.png", "page": 1},
            {"crop": "a/crops/c4.png", "page": 2, "group": "w7"},     # a group beats a page
            {"crop": "b/crops/c5.png", "group": "w7"},
            {"crop": "a/crops/c6.png", "page": 3, "group": 0},        # 0 is a group
            {"crop": "b/crops/c7.png", "page": 4, "group": 0},
            {"crop": "a/crops/c8.png", "page": 5, "group": ""},       # a blank cell is none
            {"crop": "a/crops/c9.png", "page": 5, "group": float("nan")},
            {"crop": "a/crops/c10.png", "page": None},                # alone
            {"crop": "a/crops/c11.png", "page": " ", "group": []},    # alone
            {"crop": "a/crops/c12.png", "group": False},              # false is a group
            {"crop": "a/crops/c13.png", "page": 2},                   # a's page 2, but untagged
            {"crop": "a/crops/c14.png", "group": {"doc": "x", "part": 1}},
            {"crop": "a/crops/c15.png", "group": {"part": 1, "doc": "x"}}]   # key order aside
    groups, split_by, mixed = train_strikenet.groups_of(rows)
    assert groups.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 9]
    assert split_by == {"group": 7, "page": 7, "row": 2}
    assert mixed == 1                                                 # a's page 2, c4 and c13


def test_the_rows_load_keeps_are_the_crops_it_returns(tmp_path, capsys):
    rows = [{"crop": "crops/c0.png", "label": "struck", "page": 1},
            {"crop": "crops/c1.png", "label": "maybe", "page": 2},    # not a label: skipped
            {"crop": "crops/c2.png", "label": None, "page": 3},
            {"crop": "crops/c3.png", "label": " Clean", "page": 4}]
    _at(tmp_path, rows)
    text = (tmp_path / "crops.jsonl").read_text(encoding="utf-8")
    (tmp_path / "crops.jsonl").write_text("\n" + text + "  \n", encoding="utf-8")  # blank lines
    x, y, digest, kept = train_strikenet._load(tmp_path)
    assert [r["page"] for r in kept] == [1, 4] and y.tolist() == [1.0, 0.0] and len(x) == 2
    assert digest == train_strikenet.load_dataset(tmp_path)[2]
    assert "note: skipped 2 unlabeled row(s)" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no crops.jsonl"):
        train_strikenet.load_dataset(tmp_path / "missing")
    _write(tmp_path, [rows[2]])
    with pytest.raises(SystemExit, match="no labeled rows"):
        train_strikenet.load_dataset(tmp_path)


def test_split_shortfall_needs_both_classes_on_both_sides():
    y = np.array([1.0] * 300 + [0.0] * 300, dtype=np.float32)
    pages = [0] * 300 + [1 + i // 10 for i in range(300)]            # every strike on page 0
    assert train_strikenet.split_shortfall(y, 0.2, 0, 0.05) is None   # by crop it would pass
    held = [s for s in range(30)
            if 0 in {pages[i] for i in train_strikenet.split(600, 0.2, s, pages)[0]}]
    assert held                                       # seeds that hold page 0 out and train blind
    for labels, where in ((y, "struck crops are in 1 of the 31 group(s)"),
                          (1 - y, "clean crops in 1,")):                # (or every clean crop)
        for seed in range(30):
            got = train_strikenet.split_shortfall(labels, 0.2, seed, 0.05, pages)
            assert where in got and "no split can both train and calibrate" in got
    assert "training split 0 struck and 250 clean" in train_strikenet.split_shortfall(
        y, 0.2, held[0], 0.05, pages)


def test_split_shortfall_says_what_a_grouped_split_can_still_try():
    y = np.array(([1.0] * 30 + [0.0] * 10) * 3 + [0.0] * 300, dtype=np.float32)
    pages = [i // 40 for i in range(120)] + [3 + i // 10 for i in range(300)]  # strikes on 0-2
    for seed in range(40):
        val, tr = train_strikenet.split(420, 0.2, seed, pages)
        works = y[val].sum() >= 19 and (1 - y[val]).sum() and y[tr].sum() and (1 - y[tr]).sum()
        got = train_strikenet.split_shortfall(y, 0.2, seed, 0.05, pages)
        assert (got is None) == bool(works)
        assert got is None or "(6 of 33 groups held out: another --seed holds out others" in got
    few = np.array([1.0] * 10 + [0.0] * 300, dtype=np.float32)        # each crop its own group:
    for groups in (None, list(range(310))):                             # the crop split's advice
        assert train_strikenet.split_shortfall(few, 0.2, 0, 0.05, groups).endswith(
            "(label more crops, or adjust --val-frac or raise --alpha)")


def test_split_shortfall_says_when_no_seed_can_help():
    nine = np.array(([1.0] * 12 + [0.0] * 20) * 9, dtype=np.float32)   # 9 pages of 12 struck:
    pages = [i // 32 for i in range(288)]                                 # 1 held out holds 12
    for seed in range(20):
        got = train_strikenet.split_shortfall(nine, 0.2, seed, 0.05, pages)
        assert "1 of the 9 groups" in got and "keeps at most 12 struck crops" in got
        assert "no seed can help: label crops from more pages, raise --val-frac, or raise" in got
    two = np.array([1.0] * 30 + [0.0] * 200, dtype=np.float32)           # 2 pages of 15 struck:
    pages = [i // 15 for i in range(30)] + [2 + i // 10 for i in range(200)]   # one must train
    got = train_strikenet.split_shortfall(two, 0.2, 0, 0.05, pages)
    assert "4 of the 22 groups" in got and "keeps at most 15 struck crops" in got
    assert "more pages, or raise --alpha" in got                       # more groups can't help
    edge = np.array([1.0] * 19 + [0.0] * 5 + [1.0] + [0.0] * 85, dtype=np.float32)
    pages = [0] * 24 + [1] * 6 + [2 + i // 10 for i in range(80)]      # at most exactly 19
    got = [train_strikenet.split_shortfall(edge, 0.2, s, 0.05, pages) for s in range(40)]
    assert None in got                                                  # so a seed can pass
    assert all(g is None or "another --seed holds out others" in g for g in got)
    even = np.array(([1.0] * 20 + [0.0] * 20) * 10, dtype=np.float32)
    got = train_strikenet.split_shortfall(even, 1.0, 0, 0.05, [i // 40 for i in range(400)])
    assert got.endswith("--val-frac 1.0 holds out all 10 groups, leaving none to train on: use a "
                        "--val-frac below 1")


def test_train_holds_out_the_groups_it_is_given(monkeypatch):
    class Stop(Exception):
        pass
    handed = []

    def from_numpy(a):                       # train() hands torch x[tr], then y[tr]
        handed.append(a)
        if len(handed) == 2:
            raise Stop
        return a
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        manual_seed=lambda s: None, from_numpy=from_numpy, nn=None))
    groups = [i // 4 for i in range(40)]
    x = np.repeat(np.arange(40, dtype=np.float32), 4).reshape(40, 2, 2)   # crop i is all i
    y = (np.arange(40) % 3 == 0).astype(np.float32)
    with pytest.raises(Stop):
        train_strikenet.train(x, y, groups=groups, val_frac=0.25, seed=7)
    val, tr = train_strikenet.split(40, 0.25, 7, groups)
    assert handed[0][:, 0, 0, 0].tolist() == tr.tolist()             # the training groups' crops
    assert handed[1].tolist() == y[tr].tolist() and not set(val) & set(tr)


def _stub_training(monkeypatch, seen):
    """main() up to the export, without torch: train() keeps the groups it was asked to split by,
    and export() keeps what main() hands it."""
    def train(x, y, *, groups=None, **hparams):
        seen["groups"] = groups
        return None

    def export(net, out, **kw):
        seen.update(kw)
        return out, kw

    find_spec = importlib.util.find_spec
    monkeypatch.setattr(train_strikenet, "train", train)
    monkeypatch.setattr(train_strikenet, "export", export)
    monkeypatch.setattr(train_strikenet.importlib.util, "find_spec",
                        lambda name, *a: object() if name in ("torch", "onnx") else
                        find_spec(name, *a))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="stub"))


def test_main_splits_by_group_and_records_the_split(tmp_path, monkeypatch, capsys):
    rows = [{"crop": f"crops/c{i}.png", "label": "struck" if i % 2 else "clean", "page": i // 4}
            for i in range(160)]                                     # 40 pages of 4 crops
    rows[0]["group"] = "w1"                                           # 1 of page 0's 4 rows
    _at(tmp_path, rows)
    seen = {}
    _stub_training(monkeypatch, seen)
    train_strikenet.main([str(tmp_path), "-o", str(tmp_path / "out"), "--val-frac", "0.25"])
    groups = [0] + [1 + i // 4 for i in range(1, 160)]
    assert seen["groups"].tolist() == groups
    out = capsys.readouterr().out
    assert "in 41 groups (1 row(s) by \"group\", 159 by page, 0 alone)" in out
    assert "note: 1 page(s) have rows both with and without a \"group\", so the split" in out
    val, tr = train_strikenet.split(160, 0.25, 0, groups)
    vs, ts = int(sum(i % 2 for i in val)), int(sum(i % 2 for i in tr))
    assert (f"validation: 10 of 41 groups, {len(val)} of 160 crops ({vs} struck, {len(val) - vs} "
            f"clean); training: {len(tr)} crops ({ts} struck, {len(tr) - ts} clean)") in out
    t = seen["training"]
    assert t["split_by"] == {"group": 1, "page": 159, "row": 0}
    assert t["n_groups"] == 41 and t["n_val_groups"] == 10 and t["n_val"] == len(val)
    lines = "".join(f"{g}\n" for g in groups)
    assert t["groups_sha256"] == hashlib.sha256(lines.encode("utf-8")).hexdigest()
    assert t["numpy_version"] == np.__version__


def test_main_stops_when_no_group_split_can_train_and_calibrate(tmp_path, monkeypatch, capsys):
    rows = [{"crop": f"crops/c{i}.png", "label": "struck" if i < 300 else "clean",
             "page": 0 if i < 300 else 1 + (i - 300) // 10} for i in range(600)]
    _at(tmp_path, rows)
    _stub_training(monkeypatch, {})
    pages = [r["page"] for r in rows]
    seed = next(s for s in range(30)                 # page 0 held out: nothing struck to train on
                if 0 in {pages[i] for i in train_strikenet.split(600, 0.2, s, pages)[0]})
    with pytest.raises(SystemExit, match="training split 0 struck.*no split can both train and"):
        train_strikenet.main([str(tmp_path), "-o", str(tmp_path / "out"), "--seed", str(seed)])
    _write(tmp_path, [{**r, "page": 7} for r in rows])                # a one-page dump
    with pytest.raises(SystemExit, match="in 1 of the 1 group.*\"group\" of its own"):
        train_strikenet.main([str(tmp_path), "-o", str(tmp_path / "out")])
    assert "in 1 group (0 row(s) by \"group\", 600 by page, 0 alone)" in capsys.readouterr().out
