"""The training script's thresholds: p_hi and p_lo are set on the scores the detector compares.

`training/train_strikenet.py` sets a retrained StrikeNet's p_hi and p_lo from its validation crops.
The detector scores a crop through the exported ONNX graph and rounds the probability to 3
decimals before it compares it with them (`detect.apply_cnn_verdict`), so a threshold set on
other scores, or rounded again on export, can drop the crops it was set on. The script is not
part of the package, so this runs from a source checkout and skips against an installed wheel."""
import contextlib
import importlib.util
import json
import pathlib
import shutil
import sys
import types

import numpy as np
import pytest
from PIL import Image

from pdf_strikethrough import calibration, cnn, detect

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "training" / "train_strikenet.py"
if not SCRIPT.is_file():
    pytest.skip("running against an installed package, not a source checkout",
                allow_module_level=True)

_spec = importlib.util.spec_from_file_location("train_strikenet", SCRIPT)
train_strikenet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_strikenet)

SHIPPED = pathlib.Path(cnn.__file__).parent / "strike_verdict_cnn.onnx"


@pytest.fixture
def packaged_model(monkeypatch):
    """The packaged StrikeNet loaded, whatever the environment points at."""
    monkeypatch.delenv("PDF_STRIKETHROUGH_MODEL_DIR", raising=False)
    cnn.set_model_dir(None)
    yield
    cnn.set_model_dir(None)


def _detector_calls(monkeypatch, probs, p_hi, p_lo):
    """What detect.apply_cnn_verdict decides for review words StrikeNet scores `probs`: each
    word's (final, cnn_prob)."""
    recs = [{"tier": "review", "bbox_frac": (0.1, 0.1, 0.3, 0.2)} for _ in probs]
    monkeypatch.setattr(cnn, "score_crops",
                        lambda crops, batch_size=512: np.asarray(probs, np.float64))
    detect.apply_cnn_verdict(recs, np.full((200, 400), 255, np.uint8),
                             meta={"p_hi": p_hi, "p_lo": p_lo})
    return [r["final"] for r in recs], [r["cnn_prob"] for r in recs]


def _crops(n, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.random((cnn.CROP_H, cnn.CROP_W)).astype(np.float32) for _ in range(n)]


def test_scores_are_rounded_as_the_detector_rounds_them(monkeypatch):
    p = [0.84951, 0.8494, 0.12345, 0.0005, 0.0085, 0.0125, 0.9995, 1.0, 0.0]  # .0005 edges, where
    assert train_strikenet.as_called(p).tolist() == _detector_calls(       # np.round differs
        monkeypatch, p, 0.5, 0.1)[1]


def test_p_hi_keeps_the_strikes_it_was_set_on(monkeypatch):
    clean = np.linspace(0.0, 0.04, 60)
    struck = np.r_[0.05, 0.123456, np.linspace(0.3, 0.99, 37)]       # 39 strikes: the 2nd lowest
    p_hi, p_lo = train_strikenet.conformal_pair(struck, clean, 0.05)
    assert p_hi == 0.123                                              # as the detector reads it
    assert sum(_detector_calls(monkeypatch, struck, p_hi, p_lo)[0]) == 38   # all but the lowest
    near = np.r_[0.05, 0.12344, 0.12349, 0.12351, np.linspace(0.3, 0.99, 35)]
    p_hi, p_lo = train_strikenet.conformal_pair(near, clean, 0.05)
    assert p_hi == 0.123 and sum(_detector_calls(monkeypatch, near, p_hi, p_lo)[0]) == 38


def test_p_hi_is_the_conformal_order_statistic():
    struck = np.arange(1, 41) / 100                  # 40 distinct scores: k = floor(0.05 * 41) = 2
    assert train_strikenet.conformal_pair(struck, [0.0] * 40, 0.05)[0] == 0.02   # the 2nd lowest


def test_p_lo_mirrors_p_hi_on_the_clean_crops():
    rng = np.random.default_rng(3)
    struck = 0.6 + 0.4 * rng.random(200)
    for n in (19, 100, 1000):
        clean = rng.beta(1.5, 6, n)
        p_hi, p_lo = train_strikenet.conformal_pair(struck, clean, 0.05)
        called = np.sort(train_strikenet.as_called(clean))
        k = int(np.floor(0.05 * (n + 1)))                             # conformal_threshold's k
        assert p_lo == called[n - k]                                  # the k-th largest clean
        assert p_hi == calibration.conformal_threshold(train_strikenet.as_called(struck), 0.05)
    assert train_strikenet.conformal_pair(struck, [0.1] * 5, 0.05)[1] == p_hi  # too few to set it
    assert train_strikenet.conformal_pair([0.3] * 40, [0.9] * 40, 0.05) == (0.3, 0.3)  # <= p_hi
    with pytest.raises(ValueError, match="finite"):
        train_strikenet.conformal_pair(struck, np.r_[clean, np.nan], 0.05)


def test_the_graph_is_scored_as_the_package_scores_it(packaged_model):
    crops = _crops(600)                                               # past one 512-crop batch
    got = train_strikenet.score_graph(SHIPPED.read_bytes(), crops)
    assert got.dtype == np.float64 and np.array_equal(got, cnn.score_crops(crops))
    got, want = train_strikenet.score_graph(SHIPPED.read_bytes(), []), cnn.score_crops([])
    assert got.dtype == want.dtype == np.float64 and got.shape == want.shape == (0,)


def _fake_torch(monkeypatch, offset=0.0):
    """Just enough torch for export(): its ONNX export writes the shipped graph, and the net
    scores as that graph does, its logits `offset` off (export's drift check allows 1e-3)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(SHIPPED), providers=["CPUExecutionProvider"])

    def onnx_export(net, dummy, path, **kw):
        shutil.copyfile(SHIPPED, path)

    def net(x):
        logits = sess.run(None, {sess.get_inputs()[0].name: np.asarray(x, np.float32)})[0]
        return types.SimpleNamespace(numpy=lambda: logits + np.float32(offset))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        onnx=types.SimpleNamespace(export=onnx_export), zeros=lambda *s: np.zeros(s, np.float32),
        no_grad=contextlib.nullcontext, from_numpy=lambda a: a))
    return net


def _old_model(folder):
    (folder / "strike_verdict_cnn.onnx").write_bytes(b"old")
    (folder / "strike_verdict_cnn.meta.json").write_text('{"p_hi": 0.85}', encoding="utf-8")


def _is_old_model(folder):
    return (sorted(p.name for p in folder.iterdir()) == ["strike_verdict_cnn.meta.json",
                                                        "strike_verdict_cnn.onnx"]
            and (folder / "strike_verdict_cnn.onnx").read_bytes() == b"old"
            and (folder / "strike_verdict_cnn.meta.json").read_text(encoding="utf-8")
            == '{"p_hi": 0.85}')


def test_export_writes_thresholds_set_on_its_graph(tmp_path, monkeypatch, packaged_model):
    net = _fake_torch(monkeypatch, offset=5e-4)                      # the net is not the graph
    crops = _crops(40)
    seen = {}

    def thresholds(score):
        seen["p"] = score(crops)
        return 0.123, 0.012
    cnn.set_model_dir(tmp_path / "no-model")                          # nor the loaded model
    _, meta = train_strikenet.export(net, tmp_path / "out", thresholds=thresholds, version="t")
    cnn.set_model_dir(None)
    assert np.array_equal(seen["p"], cnn.score_crops(crops))          # the graph's own scores
    on_disk = json.loads((tmp_path / "out" / "strike_verdict_cnn.meta.json").read_text("utf-8"))
    assert on_disk["p_hi"] == meta["p_hi"] == 0.123 and on_disk["p_lo"] == 0.012   # as given


@pytest.mark.parametrize("pair, reason", [
    ((0.1, 0.5), "p_lo <= p_hi"),                                     # (p_lo, p_hi) by mistake
    ((0.5, float("nan")), "p_lo <= p_hi"),
    ((0.1234, 0.05), "rounded to 3 decimals"),                        # never a score as read
    ((np.float32(0.123), 0.05), "rounded to 3 decimals"),             # 0.12300000339746475
    ((0.5,), "values to unpack"),
])
def test_export_refuses_unloadable_or_unrounded_thresholds(tmp_path, monkeypatch, pair, reason):
    net = _fake_torch(monkeypatch)
    _old_model(tmp_path)
    with pytest.raises(ValueError, match=reason):
        train_strikenet.export(net, tmp_path, thresholds=lambda score: pair, version="t")
    assert _is_old_model(tmp_path)


def test_a_failed_calibration_leaves_the_old_model(tmp_path, monkeypatch):
    net = _fake_torch(monkeypatch)
    _old_model(tmp_path)

    def thresholds(score):
        raise RuntimeError("no thresholds")
    with pytest.raises(RuntimeError, match="no thresholds"):
        train_strikenet.export(net, tmp_path, thresholds=thresholds, version="t")
    assert _is_old_model(tmp_path)


def test_an_export_that_disagrees_with_the_net_sets_nothing(tmp_path, monkeypatch):
    net = _fake_torch(monkeypatch, offset=0.01)
    _old_model(tmp_path)
    called = []
    with pytest.raises(RuntimeError, match="disagrees with the trained net"):
        train_strikenet.export(net, tmp_path, version="t",
                               thresholds=lambda score: called.append(1) or (0.5, 0.1))
    assert not called and _is_old_model(tmp_path)


def test_a_failed_replace_leaves_the_old_model(tmp_path, monkeypatch):
    net = _fake_torch(monkeypatch)
    _old_model(tmp_path)

    def replace(self, target):                                        # a read-only or open model
        raise PermissionError("in use")
    monkeypatch.setattr(pathlib.Path, "replace", replace)
    with pytest.raises(PermissionError):
        train_strikenet.export(net, tmp_path, thresholds=lambda score: (0.5, 0.1), version="t")
    assert _is_old_model(tmp_path)                                    # its meta put back


def test_an_interrupted_export_never_pairs_the_old_meta_with_the_new_graph(tmp_path, monkeypatch):
    net = _fake_torch(monkeypatch)
    _old_model(tmp_path)
    real = pathlib.Path.replace

    def replace(self, target):                                        # Ctrl-C right after it
        real(self, target)
        raise KeyboardInterrupt
    monkeypatch.setattr(pathlib.Path, "replace", replace)
    with pytest.raises(KeyboardInterrupt):
        train_strikenet.export(net, tmp_path, thresholds=lambda score: (0.5, 0.1), version="t")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["strike_verdict_cnn.onnx"]  # no meta
    assert (tmp_path / "strike_verdict_cnn.onnx").read_bytes() == SHIPPED.read_bytes()


def _labeled(tmp_path, labels):
    """A dataset of 4-crop pages labeled as given, one crop per label."""
    rows = [{"crop": f"crops/c{i}.png", "label": lab, "page": i // 4}
            for i, lab in enumerate(labels)]
    (tmp_path / "crops").mkdir()
    for i, row in enumerate(rows):
        arr = np.full((24, 96), 255, np.uint8)
        arr[8:16, i % 90:i % 90 + 6] = i % 200
        Image.fromarray(arr).save(tmp_path / row["crop"])
    (tmp_path / "crops.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                          encoding="utf-8")
    return [i // 4 for i in range(len(labels))]


def _main(tmp_path, monkeypatch, scores, *argv):
    """main() with training stubbed and an export whose graph scores the i-th crop it is handed
    scores(n)[i]; returns what export() saw."""
    seen = {}

    def train(x, y, *, groups=None, **hparams):
        seen["x"] = x
        return None

    def export(net, out, *, thresholds, **kw):
        seen["crops"] = []
        seen["thresholds"] = thresholds(lambda c: seen["crops"].extend(c) or scores(len(c)))
        return out, kw
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(train_strikenet, "train", train)
    monkeypatch.setattr(train_strikenet, "export", export)
    monkeypatch.setattr(train_strikenet.importlib.util, "find_spec",
                        lambda name, *a: object() if name in ("torch", "onnx") else
                        find_spec(name, *a))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="stub"))
    train_strikenet.main([str(tmp_path), "-o", str(tmp_path / "out"), *argv])
    return seen


def test_main_sets_both_thresholds_on_the_validation_crops_at_its_alpha(tmp_path, monkeypatch,
                                                                        capsys):
    pages = _labeled(tmp_path, ["clean", "struck"] * 120)             # 60 pages of 4 crops

    def separated(n):                     # validation crops alternate clean, struck (pages c,s)
        u = np.linspace(0.0, 1.0, n)
        return np.where(np.arange(n) % 2 == 1, 0.55 + 0.45 * u, 0.45 * u)
    seen = _main(tmp_path, monkeypatch, separated, "--alpha", "0.1")
    val, _ = train_strikenet.split(240, 0.2, 0, pages)
    assert np.array_equal(np.stack(seen["crops"]), seen["x"][val])    # the validation crops
    struck = np.array([i % 2 == 1 for i in val])
    p = separated(len(val))
    p_hi, p_lo = train_strikenet.conformal_pair(p[struck], p[~struck], 0.1)
    assert p_lo < p_hi and seen["thresholds"] == (p_hi, p_lo)
    assert "note: p_lo equals p_hi" not in capsys.readouterr().out


def test_main_says_when_p_lo_equals_p_hi(tmp_path, monkeypatch, capsys):
    _labeled(tmp_path, ["clean", "struck"] * 120)
    seen = _main(tmp_path, monkeypatch, lambda n: np.linspace(0.0, 1.0, n))   # overlapping
    p_hi, p_lo = seen["thresholds"]
    assert p_lo == p_hi
    out = capsys.readouterr().out
    assert ("note: p_lo equals p_hi, so the model calls no word unsure: at --alpha 0.05, 1 or "
            "more clean validation crops at or above p_hi is too many for p_lo to sit below p_hi. "
            "Clean validation crops at or above p_hi, so called struck: 23 of 24") in out


def test_main_counts_clean_crops_at_p_hi_as_called_struck(tmp_path, monkeypatch, capsys):
    _labeled(tmp_path, ["clean", "struck"] * 120)

    def tied(n):                          # every strike at 0.5, and 3 clean crops tie with it
        p = np.where(np.arange(n) % 2 == 1, 0.5, 0.1)
        p[[0, 2, 4]] = 0.5
        return p
    assert _main(tmp_path, monkeypatch, tied)["thresholds"] == (0.5, 0.5)
    assert "so called struck: 3 of 24" in capsys.readouterr().out


def test_main_says_when_too_few_clean_crops_set_p_lo(tmp_path, monkeypatch, capsys):
    _labeled(tmp_path, (["struck"] * 7 + ["clean"]) * 30)             # 1 clean crop in 8
    seen = _main(tmp_path, monkeypatch, lambda n: np.linspace(0.0, 1.0, n))
    assert seen["thresholds"][0] == seen["thresholds"][1]
    out = capsys.readouterr().out
    assert ("note: p_lo equals p_hi, so the model calls no word unsure: --alpha 0.05 needs 19 "
            "clean validation crops for p_lo to sit below p_hi and the validation split holds ") \
        in out


def test_split_shortfall_wants_the_strikes_alpha_really_needs():
    alpha = 0.006211180124223602                                      # 1/161: 1/alpha reads 161.0
    assert calibration.conformal_threshold(np.linspace(0.5, 1.0, 160), alpha) == 0.0
    y = np.array([1.0] * 160 + [0.0] * 5 + [1.0] * 10 + [0.0] * 50, dtype=np.float32)
    pages = [0] * 165 + [1] * 60
    seed = next(s for s in range(20)                                  # page 0 held out: 160
                if train_strikenet.split(225, 0.5, s, pages)[0][0] == 0)    # strikes to calibrate
    got = train_strikenet.split_shortfall(y, 0.5, seed, alpha, pages)
    assert got is not None and "at least 161 struck" in got
    assert train_strikenet.fewest(alpha) == 161 and train_strikenet.fewest(0.05) == 19


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.3, 1.5, float("nan")])
def test_an_alpha_outside_0_to_1_is_refused(alpha, tmp_path):
    with pytest.raises(ValueError, match="alpha must be in"):
        train_strikenet.fewest(alpha)                     # without the check, -0.3 would loop
    with pytest.raises(SystemExit) as stop:
        train_strikenet.main([str(tmp_path), "--alpha", str(alpha)])
    assert stop.value.code == 2                                        # a usage error, at once
