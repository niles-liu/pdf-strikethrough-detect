"""The training script's thresholds: p_hi and p_lo are matched to the model the package loads, on
the scores the detector compares with them.

`training/train_strikenet.py` sets a retrained StrikeNet's p_hi and p_lo from its validation crops,
so that it calls at least as many of the struck ones struck, in each source, as the reference
model (the one the package loads) does at its own thresholds, and at least as many of the clean
ones clean, unless that would put p_lo at or above p_hi; the split-conformal pair is reported
beside them. The detector scores a crop through the exported ONNX graph and rounds the probability
to 3 decimals before it compares it with them (`detect.apply_cnn_verdict`), so a threshold set on
other scores, or rounded again on export, can drop the crops it was set on. The script is not
part of the package, so this runs from a source checkout and skips against an installed wheel."""
import contextlib
import importlib.util
import io
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
REFERENCE = {"version": "ref", "sha256": "f" * 64, "p_hi": 0.85, "p_lo": 0.15, "runtime": "onnx"}


@pytest.fixture
def packaged_model(monkeypatch):
    """The packaged StrikeNet loaded, whatever the environment points at."""
    monkeypatch.delenv("PDF_STRIKETHROUGH_MODEL_DIR", raising=False)
    cnn.set_model_dir(None)
    yield
    cnn.set_model_dir(None)


def _review(monkeypatch, probs, p_hi, p_lo):
    """detect.apply_cnn_verdict's records for review words StrikeNet scores `probs`."""
    recs = [{"tier": "review", "bbox_frac": (0.1, 0.1, 0.3, 0.2)} for _ in probs]
    monkeypatch.setattr(cnn, "score_crops",
                        lambda crops, batch_size=512: np.asarray(probs, np.float64))
    detect.apply_cnn_verdict(recs, np.full((200, 400), 255, np.uint8),
                             meta={"p_hi": p_hi, "p_lo": p_lo})
    return recs


def _detector_calls(monkeypatch, probs, p_hi, p_lo):
    """What detect.apply_cnn_verdict decides for review words StrikeNet scores `probs`: each
    word's (final, cnn_prob)."""
    recs = _review(monkeypatch, probs, p_hi, p_lo)
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


def test_matched_high_calls_as_many_strikes_as_the_reference_and_no_higher_one_does(monkeypatch):
    rng = np.random.default_rng(5)
    ref = np.r_[0.84951, 0.8494, rng.random(38)]                     # read as 0.85 and 0.849
    new = rng.random(40)
    for reference in ((0.85, 0.15), (0.5, 0.5), (0.3, 0.1)):
        t = train_strikenet.matched_high(new, ref, reference)
        want = sum(_detector_calls(monkeypatch, ref, *reference)[0])
        assert t == round(t, 3) and sum(_detector_calls(monkeypatch, new, t, 0.0)[0]) >= want
        assert sum(_detector_calls(monkeypatch, new, round(t + 0.001, 3), 0.0)[0]) < want
    assert train_strikenet.matched_high(new, np.full(40, 0.2), (0.85, 0.15)) is None


def test_matched_low_clears_as_many_crops_as_the_reference_and_no_lower_one_does(monkeypatch):
    rng = np.random.default_rng(6)
    ref = np.r_[0.15049, 0.1506, 0.5, rng.random(37)]                # read as 0.15 and 0.151
    new = rng.random(40)

    def clean(probs, p_hi, p_lo):
        return [r["verdict"] for r in _review(monkeypatch, probs, p_hi, p_lo)].count("clean")
    for reference in ((0.85, 0.15), (0.6, 0.5), (0.5, 0.5)):      # at p_lo == p_hi, 0.5 is struck
        t = train_strikenet.matched_low(new, ref, reference)
        want = clean(ref, *reference)
        assert t == round(t, 3) and clean(new, 1.0, t) >= want
        assert clean(new, 1.0, round(t - 0.001, 3)) < want
    assert train_strikenet.matched_low(new, np.full(40, 0.9), (0.85, 0.15)) == 0.0


def test_p_hi_is_matched_in_each_source_and_the_lowest_bound_kept(monkeypatch):
    sources = np.array(['"print"'] * 40 + ['"hand"'] * 40 + [None] * 40 + ['"print"', None] * 30,
                       dtype=object)
    struck = np.r_[np.ones(120, bool), np.zeros(60, bool)]
    ref = np.r_[np.full(36, 0.95), np.full(4, 0.5),                 # print: the reference calls 36
                np.full(10, 0.9), np.full(30, 0.3),                 # hand: 10
                np.full(40, 0.2), np.full(60, 0.1)]                 # no source: none, so no bound
    new = np.r_[np.linspace(0.6, 0.99, 40), np.linspace(0.2, 0.8, 40), np.full(40, 0.01),
                np.linspace(0.0, 0.5, 60)]
    groups = np.arange(180) // 4
    p_hi, p_lo, how = train_strikenet.matched_pair(new, ref, struck, sources, (0.85, 0.15), groups)
    p = train_strikenet.as_called(new)
    by = how["p_hi"]["by_source"]
    assert [e["source"] for e in by] == [None, "hand", "print"]      # rows without one first
    assert by[0] == {"source": None, "struck": 40, "groups": 10, "reference_calls": 0,
                     "matched": None}
    for e, key, k in ((by[1], '"hand"', 10), (by[2], '"print"', 36)):
        mask = struck & (sources == key)
        assert e == {"source": json.loads(key), "struck": 40, "groups": 10, "reference_calls": k,
                     "matched": np.sort(p[mask])[::-1][k - 1]}       # the k-th highest strike
        assert sum(_detector_calls(monkeypatch, new[mask], p_hi, p_lo)[0]) >= k   # none loses
    assert p_hi == min(by[1]["matched"], by[2]["matched"]) == by[2]["matched"] == 0.64
    assert how["p_hi"]["set_by"] == "print"
    assert how["p_hi"]["pooled"] == np.sort(p[struck])[::-1][46 - 1]
    caught = int((p[struck] >= 0.64).sum())
    assert how["called_struck"] == {"struck": {"reference": 46, "new": caught},
                                    "clean": {"reference": 0, "new": 0}}


def test_p_lo_is_matched_on_all_clean_crops_at_once_and_capped_at_p_hi():
    struck = np.r_[np.ones(20, bool), np.zeros(40, bool)]
    sources = np.array(['"a"'] * 30 + ['"b"'] * 30, dtype=object)
    ref = np.r_[np.full(20, 0.9), np.full(10, 0.1),                 # a: 10 clean, all cleared
                np.full(20, 0.1), np.full(10, 0.5)]                 # b: 20 of 30 cleared
    clean_new = np.r_[np.linspace(0.30, 0.39, 10), np.linspace(0.0, 0.29, 30)]
    p_hi, p_lo, how = train_strikenet.matched_pair(np.r_[np.full(20, 0.8), clean_new], ref,
                                                   struck, sources, (0.85, 0.15))
    assert how["p_lo"] == {"clean": 40, "reference_calls": 30, "matched": 0.29}   # 30th lowest of
    assert (p_hi, p_lo) == (0.8, 0.29)                    # all 40 (by source: 0.39 and 0.19)
    p_hi, p_lo, how = train_strikenet.matched_pair(np.r_[np.full(20, 0.2), clean_new], ref,
                                                   struck, sources, (0.85, 0.15))
    assert how["p_lo"]["matched"] == 0.29 and p_hi == p_lo == 0.2   # capped


def test_matching_refuses_without_a_recall_to_match():
    struck = np.r_[np.ones(20, bool), np.zeros(20, bool)]
    with pytest.raises(ValueError, match="calls none of the 20 struck crops struck"):
        train_strikenet.matched_pair(np.full(40, 0.9), np.full(40, 0.5), struck, [None] * 40,
                                     (0.85, 0.15))
    ok = np.r_[np.full(20, 0.9), np.full(20, 0.1)]
    for new, ref in ((np.r_[ok[:-1], np.nan], ok), (ok, np.r_[ok[:-1], np.inf])):
        with pytest.raises(ValueError, match="finite"):
            train_strikenet.matched_pair(new, ref, struck, [None] * 40, (0.85, 0.15))


def test_a_source_is_the_source_field_as_json_text():
    rows = [{"source": "print"}, {"source": "print "}, {"source": 1}, {"source": 1.0}, {},
            {"source": None}, {"source": " "}, {"source": float("nan")}, {"source": []},
            {"source": {"b": 1, "a": 2}}, {"source": 0}, {"source": False}]
    assert train_strikenet.sources_of(rows).tolist() == [
        '"print"', '"print "', "1", "1.0", None, None, None, None, None, '{"a": 2, "b": 1}', "0",
        "false"]
    for source in (float("inf"), ["print", float("nan")], {"w": float("-inf")}):  # NaN, Infinity
        with pytest.raises(SystemExit, match=r"source\" of crops/c7\.png holds NaN or Infinity"):
            train_strikenet.sources_of([{"crop": "crops/c7.png", "source": source}])


def test_a_source_with_no_struck_crop_here_is_listed_with_no_bound():
    struck = np.r_[np.ones(4, bool), np.zeros(4, bool)]
    p_hi, p_lo, how = train_strikenet.matched_pair(
        np.full(8, 0.9), np.full(8, 0.9), struck, ['"print"'] * 8, (0.85, 0.15), np.arange(8),
        every=['"hand"', '"print"'])
    assert how["p_hi"]["by_source"][0] == {"source": "hand", "struck": 0, "groups": 0,
                                           "reference_calls": 0, "matched": None}
    assert how["p_hi"]["set_by"] == "print" and p_hi == 0.9


def test_a_reference_off_the_3_decimal_grid_is_read_as_the_detector_reads_it(monkeypatch):
    ref = [0.9979, 0.99749, 0.9975, 0.998, 0.997]          # an older model's 4-decimal p_hi
    assert sum(_detector_calls(monkeypatch, ref, 0.9979, 0.0021)[0]) == 3
    assert train_strikenet.matched_high([0.5, 0.6, 0.7, 0.8, 0.9], ref, (0.9979, 0.0021)) == 0.7
    ref = [0.0085, 0.001]                                  # round() reads 0.009, np.round 0.008
    assert _detector_calls(monkeypatch, ref, 0.009, 0.005)[0] == [True, False]
    assert train_strikenet.matched_high([0.4, 0.2], ref, (0.009, 0.005)) == 0.4
    ref = [0.998, 0.997, 0.5, 0.003, 0.002]                # a p_hi that rounds down, a p_lo up
    assert _detector_calls(monkeypatch, ref, 0.9971, 0.0026)[0] == [True] + [False] * 4
    assert [r["verdict"] for r in _review(monkeypatch, ref, 0.9971, 0.0026)].count("clean") == 1
    new = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert train_strikenet.matched_high(new, ref, (0.9971, 0.0026)) == 0.5   # at 0.997: 0.4
    assert train_strikenet.matched_low(new, ref, (0.9971, 0.0026)) == 0.1    # at 0.003: 0.2


def test_one_caught_strike_sets_a_bound_and_one_cleared_crop_a_match():
    assert train_strikenet.matched_high([0.3, 0.2], [0.9, 0.1], (0.85, 0.15)) == 0.3
    assert train_strikenet.matched_low([0.3, 0.2], [0.1, 0.5], (0.85, 0.15)) == 0.2


def test_only_sources_with_strikes_are_listed_and_pooled_is_matched_on_the_strikes():
    struck = np.array([True] * 4 + [False] * 4)
    sources = np.array(['"b"', '"b"', '"a"', '"a"', '"c"', '"c"', None, None], dtype=object)
    ref = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1])     # and a clean crop called struck
    new = np.array([0.7, 0.6, 0.7, 0.2, 0.3, 0.1, 0.1, 0.1])
    p_hi, _, how = train_strikenet.matched_pair(new, ref, struck, sources, (0.85, 0.15))
    assert [e["source"] for e in how["p_hi"]["by_source"]] == ["a", "b"]   # c, None: clean only
    assert p_hi == 0.7 and how["p_hi"]["set_by"] == "a"                    # a tie: the first
    assert how["p_hi"]["pooled"] == 0.7                  # the 2nd highest strike; of all 8: 0.6


def test_a_tie_for_the_lowest_bound_names_the_first_source_in_order():
    how = train_strikenet.matched_pair([0.6] * 3, [0.9] * 3, [True] * 3, ['"z"', '"a"', None],
                                       (0.85, 0.15))[2]
    assert how["p_hi"]["set_by"] is None and [e["matched"] for e in how["p_hi"]["by_source"]] \
        == [0.6, 0.6, 0.6]


def test_split_shortfall_wants_19_strikes_for_the_match_whatever_alpha():
    y = np.array(([1.0] * 10 + [0.0] * 30) * 5, dtype=np.float32)       # 10 struck per 40 crops
    for alpha in (0.5, 0.1, 0.05):
        got = train_strikenet.split_shortfall(y, 0.2, 0, alpha)
        assert "the thresholds need at least 19 struck" in got and "--alpha" not in got
    got = train_strikenet.split_shortfall(y, 0.2, 0, 0.01)            # the report needs more
    assert ("the split-conformal pair at --alpha 0.01 needs at least 99 struck (19 at --alpha "
            "0.05 or above)") in got


def test_the_new_models_scores_are_read_as_the_detector_reads_them():
    # np.round reads 0.0055 as 0.006 (the detector: 0.005) and 0.0085 as 0.008 (0.009); at those
    # the detector calls the strike unsure and the clean crop unsure
    assert train_strikenet.matched_high([0.0055, 0.001], [0.9, 0.1], (0.85, 0.15)) == 0.005
    assert train_strikenet.matched_low([0.0085, 0.9], [0.1, 0.9], (0.85, 0.15)) == 0.009


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
        no_grad=contextlib.nullcontext, from_numpy=lambda a: a, __version__="fake"))
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


def _labeled(tmp_path, labels, source=None):
    """A dataset of 4-crop pages labeled as given, one crop per label; with `source`, each row's
    ``source`` is source(page)."""
    rows = [{"crop": f"crops/c{i}.png", "label": lab, "page": i // 4}
            for i, lab in enumerate(labels)]
    if source is not None:
        for row in rows:
            row["source"] = source(row["page"])
    (tmp_path / "crops").mkdir()
    for i, row in enumerate(rows):
        arr = np.full((24, 96), 255, np.uint8)
        arr[8:16, i % 90:i % 90 + 6] = i % 200
        Image.fromarray(arr).save(tmp_path / row["crop"])
    (tmp_path / "crops.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                          encoding="utf-8")
    return [i // 4 for i in range(len(labels))]


def _reference(monkeypatch, scores, seen, meta=REFERENCE):
    """The model the package loads, as main() reads it: `meta`, and a score for each crop it is
    handed, the i-th of n scores(n)[i]."""
    def score_crops(crops, batch_size=512):
        seen["reference_crops"] = list(crops)
        return np.asarray(scores(len(crops)), np.float64)
    monkeypatch.setattr(cnn, "score_crops", score_crops)
    monkeypatch.setattr(cnn, "get_model_meta", lambda: dict(meta))


def _main(tmp_path, monkeypatch, scores, reference, *argv, seen=None, meta=REFERENCE):
    """main() with training stubbed, the reference model (`meta`) scoring the crops it is handed
    reference(n), and an export whose graph scores the i-th crop it is handed scores(n)[i];
    returns what export() saw."""
    seen = {} if seen is None else seen
    _reference(monkeypatch, reference, seen, meta)

    def train(x, y, *, groups=None, **hparams):
        seen["x"] = x
        return None

    def export(net, out, *, thresholds, **kw):
        seen["crops"] = []
        seen["thresholds"] = thresholds(lambda c: seen["crops"].extend(c) or scores(len(c)))
        seen.update(kw)
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


def _alternating(struck, clean):
    """Scores for validation crops that alternate clean, struck (a page of _labeled's holds clean,
    struck, clean, struck): struck(u) for the strikes and clean(u) for the rest, u running 0 to 1
    across them all."""
    def scores(n):
        u = np.linspace(0.0, 1.0, n)
        return np.where(np.arange(n) % 2 == 1, struck(u), clean(u))
    return scores


CAUGHT = _alternating(lambda u: 0.9, lambda u: 0.1)       # the reference calls each crop right


def test_main_matches_the_reference_on_the_validation_crops(tmp_path, monkeypatch, capsys):
    pages = _labeled(tmp_path, ["clean", "struck"] * 120)             # 60 pages of 4 crops
    separated = _alternating(lambda u: 0.55 + 0.45 * u, lambda u: 0.45 * u)
    reference = _alternating(lambda u: 0.3 + 0.7 * u, lambda u: 0.4 * u)
    seen = _main(tmp_path, monkeypatch, separated, reference, "--alpha", "0.1")
    val, _ = train_strikenet.split(240, 0.2, 0, pages)
    assert np.array_equal(np.stack(seen["crops"]), seen["x"][val])    # the validation crops,
    assert np.array_equal(np.stack(seen["reference_crops"]), seen["x"][val])   # by both models
    struck = np.array([i % 2 == 1 for i in val])
    p = train_strikenet.as_called(separated(len(val)))
    r = train_strikenet.as_called(reference(len(val)))
    k_hi, k_lo = int((r[struck] >= 0.85).sum()), int((r[~struck] <= 0.15).sum())
    p_hi = np.sort(p[struck])[::-1][k_hi - 1]                         # the k-th highest strike
    p_lo = np.sort(p[~struck])[k_lo - 1]                              # the k-th lowest clean crop
    assert 0 < k_hi < struck.sum() and 0 < k_lo < (~struck).sum() and p_lo < p_hi
    assert seen["thresholds"] == (p_hi, p_lo)
    calibration = seen["training"]["calibration"]
    assert calibration["reference"] == {k: REFERENCE[k] for k in ("version", "sha256", "p_hi",
                                                                  "p_lo")}
    c_hi, c_lo = train_strikenet.conformal_pair(p[struck], p[~struck], 0.1)
    conformal = calibration["conformal"]
    assert (conformal["alpha"], conformal["p_hi"], conformal["p_lo"]) == (0.1, c_hi, c_lo)
    assert conformal["mirror"] == c_lo < c_hi                         # not capped
    out = capsys.readouterr().out
    assert (f"reference: ref (sha256 ffffffffffff, p_hi 0.85, p_lo 0.15), the model the package "
            f"loads, calls {k_hi} of the 24 struck validation crops struck") in out
    assert f"matched thresholds: p_hi={p_hi:.3f}, p_lo={p_lo:.3f}" in out
    assert "note: p_lo equals p_hi" not in out and "capped at p_hi" not in out


def test_main_matches_p_hi_in_each_source(tmp_path, monkeypatch, capsys):
    pages = _labeled(tmp_path, ["clean", "struck"] * 120,
                     source=lambda page: ("print", "hand")[page % 2])
    val, _ = train_strikenet.split(240, 0.2, 0, pages)
    struck = np.array([i % 2 == 1 for i in val])
    hand = np.array([pages[i] % 2 == 1 for i in val])
    u = np.linspace(0.0, 1.0, len(val))
    new = np.where(struck, np.where(hand, 0.2 + 0.6 * u, 0.7 + 0.29 * u), 0.1 * u)
    ref = np.where(struck, np.where(hand & (u < 0.5), 0.3, 0.95), 0.1)
    seen = _main(tmp_path, monkeypatch, lambda n: new, lambda n: ref)
    p, r = train_strikenet.as_called(new), train_strikenet.as_called(ref)
    bounds = {}
    for name, mask in (("hand", struck & hand), ("print", struck & ~hand)):
        k = int((r[mask] >= 0.85).sum())
        assert 0 < k and mask.sum()
        bounds[name] = np.sort(p[mask])[::-1][k - 1]
    assert seen["thresholds"][0] == min(bounds.values()) and bounds["hand"] != bounds["print"]
    p_hi = seen["training"]["calibration"]["p_hi"]
    assert {e["source"]: e["matched"] for e in p_hi["by_source"]} == bounds
    assert p_hi["set_by"] == min(bounds, key=bounds.get)
    out = capsys.readouterr().out
    for name in ("hand", "print"):
        assert f'  source "{name}": ' in out
    assert (f'matched thresholds: p_hi={min(bounds.values()):.3f} (set by source '
            f'"{p_hi["set_by"]}")') in out


def test_main_matches_the_reference_at_its_own_thresholds(tmp_path, monkeypatch):
    pages = _labeled(tmp_path, ["clean", "struck"] * 120)
    meta = {**REFERENCE, "p_hi": 0.7, "p_lo": 0.25,
            "external_data": [{"location": "w.bin", "sha256": "e" * 64}]}
    separated = _alternating(lambda u: 0.55 + 0.45 * u, lambda u: 0.45 * u)
    reference = _alternating(lambda u: 0.3 + 0.7 * u, lambda u: 0.4 * u)
    seen = _main(tmp_path, monkeypatch, separated, reference, meta=meta)
    val, _ = train_strikenet.split(240, 0.2, 0, pages)
    struck = np.array([i % 2 == 1 for i in val])
    p = train_strikenet.as_called(separated(len(val)))
    r = train_strikenet.as_called(reference(len(val)))
    k_hi, k_lo = int((r[struck] >= 0.7).sum()), int((r[~struck] <= 0.25).sum())
    assert k_hi != int((r[struck] >= 0.85).sum()) and k_lo != int((r[~struck] <= 0.15).sum())
    assert seen["thresholds"] == (np.sort(p[struck])[::-1][k_hi - 1],
                                  np.sort(p[~struck])[k_lo - 1])
    assert seen["training"]["calibration"]["reference"] == {
        k: meta[k] for k in ("version", "sha256", "external_data", "p_hi", "p_lo")}


def test_main_prints_each_source_and_both_models_counts(tmp_path, monkeypatch, capsys):
    pages = _labeled(tmp_path, ["clean", "struck"] * 120,
                     source=lambda page: ("print", "hand")[page % 2])
    val, _ = train_strikenet.split(240, 0.2, 0, pages)
    struck = np.array([i % 2 == 1 for i in val])
    hand = np.array([pages[i] % 2 == 1 for i in val])
    u = np.linspace(0.0, 1.0, len(val))
    new = np.where(struck, np.where(hand, 0.2 + 0.6 * u, 0.6), 0.1 * u)     # print's strikes tie
    ref = np.where(struck, np.where(u < 0.3, 0.3, 0.95), 0.1)              # most of each caught
    seen = _main(tmp_path, monkeypatch, lambda n: new, lambda n: ref, "--alpha", "0.1")
    c = seen["training"]["calibration"]
    out = capsys.readouterr().out
    bounds = [e["matched"] for e in c["p_hi"]["by_source"]]
    assert len(bounds) == 2 and None not in bounds and bounds[0] != bounds[1]
    for e in c["p_hi"]["by_source"]:
        assert (f'  source "{e["source"]}": {e["struck"]} struck validation crops from '
                f'{e["groups"]} group(s), {e["reference_calls"]} called struck by the reference: '
                f'matched at {e["matched"]:.3f}\n') in out
    s, k = c["called_struck"]["struck"], c["called_struck"]["clean"]
    assert s["new"] > s["reference"]                # the source that did not set p_hi gains
    assert (f"called struck by the reference and the new model: {s['reference']} and {s['new']} "
            f"of the 24 struck validation crops, {k['reference']} and {k['new']} of the 24 "
            f"clean\n") in out
    conformal = c["conformal"]
    assert conformal["p_lo"] < conformal["p_hi"]
    assert (f"split-conformal pair at --alpha 0.1, reported only: p_hi={conformal['p_hi']:.3f}, "
            f"p_lo={conformal['p_lo']:.3f}\n") in out


def test_main_counts_the_validation_groups_of_each_source(tmp_path, monkeypatch):
    _labeled(tmp_path, ["clean", "struck"] * 240)
    manifest = tmp_path / "crops.jsonl"
    rows = [json.loads(line) for line in manifest.read_text("utf-8").splitlines()]
    for row in rows:
        row["group"] = f"w{row['page'] % 7}"            # 7 writers of 17 or 18 pages
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    seen = _main(tmp_path, monkeypatch, lambda n: np.linspace(0.0, 1.0, n), CAUGHT)
    by = seen["training"]["calibration"]["p_hi"]["by_source"]
    assert seen["training"]["n_val_groups"] == 1 and by[0]["groups"] == 1


def test_main_stops_before_training_when_the_reference_calls_no_strike(tmp_path, monkeypatch):
    _labeled(tmp_path, ["clean", "struck"] * 120)
    seen = {}
    misses = _alternating(lambda u: 0.8, lambda u: 0.9)    # every strike missed, clean crops not
    with pytest.raises(SystemExit, match="calls none of the 24 struck validation crops struck"):
        _main(tmp_path, monkeypatch, lambda n: np.zeros(n), misses, seen=seen)
    assert "reference_crops" in seen and "x" not in seen                 # before the epochs
    with pytest.raises(SystemExit, match="calls none of the 24 struck validation crops struck"):
        _main(tmp_path, monkeypatch, lambda n: np.zeros(n), CAUGHT,   # 0.9, below a p_hi of
              meta={**REFERENCE, "p_hi": 0.9001})                     # 0.9001, which rounds to it


def test_main_stops_before_training_when_the_reference_does_not_load(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    _labeled(data, ["clean", "struck"] * 120)
    broken = tmp_path / "model"                                          # an export cut short
    broken.mkdir()
    shutil.copyfile(SHIPPED, broken / "strike_verdict_cnn.onnx")
    cut = tmp_path / "cut"                                               # a graph cut short
    cut.mkdir()
    (cut / "strike_verdict_cnn.onnx").write_bytes(SHIPPED.read_bytes()[:1000])
    shutil.copyfile(SHIPPED.with_name("strike_verdict_cnn.meta.json"),
                    cut / "strike_verdict_cnn.meta.json")
    monkeypatch.setattr(train_strikenet, "train", lambda *a, **kw: pytest.fail("trained"))
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(train_strikenet.importlib.util, "find_spec",
                        lambda name, *a: object() if name in ("torch", "onnx") else
                        find_spec(name, *a))
    try:
        for folder, error in ((broken, "FileNotFoundError"), (cut, "")):   # ORT's own error
            monkeypatch.setenv("PDF_STRIKETHROUGH_MODEL_DIR", str(folder))
            cnn.set_model_dir(None)
            with pytest.raises(SystemExit, match=rf"no model loads from .*{folder.name}: {error}"):
                train_strikenet.main([str(data), "-o", str(tmp_path / "out")])
    finally:
        cnn.set_model_dir(None)


def test_main_stops_before_training_when_the_meta_cannot_record_the_reference(tmp_path,
                                                                              monkeypatch):
    _labeled(tmp_path, ["clean", "struck"] * 120)
    seen = {}
    with pytest.raises(ValueError, match="not JSON compliant"):
        _main(tmp_path, monkeypatch, CAUGHT, CAUGHT, seen=seen,
              meta={**REFERENCE, "version": float("nan")})
    assert "x" not in seen                                                 # before the epochs


def test_main_stops_before_training_when_the_reference_scores_nan(tmp_path, monkeypatch):
    _labeled(tmp_path, ["clean", "struck"] * 120)
    for reference in (lambda n: np.r_[np.nan, np.full(n - 1, 0.9)],     # on a clean crop
                      lambda n: np.r_[np.full(n - 1, 0.9), np.nan]):    # on a struck one
        seen = {}
        with pytest.raises(SystemExit, match="gives 1 of the 48 validation crops no finite"):
            _main(tmp_path, monkeypatch, lambda n: np.zeros(n), reference, seen=seen)
        assert "x" not in seen


def test_main_names_a_source_whose_strikes_are_all_in_training(tmp_path, monkeypatch, capsys):
    pages = _labeled(tmp_path, ["clean", "struck"] * 120,
                     source=lambda page: "hand" if page in (7, 33) else "print")
    val, _ = train_strikenet.split(240, 0.2, 0, pages)
    assert not {pages[i] for i in val} & {7, 33}                       # seed 0 holds neither out
    seen = _main(tmp_path, monkeypatch, CAUGHT, CAUGHT)
    out = capsys.readouterr().out
    note = ('note: source "hand" has struck crops only in the training split, so p_hi is not '
            'matched on them')
    assert note in out and out.index(note) < out.index("matched thresholds")   # before training
    hand = seen["training"]["calibration"]["p_hi"]["by_source"][0]
    assert hand == {"source": "hand", "struck": 0, "groups": 0, "reference_calls": 0,
                    "matched": None}
    assert '  source "hand": no struck validation crops, so no bound' in out


def test_main_lists_every_source_with_strikes_and_only_those(tmp_path, monkeypatch, capsys):
    pages = _labeled(tmp_path, ["clean", "struck"] * 120)
    val, _ = train_strikenet.split(240, 0.2, 0, pages)
    held = sorted({pages[i] for i in val})
    assert not set(held) & {0, 7, 33}
    manifest = tmp_path / "crops.jsonl"
    rows = [json.loads(line) for line in manifest.read_text("utf-8").splitlines()]
    for row in rows:
        if row["page"] not in (7, 33):          # pages 7 and 33: no source, strikes all trained
            row["source"] = "hand" if row["page"] == held[1] else "print"
        if row["page"] == held[0] and row["label"] == "clean":
            del row["source"]                   # no source: clean crops held out, no strike
        if row["page"] == 0 and row["label"] == "clean":
            row["source"] = "blank"             # a source with no strike at all
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    hand = np.array([pages[i] == held[1] for i in val])
    ref = np.where(hand, 0.5, CAUGHT(len(val)))                       # hand's strikes: unsure
    seen = _main(tmp_path, monkeypatch, CAUGHT, lambda n: ref)
    out = capsys.readouterr().out
    assert ("note: the rows without a source have struck crops only in the training split" in out
            and "blank" not in out)
    assert ('  source "hand": 2 struck validation crops from 1 group(s), 0 called struck by the '
            'reference: no bound\n') in out
    assert "  rows without a source: no struck validation crops, so no bound\n" in out
    by = seen["training"]["calibration"]["p_hi"]["by_source"]
    assert [e["source"] for e in by] == [None, "hand", "print"]
    assert by[0] == {"source": None, "struck": 0, "groups": 0, "reference_calls": 0,
                     "matched": None}


def test_main_prints_text_a_cp1252_log_cannot_encode(tmp_path, monkeypatch):
    _labeled(tmp_path, ["clean", "struck"] * 120, source=lambda page: "手写")
    log = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")              # stdout redirected
    monkeypatch.setattr(sys, "stdout", log)
    _main(tmp_path, monkeypatch, CAUGHT, CAUGHT)
    log.flush()
    assert b"'source': '\\u624b\\u5199'" in log.buffer.getvalue()      # the meta, escaped


def test_main_says_when_p_lo_equals_p_hi(tmp_path, monkeypatch, capsys):
    _labeled(tmp_path, ["clean", "struck"] * 120)
    seen = _main(tmp_path, monkeypatch, lambda n: np.linspace(0.0, 1.0, n), CAUGHT)  # overlapping
    p_hi, p_lo = seen["thresholds"]
    assert p_lo == p_hi
    out = capsys.readouterr().out
    assert ("note: p_lo equals p_hi, so the model calls no word unsure: to call as many clean "
            "validation crops clean as the reference (24 of 24), p_lo would sit at 0.979, at or "
            "above p_hi. Clean validation crops at or above p_hi, so called struck: 23 of 24"
            ) in out
    assert ("  24 clean validation crops, 24 called clean by the reference: matched at 0.979, "
            "capped at p_hi 0.021\n") in out
    conformal = seen["training"]["calibration"]["conformal"]           # capped there too
    assert conformal["p_lo"] == conformal["p_hi"] == 0.021 and conformal["mirror"] == 0.979


def test_main_counts_clean_crops_at_p_hi_as_called_struck(tmp_path, monkeypatch, capsys):
    _labeled(tmp_path, ["clean", "struck"] * 120)

    def tied(n):                          # every strike at 0.5, and 3 clean crops tie with it
        p = np.where(np.arange(n) % 2 == 1, 0.5, 0.1)
        p[[0, 2, 4]] = 0.5
        return p
    seen = _main(tmp_path, monkeypatch, tied, CAUGHT)
    assert seen["thresholds"] == (0.5, 0.5)
    conformal = seen["training"]["calibration"]["conformal"]           # the pair, now reported
    assert conformal["p_hi"] == conformal["p_lo"] == conformal["mirror"] == 0.5
    out = capsys.readouterr().out
    assert "so called struck: 3 of 24" in out
    assert ("  24 clean validation crops, 24 called clean by the reference: matched at 0.500\n"
            in out)                          # the match lands on p_hi: the cap changes nothing


def test_main_reports_too_few_clean_crops_for_the_conformal_p_lo(tmp_path, monkeypatch):
    _labeled(tmp_path, (["struck"] * 7 + ["clean"]) * 30)             # 1 clean crop in 8
    seen = _main(tmp_path, monkeypatch, lambda n: np.linspace(0.0, 1.0, n),
                 lambda n: np.full(n, 0.9))
    conformal = seen["training"]["calibration"]["conformal"]
    assert conformal["mirror"] is None and conformal["p_lo"] == conformal["p_hi"]


def test_main_writes_how_it_set_the_thresholds_into_the_meta(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    pages = _labeled(data, ["clean", "struck"] * 120)
    net = _fake_torch(monkeypatch)
    _reference(monkeypatch, CAUGHT, {})
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(train_strikenet, "train", lambda x, y, **kw: net)
    monkeypatch.setattr(train_strikenet.importlib.util, "find_spec",
                        lambda name, *a: object() if name in ("torch", "onnx") else
                        find_spec(name, *a))
    train_strikenet.main([str(data), "-o", str(tmp_path / "out")])
    meta = json.loads((tmp_path / "out" / "strike_verdict_cnn.meta.json").read_text("utf-8"))
    val, _ = train_strikenet.split(240, 0.2, 0, pages)
    struck = np.array([i % 2 == 1 for i in val])
    x = train_strikenet.load_dataset(data)[0]
    p = train_strikenet.as_called(train_strikenet.score_graph(SHIPPED.read_bytes(), x[val]))
    assert meta["p_hi"] == p[struck].min()                            # every strike was caught
    assert meta["p_lo"] == min(p[~struck].max(), meta["p_hi"])        # and every clean crop
    calibration = meta["training"]["calibration"]
    assert calibration["reference"]["sha256"] == REFERENCE["sha256"]
    assert calibration["p_hi"]["by_source"] == [{"source": None, "struck": 24, "groups": 12,
                                                 "reference_calls": 24, "matched": meta["p_hi"]}]
    c_hi, c_lo = train_strikenet.conformal_pair(p[struck], p[~struck], 0.05)
    assert (calibration["conformal"]["p_hi"], calibration["conformal"]["p_lo"]) == (c_hi, c_lo)


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
