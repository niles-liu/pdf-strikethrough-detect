"""StrikeNet loading and threshold calibration: digests that verify what they promise, thresholds a
model cannot be loaded without, and calibration helpers that read the labels and ties a
``dump_crops`` manifest actually holds."""
import hashlib
import json
import pathlib

import numpy as np
import pytest

import pdf_strikethrough as st
from pdf_strikethrough import calibration as C, cnn


def _packaged():
    here = pathlib.Path(cnn.__file__).parent
    onnx = here / "strike_verdict_cnn.onnx"
    meta = here / "strike_verdict_cnn.meta.json"
    return onnx, meta, hashlib.sha256(onnx.read_bytes()).hexdigest()


# ----------------------------------------------------------------------- ensure_model

def test_uppercase_digest_is_accepted(tmp_path):
    onnx, meta, sha = _packaged()
    try:
        st.ensure_model(onnx.as_uri(), sha.upper(), meta=json.loads(meta.read_text()),
                        cache_dir=str(tmp_path / "c"))
    finally:
        cnn.set_model_dir(None)


@pytest.mark.parametrize("digest", ["", None, "sha256:" + "0" * 64, "abc"])
def test_malformed_digest_is_rejected_before_any_download(tmp_path, digest):
    onnx, meta, _ = _packaged()
    with pytest.raises(ValueError, match="64-character"):
        st.ensure_model(onnx.as_uri(), digest, meta=json.loads(meta.read_text()),
                        cache_dir=str(tmp_path / "c"))
    assert not (tmp_path / "c").exists()


def test_meta_url_needs_its_digest(tmp_path):
    onnx, meta, sha = _packaged()
    with pytest.raises(ValueError, match="meta_sha256"):
        st.ensure_model(onnx.as_uri(), sha, meta_url=meta.as_uri(), cache_dir=str(tmp_path))


def test_tampered_meta_is_rejected(tmp_path):
    onnx, meta, sha = _packaged()
    evil = tmp_path / "evil.meta.json"
    evil.write_text(json.dumps({"p_hi": 0.0, "p_lo": 0.0}))
    good = hashlib.sha256(meta.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="sha256 mismatch"):
        st.ensure_model(onnx.as_uri(), sha, meta_url=evil.as_uri(), meta_sha256=good,
                        cache_dir=str(tmp_path / "c"))


def test_meta_url_with_digest_loads(tmp_path):
    onnx, meta, sha = _packaged()
    try:
        st.ensure_model(onnx.as_uri(), sha, meta_url=meta.as_uri(),
                        meta_sha256=hashlib.sha256(meta.read_bytes()).hexdigest(),
                        cache_dir=str(tmp_path / "c"))
        assert st.get_model_meta()["p_hi"] == json.loads(meta.read_text())["p_hi"]
    finally:
        cnn.set_model_dir(None)


@pytest.mark.parametrize("thresholds", [{"p_hi": 0.2, "p_lo": 0.6}, {"p_hi": 85, "p_lo": 15},
                                        {"p_lo": 0.15}])
def test_model_with_impossible_thresholds_is_refused(tmp_path, thresholds):
    onnx, meta, _ = _packaged()
    (tmp_path / "strike_verdict_cnn.onnx").write_bytes(onnx.read_bytes())
    bad = {**json.loads(meta.read_text()), "p_hi": None, **thresholds}
    bad = {k: v for k, v in bad.items() if v is not None}
    (tmp_path / "strike_verdict_cnn.meta.json").write_text(json.dumps(bad))
    try:
        cnn.set_model_dir(str(tmp_path))
        with pytest.raises(ValueError, match="p_lo <= p_hi"):
            st.get_model_meta()
    finally:
        cnn.set_model_dir(None)


def test_loader_hands_back_the_model_it_loaded():
    cnn.set_model_dir(None)                              # the global is reset to None here
    score, meta = cnn._ensure_loaded()                   # callers use the tuple, not the global
    assert callable(score) and "p_hi" in meta


# ----------------------------------------------------------------------- the loaded model's digest

VARIANT = b"\x32\x07variant"        # a ModelProto doc_string field: a valid model, another digest


def _model_dir(tmp_path, onnx_bytes, name="m"):
    _, meta, _ = _packaged()
    d = tmp_path / name
    d.mkdir()
    (d / "strike_verdict_cnn.onnx").write_bytes(onnx_bytes)
    (d / "strike_verdict_cnn.meta.json").write_bytes(meta.read_bytes())
    return d


def _varint(n):
    out = bytearray()
    while True:
        out.append((n & 0x7F) | (0x80 if n > 0x7F else 0))
        n >>= 7
        if not n:
            return bytes(out)


def _field(number, payload):
    """A protobuf field: an int as a varint, bytes or str length-delimited."""
    if isinstance(payload, int):
        return _varint(number << 3) + _varint(payload)
    payload = payload.encode() if isinstance(payload, str) else payload
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def _external_tensor(name, location):
    """A float[1] TensorProto whose data is in the file `location` (data_location EXTERNAL)."""
    entry = lambda k, v: _field(13, _field(1, k) + _field(2, v))          # noqa: E731
    return (_field(1, 1) + _field(2, 1) + _field(8, name) + entry("location", location)
            + entry("offset", "0") + entry("length", "4") + _field(14, 1))


def _external_add_model(location="w.bin", unused=()):
    """An ONNX model y = x + w over float[1], its initializer w stored in the file `location`,
    built by hand so the test needs no `onnx` install. Each of `unused` (bytes or str) is the
    location of one more external initializer that no node reads."""
    vec = _field(2, _field(1, _field(1, 1) + _field(2, _field(1, _field(1, 1)))))
    graph = (_field(1, _field(1, "x") + _field(1, "w") + _field(2, "y") + _field(4, "Add"))
             + _field(2, "g") + _field(5, _external_tensor("w", location))
             + b"".join(_field(5, _external_tensor(f"u{i}", loc)) for i, loc in enumerate(unused))
             + _field(11, _field(1, "x") + vec) + _field(12, _field(1, "y") + vec))
    return _field(1, 8) + _field(8, _field(1, "") + _field(2, 13)) + _field(7, graph)


def test_meta_carries_the_digest_of_the_model_it_loaded(tmp_path):
    onnx, _, sha = _packaged()
    variant = onnx.read_bytes() + VARIANT
    d = _model_dir(tmp_path, variant)
    crop = np.random.default_rng(0).random((cnn.CROP_H, cnn.CROP_W)).astype(np.float32)
    try:
        cnn.set_model_dir(None)
        assert st.get_model_meta()["sha256"] == sha        # what ensure_model pins
        want = cnn.score_crops([crop])
        cnn.set_model_dir(str(d))
        meta = st.get_model_meta()
        assert meta["sha256"] == hashlib.sha256(variant).hexdigest() != sha
        assert "external_data" not in meta
        assert cnn.score_crops([crop]) == pytest.approx(want)   # the same weights, scored alike
    finally:
        cnn.set_model_dir(None)


def test_external_data_is_read_beside_the_model_and_hashed(tmp_path, monkeypatch):
    # ONNX Runtime reads a model's external data beside it only when it loads the model by path;
    # from bytes it looks in the working directory, which here holds a decoy of the same name.
    d = _model_dir(tmp_path, _external_add_model())
    (d / "w.bin").write_bytes(np.float32(2.0).tobytes())
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "w.bin").write_bytes(np.float32(-5.0).tobytes())
    monkeypatch.chdir(cwd)
    try:
        cnn.set_model_dir(str(d))
        score, meta = cnn._ensure_loaded()
        assert score(np.zeros(1, np.float32))[0] == pytest.approx(1 / (1 + np.exp(-2.0)))
        assert meta["external_data"] == [
            {"location": "w.bin", "sha256": hashlib.sha256((d / "w.bin").read_bytes()).hexdigest()}]
        assert meta["sha256"] == hashlib.sha256(_external_add_model()).hexdigest()
        # new weights, same graph file (a second dir: ONNX Runtime keeps the first file mapped)
        d2 = _model_dir(tmp_path, _external_add_model(), name="m2")
        (d2 / "w.bin").write_bytes(np.float32(3.0).tobytes())
        cnn.set_model_dir(str(d2))
        meta2 = st.get_model_meta()
        assert meta2["sha256"] == meta["sha256"]
        assert meta2["external_data"][0]["sha256"] != meta["external_data"][0]["sha256"]
    finally:
        cnn.set_model_dir(None)


def test_external_data_in_a_subfolder(tmp_path):
    d = _model_dir(tmp_path, _external_add_model("weights/w.bin"))
    (d / "weights").mkdir()
    (d / "weights" / "w.bin").write_bytes(np.float32(2.0).tobytes())
    try:
        cnn.set_model_dir(str(d))
        digest = hashlib.sha256((d / "weights" / "w.bin").read_bytes()).hexdigest()
        assert st.get_model_meta()["external_data"] == [
            {"location": "weights/w.bin", "sha256": digest}]
    finally:
        cnn.set_model_dir(None)


def test_stale_locations_on_tensors_nothing_reads_do_not_stop_the_load(tmp_path):
    # Initializers no node uses: ONNX Runtime never opens their files, so a missing one, a folder
    # or a name that is not UTF-8 must not stop a model it loads (0.12.1 loaded these).
    d = _model_dir(tmp_path, _external_add_model(unused=["missing.bin", ".", b"\xff\xfe.bin"]))
    (d / "w.bin").write_bytes(np.float32(2.0).tobytes())
    try:
        cnn.set_model_dir(str(d))
        meta = st.get_model_meta()
        assert [e["location"] for e in meta["external_data"]] == ["w.bin"]
    finally:
        cnn.set_model_dir(None)


def test_meta_json_cannot_supply_the_digests_and_callers_get_a_copy(tmp_path):
    onnx, meta, sha = _packaged()
    d = _model_dir(tmp_path, onnx.read_bytes())
    stale = {**json.loads(meta.read_text()), "sha256": "0" * 64,
             "external_data": [{"location": "old.data", "sha256": "1" * 64}]}
    (d / "strike_verdict_cnn.meta.json").write_text(json.dumps(stale))
    try:
        cnn.set_model_dir(str(d))
        got = st.get_model_meta()
        assert got["sha256"] == sha and "external_data" not in got
        got["sha256"], got["p_hi"] = "x", 0.0
        assert st.get_model_meta()["sha256"] == sha and st.get_model_meta()["p_hi"] != 0.0
        e = _model_dir(tmp_path, _external_add_model(), name="e")
        (e / "w.bin").write_bytes(np.float32(2.0).tobytes())
        cnn.set_model_dir(str(e))
        st.get_model_meta()["external_data"].append({"location": "extra"})
        assert len(st.get_model_meta()["external_data"]) == 1
    finally:
        cnn.set_model_dir(None)


def test_a_model_the_walker_cannot_read_is_refused_by_name(tmp_path):
    # An unknown field encoded as a group (wire type 3): ONNX Runtime skips it, the walker
    # cannot, and the loader says which file it could not read rather than report a digest.
    group = _varint(100 << 3 | 3) + _varint(100 << 3 | 4)
    d = _model_dir(tmp_path, _external_add_model() + group)
    (d / "w.bin").write_bytes(np.float32(2.0).tobytes())
    try:
        cnn.set_model_dir(str(d))
        with pytest.raises(ValueError, match="cannot read the model's external-data entries"):
            st.get_model_meta()
    finally:
        cnn.set_model_dir(None)


def test_external_data_found_in_every_tensor_holder():
    def tensor(loc):
        return _field(8, "t") + _field(13, _field(1, "location") + _field(2, loc))

    sub = _field(5, tensor("in_subgraph.bin"))                               # graph initializer
    node = _field(4, "If") + _field(5, _field(1, "then_branch") + _field(6, sub))
    loop = _field(4, "Scan") + _field(5, _field(1, "bodies")
                                      + _field(11, _field(5, tensor("in_graphs.bin"))))
    graph = (_field(1, node) + _field(1, loop) + _field(5, tensor("init.bin"))
             + _field(15, _field(1, tensor("sparse_values.bin"))
                      + _field(2, tensor("sparse_indices.bin")))
             + _field(1, _field(5, _field(10, tensor("attr_tensors.bin"))))
             + _field(1, _field(5, _field(22, _field(1, tensor("attr_sparse.bin")))))
             + _field(1, _field(5, _field(23, _field(2, tensor("attr_sparses.bin"))))))
    function = (_field(7, _field(5, _field(5, tensor("in_function.bin"))))
                + _field(11, _field(5, tensor("function_default.bin"))))
    training = _field(1, _field(5, tensor("training_init.bin"))) + _field(
        2, _field(5, tensor("training_algorithm.bin")))
    model = _field(1, 8) + _field(7, graph) + _field(20, training) + _field(25, function)
    assert cnn._external_data_locations(model) == sorted(
        ["in_subgraph.bin", "in_graphs.bin", "init.bin", "sparse_values.bin",
         "sparse_indices.bin", "attr_tensors.bin", "attr_sparse.bin", "attr_sparses.bin",
         "in_function.bin", "function_default.bin", "training_init.bin",
         "training_algorithm.bin"])
    onnx, _, _ = _packaged()
    assert cnn._external_data_locations(onnx.read_bytes() + VARIANT) == []


def test_a_tag_is_read_as_32_bits():
    # protobuf reads a five-byte tag BA 80 80 80 10 as field 7 (the graph), as ONNX Runtime does
    graph = _field(5, _field(8, "t") + _field(13, _field(1, "location") + _field(2, "w.bin")))
    model = b"\xba\x80\x80\x80\x10" + _varint(len(graph)) + graph
    assert cnn._external_data_locations(model) == ["w.bin"]


@pytest.mark.parametrize("data", [b"\x3a\x05ab", b"\x08\xff", b"\x0b",
                                  b"\x08" + b"\xff" * 10 + b"\x01"])     # an 11-byte varint
def test_malformed_model_bytes_raise(data):
    with pytest.raises(ValueError):
        cnn._external_data_locations(data)


# ----------------------------------------------------------------------- calibration

def test_manifest_string_labels_are_read_as_labels():
    probs = [0.9, 0.8, 0.3, 0.2]
    assert (C.threshold_for_precision(probs, ["struck", "struck", "clean", "clean"], 1.0)
            == C.threshold_for_precision(probs, [1, 1, 0, 0], 1.0) == 0.8)


@pytest.mark.parametrize("bad", [None, "unsure", 2])
def test_unfilled_or_unknown_label_raises(bad):
    with pytest.raises(ValueError, match="label 1"):
        C.pr_curve([0.9, 0.5], ["struck", bad])


def test_precision_threshold_respects_ties():
    probs, labels = [1, 1, 1, 1, 0.4], [1, 1, 1, 0, 1]   # at 1.0 the precision is 3/4, not 1
    with pytest.raises(ValueError, match="no threshold"):
        C.threshold_for_precision(probs, labels, 1.0)
    # 0.5 admits both tied words (one clean): precision 2/3, so only 0.9 reaches 1.0
    assert C.threshold_for_precision([0.9, 0.5, 0.5, 0.1], [1, 1, 0, 0], 1.0) == 0.9


def test_recall_threshold_is_the_highest_that_qualifies():
    pos = list(np.round(np.linspace(0.05, 0.95, 10), 2))
    t = C.threshold_for_recall(pos, [1] * 10, 0.9)
    assert t == pos[1]                                    # 9 of 10 stay at or above it
    assert sum(p >= t for p in pos) / 10 >= 0.9
