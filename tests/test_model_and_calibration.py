"""StrikeNet loading and threshold calibration: digests that verify what they promise, thresholds a
model cannot be loaded without, and calibration helpers that read the labels and ties a
``dump_crops`` manifest actually holds."""
import hashlib
import json
import os
import pathlib
import subprocess

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


# ------------------------------------------------- ensure_model and a model's external data

def _hosted_external(tmp_path, location="w.bin", weight=2.0):
    """A host folder with an external-data model and its weights file; the meta that pins it."""
    host = tmp_path / "host"
    (host / location).parent.mkdir(parents=True, exist_ok=True)
    (host / "strike_verdict_cnn.onnx").write_bytes(_external_add_model(location))
    (host / location).write_bytes(np.float32(weight).tobytes())
    _, meta, _ = _packaged()
    pinned = [{"location": location,
               "sha256": hashlib.sha256((host / location).read_bytes()).hexdigest()}]
    return (host / "strike_verdict_cnn.onnx", hashlib.sha256(_external_add_model(location))
            .hexdigest(), {**json.loads(meta.read_text()), "external_data": pinned})


@pytest.mark.parametrize("location", ["w.bin", "weights/w.bin", "w x.bin", "w#1.bin", "w%41.bin"])
def test_ensure_model_fetches_the_external_data_its_meta_pins(tmp_path, location):
    model, sha, meta = _hosted_external(tmp_path, location)
    c = tmp_path / "c"
    try:
        st.ensure_model(model.as_uri(), sha, meta=meta, cache_dir=str(c))
        assert (c / location).read_bytes() == np.float32(2.0).tobytes()
        score, got = cnn._ensure_loaded()
        assert score(np.zeros(1, np.float32))[0] == pytest.approx(1 / (1 + np.exp(-2.0)))
        assert got["external_data"] == meta["external_data"]   # what get_model_meta() reports
    finally:
        cnn.set_model_dir(None)


def test_external_data_pinned_by_a_meta_url_and_reused_offline(tmp_path):
    model, sha, meta = _hosted_external(tmp_path)
    meta_file = tmp_path / "host" / "strike_verdict_cnn.meta.json"
    meta_file.write_text(json.dumps(meta))
    meta_sha = hashlib.sha256(meta_file.read_bytes()).hexdigest()
    c = tmp_path / "c"
    try:
        st.ensure_model(model.as_uri(), sha, meta_url=meta_file.as_uri(), meta_sha256=meta_sha,
                        cache_dir=str(c))
        (tmp_path / "host" / "w.bin").unlink()                  # the host is gone
        cnn.set_model_dir(None)
        st.ensure_model(model.as_uri(), sha, meta_url=meta_file.as_uri(), meta_sha256=meta_sha,
                        cache_dir=str(c))
        assert st.get_model_meta()["external_data"] == meta["external_data"]
    finally:
        cnn.set_model_dir(None)


def test_tampered_external_data_is_rejected_and_not_written(tmp_path):
    model, sha, meta = _hosted_external(tmp_path)
    (tmp_path / "host" / "w.bin").write_bytes(np.float32(-5.0).tobytes())
    c = tmp_path / "c"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        st.ensure_model(model.as_uri(), sha, meta=meta, cache_dir=str(c))
    assert not (c / "w.bin").exists()
    assert cnn._MODEL_DIR_OVERRIDE is None


@pytest.mark.parametrize("listed", [None, [], [{"location": "other.bin", "sha256": "0" * 64}]])
def test_external_data_the_meta_does_not_pin_is_refused(tmp_path, listed):
    model, sha, meta = _hosted_external(tmp_path)
    meta = {k: v for k, v in meta.items() if k != "external_data"}
    if listed is not None:
        meta["external_data"] = listed
    c = tmp_path / "c"
    with pytest.raises(ValueError, match="gives no sha256"):
        st.ensure_model(model.as_uri(), sha, meta=meta, cache_dir=str(c))
    assert not (c / "w.bin").exists()
    assert cnn._MODEL_DIR_OVERRIDE is None


def test_a_pin_for_a_file_the_model_does_not_name_is_refused(tmp_path):
    """The loader would refuse the folder at the first score, so ensure_model refuses it first."""
    model, sha, meta = _hosted_external(tmp_path)
    meta["external_data"].append({"location": "other.bin", "sha256": "0" * 64})
    c = tmp_path / "c"
    with pytest.raises(ValueError, match="does not name"):
        st.ensure_model(model.as_uri(), sha, meta=meta, cache_dir=str(c))
    assert [p.name for p in c.rglob("*")] == []
    assert cnn._MODEL_DIR_OVERRIDE is None


@pytest.mark.parametrize("listed, match", [
    ({"location": "w.bin"}, "list"),
    (["w.bin"], "no location"),
    ([{"sha256": "0" * 64}], "no location"),
    ([{"location": "w.bin", "sha256": "abc"}], "64-character"),
    ([{"location": "w.bin", "sha256": "0" * 64}, {"location": "w.bin", "sha256": "1" * 64}],
     "twice"),
])
def test_a_malformed_external_data_list_is_refused(tmp_path, listed, match):
    model, sha, meta = _hosted_external(tmp_path)
    with pytest.raises(ValueError, match=match):
        st.ensure_model(model.as_uri(), sha, meta={**meta, "external_data": listed},
                        cache_dir=str(tmp_path / "c"))
    assert cnn._MODEL_DIR_OVERRIDE is None


@pytest.mark.parametrize("location", ["../w.bin", "/w.bin", "C:w.bin", "a\\w.bin", "a//w.bin",
                                      "./w.bin"])
def test_an_external_location_outside_the_cache_is_refused(tmp_path, location):
    host = tmp_path / "host"
    host.mkdir()
    (host / "strike_verdict_cnn.onnx").write_bytes(_external_add_model(location))
    _, meta, _ = _packaged()
    meta = {**json.loads(meta.read_text()),
            "external_data": [{"location": location, "sha256": "0" * 64}]}
    c = tmp_path / "deep" / "c"
    with pytest.raises(ValueError, match="plain relative path"):
        st.ensure_model((host / "strike_verdict_cnn.onnx").as_uri(),
                        hashlib.sha256(_external_add_model(location)).hexdigest(), meta=meta,
                        cache_dir=str(c))
    assert [p.name for p in (tmp_path / "deep").rglob("*")] == ["c"]   # nothing written
    assert cnn._MODEL_DIR_OVERRIDE is None


def _host(tmp_path, locations, write=True):
    """A host folder with a model whose first location is read by its node, the rest by nothing,
    each file written (`write`), and a meta pinning all of them."""
    host = tmp_path / "host"
    host.mkdir(exist_ok=True)
    model = _external_add_model(locations[0], unused=locations[1:])
    (host / "strike_verdict_cnn.onnx").write_bytes(model)
    _, meta, _ = _packaged()
    data = np.float32(2.0).tobytes()
    pins = [{"location": loc, "sha256": hashlib.sha256(data).hexdigest()} for loc in locations]
    for loc in locations if write else []:
        (host / loc).parent.mkdir(parents=True, exist_ok=True)
        (host / loc).write_bytes(data)
    return (host / "strike_verdict_cnn.onnx", hashlib.sha256(model).hexdigest(),
            {**json.loads(meta.read_text()), "external_data": pins})


def test_a_failed_external_fetch_writes_nothing(tmp_path):
    model, sha, meta = _hosted_external(tmp_path)
    c = tmp_path / "c"
    st.ensure_model(model.as_uri(), sha, meta=meta, cache_dir=str(c))       # model A, working
    cnn.set_model_dir(None)
    before = {p.relative_to(c): p.read_bytes() for p in c.rglob("*") if p.is_file()}
    # model B: the same graph file, other weights, and its weights missing on the host
    (tmp_path / "host" / "w.bin").unlink()
    meta_b = {**meta, "p_hi": 0.9, "external_data": [{"location": "w.bin", "sha256": "1" * 64}]}
    with pytest.raises(OSError) as err:
        st.ensure_model(model.as_uri(), sha, meta=meta_b, cache_dir=str(c))
    assert {p.relative_to(c): p.read_bytes() for p in c.rglob("*") if p.is_file()} == before
    assert cnn._MODEL_DIR_OVERRIDE is None
    if hasattr(err.value, "add_note"):
        assert any("'w.bin'" in n for n in err.value.__notes__)


@pytest.mark.parametrize("locations", [
    ["strike_verdict_cnn.meta.json"], ["STRIKE_VERDICT_CNN.ONNX"], ["strike_verdict_cnn.pt"],
    ["strike_verdict_cnn.meta.verified.json"], ["NUL"], ["a/con.bin"], ["COM1.dat"],
    ["W.bin", "w.bin"], ["w.bin", "w.bin."], ["w.bin", "w.bin "], ["a.bin", "../c.bin"],
    ["a.bin", "b/../c.bin"], ["a /w.bin"], ["a./w.bin"], ["a<b.bin"], ["a\tb.bin"],
    ["w" * 256 + ".bin"], ["STRIKE~2.JSO"], ["sub/STRIKE~1.ONN"], ["sub", "sub/w.bin"],
    ["sub/w.bin", "SUB"], ["A/w.bin", "a"], ["strike_verdict_cnn.onnx/w.bin"]])
def test_locations_that_are_not_a_file_of_their_own_are_refused_before_any_download(
        tmp_path, locations):
    model, sha, meta = _host(tmp_path, [loc for loc in locations if ".." not in loc], write=False)
    if any(".." in loc for loc in locations):                   # the model names it all the same
        model.write_bytes(_external_add_model(locations[0], unused=locations[1:]))
        sha = hashlib.sha256(model.read_bytes()).hexdigest()
        meta["external_data"] += [{"location": loc, "sha256": "0" * 64}
                                  for loc in locations if ".." in loc]
    c = tmp_path / "c"
    with pytest.raises(ValueError, match="plain relative path"):
        st.ensure_model(model.as_uri(), sha, meta=meta, cache_dir=str(c))
    assert [p.name for p in c.rglob("*")] == []
    assert cnn._MODEL_DIR_OVERRIDE is None


def test_a_location_through_a_link_out_of_the_cache_is_refused(tmp_path):
    model, sha, meta = _hosted_external(tmp_path, "sub/w.bin")
    c, outside = tmp_path / "c", tmp_path / "outside"
    c.mkdir()
    outside.mkdir()
    try:
        os.symlink(outside, c / "sub", target_is_directory=True)
    except OSError:
        if os.name != "nt" or subprocess.run(["cmd", "/c", "mklink", "/J", str(c / "sub"),
                                              str(outside)], capture_output=True).returncode:
            pytest.skip("no permission to create a directory link here")
    with pytest.raises(ValueError, match="plain relative path"):
        st.ensure_model(model.as_uri(), sha, meta=meta, cache_dir=str(c))
    assert list(outside.iterdir()) == []


def test_a_signed_model_url_with_external_data_is_refused_up_front(tmp_path):
    """The model and meta are already cached, so nothing is fetched; the files' URLs would lose
    the query string, so the call says so rather than fail on a 403."""
    model, sha, meta = _hosted_external(tmp_path)
    c = tmp_path / "c"
    st.ensure_model(model.as_uri(), sha, meta=meta, cache_dir=str(c))
    with pytest.raises(ValueError, match="query string"):
        st.ensure_model(model.as_uri() + "?sig=1", sha, meta=meta, cache_dir=str(c))
    onnx, packaged_meta, packaged_sha = _packaged()             # one file: a query is fine
    d = tmp_path / "d"
    st.ensure_model(onnx.as_uri(), packaged_sha, meta=json.loads(packaged_meta.read_text()),
                    cache_dir=str(d))
    try:
        st.ensure_model(onnx.as_uri() + "?sig=1", packaged_sha,
                        meta=json.loads(packaged_meta.read_text()), cache_dir=str(d))
    finally:
        cnn.set_model_dir(None)


def test_the_record_is_written_before_the_external_files(tmp_path, monkeypatch):
    """A call stopped while writing the record has written nothing else: had it written new
    weights first, the old record would vouch for the old meta beside them."""
    model, sha, meta = _hosted_external(tmp_path)
    c = tmp_path / "c"
    st.ensure_model(model.as_uri(), sha, meta=meta, cache_dir=str(c))
    before = {p.relative_to(c): p.read_bytes() for p in c.rglob("*") if p.is_file()}
    (tmp_path / "host" / "w.bin").write_bytes(np.float32(3.0).tobytes())   # retrained weights
    meta_b = {**meta, "p_hi": 0.9, "external_data": [
        {"location": "w.bin",
         "sha256": hashlib.sha256(np.float32(3.0).tobytes()).hexdigest()}]}
    real = pathlib.Path.write_text

    def stop_at_the_record(self, *a, **k):
        if self.name == cnn._META_RECORD:
            raise RuntimeError("stopped")
        return real(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "write_text", stop_at_the_record)
    with pytest.raises(RuntimeError):
        st.ensure_model(model.as_uri(), sha, meta=meta_b, cache_dir=str(c))
    assert {p.relative_to(c): p.read_bytes() for p in c.rglob("*") if p.is_file()} == before
    cnn.set_model_dir(None)


def test_a_file_that_does_not_read_back_as_written_is_refused(tmp_path, monkeypatch):
    """Two names for one file (case, trailing dots, a device) are refused by name; whatever the
    names miss, each file is hashed again after writing."""
    model, sha, meta = _hosted_external(tmp_path)
    real = pathlib.Path.write_bytes

    def short_write(self, data):
        return real(self, data[:-1] if self.name == "w.bin" else data)

    monkeypatch.setattr(pathlib.Path, "write_bytes", short_write)
    with pytest.raises(ValueError, match="does not read back"):
        st.ensure_model(model.as_uri(), sha, meta=meta, cache_dir=str(tmp_path / "c"))
    assert cnn._MODEL_DIR_OVERRIDE is None


def test_weights_that_fail_to_write_leave_the_old_model_whole(tmp_path, monkeypatch):
    """Model A cached, then retrained weights B under the same graph file, whose weights cannot
    be written (ONNX Runtime holding the file open, say): the folder still loads A, its weights
    with its meta, and a call with neither meta argument is refused."""
    model, sha, meta = _hosted_external(tmp_path)
    c = tmp_path / "c"
    st.ensure_model(model.as_uri(), sha, meta=meta, cache_dir=str(c))
    cnn.set_model_dir(None)
    (tmp_path / "host" / "w.bin").write_bytes(np.float32(3.0).tobytes())
    meta_b = {**meta, "p_hi": 0.9, "external_data": [
        {"location": "w.bin", "sha256": hashlib.sha256(np.float32(3.0).tobytes()).hexdigest()}]}
    real = pathlib.Path.write_bytes

    def held_open(self, data):
        if self.name == "w.bin":
            raise PermissionError("in use")
        return real(self, data)

    monkeypatch.setattr(pathlib.Path, "write_bytes", held_open)
    with pytest.raises(PermissionError):
        st.ensure_model(model.as_uri(), sha, meta=meta_b, cache_dir=str(c))
    monkeypatch.undo()
    try:
        cnn.set_model_dir(str(c))
        score, got = cnn._ensure_loaded()
        assert got["p_hi"] == meta["p_hi"] != 0.9
        assert score(np.zeros(1, np.float32))[0] == pytest.approx(1 / (1 + np.exp(-2.0)))
        with pytest.raises(ValueError, match="not verified for this model"):
            st.ensure_model(model.as_uri(), sha, cache_dir=str(c))
    finally:
        cnn.set_model_dir(None)


@pytest.mark.parametrize("pins, match", [
    ([{"location": "w.bin", "sha256": "1" * 64}], "do not match"),
    ([], "do not match"),
    ([{"location": "w.bin", "sha256": None}], "must be a list"),
    ("w.bin", "must be a list"),
])
def test_a_folder_whose_weights_and_meta_disagree_is_refused_at_load(tmp_path, pins, match):
    d = _model_dir(tmp_path, _external_add_model())
    (d / "w.bin").write_bytes(np.float32(2.0).tobytes())
    meta = json.loads((d / "strike_verdict_cnn.meta.json").read_text())
    (d / "strike_verdict_cnn.meta.json").write_text(json.dumps({**meta, "external_data": pins}))
    try:
        cnn.set_model_dir(str(d))
        with pytest.raises(ValueError, match=match):
            st.get_model_meta()
    finally:
        cnn.set_model_dir(None)


def test_a_meta_whose_pins_match_loads_and_reports_them(tmp_path):
    d = _model_dir(tmp_path, _external_add_model())
    (d / "w.bin").write_bytes(np.float32(2.0).tobytes())
    pins = [{"location": "w.bin",
             "sha256": hashlib.sha256((d / "w.bin").read_bytes()).hexdigest().upper()}]
    meta = json.loads((d / "strike_verdict_cnn.meta.json").read_text())
    (d / "strike_verdict_cnn.meta.json").write_text(json.dumps({**meta, "external_data": pins}))
    try:
        cnn.set_model_dir(str(d))
        assert st.get_model_meta()["external_data"] == [
            {"location": "w.bin", "sha256": pins[0]["sha256"].lower()}]
    finally:
        cnn.set_model_dir(None)


def test_a_meta_that_is_not_an_object_pins_nothing(tmp_path):
    model, sha, _ = _hosted_external(tmp_path)
    with pytest.raises(ValueError, match="gives no sha256"):
        st.ensure_model(model.as_uri(), sha, meta=["not", "an", "object"],
                        cache_dir=str(tmp_path / "c"))


def test_a_model_without_external_data_never_reads_the_list(tmp_path):
    onnx, meta, sha = _packaged()
    try:
        st.ensure_model(onnx.as_uri(), sha, cache_dir=str(tmp_path / "c"),
                        meta={**json.loads(meta.read_text()), "external_data": "not a list"})
        assert "external_data" not in st.get_model_meta()
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
