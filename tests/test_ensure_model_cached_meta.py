"""ensure_model called with neither meta_url nor meta reuses the cached meta only for the model it
was verified with, so two models sharing a cache folder never run with each other's thresholds;
a rejected download leaves the cache as it was."""
import hashlib
import json
import os
import pathlib
import stat

import pytest

import pdf_strikethrough as st
from pdf_strikethrough import cnn

MODEL, META = "strike_verdict_cnn.onnx", "strike_verdict_cnn.meta.json"
# root can write to a read-only file, so these tests would fail or prove nothing as root
needs_non_root = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                                    reason="root can write to read-only files")


def _packaged():
    here = pathlib.Path(cnn.__file__).parent
    onnx = here / MODEL
    return onnx, here / META, hashlib.sha256(onnx.read_bytes()).hexdigest()


def _other_model(tmp_path, name="other.onnx", data=b"another model"):
    """A second model file. ensure_model never loads what it fetches, so any bytes do."""
    p = tmp_path / name
    p.write_bytes(data)
    return p, hashlib.sha256(data).hexdigest()


def _meta(p_hi=0.85):
    _, meta, _ = _packaged()
    return {**json.loads(meta.read_text()), "p_hi": p_hi}


def _meta_file(tmp_path, p_hi=0.85, name="meta.json"):
    p = tmp_path / name
    p.write_text(json.dumps(_meta(p_hi)))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def _old_cache(c, model_bytes, meta_bytes):
    """A cache folder as a version before the record left it: the files, no record."""
    c.mkdir(parents=True)
    (c / MODEL).write_bytes(model_bytes)
    (c / META).write_bytes(meta_bytes)


@pytest.fixture(autouse=True)
def _packaged_model_after():
    yield
    cnn.set_model_dir(None)


def test_a_meta_given_once_is_reused_for_its_model(tmp_path):
    onnx, _, sha = _packaged()
    c = tmp_path / "c"
    st.ensure_model(onnx.as_uri(), sha, meta=_meta(0.9), cache_dir=str(c))
    cnn.set_model_dir(None)
    assert st.ensure_model(onnx.as_uri(), sha, cache_dir=str(c)) == str(c)
    assert st.get_model_meta()["p_hi"] == 0.9


def test_a_meta_url_verified_once_is_reused_unchanged(tmp_path):
    onnx, _, sha = _packaged()
    meta_file, meta_sha = _meta_file(tmp_path, 0.9)
    c = tmp_path / "c"
    st.ensure_model(onnx.as_uri(), sha, meta_url=meta_file.as_uri(), meta_sha256=meta_sha,
                    cache_dir=str(c))
    cnn.set_model_dir(None)
    st.ensure_model(onnx.as_uri(), sha, cache_dir=str(c))
    assert (c / META).read_bytes() == meta_file.read_bytes()
    assert st.get_model_meta()["p_hi"] == 0.9


def test_the_record_pairs_each_model_with_the_meta_verified_with_it(tmp_path):
    onnx, _, sha = _packaged()
    other, other_sha = _other_model(tmp_path)
    meta_file, meta_sha = _meta_file(tmp_path, 0.7)
    c = tmp_path / "c"
    st.ensure_model(onnx.as_uri(), sha, meta=_meta(), cache_dir=str(c))
    st.ensure_model(other.as_uri(), other_sha, meta_url=meta_file.as_uri(), meta_sha256=meta_sha,
                    cache_dir=str(c))
    record = json.loads((c / cnn._META_RECORD).read_text(encoding="utf-8"))
    assert record == {sha: hashlib.sha256(json.dumps(_meta()).encode()).hexdigest(),
                      other_sha: meta_sha}


def test_another_models_meta_in_a_shared_cache_is_refused(tmp_path):
    onnx, _, sha = _packaged()
    other, other_sha = _other_model(tmp_path)
    c = tmp_path / "c"
    st.ensure_model(onnx.as_uri(), sha, meta=_meta(), cache_dir=str(c))
    cnn.set_model_dir(None)
    with pytest.raises(ValueError, match="not verified for this model"):
        st.ensure_model(other.as_uri(), other_sha, cache_dir=str(c))
    assert cnn._MODEL_DIR_OVERRIDE is None                 # the loader was not pointed at it
    assert hashlib.sha256((c / MODEL).read_bytes()).hexdigest() == sha   # nothing written


def test_two_models_with_the_same_meta_share_a_folder(tmp_path):
    onnx, _, sha = _packaged()
    other, other_sha = _other_model(tmp_path)
    c = tmp_path / "c"
    st.ensure_model(onnx.as_uri(), sha, meta=_meta(), cache_dir=str(c))
    st.ensure_model(other.as_uri(), other_sha, meta=_meta(), cache_dir=str(c))
    cnn.set_model_dir(None)
    st.ensure_model(onnx.as_uri(), sha, cache_dir=str(c))  # downloaded again over the other
    assert st.get_model_meta()["sha256"] == sha


def test_a_rejected_meta_leaves_the_cache_as_it_was(tmp_path):
    onnx, _, sha = _packaged()
    other, other_sha = _other_model(tmp_path)
    meta_file, _ = _meta_file(tmp_path)
    c = tmp_path / "c"
    st.ensure_model(onnx.as_uri(), sha, meta=_meta(0.9), cache_dir=str(c))
    before = {p.name: p.read_bytes() for p in c.iterdir()}
    for bad in ({"meta_url": meta_file.as_uri(), "meta_sha256": "0" * 64},
                {"meta": {"p_hi": object()}}):                      # not JSON-serializable
        with pytest.raises((ValueError, TypeError)):
            st.ensure_model(other.as_uri(), other_sha, cache_dir=str(c), **bad)
        assert {p.name: p.read_bytes() for p in c.iterdir()} == before
    with pytest.raises(ValueError, match="not verified for this model"):
        st.ensure_model(other.as_uri(), other_sha, cache_dir=str(c))


def test_a_meta_changed_since_it_was_recorded_is_refused(tmp_path):
    onnx, _, sha = _packaged()
    c = tmp_path / "c"
    st.ensure_model(onnx.as_uri(), sha, meta=_meta(), cache_dir=str(c))
    (c / META).write_text(json.dumps(_meta(0.0)), encoding="utf-8")
    with pytest.raises(ValueError, match="not verified for this model"):
        st.ensure_model(onnx.as_uri(), sha, cache_dir=str(c))


@pytest.mark.parametrize("record", [b"\x00\xff not json", b"[]", b"null", b'"x"', b"",
                                    b"[" * 100_000, b'{"model_sha256": 1}'],
                         ids=["bytes", "list", "null", "string", "empty", "nested", "other-keys"])
def test_an_unreadable_record_vouches_for_nothing(tmp_path, record):
    onnx, _, sha = _packaged()
    c = tmp_path / "c"
    st.ensure_model(onnx.as_uri(), sha, meta=_meta(), cache_dir=str(c))
    (c / cnn._META_RECORD).write_bytes(record)
    with pytest.raises(ValueError, match="not verified for this model"):
        st.ensure_model(onnx.as_uri(), sha, cache_dir=str(c))
    st.ensure_model(onnx.as_uri(), sha, meta=_meta(), cache_dir=str(c))   # a meta rewrites it
    st.ensure_model(onnx.as_uri(), sha, cache_dir=str(c))


def test_a_new_meta_replaces_the_models_entry(tmp_path):
    onnx, _, sha = _packaged()
    c = tmp_path / "c"
    st.ensure_model(onnx.as_uri(), sha, meta=_meta(0.9), cache_dir=str(c))
    st.ensure_model(onnx.as_uri(), sha, meta=_meta(0.8), cache_dir=str(c))
    cnn.set_model_dir(None)
    st.ensure_model(onnx.as_uri(), sha, cache_dir=str(c))
    assert st.get_model_meta()["p_hi"] == 0.8


@needs_non_root
def test_a_read_only_cache_that_holds_the_files_still_works(tmp_path):
    onnx, _, sha = _packaged()
    meta_file, meta_sha = _meta_file(tmp_path)
    c = tmp_path / "c"
    kw = {"meta_url": meta_file.as_uri(), "meta_sha256": meta_sha, "cache_dir": str(c)}
    st.ensure_model(onnx.as_uri(), sha, **kw)
    files = list(c.iterdir())
    for p in files:
        os.chmod(p, stat.S_IREAD)
    try:
        st.ensure_model(onnx.as_uri(), sha, **kw)                        # nothing to write
        st.ensure_model(onnx.as_uri(), sha, meta=_meta(), cache_dir=str(c))
        st.ensure_model(onnx.as_uri(), sha, cache_dir=str(c))
    finally:
        for p in files:
            os.chmod(p, stat.S_IREAD | stat.S_IWRITE)


def test_a_call_with_nothing_new_writes_nothing(tmp_path, monkeypatch):
    onnx, _, sha = _packaged()
    meta_file, meta_sha = _meta_file(tmp_path)
    c = tmp_path / "c"
    calls = [{"meta_url": meta_file.as_uri(), "meta_sha256": meta_sha}, {"meta": _meta(0.9)}, {}]
    for kw in calls:
        st.ensure_model(onnx.as_uri(), sha, cache_dir=str(c), **kw)

    def no_write(self, *a, **k):
        raise AssertionError(f"wrote {self}")

    monkeypatch.setattr(pathlib.Path, "write_bytes", no_write)
    monkeypatch.setattr(pathlib.Path, "write_text", no_write)
    for kw in calls[1:]:
        st.ensure_model(onnx.as_uri(), sha, cache_dir=str(c), **kw)


def test_a_record_that_cannot_be_written_is_skipped(tmp_path):
    """The record's name is taken by a folder: the call goes through, and a later call with
    neither meta argument is refused rather than trusting the meta."""
    onnx, _, sha = _packaged()
    c = tmp_path / "c"
    (c / cnn._META_RECORD).mkdir(parents=True)
    st.ensure_model(onnx.as_uri(), sha, meta=_meta(), cache_dir=str(c))
    assert st.get_model_meta()["sha256"] == sha
    with pytest.raises(ValueError, match="not verified for this model"):
        st.ensure_model(onnx.as_uri(), sha, cache_dir=str(c))


def test_a_cache_with_no_record_is_reused_as_before(tmp_path):
    """A folder written before the record existed, already holding the model: its meta is reused
    as it was, and no record is made for a meta this call did not verify."""
    onnx, meta, sha = _packaged()
    c = tmp_path / "c"
    _old_cache(c, onnx.read_bytes(), meta.read_bytes())
    st.ensure_model(onnx.as_uri(), sha, cache_dir=str(c))
    assert st.get_model_meta()["sha256"] == sha
    assert not (c / cnn._META_RECORD).exists()


@pytest.mark.parametrize("stale_model", [None, b"an older model"])
def test_no_record_and_another_model_is_refused_and_stays_refused(tmp_path, stale_model):
    """With no record, the meta came with whatever model is in the folder (or none): asked for
    another, the call writes nothing, so asking again is refused again."""
    onnx, meta, sha = _packaged()
    c = tmp_path / "c"
    c.mkdir()
    (c / META).write_bytes(meta.read_bytes())
    if stale_model is not None:
        (c / MODEL).write_bytes(stale_model)
    for _ in range(2):
        with pytest.raises(ValueError, match="not verified for this model"):
            st.ensure_model(onnx.as_uri(), sha, cache_dir=str(c))
        assert cnn._MODEL_DIR_OVERRIDE is None
        assert sorted(p.name for p in c.iterdir()) == sorted(
            [META] + ([MODEL] if stale_model is not None else []))
        if stale_model is not None:
            assert (c / MODEL).read_bytes() == stale_model


def test_no_record_and_a_rejected_meta_still_refuses_the_new_model(tmp_path):
    onnx, meta, sha = _packaged()
    other, other_sha = _other_model(tmp_path)
    meta_file, _ = _meta_file(tmp_path)
    c = tmp_path / "c"
    _old_cache(c, onnx.read_bytes(), meta.read_bytes())
    with pytest.raises(ValueError, match="sha256 mismatch"):
        st.ensure_model(other.as_uri(), other_sha, meta_url=meta_file.as_uri(),
                        meta_sha256="0" * 64, cache_dir=str(c))
    assert hashlib.sha256((c / MODEL).read_bytes()).hexdigest() == sha
    with pytest.raises(ValueError, match="not verified for this model"):
        st.ensure_model(other.as_uri(), other_sha, cache_dir=str(c))


@needs_non_root
def test_a_meta_that_fails_to_write_leaves_a_record_that_refuses_the_old_one(tmp_path):
    """A folder from before the record, holding another model and its meta: a call for this model
    writes the record and the model, then cannot write the meta. The record, written first,
    keeps the other model's meta from being taken for this one."""
    onnx, _, sha = _packaged()
    other, other_sha = _other_model(tmp_path, data=b"the folder's old model")
    c = tmp_path / "c"
    _old_cache(c, other.read_bytes(), json.dumps(_meta(0.6)).encode())
    os.chmod(c / META, stat.S_IREAD)
    try:
        with pytest.raises(PermissionError):
            st.ensure_model(onnx.as_uri(), sha, meta=_meta(0.9), cache_dir=str(c))
        with pytest.raises(ValueError, match="not verified for this model"):
            st.ensure_model(onnx.as_uri(), sha, cache_dir=str(c))
    finally:
        os.chmod(c / META, stat.S_IREAD | stat.S_IWRITE)


def test_no_record_in_the_models_own_default_folder_is_reused(tmp_path, monkeypatch):
    """The default folder is named after the model's digest, so its meta came with this model,
    even when the model file has to be fetched again."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    onnx, meta, sha = _packaged()
    c = tmp_path / ".cache" / "pdf_strikethrough" / "models" / sha[:12]
    assert pathlib.Path.home() == tmp_path
    _old_cache(c, b"a partial download", meta.read_bytes())
    assert st.ensure_model(onnx.as_uri(), sha) == str(c)
    assert (c / MODEL).read_bytes() == onnx.read_bytes()
    assert st.get_model_meta()["sha256"] == sha
