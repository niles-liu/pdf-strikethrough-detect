"""StrikeNet — a tiny (79k-param) CNN that answers the physical question
"is there a strike through this word crop?" from pixels alone.

It is OCR- and domain-independent: the input is a grayscale word crop, the output is a
strike probability in [0, 1]. It exists to resolve cases geometry alone cannot decide
(a real thin strike over an ascender-less word looks, pixel-for-pixel, like a serif glyph
chain) and as an agreement check on geometric detections.

Inference prefers ONNX Runtime (small dependency, CPU-fast); PyTorch is an optional fallback
for dev environments that supply a ``.pt`` checkpoint. The package ships ONNX only:
  strike_verdict_cnn.onnx + strike_verdict_cnn.meta.json
Override the model location with the PDF_STRIKETHROUGH_MODEL_DIR environment variable.

Thresholds live in the checkpoint metadata: p >= p_hi -> 'struck', p <= p_lo -> 'clean',
between -> 'unsure' (guardrailed 0.85 / 0.15).
"""
import json
import os
import threading

import numpy as np
from PIL import Image

from .lines import to_gray_u8

CROP_H, CROP_W = 32, 160          # net input: ink-positive [0,1], height-normalized isotropically
PAD_X, PAD_Y = 5, 7               # crop margin around the word box, in PIXELS

# --- Calibration note --------------------------------------------------------------------------
# PAD_X/PAD_Y are FIXED PIXEL margins calibrated for RENDER_DPI = 200 (detect.py). Their share of a
# word box shifts with resolution (large around a 72-dpi glyph, tiny at 600 dpi), which would push
# the CNN input off its training distribution, so detect_pdf / detect_image_file bring any raster
# above HIGH_DPI_CAP down to RENDER_DPI before cropping (see detect._working_dpi). A crop from a
# caller-supplied raster far below 200 dpi (score_word on a 72-dpi image, say) is still
# off-distribution: nothing upsamples it.

SHADE_WHITE_FRAC = 0.85           # shaded-crop white point, as a fraction of the paper ground.
                                  # PROVISIONAL: only issue #15's scan needs it below 1.0. On
                                  # manufactured shading every value 1.0..0.75 clears the same
                                  # false positives and lower ones only cost recall, so 0.85 is
                                  # the edge of the safe range, not a measured optimum.

_MODEL_DIR_OVERRIDE = None        # set via set_model_dir(); wins over the env var
_META_RECORD = "strike_verdict_cnn.meta.verified.json"  # ensure_model's record: the sha256 of
                                  # each model in the folder -> the sha256 of the meta verified or
                                  # written with it
_OWN_FILES = {n.casefold() for n in ("strike_verdict_cnn.onnx", "strike_verdict_cnn.meta.json",
                                     "strike_verdict_cnn.pt", _META_RECORD)}
_DEVICE_NAMES = {"con", "prn", "aux", "nul", "conin$", "conout$",
                 *(f"{d}{i}" for d in ("com", "lpt") for i in range(10))}  # Windows devices
_BAD_NAME_CHARS = frozenset('\\:<>"|?*' + "".join(map(chr, range(32))))  # not in a Windows name
_lock = threading.Lock()
_model = None                     # lazy singleton: (score_fn, meta dict)


def _current_model_dir():
    """Resolve the model directory at LOAD time, not import time, so setting
    PDF_STRIKETHROUGH_MODEL_DIR or calling set_model_dir() any time before the first score takes
    effect, even after the package is imported."""
    return (_MODEL_DIR_OVERRIDE
            or os.environ.get("PDF_STRIKETHROUGH_MODEL_DIR")
            or os.path.dirname(os.path.abspath(__file__)))


def set_model_dir(path):
    """Point the loader at a directory holding strike_verdict_cnn.onnx + .meta.json (or a .pt),
    clearing any already-loaded model so the next score reloads from `path`. Pass None to revert
    to PDF_STRIKETHROUGH_MODEL_DIR / the packaged model."""
    global _MODEL_DIR_OVERRIDE, _model
    with _lock:
        _MODEL_DIR_OVERRIDE = path
        _model = None


def ensure_model(url, sha256, *, meta_url=None, meta_sha256=None, meta=None,
                 cache_dir=None, timeout=30):
    """Download a StrikeNet model to a local cache, VERIFY its sha256, and point the loader at it.

    `url` is the ONNX model and `sha256` its expected hex digest (any case). The bytes are hashed
    after download and a mismatch raises ``ValueError`` (nothing is written) — so a tampered host or
    a man-in-the-middle cannot swap the graph you run; nothing unverified is ever loaded. The ONNX
    loader also needs a meta JSON (thresholds + crop geometry): supply it as `meta_url` with its
    required `meta_sha256` (the thresholds decide what counts as struck), or as a `meta` dict
    written verbatim. A digest that is not 64 hex characters raises ``ValueError``.

    With neither, the meta already in the cache is reused only for the model it was verified with:
    each call given `meta_url` or `meta` records the model's sha256 with the meta's in the cache
    folder (when the record can be written), and a call with neither raises ``ValueError``,
    writing no file, unless that record pairs this model with the meta as it is now. A folder with
    no record (one written before there was a record) is trusted as before when it is this
    model's default folder or already holds this model; pass `meta_url` or `meta` once to record
    it. The model and its meta are both verified before either is written, so a rejected download
    leaves the cache as it was.

    A model that keeps its weights in external data files (torch's newer exporter writes them by
    default) needs each file hosted beside the model, at its location relative to the model's URL,
    and its sha256 listed in the meta's ``external_data``, the shape :func:`get_model_meta`
    reports: ``[{"location": ..., "sha256": ...}, ...]``. Each is downloaded and verified with the
    model and the meta, before any of them is written. A file the meta does not list, a listed
    file the model does not name, a location that is not a plain relative path to a file of its
    own inside the cache folder, or a model URL with a query string raises ``ValueError`` before
    any of the files is downloaded. The files are written
    weights first and the meta last, then hashed again; the loader refuses a folder whose files
    do not match its meta's ``external_data``. For such a model `sha256` covers the graph, not the
    weights, and the cached-meta record and the default folder go by it: give retrained weights
    under the same graph file their own `cache_dir`.

    Downloads land in `cache_dir` (default:
    ``~/.cache/pdf_strikethrough/models/<sha256[:12]>``); an already-present, hash-matching file is
    reused. Accepts ``http(s)`` and ``file`` URLs. Returns the cache directory (already applied via
    :func:`set_model_dir`)."""
    import hashlib
    import pathlib
    import re
    import urllib.parse
    import urllib.request

    def _digest(value, name):
        d = value.strip().lower() if isinstance(value, str) else ""
        if not re.fullmatch(r"[0-9a-f]{64}", d):
            raise ValueError(f"{name} must be a 64-character hex sha256 digest, got {value!r}")
        return d

    def _fetch_verify(u, digest, dest):
        """The verified bytes at `u`, or None when `dest` already holds them. Writes nothing."""
        scheme = urllib.parse.urlparse(u).scheme
        if scheme not in ("http", "https", "file"):
            raise ValueError(f"unsupported URL scheme {scheme!r} (use http, https, or file)")
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == digest:
            return None
        with urllib.request.urlopen(u, timeout=timeout) as r:
            data = r.read()
        got = hashlib.sha256(data).hexdigest()
        if got != digest:
            raise ValueError(f"sha256 mismatch for {u}: expected {digest}, got {got} "
                             "(download rejected; nothing was written)")
        return data

    sha256 = _digest(sha256, "sha256")
    if meta_url is not None:
        if meta_sha256 is None:
            raise ValueError("meta_url needs meta_sha256: the meta JSON holds the struck/clean "
                             "thresholds, so it is verified like the model (or pass meta={...})")
        meta_sha256 = _digest(meta_sha256, "meta_sha256")
    default_dir = cache_dir is None
    if cache_dir is None:
        cache_dir = pathlib.Path.home() / ".cache" / "pdf_strikethrough" / "models" / sha256[:12]
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "strike_verdict_cnn.onnx"
    meta_path = cache_dir / "strike_verdict_cnn.meta.json"
    record_path = cache_dir / _META_RECORD

    def _read_record():
        """(record, whether there is one); an unreadable record is one that vouches for nothing."""
        try:
            got, have = json.loads(record_path.read_text(encoding="utf-8")), True
        except FileNotFoundError:
            got, have = {}, False
        except (OSError, ValueError, RecursionError):
            got, have = {}, True
        return (got if isinstance(got, dict) else {}), have

    def _fetch_external(model_bytes, meta_bytes):
        """[(location, path, verified bytes or None, sha256)] for each external data file the
        model names, pinned by the meta's ``external_data``, every location checked before the
        first download and nothing written."""
        if urllib.parse.urlparse(url).query:
            raise ValueError("a model with external data files cannot be fetched from a URL with "
                             "a query string (a signed URL): the files' URLs are the model's with "
                             "the file name swapped, which drops it")
        listed = json.loads(meta_bytes.decode("utf-8"))
        listed = (listed.get("external_data") if isinstance(listed, dict) else None) or []
        if not isinstance(listed, list):
            raise ValueError("the meta's external_data must be a list of {location, sha256}")
        pinned = {}
        for e in listed:
            loc = e.get("location") if isinstance(e, dict) else None
            if not isinstance(loc, str):
                raise ValueError(f"the meta's external_data entry {e!r} has no location")
            digest = _digest(e.get("sha256"), f"external_data sha256 for {loc!r}")
            if pinned.setdefault(loc, digest) != digest:
                raise ValueError(f"the meta's external_data lists {loc!r} twice")
        locations = _external_data_locations(model_bytes)
        missing = [loc for loc in locations if loc not in pinned]
        if missing:
            raise ValueError(
                f"the model names external data files {missing} that the meta's external_data "
                "gives no sha256 for: list each file as get_model_meta() reports it for the model "
                "loaded from its folder, and remove from the model the tensors no node reads, "
                "whose files it leaves out")
        extra = sorted(set(pinned) - set(locations))
        if extra:                                  # the loader would refuse the folder
            raise ValueError(f"the meta's external_data lists {extra}, which the model does not "
                             "name: list only the model's own external data files")
        root, keys, folders, out = cache_dir.resolve(), set(), set(), []
        for loc in locations:
            parts = loc.split("/")
            names = [p.casefold() for p in parts]                 # Windows ignores case
            key, above = "/".join(names), {"/".join(names[:i]) for i in range(1, len(names))}
            dest = cache_dir.joinpath(*parts)
            # Refused: a part Windows cannot hold or would rename (a trailing dot or space, so
            # also "." and ".."), a device, a short name (STRIKE~1.JSO can be the meta), the
            # folder's own files, and two locations naming one file, or a file and its folder.
            if (any(c in _BAD_NAME_CHARS for c in loc)
                    or any(not p or p != p.rstrip(". ") or len(p) > 255 or re.search(r"~\d", p)
                           for p in parts)
                    or any(n.split(".")[0] in _DEVICE_NAMES for n in names)
                    or key in _OWN_FILES or key in keys or key in folders
                    or above & (_OWN_FILES | keys)
                    or not dest.resolve().is_relative_to(root)):
                raise ValueError(f"external data location {loc!r} is not a plain relative path "
                                 "to a file of its own inside the cache folder")
            keys.add(key)
            folders |= above
        for loc in locations:
            u = urllib.parse.urljoin(url, urllib.parse.quote(loc))
            dest = cache_dir.joinpath(*loc.split("/"))
            try:
                data = _fetch_verify(u, pinned[loc], dest)
            except OSError as e:
                if hasattr(e, "add_note"):                         # Python 3.11+
                    e.add_note(f"fetching the model's external data {loc!r} from {u}")
                raise
            out.append((loc, dest, data, pinned[loc]))
        return out

    if meta_url is None and meta is None:
        if not meta_path.exists():
            raise ValueError(
                "no model meta: pass meta_url=(+meta_sha256) or meta={...} (the ONNX loader needs "
                "strike_verdict_cnn.meta.json, with the thresholds and crop geometry, alongside "
                "the model)")
        # A shared cache_dir can hold another model's thresholds. Without a record, the meta
        # came with the model already there, or, in the default folder, with this model.
        record, have_record = _read_record()
        if have_record:
            ok = record.get(sha256) == _file_sha256(meta_path)
        else:
            ok = default_dir or (model_path.exists() and _file_sha256(model_path) == sha256)
        if not ok:
            raise ValueError(
                f"the meta cached in {cache_dir} was not verified for this model (sha256 "
                f"{sha256}): pass meta_url=(+meta_sha256) or meta={{...}}")
        model_data, meta_data, meta_digest = _fetch_verify(url, sha256, model_path), None, None
    else:
        model_data = _fetch_verify(url, sha256, model_path)
        if meta_url is not None:
            meta_data, meta_digest = _fetch_verify(meta_url, meta_sha256, meta_path), meta_sha256
        else:
            meta_data = json.dumps(meta).encode("utf-8")
            meta_digest = hashlib.sha256(meta_data).hexdigest()
            if meta_path.exists() and meta_path.read_bytes() == meta_data:
                meta_data = None
    model_bytes = model_path.read_bytes() if model_data is None else model_data
    external = []
    if b"location" in model_bytes and _external_data_locations(model_bytes):
        meta_bytes = meta_path.read_bytes() if meta_data is None else meta_data
        external = _fetch_external(model_bytes, meta_bytes)
        external += [("strike_verdict_cnn.onnx", model_path, model_data, sha256),
                     ("strike_verdict_cnn.meta.json", meta_path, meta_data,
                      hashlib.sha256(meta_bytes).hexdigest())]
    if meta_digest is not None:
        # The record is written first: if writing the model or the meta then fails, it already
        # names the meta this model needs. It is read just before it is written, so entries other
        # calls wrote by then are kept. One that cannot be written is skipped; a later call with
        # neither meta argument is then refused or falls back as above.
        record, _ = _read_record()
        if record.get(sha256) != meta_digest:
            try:
                record_path.write_text(json.dumps({**record, sha256: meta_digest}),
                                       encoding="utf-8")
            except OSError:
                pass
    if external:
        # The weights, then the model, then the meta, so a write that fails (a file another
        # process has open) leaves the old meta, whose pins the loader then finds do not match
        # the files beside it, or which still matches them. Then every file is hashed again: two
        # names for one file, or another process writing here, would leave other bytes.
        for loc, dest, data, digest in external:
            if data is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
        for loc, dest, data, digest in external:
            if not dest.is_file() or _file_sha256(dest) != digest:
                raise ValueError(f"{loc!r} does not read back as written: another location in "
                                 "the model names the same file here, or another process wrote "
                                 "to the folder")
    else:
        if model_data is not None:
            model_path.write_bytes(model_data)
        if meta_data is not None:
            meta_path.write_bytes(meta_data)
    set_model_dir(str(cache_dir))
    return str(cache_dir)


def word_crop_px(gray, bbox_frac, pad_x=PAD_X, pad_y=PAD_Y):
    """Grey page raster (0=black..255=white, uint8 or float; `std_crop` rescales [0,1] floats) +
    normalized word box (x0,y0,x1,y1 as PAGE FRACTIONS in [0,1], origin top-left) -> padded
    word crop (float32 HxW), or None if the box is too small. Raises ValueError when the box
    looks like pixel coordinates rather than fractions."""
    H, W = gray.shape
    wx0, wy0, wx1, wy1 = bbox_frac
    if max(abs(wx0), abs(wy0), abs(wx1), abs(wy1)) > 1.5:
        raise ValueError(
            f"bbox_frac must be normalized page fractions in [0,1], got {bbox_frac!r} "
            "(these look like pixel coordinates — divide by the image width/height)")
    x0, x1 = max(0, int(wx0 * W) - pad_x), min(W, int(wx1 * W) + pad_x)
    y0, y1 = max(0, int(wy0 * H) - pad_y), min(H, int(wy1 * H) + pad_y)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return gray[y0:y1, x0:x1].astype(np.float32)


def std_crop(crop):
    """Raw grey crop (0=black..255=white; [0,1] floats are rescaled, values clipped — no mod-256
    wraparound) -> (CROP_H, CROP_W) float32, ink-positive, height-normalized isotropically;
    width center-cropped/padded (a strike spans the word, so any window still shows it).
    A shaded ground (a grey highlight block) is flattened back to white first; ordinary paper is
    untouched."""
    crop = np.asarray(crop)
    if np.issubdtype(crop.dtype, np.floating):
        if crop.size and float(crop.max()) <= 1.0:
            crop = crop * 255.0
        crop = np.clip(crop, 0.0, 255.0)
    elif crop.dtype != np.uint8 and np.issubdtype(crop.dtype, np.integer):
        # rescale wide integer crops (16-bit and up); a bare .astype(uint8) below would wrap mod-256
        if np.iinfo(crop.dtype).max > 255:
            crop = crop.astype(np.float64) * (255.0 / np.iinfo(crop.dtype).max)
        crop = np.clip(crop, 0.0, 255.0)

    crop = np.asarray(crop, dtype=np.float32)
    light = crop[crop >= 128]
    if light.size:
        bg = float(np.bincount(light.astype(np.intp), minlength=256).argmax())
        if 120 <= bg < 240:                     # else ordinary paper, or too dark to be a highlight
            cut = SHADE_WHITE_FRAC * bg
            crop = np.where(crop >= cut, 255.0, crop * (255.0 / cut))
    ch, cw = crop.shape
    nw = max(12, int(round(cw * CROP_H / ch)))
    arr = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize((nw, CROP_H), Image.LANCZOS),
                     dtype=np.float32) / 255.0
    arr = 1.0 - arr
    if nw >= CROP_W:
        o = (nw - CROP_W) // 2
        return arr[:, o:o + CROP_W].copy()
    out = np.zeros((CROP_H, CROP_W), np.float32)
    out[:, (CROP_W - nw) // 2:(CROP_W - nw) // 2 + nw] = arr
    return out


def _build_torch_net():
    import torch.nn as nn

    class StrikeNet(nn.Module):
        def __init__(self):
            super().__init__()
            def blk(ci, co):
                return [nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                        nn.MaxPool2d(2)]
            self.net = nn.Sequential(*blk(1, 16), *blk(16, 32), *blk(32, 64),
                                     nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return StrikeNet()


def _check_geometry(meta):
    """The shipped meta records the crop/pad geometry the model was trained with. If a
    retrained model ships different values than the code constants, preprocessing would silently
    feed off-distribution crops — fail loudly instead."""
    for key, const in (("crop_h", CROP_H), ("crop_w", CROP_W), ("pad_x", PAD_X), ("pad_y", PAD_Y)):
        if key in meta and meta[key] != const:
            raise ValueError(
                f"model meta {key}={meta[key]} disagrees with the code constant {const}: the "
                "model was trained with different preprocessing geometry. Re-export the model or "
                "align the constants in cnn.py before using it.")


def _check_thresholds(meta):
    """Refuse a model whose thresholds are not ``0 <= p_lo <= p_hi <= 1``."""
    p_lo, p_hi = meta.get("p_lo"), meta.get("p_hi")
    ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (p_lo, p_hi))
    if not ok or not 0.0 <= p_lo <= p_hi <= 1.0:
        raise ValueError(f"model meta thresholds must satisfy 0 <= p_lo <= p_hi <= 1, got "
                         f"p_lo={p_lo!r}, p_hi={p_hi!r}")


# The ONNX messages that can hold a tensor, and the fields leading to them (onnx.proto): the walk
# that finds a model's external data. The package does not depend on `onnx`, so it reads them here.
_ONNX_CHILDREN = {
    "model": {7: "graph", 20: "training", 25: "function"},
    "training": {1: "graph", 2: "graph"},                        # TrainingInfoProto
    "function": {7: "node", 11: "attribute"},                    # nodes, attribute defaults
    "graph": {1: "node", 5: "tensor", 15: "sparse"},             # initializer, sparse_initializer
    "node": {5: "attribute"},
    "attribute": {5: "tensor", 6: "graph", 10: "tensor", 11: "graph", 22: "sparse", 23: "sparse"},
    "sparse": {1: "tensor", 2: "tensor"},                        # values, indices
    "tensor": {13: "entry"},                                     # external_data
}


def _pb_varint(buf, i, end):
    value = shift = 0
    while True:
        if i >= end or shift > 63:
            raise ValueError("malformed varint")
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, i
        shift += 7


def _pb_fields(buf, start, end):
    """(field number, wire type, value) for each field of the protobuf message in buf[start:end];
    a length-delimited value is its (start, end) span, a fixed-width one None."""
    i = start
    while i < end:
        key, i = _pb_varint(buf, i, end)
        key &= 0xFFFFFFFF                        # protobuf reads a tag as 32 bits
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _pb_varint(buf, i, end)
        elif wire == 2:
            n, i = _pb_varint(buf, i, end)
            value, i = (i, i + n), i + n
        elif wire in (1, 5):
            value, i = None, i + (8 if wire == 1 else 4)
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        if i > end:
            raise ValueError("truncated protobuf field")
        yield field, wire, value


def _external_data_locations(model_bytes):
    """The files an ONNX model's tensors keep their data in (each tensor's external_data
    ``location``), sorted; empty for a model that holds all its weights. Raises ValueError when
    the bytes are not a well-formed protobuf."""
    buf, found, todo = bytes(model_bytes), set(), [("model", 0, len(model_bytes))]
    while todo:                                  # a stack, so deep nesting can't overflow
        kind, start, end = todo.pop()
        for field, wire, span in _pb_fields(buf, start, end):
            child = _ONNX_CHILDREN.get(kind, {}).get(field)
            if child is None or wire != 2:
                continue
            if child != "entry":
                todo.append((child, *span))
                continue
            entry = {f: buf[s:e] for f, w, (s, e) in
                     ((f, w, v) for f, w, v in _pb_fields(buf, *span) if w == 2)}
            if entry.get(1) == b"location":
                try:
                    found.add(entry.get(2, b"").decode("utf-8"))
                except UnicodeDecodeError:       # no file has that name: a stale entry
                    pass
    return sorted(found)


def _file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pins(pins, onnx_path):
    """A meta's ``external_data`` as {location: sha256}; one that is not a list of
    ``{location, sha256}`` raises ``ValueError``."""
    try:
        return {e["location"]: e["sha256"].strip().lower() for e in pins}
    except (TypeError, KeyError, AttributeError):
        raise ValueError(f"{onnx_path}: its meta's external_data must be a list of "
                         "{location, sha256}") from None


def _load_model():
    """Try ONNX Runtime first, then a torch checkpoint. Returns (score_fn, meta)."""
    model_dir = _current_model_dir()
    onnx_path = os.path.join(model_dir, "strike_verdict_cnn.onnx")
    meta_path = os.path.join(model_dir, "strike_verdict_cnn.meta.json")
    pt_path = os.path.join(model_dir, "strike_verdict_cnn.pt")

    if os.path.exists(onnx_path) and os.path.exists(meta_path):
        import hashlib

        import onnxruntime as ort
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        _check_geometry(meta)
        _check_thresholds(meta)
        with open(onnx_path, "rb") as f:
            model_bytes = f.read()
        # Loaded by path, not from the bytes just read: ONNX Runtime finds a model's external data
        # beside the model only when it has the path (from bytes it looks in the working directory).
        # So a file replaced while it loads can leave the digests naming another model.
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        try:                                     # no b"location" in the bytes: no external data
            external = _external_data_locations(model_bytes) if b"location" in model_bytes else []
        except ValueError as e:
            raise ValueError(f"{onnx_path}: cannot read the model's external-data entries to "
                             f"compute its digest ({e})") from None

        def score(batch):
            logits = sess.run(None, {iname: batch})[0].reshape(-1)
            return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

        meta["runtime"] = "onnx"
        meta["sha256"] = hashlib.sha256(model_bytes).hexdigest()
        pins = meta.pop("external_data", None)   # computed here, never taken from the meta JSON
        # The session is built, so every file it read is a regular file beside the model. A
        # location that is not (a stale entry on a tensor nothing reads, a pipe) is skipped.
        paths = [(loc, os.path.join(model_dir, loc)) for loc in external]
        files = [{"location": loc, "sha256": _file_sha256(p)} for loc, p in paths
                 if os.path.isfile(p)]
        # A meta that pins a model's external data (ensure_model writes one) must match the files:
        # a folder whose weights and meta come from two models is refused, not scored.
        if files and pins is not None and _pins(pins, onnx_path) != {
                f["location"]: f["sha256"] for f in files}:
            raise ValueError(f"{onnx_path}: the external data files beside the model do not "
                             "match the sha256 its meta's external_data gives them (a folder "
                             "holding two models' files?)")
        if files:
            meta["external_data"] = files
        return score, meta

    if os.path.exists(pt_path):
        import torch
        # weights_only=True: never execute arbitrary pickle from a user-pointed model dir
        # (the checkpoint holds only tensors + scalar thresholds, which load fine under it).
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
        net = _build_torch_net()
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        meta = {"p_hi": float(ckpt["p_hi"]), "p_lo": float(ckpt["p_lo"]),
                "version": ckpt.get("version", "unknown"), "runtime": "torch",
                "sha256": _file_sha256(pt_path)}
        _check_geometry({k: ckpt[k] for k in ("crop_h", "crop_w", "pad_x", "pad_y") if k in ckpt})
        _check_thresholds(meta)

        def score(batch):
            with torch.no_grad():
                return torch.sigmoid(net(torch.from_numpy(batch))).numpy().astype(np.float64)

        return score, meta

    raise FileNotFoundError(
        f"no strike-verdict model in {model_dir} "
        "(need strike_verdict_cnn.onnx + .meta.json, or strike_verdict_cnn.pt with torch)")


def get_model_meta():
    """Model metadata: {version, p_hi, p_lo, runtime, sha256}. Loads the model on first call.
    ``sha256`` is the hex digest of the model file loaded (``strike_verdict_cnn.onnx``, or the
    ``.pt``), the digest :func:`ensure_model` verifies. An ONNX model that keeps weights in
    external data files also gets ``external_data``: one ``{location, sha256}`` per file, sorted by
    location. Together they tell retrained models apart where ``version`` may not: for a model
    with external data, compare ``external_data`` too, since its weights are not in the ``.onnx``.
    The digests are of the files as read while the model loads. Returns a copy."""
    import copy
    return copy.deepcopy(_ensure_loaded()[1])


def _loaded_meta():
    """A copy of the loaded model's meta, or None when no model is loaded. Never loads one."""
    import copy
    model = _model
    return copy.deepcopy(model[1]) if model is not None else None


def _ensure_loaded():
    """The loaded ``(score_fn, meta)``, loading it on first use. Callers use the returned tuple,
    never the global, which a concurrent ``set_model_dir`` / ``ensure_model`` may reset."""
    global _model
    model = _model
    if model is None:
        with _lock:
            if _model is None:
                _model = _load_model()
            model = _model
    return model


def score_crops(std_crops, batch_size=512):
    """Standardized crops (list/array of (CROP_H, CROP_W) float32) -> strike probabilities."""
    if not len(std_crops):
        return np.zeros(0)
    score, _ = _ensure_loaded()
    x = np.stack(std_crops).astype(np.float32)[:, None, :, :]
    return np.concatenate([score(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def score_word(gray, bbox_frac):
    """Convenience: page image + word box in [0,1] PAGE FRACTIONS -> strike probability (or None
    if the box is too small to crop). A color or wide-integer page goes through the detectors' own
    conversion (:func:`~pdf_strikethrough.lines.to_gray_u8`) first. Raises ValueError if the box
    looks like pixel coordinates."""
    gray = np.asarray(gray)
    if gray.ndim != 2 or (np.issubdtype(gray.dtype, np.integer) and gray.dtype != np.uint8):
        gray = to_gray_u8(gray)
    crop = word_crop_px(gray, bbox_frac)
    if crop is None:
        return None
    return float(score_crops([std_crop(crop)])[0])


def verdict_of(p, meta=None):
    """CNN probability -> 'struck' / 'clean' / 'unsure' using the checkpoint thresholds."""
    if meta is None:
        meta = _ensure_loaded()[1]
    return "struck" if p >= meta["p_hi"] else ("clean" if p <= meta["p_lo"] else "unsure")
