"""Shared corpus loader for the benchmark scripts.

The corpus itself is NOT committed (public PDFs, but large and re-downloadable). `manifest.json`
lists each document with its source URL and a sha256; the actual files live under
``benchmarks/corpus/`` (git-ignored). This module loads the manifest, resolves each file, and
verifies its sha256 so a benchmark run is reproducible and can't silently drift when a source
document changes upstream.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path=MANIFEST) -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def corpus_dir(manifest_path=MANIFEST) -> pathlib.Path:
    """The directory holding the corpus files, resolved relative to the manifest's location."""
    manifest_path = pathlib.Path(manifest_path)
    base = manifest_path.resolve().parent
    return (base / load_manifest(manifest_path).get("corpus_dir", "corpus")).resolve()


def iter_corpus(manifest_path=MANIFEST, verify=True):
    """Yield ``(entry, path)`` for each PDF in the manifest. `entry` is the manifest dict (name,
    url, sha256, file, ...); `path` is the resolved local file. Raises as :func:`check_file`
    does on a missing file or a sha256 mismatch."""
    manifest_path = pathlib.Path(manifest_path)
    manifest = load_manifest(manifest_path)
    cdir = corpus_dir(manifest_path)
    pdfs = manifest.get("pdfs", [])
    if not pdfs:
        raise SystemExit(
            f"no PDFs in {manifest_path.name}. Populate `pdfs` (see benchmarks/README.md for the "
            f"entry schema) and drop the files under {cdir}/ before running a benchmark.")
    for entry in pdfs:
        path = cdir / entry["file"]
        check_file(path, entry.get("sha256") if verify else None, entry.get("url"))
        yield entry, path


def check_file(path, sha256=None, url=None):
    """Raise FileNotFoundError if the corpus file `path` is missing (naming where to get it) and
    ValueError if it does not hash to `sha256`. No digest, no hash check."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found under {path.parent}/ — run `python benchmarks/fetch_corpus.py` "
            f"or download it from {url or '<no url in manifest>'}")
    if sha256:
        got = _sha256(path)
        if got != sha256:
            raise ValueError(
                f"sha256 mismatch for {path.name}: manifest {sha256} != {got} "
                f"(the source may have changed; update the manifest deliberately)")
