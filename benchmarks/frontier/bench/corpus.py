"""The G1 corpus: the public redline PDFs in `corpus_manifest.json`, each checked by sha256.

The files are not committed; `python -m bench fetch corpus` downloads them into `data/corpus/`.
The manifest is kept byte-identical to the package repo's `benchmarks/manifest.json`, so its own
`corpus_dir` is ignored here.
"""

import json
from collections.abc import Iterator
from pathlib import Path

from bench.common import DATA, ROOT, sha256_file

MANIFEST = json.loads((ROOT / "corpus_manifest.json").read_text(encoding="utf-8"))
CORPUS = DATA / "corpus"


def iter_corpus() -> Iterator[tuple[dict, Path]]:
    """Yield each manifest entry with its local file, after checking the file's sha256."""
    for entry in MANIFEST["pdfs"]:
        path = CORPUS / entry["file"]
        if not path.exists():
            raise FileNotFoundError(f"{path.name} is missing; run `python -m bench fetch corpus`")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{path.name} does not match its sha256 in corpus_manifest.json")
        yield entry, path
