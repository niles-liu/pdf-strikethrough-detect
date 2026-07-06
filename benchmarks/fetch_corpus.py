"""Download the benchmark corpus listed in manifest.json into corpus/, verifying each sha256.

    python benchmarks/fetch_corpus.py

Reads every manifest entry's ``url`` and ``sha256``, downloads any file not already present (or
whose hash doesn't match), and verifies the digest after download so a drifted source fails loudly.
Files already present with a matching hash are skipped. This is the "collect the corpus" step for
the benchmarks (`confirmation_rate.py` et al.) and the model/calibration program — populate
`manifest.json` (name/file/url/sha256 per entry; see benchmarks/README.md) and run this once.

An entry with no ``url`` (a file you must obtain manually — e.g. a per-doc Azure DI result) is
reported and skipped rather than treated as an error.
"""
from __future__ import annotations

import sys
import urllib.parse
import urllib.request

from _corpus import _sha256, corpus_dir, load_manifest


def _download(url, dest, timeout=60):
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https", "file"):
        raise ValueError(f"unsupported URL scheme {scheme!r} for {url} (use http/https/file)")
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = r.read()
    dest.write_bytes(data)


def main() -> None:
    manifest = load_manifest()
    cdir = corpus_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    pdfs = manifest.get("pdfs", [])
    if not pdfs:
        sys.exit("manifest.json lists no PDFs — populate `pdfs` first (see benchmarks/README.md).")

    fetched = skipped = manual = failed = 0
    for entry in pdfs:
        dest = cdir / entry["file"]
        want = entry.get("sha256")
        if dest.exists() and want and _sha256(dest) == want:
            print(f"ok      {entry['file']} (present, hash matches)")
            skipped += 1
            continue
        url = entry.get("url")
        if not url:
            print(f"MANUAL  {entry['file']} — no url in manifest; obtain it by hand")
            manual += 1
            continue
        try:
            print(f"fetch   {entry['file']} <- {url}")
            _download(url, dest)
        except Exception as e:                       # noqa: BLE001 - report and continue the batch
            print(f"FAILED  {entry['file']}: {e}", file=sys.stderr)
            failed += 1
            continue
        got = _sha256(dest)
        if want and got != want:
            print(f"FAILED  {entry['file']}: sha256 {got} != manifest {want} "
                  f"(source may have changed; update the manifest deliberately)", file=sys.stderr)
            dest.unlink(missing_ok=True)
            failed += 1
        else:
            if not want:
                print(f"        {entry['file']}: no sha256 in manifest; downloaded hash is {got}")
            fetched += 1

    print(f"\n{fetched} fetched, {skipped} already present, {manual} manual, {failed} failed "
          f"-> {cdir}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
