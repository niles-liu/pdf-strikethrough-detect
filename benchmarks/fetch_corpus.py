"""Download the benchmark corpus listed in manifest.json into corpus/, verifying each sha256.

    python benchmarks/fetch_corpus.py

Reads every manifest entry's ``url`` and ``sha256`` (and, on an entry with a cached Azure DI result,
``scanned_di_url`` and ``scanned_di_sha256``), downloads any file not already present (or whose
hash doesn't match), and verifies the digest after download so a drifted source fails loudly.
Files already present with a matching hash are skipped. This is the "collect the corpus" step for
the benchmarks (`confirmation_rate.py` et al.) and the model/calibration program — populate
`manifest.json` (name/file/url/sha256 per entry; see benchmarks/README.md) and run this once.

A file with no URL in the manifest (one you must obtain by hand) is reported and skipped rather
than treated as an error.
"""
from __future__ import annotations

import sys
import urllib.parse
import urllib.request
from collections import Counter

from _corpus import _sha256, corpus_dir, load_manifest

# Some publishers (copyright.gov, gretnala.com) answer urllib's default User-Agent with a 403.
USER_AGENT = ("pdf-strikethrough-detect-benchmarks "
              "(+https://github.com/niles-liu/pdf-strikethrough-detect)")


def _download(url, dest, timeout=60):
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https", "file"):
        raise ValueError(f"unsupported URL scheme {scheme!r} for {url} (use http/https/file)")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    dest.write_bytes(data)


def _files(entry):
    """The files a manifest entry lists, as ``(file, url, sha256)``: the PDF, then its cached Azure
    DI result if it has one."""
    yield entry["file"], entry.get("url"), entry.get("sha256")
    if entry.get("scanned_di_result"):
        yield (entry["scanned_di_result"], entry.get("scanned_di_url"),
               entry.get("scanned_di_sha256"))


def _fetch(name, url, want, dest):
    """Bring ``dest`` up to date with the manifest: 'ok', 'fetched', 'manual' or 'failed'."""
    if dest.exists() and want and _sha256(dest) == want:
        print(f"ok      {name} (present, hash matches)")
        return "ok"
    if not url:
        print(f"MANUAL  {name} — no url in manifest; obtain it by hand")
        return "manual"
    try:
        print(f"fetch   {name} <- {url}")
        _download(url, dest)
    except Exception as e:                           # noqa: BLE001 - report and continue the batch
        print(f"FAILED  {name}: {e}", file=sys.stderr)
        return "failed"
    got = _sha256(dest)
    if want and got != want:
        print(f"FAILED  {name}: sha256 {got} != manifest {want} "
              f"(source may have changed; update the manifest deliberately)", file=sys.stderr)
        dest.unlink(missing_ok=True)
        return "failed"
    if not want:
        print(f"        {name}: no sha256 in manifest; downloaded hash is {got}")
    return "fetched"


def main() -> None:
    manifest = load_manifest()
    cdir = corpus_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    pdfs = manifest.get("pdfs", [])
    if not pdfs:
        sys.exit("manifest.json lists no PDFs — populate `pdfs` first (see benchmarks/README.md).")

    counts = Counter(_fetch(name, url, want, cdir / name)
                     for entry in pdfs for name, url, want in _files(entry))
    print(f"\n{counts['fetched']} fetched, {counts['ok']} already present, "
          f"{counts['manual']} manual, {counts['failed']} failed -> {cdir}")
    if counts["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
