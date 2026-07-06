"""One-time asset generator for the scanned-recovery benchmark (needs an Azure DI key).

For each configured document it rasterizes the most-struck born-digital pages into an image-only
PDF (a genuine "scan"), sends that PDF to Azure Document Intelligence (prebuilt-layout) once, and
caches both the image PDF and the DI analyze-result JSON under benchmarks/corpus/ (git-ignored).
It then records ``scanned_pages`` / ``scanned_pdf`` / ``scanned_di_result`` on the manifest entry so
``scanned_recovery.py`` can reproduce the numbers offline (RapidOCR runs live; DI is read from the
cached JSON).

    # credentials in the repo .env: AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT / _KEY / _API_VERSION
    python benchmarks/prep_scanned_di.py

This is the only script that calls the cloud, and only for the handful of pages listed in CONFIG.
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.request

from _corpus import MANIFEST, corpus_dir, load_manifest
from _scanned import build_scanned_pdf, struck_pages

# (manifest `file`, number of most-struck pages to rasterize) — kept small; DI bills per page.
CONFIG = [
    ("copyright-201-10-redline.pdf", 4),
    ("gretna-udo-edits.pdf", 8),
    ("fdic-fair-lending-redline.pdf", 12),
]


def _load_env(repo_root):
    env = {}
    p = repo_root / ".env"
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _analyze_layout(pdf_bytes, endpoint, key, api_version):
    """Run prebuilt-layout on ``pdf_bytes`` and return the full analyze-result JSON (with the
    ``analyzeResult`` envelope, which detect_pdf's _di_pages accepts)."""
    base = endpoint.rstrip("/")
    url = f"{base}/documentintelligence/documentModels/prebuilt-layout:analyze?api-version={api_version}"
    req = urllib.request.Request(
        url, data=pdf_bytes, method="POST",
        headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/pdf"})
    with urllib.request.urlopen(req, timeout=120) as r:
        op = r.headers.get("Operation-Location")
    if not op:
        raise RuntimeError("DI accepted the document but returned no Operation-Location to poll")
    for _ in range(90):                                   # up to ~3 min at 2 s
        time.sleep(2)
        poll = urllib.request.Request(op, headers={"Ocp-Apim-Subscription-Key": key})
        with urllib.request.urlopen(poll, timeout=60) as r:
            body = json.loads(r.read())
        status = body.get("status")
        if status == "succeeded":
            return body
        if status == "failed":
            raise RuntimeError(f"DI analyze failed: {body.get('error')}")
    raise TimeoutError("DI analyze did not finish within the poll budget")


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    env = _load_env(repo_root)
    endpoint = env.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = env.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    api_version = env.get("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION", "2024-11-30")
    if not endpoint or not key:
        raise SystemExit("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT / _KEY not found in .env")

    manifest = load_manifest()
    cdir = corpus_dir()
    entries = {e["file"]: e for e in manifest["pdfs"]}
    total_pages = 0
    for fname, top_k in CONFIG:
        entry = entries.get(fname)
        if entry is None:
            print(f"SKIP  {fname}: not in manifest")
            continue
        orig = cdir / fname
        page_indices, _gt = struck_pages(orig, top_k)
        pdf_bytes = build_scanned_pdf(orig, page_indices)
        scanned_pdf = f"{orig.stem}.scanned.pdf"
        di_json = f"{orig.stem}.scanned.di.json"
        (cdir / scanned_pdf).write_bytes(pdf_bytes)
        print(f"analyze {fname}: {len(page_indices)} page(s) -> DI ...", flush=True)
        result = _analyze_layout(pdf_bytes, endpoint, key, api_version)
        (cdir / di_json).write_text(json.dumps(result), encoding="utf-8")
        n_words = sum(len(p.get("words", [])) for p in result["analyzeResult"]["pages"])
        entry["scanned_pages"] = page_indices
        entry["scanned_pdf"] = scanned_pdf
        entry["scanned_di_result"] = di_json
        total_pages += len(page_indices)
        print(f"  ok: {len(page_indices)} page(s), {n_words} DI words -> {di_json}")

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\n{total_pages} page(s) analyzed; manifest updated with scanned_* fields.")


if __name__ == "__main__":
    main()
