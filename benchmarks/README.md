# Benchmarks

Makes the package's headline accuracy claims **falsifiable**: each script recomputes a claimed
number from a corpus of public redline PDFs, rather than asking you to trust a figure baked into a
docstring.

The PDFs are **not committed** (they're public but large and re-downloadable). `manifest.json`
lists each document with its source URL and a sha256; you download the files into
`corpus/` (git-ignored) and the loader verifies the hashes so a run is reproducible.

## Scripts

| Script | Reproduces | Needs |
|---|---|---|
| [`confirmation_rate.py`](confirmation_rate.py) | "99.8% of vector detections confirmed by the flag signal (92.5–100% per doc)" (README + `native.py`) | just the PDFs |
| [`scanned_recovery.py`](scanned_recovery.py) | the "Choosing an OCR backend" recovery table — "95–97% of the native strike set recovered by the scanned path" | PDFs + `scanned_pages`/`scanned_di_result` (from `prep_scanned_di.py`) + `[rapidocr]` |
| [`prep_scanned_di.py`](prep_scanned_di.py) | *(one-time asset generator for the above)* rasterizes struck pages, runs Azure DI once, caches the result | an Azure DI key in the repo `.env` |
| [`ocr_backend_table.py`](ocr_backend_table.py) | *(legacy)* the OCR-backend table against a **scanned** corpus with DI references | a scanned corpus + per-doc DI result + `[rapidocr,tesseract]` |
| [`di_parity.py`](di_parity.py) | *(legacy)* "1477 vs 1484 (99.5% parity)" against the **original** Azure-DI pipeline | a scanned corpus + per-doc DI result + the original pipeline's reference count |

`confirmation_rate.py` needs only the PDFs and no cloud access — start there. `scanned_recovery.py`
is the reproducible scanned-path benchmark on this (born-digital) corpus: it rasterizes the redline
pages into image-only "scans" and scores recovery against the native detector's known strikes.
`ocr_backend_table.py` / `di_parity.py` are the older scanned-corpus scripts — kept for anyone with
a genuinely scanned corpus and (for parity) the original pipeline's recorded counts, which
`scanned_recovery.py` no longer needs.

## Manifest schema

`manifest.json`:

```json
{
  "corpus_dir": "corpus",
  "pdfs": [
    {
      "name": "US CFR Title 12 redline (2023)",
      "file": "cfr-title12-redline.pdf",
      "url": "https://example.gov/.../cfr-title12-redline.pdf",
      "sha256": "<64-hex sha256 of the downloaded file>",

      "di_result": "cfr-title12.di.json",   /* optional: Azure DI analyze-result JSON in corpus/  */
      "di_reference_struck": 1484            /* optional: struck count from the original DI pipeline */
    }
  ]
}
```

- `name`, `file`, `url`, `sha256` — required for every entry. `file` is resolved under `corpus_dir`.
- `di_result` — optional path (under `corpus_dir`) to that document's Azure DI analyze-result JSON;
  required only for `ocr_backend_table.py` and `di_parity.py`.
- `di_reference_struck` — optional; the struck-word count the original Azure-DI pipeline produced,
  used only by `di_parity.py`.

Compute a file's hash with `python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" corpus/xyz.pdf`.

## Running

```bash
pip install -e ".[dev,rapidocr,tesseract]"   # from the repo root
# populate manifest.json and drop the PDFs into benchmarks/corpus/
cd benchmarks
python confirmation_rate.py
```

Each script errors clearly if the manifest is empty, a file is missing (printing its download URL),
or a sha256 doesn't match — so a stale or drifted corpus fails loudly instead of quietly changing
the numbers.
