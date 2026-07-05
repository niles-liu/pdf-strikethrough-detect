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
| [`confirmation_rate.py`](confirmation_rate.py) | "99.9–100% of vector detections confirmed by the flag signal" (README + `native.py`) | just the PDFs |
| [`ocr_backend_table.py`](ocr_backend_table.py) | the "Choosing an OCR backend" coverage/agreement table | PDFs + per-doc DI reference + `[rapidocr,tesseract]` |
| [`di_parity.py`](di_parity.py) | "1477 vs 1484 struck words (99.5% parity)" with the original Azure-DI pipeline | PDFs + per-doc DI result + reference count |

`confirmation_rate.py` needs only the PDFs and no cloud access — start there.

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
