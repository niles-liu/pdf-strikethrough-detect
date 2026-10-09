# Benchmarks

Makes the package's headline accuracy claims **falsifiable**: each script recomputes a claimed
number from a corpus of public redline PDFs, rather than asking you to trust a figure baked into a
docstring.

The PDFs are **not committed** (they're public but large and re-downloadable). `manifest.json`
lists each document with its source URL and a sha256; [`fetch_corpus.py`](fetch_corpus.py)
downloads the files into `corpus/` (git-ignored) and the loader verifies the hashes so a run is
reproducible. The cached Azure DI results that `scanned_recovery.py` reads are hosted in the
[`niles-liu/strikethrough-benchmark`](https://huggingface.co/datasets/niles-liu/strikethrough-benchmark)
dataset and fetched and verified the same way, so that benchmark needs no Azure account.

## Scripts

| Script | Reproduces | Needs |
|---|---|---|
| [`fetch_corpus.py`](fetch_corpus.py) | *(setup)* downloads every file the manifest lists into `corpus/` and verifies its sha256 | network access |
| [`confirmation_rate.py`](confirmation_rate.py) | "99.8% of vector detections confirmed by the flag signal (99.6–100% per doc)" (README + `native.py`) | just the PDFs |
| [`scanned_recovery.py`](scanned_recovery.py) | the "Choosing an OCR backend" recovery table: RapidOCR, and Azure DI under its default calibration and `confidence_free()` | the PDFs + the cached DI results + `[rapidocr]` |
| [`prep_scanned_di.py`](prep_scanned_di.py) | *(one-time asset generator for the above)* rasterizes struck pages, runs Azure DI once, caches the result | an Azure DI key in the repo `.env` |
| [`ocr_backend_table.py`](ocr_backend_table.py) | *(legacy)* the OCR-backend table against a **scanned** corpus with DI references | a scanned corpus + per-doc DI result + `[rapidocr,tesseract]` |
| [`di_parity.py`](di_parity.py) | *(legacy)* "1477 vs 1484 (99.5% parity)" against the **original** Azure-DI pipeline | a scanned corpus + per-doc DI result + the original pipeline's reference count |
| [`confidence_veto.py`](confidence_veto.py) | the ruled-forms precision numbers (issues #4, #7) — `--ab` scores the printed-rule veto, `--switches` scores all four combinations of the two provisional switches | a **private** ruled-form corpus + per-doc `di-result.json` + a `ground-truth.json` label set |

`confirmation_rate.py` needs only the PDFs and no cloud access — start there. `scanned_recovery.py`
is the reproducible scanned-path benchmark on this (born-digital) corpus: it rasterizes the redline
pages into image-only "scans" and scores recovery against the native detector's known strikes.
`ocr_backend_table.py` / `di_parity.py` are the older scanned-corpus scripts — kept for anyone with
a genuinely scanned corpus and (for parity) the original pipeline's recorded counts, which
`scanned_recovery.py` no longer needs.

`confidence_veto.py` is the odd one out: it scores **precision on degraded ruled forms**, and the
corpus it needs is private (real ruled-form paperwork), so it is not reproducible from this
repo alone — point it at your own labeled set with `PDF_STRIKETHROUGH_CORPUS_DIR`. It is still the
script that produces every ruled-forms figure quoted in the README and CHANGELOG, and it scores
against a label set rather than counting detections, because **a struck-final count is not a
false-positive count** — it includes the real strikes.

[`frontier/`](frontier/) holds the plan for a benchmark against frontier models: can they do this
package's job, finding struck text and keeping it out of the live text? Nothing there runs yet, and
it has its own pinned environment; read [`frontier/PLAN.md`](frontier/PLAN.md), then the draft
pre-registration, [`frontier/PREREG.md`](frontier/PREREG.md).

## Manifest schema

`manifest.json`:

```json
{
  "corpus_dir": "corpus",
  "pdfs": [
    {
      "name": "FDIC Fair Lending laws & regulations (redlined changes)",
      "file": "fdic-fair-lending-redline.pdf",
      "url": "https://www.fdic.gov/redlined-document-identifying-changes.pdf",
      "sha256": "<64-hex sha256 of the downloaded file>",

      "scanned_pages": [0, 1, 2, 23],                       /* optional: the scanned-recovery set */
      "scanned_pdf": "fdic-fair-lending-redline.scanned.pdf",
      "scanned_di_result": "fdic-fair-lending-redline.scanned.di.json",
      "scanned_di_url": "https://huggingface.co/datasets/.../fdic-fair-lending-redline.scanned.di.json",
      "scanned_di_sha256": "<64-hex sha256 of the DI result>"
    }
  ]
}
```

- `name`, `file`, `url`, `sha256` — required for every entry. `file` is resolved under `corpus_dir`.
- `scanned_pages` — optional; the original page indices `scanned_recovery.py` rasterizes and
  scores, which are the pages the cached DI result was captured on. Entries without it are not part
  of that benchmark.
- `scanned_pdf` — the rasterized pages `prep_scanned_di.py` sent to Azure DI, kept under
  `corpus_dir` for reference. The benchmark rebuilds the same raster in memory and does not read it.
- `scanned_di_result`, `scanned_di_url`, `scanned_di_sha256` — the cached Azure DI analyze-result
  JSON for those pages, where to download it, and its digest. `fetch_corpus.py` downloads and
  verifies it like a PDF, and `scanned_recovery.py` checks the digest before reading it.
  `prep_scanned_di.py` writes all three except the URL, which is set once the file is uploaded.
- `di_result`, `di_reference_struck` — the per-document DI result and original-pipeline struck
  count the legacy `ocr_backend_table.py` and `di_parity.py` read. No current entry carries them.

Compute a file's hash with `python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" corpus/xyz.pdf`.

## Running

```bash
pip install -e ".[dev,rapidocr]"             # from the repo root
python benchmarks/fetch_corpus.py            # the PDFs + cached DI results, sha256-verified
cd benchmarks
python confirmation_rate.py
python scanned_recovery.py
```

Each script errors clearly if the manifest is empty, a file is missing (printing its download URL),
or a sha256 doesn't match — so a stale or drifted corpus fails loudly instead of quietly changing
the numbers.
