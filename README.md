# pdf-strikethrough-detect

[![PyPI](https://img.shields.io/pypi/v/pdf-strikethrough-detect.svg)](https://pypi.org/project/pdf-strikethrough-detect/)
[![Python versions](https://img.shields.io/pypi/pyversions/pdf-strikethrough-detect.svg)](https://pypi.org/project/pdf-strikethrough-detect/)
[![CI](https://github.com/niles-liu/pdf-strikethrough-detect/actions/workflows/ci.yml/badge.svg)](https://github.com/niles-liu/pdf-strikethrough-detect/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/LICENSE)

Detect **struck-through (deleted) text** in PDFs and scanned document images.

Strikethrough detection is a surprisingly unserved niche: most "redline"/diff tools assume clean
born-digital PDFs and fall apart on scans — which is the real-world case. `pdf-strikethrough-detect`
handles both, does the hard part (scanned images) with a tiny CPU model, and makes no cloud calls.

```python
import pdf_strikethrough as st

# born-digital PDF — exact, no OCR
for w in st.strikethroughs_in_pdf("contract.pdf"):
    print(w["page"], repr(w["chars"]), "partial" if w["partial"] else "full")
```

## Install

```bash
pip install pdf-strikethrough-detect
```

Pure pip, no system binaries required: the CNN runs on ONNX Runtime (CPU) and the ~318 KB model
ships inside the wheel. Extras:

```bash
pip install "pdf-strikethrough-detect[markdown]"    # clean_markdown() via pymupdf4llm
pip install "pdf-strikethrough-detect[rapidocr]"    # free scanned-word OCR backend (no binary)
pip install "pdf-strikethrough-detect[tesseract]"   # word-level OCR (also needs the tesseract binary)
pip install "pdf-strikethrough-detect[torch]"       # optional .pt CNN fallback for dev/retraining
```

The CNN ships as ONNX and needs nothing extra. `[torch]` is only for development — the loader
prefers `strike_verdict_cnn.onnx`, falling back to a `strike_verdict_cnn.pt` checkpoint if you
point `PDF_STRIKETHROUGH_MODEL_DIR` at one. Regenerate the shipped ONNX from a trained checkpoint
with [`tools/export_model.py`](tools/export_model.py).

## Native / born-digital PDFs — exact

In a born-digital PDF a strikethrough is a *vector drawing* (a line or thin rect over the text),
so detection is exact ground truth — no OCR, no model, no guessing. Both vector-rule and
filled-rect strikethrough styles are handled.

```python
import pdf_strikethrough as st

for w in st.strikethroughs_in_pdf("contract.pdf"):
    print(w["page"], repr(w["chars"]))          # 'chars' = the struck substring
print(st.clean_markdown("contract.pdf"))        # surviving text, deletions removed (needs [markdown])
```

Each native record: `{page, text, chars, char_span, partial, bbox_frac, tier, coverage, verdict,
final}` (`tier` is `"vector"`, `"flag"`, or `"annot"`). Vector records also carry
`stroke_color`/`stroke_width` (the paint + thickness of the dominant stroke — red = opposing
counsel is evidence); annotation records carry `annot_author`/`annot_created`/`annot_modified`/
`annot_color`/`annot_id` (the redline's "who and when"). Scanned records replace `coverage` with
the evidence that decided them — `score`, `cnn_prob`, `cnn_agrees`, `conf` — and `tier` is
`"auto"`/`"review"`. The keys are documented as `TypedDict`s in `pdf_strikethrough.types`
(`StruckWord`, `DetectResult`, `Passage`); the package ships `py.typed`, so type checkers see them.
Partial strikes (`semi-` of `semi-monthly`) are resolved to a char range. `bbox_frac` is in
fractions of the rendered page (rotation-aware), so it maps directly onto a rendered pixmap.

Three native detectors, all **base-PyMuPDF only** (no pymupdf4llm), selected by `method`:

```python
st.strikethroughs_in_pdf("contract.pdf", method="vector")  # stroke geometry (default) —
                                                           #   precise partial-char spans + color/width
st.strikethroughs_in_pdf("contract.pdf", method="flag")    # MuPDF's FZ_STEXT_STRIKEOUT signal —
                                                           #   also catches font-attribute strikes
st.strikethroughs_in_pdf("contract.pdf", method="annot")   # explicit /StrikeOut annotations —
                                                           #   Acrobat/Preview redlines + forensics
st.strikethroughs_in_pdf("contract.pdf", method="both")    # union of all three — maximum recall
```

**Validated across domains.** On 12 public redline PDFs (federal & state regulations, court
rules, procurement clauses, municipal codes, university policy; 33k struck words),
**99.9–100% of vector detections are independently confirmed by MuPDF's strikeout signal**,
and the flag method adds ~2% more words (font-attribute strikes and edge cases) — use
`method="both"` to capture them. `pymupdf4llm` is not used for detection at all; it is only an
optional `[markdown]` extra for richer layout in `clean_markdown()`.

## Any PDF — routed per page, scanned pages use OCR + CNN

```python
import pdf_strikethrough as st
from pdf_strikethrough.ocr import rapidocr_backend
from pdf_strikethrough.scanned import ScanConfig

res = st.detect_pdf("mixed.pdf",
                    ocr=rapidocr_backend(),                 # for scanned pages
                    scan_config=ScanConfig.confidence_free())

struck   = [w for w in res["words"] if w["final"]]          # struck words (boxes, char spans)
markdown = res["markdown"]                                  # deletions as ~~struck~~
clean    = res["clean_text"]                                # surviving text, deletions removed
passages = res["passages"]                                  # grouped deletion sections
```

`detect_pdf` classifies each page native-vs-scanned, runs the exact path on native pages
(`method="vector"|"flag"|"annot"|"both"`) and the geometry→OCR→CNN pipeline on scanned ones, and
assembles `markdown` / `clean_text` / `passages` for **both** page kinds from its own strike
decisions (so the text and the word records always agree — no dependence on an external markdown
engine). Already have an Azure Document Intelligence result? Pass `di_result=...` (the REST JSON
dict, an `{'analyzeResult': ...}` envelope, or `sdk_result.as_dict()`) to skip re-OCR and use
DI's word boxes.

Scanned pages with no OCR backend raise `OcrRequiredError` by default; pass
`on_missing_ocr="skip"` to skip them (with a warning in `res["warnings"]`) and still get
everything from the native pages. Password-protected PDFs raise `EncryptedPdfError`.

> `clean_markdown()` remains a separate, higher-fidelity **native-only** path that borrows
> pymupdf4llm's layout (headings, paragraphs). `detect_pdf`'s `markdown` is layout-plain but works
> uniformly on scanned pages too.

### Choosing an OCR backend

The geometry + CNN carry the detection and are **OCR-independent**; OCR only supplies word boxes
to attribute strikes to, plus a confidence prior. Benchmarked on a heavily-edited document
(Azure DI as reference):

| Backend | Setup | Struck **regions** | Spatial agreement | Word granularity |
|---|---|---|---|---|
| Azure Document Intelligence | cloud, paid | reference | — | exact word boxes |
| **RapidOCR** | `pip`, no binary | **100% covered** | **~99%** | ~4× coarser (phrase-level) |
| Tesseract | needs system binary | — | — | genuine word-level |

Use `ScanConfig.confidence_free()` with RapidOCR (its confidences cluster near 1.0 and don't
separate struck from clean text); the default `ScanConfig()` is calibrated to Azure DI, whose
struck words drop to 0.43–0.94. The DI-decoupled classifier reproduces the original Azure-DI
pipeline to **99.5%** (1477 vs 1484 struck words on the validation doc).

## Low-level building blocks

```python
gray = st.render_page_gray(doc[0], dpi=200)     # always returns a HxW uint8 grayscale array

lines = st.strike_lines(gray, dpi=200)          # OCR-free stroke geometry (strike/underline/rule)
                                                # pass the dpi the image was rendered/scanned at

# word boxes are PAGE FRACTIONS in [0,1], origin top-left — not pixels
p = st.score_word(gray, (0.12, 0.34, 0.38, 0.36))   # CNN strike probability (0..1)

from pdf_strikethrough.ocr import Word
# detect_scanned_image accepts your own image; RGB / float arrays are coerced to grayscale uint8
recs = st.detect_scanned_image(gray, [Word("foo", (0.12, 0.34, 0.38, 0.36), 0.6)])

# visual overlay — render each struck page with the strikes boxed (red=full, orange=partial)
for pg in st.render_overlay("contract.pdf", dpi=150):
    pg["image"].save(f"overlay-p{pg['page']}.png")     # or st.save_overlays(src, "out/")
```

**Logging.** Diagnostics (page routing, native method + record counts, OCR/CNN timings) are
emitted at `DEBUG` under the `pdf_strikethrough` logger — silent by default (a `NullHandler` is
attached). Opt in with `logging.getLogger("pdf_strikethrough").setLevel(logging.DEBUG)` and a
handler; `warnings` stay reserved for caller-facing hazards.

## CLI

```bash
pdf-strikethrough detect contract.pdf                       # native pages (scanned pages are
                                                            #   skipped with a warning)
pdf-strikethrough detect scan.pdf --ocr rapidocr            # include scanned pages
pdf-strikethrough detect scan.pdf --di-result di.json       # use a pre-fetched Azure DI result
pdf-strikethrough detect doc.pdf --method both              # max-recall native detection
pdf-strikethrough detect doc.pdf --method annot             # explicit /StrikeOut annotations only
pdf-strikethrough detect doc.pdf --pages 1-5,12             # only these pages (1-based)
pdf-strikethrough detect doc.pdf --json out.json            # full struck words + passages + evidence
pdf-strikethrough detect doc.pdf --json -                   # ...or stream JSON to stdout
pdf-strikethrough detect doc.pdf --clean-text clean.txt     # surviving text, deletions removed
pdf-strikethrough detect doc.pdf --markdown marked.md       # deletions as ~~struck~~
pdf-strikethrough detect doc.pdf --overlay out/             # page images with strikes boxed
pdf-strikethrough detect doc.pdf --fail-if-found            # exit 3 if any strike found (CI gate)
cat doc.pdf | pdf-strikethrough detect -                    # read the PDF from stdin
pdf-strikethrough --version
```

Other flags: `--dpi` (raster DPI for scanned pages, default 200), `--overlay-dpi` (overlay render
DPI, default 150), `--limit` (max words in plain output), `--scan-config
auto|azure-di|confidence-free`. Exit codes: `0` ok, `1` usage/file error,
`2` encrypted / OCR required, `3` `--fail-if-found` matched. Full help: `pdf-strikethrough detect -h`.

## Examples

Runnable, dependency-light scripts are in [`examples/`](examples/) — they generate their own
sample PDFs (no assets to download) so every snippet is copy-paste runnable:

```bash
python examples/native_quickstart.py     # build a redline PDF, detect strikes, print clean text
python examples/scanned_quickstart.py    # rasterize it to a "scan", run OCR + CNN  ([rapidocr])
python examples/overlay_quickstart.py     # print strike evidence + write a before/after overlay
```

## How it works

- **Native**: merged horizontal vector strokes through a word's middle band (excludes under/over-
  lines); coverage ≥ 50% → struck, partials resolved to a char range.
- **Scanned geometry** (`lines.py`): per-angle morphological opening extracts stroke *fragments*,
  collinear fragments are stitched, then strict filters (spine fill, stroke run-thickness,
  angle/length) separate real strikes from bold crossbars and serif-glyph chains.
- **CNN** (`cnn.py`, StrikeNet, 79k params): resolves pixel-ambiguous cases — a thin strike over
  an ascender-less word is pixel-identical to a glyph chain, and only a learned model tells them
  apart. Ships as ONNX; set `PDF_STRIKETHROUGH_MODEL_DIR` to use your own weights.
- **Attribution** (`scanned.py`): assigns strokes to OCR words with char spans and full/partial
  resolution, plus a visual-row "orphan" pass for words the detector's stroke evidence missed.

## API surface

The top-level package (`import pdf_strikethrough as st`) exports ~30 names; the most-used:

| Group | Names |
|---|---|
| High-level | `strikethroughs_in_pdf`, `clean_markdown`, `detect_pdf`, `detect_scanned_image`, `open_pdf`, `render_page_gray`, `render_overlay`, `save_overlays` |
| Native detectors | `native_page_strikes`, `native_flag_strikes`, `native_annot_strikes`, `native_doc_strikes`, `page_strikes`, `native_markdown`, `strip_struck_markdown` |
| Scanned geometry + classifier | `strike_lines`, `ink_mask`, `to_gray_u8`, `analyze_scanned_page`, `ScanConfig`, `classify_page_source`, `apply_cnn_verdict` |
| CNN | `score_word`, `score_crops`, `std_crop`, `word_crop_px`, `verdict_of`, `get_model_meta` |
| OCR | `Word`, `rapidocr_backend`, `tesseract_backend`, `words_from_azure_di` |
| Errors | `OcrRequiredError`, `EncryptedPdfError` |
| Types | `StruckWord`, `DetectResult`, `Passage` (in `pdf_strikethrough.types`) |

Everything is docstringed; `help(st.detect_pdf)` is the reference.

## Contributing, security, citation

- Development setup, running the tests, and regenerating the model: [`CONTRIBUTING.md`](CONTRIBUTING.md).
- This package parses untrusted PDFs — reporting policy in [`SECURITY.md`](SECURITY.md).
- Citing it in research: [`CITATION.cff`](CITATION.cff).

## License

MIT.
