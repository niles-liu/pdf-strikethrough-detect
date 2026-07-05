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
```

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

Each record: `{page, text, chars, char_span, partial, bbox_frac, coverage, verdict, final}`.
Partial strikes (`semi-` of `semi-monthly`) are resolved to a char range. `bbox_frac` is in
fractions of the rendered page (rotation-aware), so it maps directly onto a rendered pixmap.

Two native detectors, both **base-PyMuPDF only** (no pymupdf4llm), selected by `method`:

```python
st.strikethroughs_in_pdf("contract.pdf", method="vector")  # stroke geometry (default) —
                                                           #   precise partial-char spans
st.strikethroughs_in_pdf("contract.pdf", method="flag")    # MuPDF's FZ_STEXT_STRIKEOUT signal —
                                                           #   also catches font-attribute strikes
st.strikethroughs_in_pdf("contract.pdf", method="both")    # union — maximum recall
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
(`native_method="vector"|"flag"|"both"`) and the geometry→OCR→CNN pipeline on scanned ones, and
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
gray = st.render_page_gray(doc[0], dpi=200)     # HxW grayscale; RGB/float arrays are coerced

lines = st.strike_lines(gray, dpi=200)          # OCR-free stroke geometry (strike/underline/rule)
                                                # pass the dpi the image was rendered/scanned at

# word boxes are PAGE FRACTIONS in [0,1], origin top-left — not pixels
p = st.score_word(gray, (0.12, 0.34, 0.38, 0.36))   # CNN strike probability (0..1)

from pdf_strikethrough.ocr import Word
recs = st.detect_scanned_image(gray, [Word("foo", (0.12, 0.34, 0.38, 0.36), 0.6)])
```

## CLI

```bash
pdf-strikethrough detect contract.pdf                       # native pages (scanned pages are
                                                            #   skipped with a warning)
pdf-strikethrough detect scan.pdf --ocr rapidocr            # include scanned pages
pdf-strikethrough detect doc.pdf --method both              # max-recall native detection
pdf-strikethrough detect doc.pdf --json out.json            # full struck words + passages
pdf-strikethrough detect doc.pdf --clean-text clean.txt     # surviving text, deletions removed
pdf-strikethrough detect doc.pdf --markdown marked.md       # deletions as ~~struck~~
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

## License

MIT.
