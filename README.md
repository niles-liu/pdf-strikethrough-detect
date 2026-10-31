# pdf-strikethrough-detect

[![PyPI](https://img.shields.io/pypi/v/pdf-strikethrough-detect.svg)](https://pypi.org/project/pdf-strikethrough-detect/)
[![Python versions](https://img.shields.io/pypi/pyversions/pdf-strikethrough-detect.svg)](https://pypi.org/project/pdf-strikethrough-detect/)
[![CI](https://github.com/niles-liu/pdf-strikethrough-detect/actions/workflows/ci.yml/badge.svg)](https://github.com/niles-liu/pdf-strikethrough-detect/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/LICENSE)

Detect **struck-through (deleted) text** in PDFs and scanned document images.

Born-digital PDFs are read exactly. Scans go through OCR word boxes, stroke geometry and a small
CPU model. No cloud calls. Try it in the browser at
[huggingface.co/spaces/niles-liu/strikethrough-demo](https://huggingface.co/spaces/niles-liu/strikethrough-demo).

```python
import pdf_strikethrough as st

# born-digital PDF: exact, no OCR
for w in st.strikethroughs_in_pdf("contract.pdf"):
    print(w["page"], repr(w["chars"]), "partial" if w["partial"] else "full")

# any PDF, scanned pages included
res = st.detect_pdf("mixed.pdf", ocr=st.rapidocr_backend())
res["clean_text"]     # the surviving text, deletions removed
res["markdown"]       # deletions as ~~struck~~
```

## Install

```bash
pip install pdf-strikethrough-detect
pip install "pdf-strikethrough-detect[rapidocr]"    # free OCR for scans, no system binary
pip install "pdf-strikethrough-detect[markdown]"    # clean_markdown() via pymupdf4llm
pip install "pdf-strikethrough-detect[tesseract]"   # word-level OCR (needs the tesseract binary)
pip install "pdf-strikethrough-detect[torch]"       # a .pt model fallback (training: + onnx)
```

The model (~318 KB) ships in the wheel and runs on ONNX Runtime on the CPU. On Python 3.11+
`[markdown]` installs PyMuPDF 2.0; to stay below 2.0 there, install `pymupdf4llm<2` instead.

## Born-digital PDFs

A strikethrough in a born-digital PDF is a drawing over the text, so detection is exact: no OCR,
no model. Pick the detector with `method=`:

| `method=` | Reads | Gives |
|---|---|---|
| `"vector"` (default) | stroke geometry, and U+0336 overlay characters (`o̶f̶`) | exact partial spans, stroke color and width |
| `"flag"` | MuPDF's strikeout flag | also font-attribute strikes |
| `"annot"` | `/StrikeOut` annotations | the redline's author and dates |
| `"both"` | all three | the most recall |

A word counts as fully struck only when every character is, so in `~~December~~May` the `May`
stays in `clean_text`; `chars` and `char_span` give the struck characters. `bbox_frac` is in
fractions of the rendered page. Every record key is a `TypedDict` in `pdf_strikethrough.types`.

**Accuracy.** On 10 public redline PDFs (federal and California regulatory redlines and a
municipal code, 55.7k struck words), 99.8% of vector detections are confirmed by MuPDF's
strikeout flag (99.6 to 100% per document), and the flag finds 1.3% more words. The flag can
also include the letter right after a strike when no space follows (`DecemberM` of `DecemberMay`),
more often under PyMuPDF 2.0; `vector` and `both` stop at `December`.

**Garbled text layers.** Some PDFs map glyphs to the wrong characters: Word's Cambria Math
equations read `·(0.955) −` as `⋅ሺ0.955ሻെ`. `detect_pdf` lists such words in
`res["garbled_words"]`, marks a struck one `garbled: True`, notes each page in `res["warnings"]`
and issues one `GarbledTextWarning`. `on_garbled="drop"` (CLI `--on-garbled drop`) leaves them out
of the text. Not caught: a broken map that gives Latin, Greek or Cyrillic letters or U+FFFD, or
one that gives only letters of a single script in a non-math font, since that reads like real
text.

`clean_markdown()` (needs `[markdown]`) keeps pymupdf4llm's layout, but it reads MuPDF's flag, not
the vector strokes, and covers born-digital pages only: `help(st.clean_markdown)`.

## Scanned pages and images

```python
res = st.detect_pdf("scan.pdf", ocr=st.rapidocr_backend())          # scanned pages use OCR
res = st.detect_image_file("scan.tiff", ocr=st.rapidocr_backend())   # PNG, JPEG, TIFF (multi-page)
res = st.detect_pdf("scan.pdf", di_result=di_json)                   # an Azure DI result you have
struck = [w for w in res["words"] if w["final"]]
```

`detect_pdf` routes each page: native pages take the exact path, scanned pages go through stroke
geometry, OCR word boxes and StrikeNet (the bundled model), and `markdown`, `clean_text` and
`passages` cover both. A scanned page with no OCR raises `OcrRequiredError` (`on_missing_ocr="skip"`
skips it); a password-protected PDF raises `EncryptedPdfError`. An image's dpi comes from its
metadata (else 200), and a photo's EXIF rotation is applied.

### OCR backends

OCR supplies the word boxes; geometry and the model do the detecting. Measured by rasterizing
born-digital redline pages and scoring against the native detector's strikes (3 documents, 24
pages, 2,170 strikes;
[`benchmarks/scanned_recovery.py`](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/benchmarks/scanned_recovery.py)):

| Backend | Setup | Strikes recovered | Word boxes |
|---|---|---|---|
| Azure Document Intelligence (`di_result=`) | cloud, paid | 95.2% with the default; 97.9% with `confidence_free()` | exact words |
| RapidOCR (`[rapidocr]`) | pip only | 97.3% | phrases |
| Tesseract (`[tesseract]`) | system binary | not benchmarked | words |

**Confidence.** RapidOCR's confidences are near 1.0 whether a word is struck or not, so every entry
point runs `ScanConfig.confidence_free()`, except `detect_pdf` on the pages a `di_result` covers,
which uses the Azure DI calibration (`ScanConfig()`). That calibration was built on degraded ruled
forms, where DI reads struck words lower than clean ones; on a crisp printed scan it reads them
alike. On 472 printed pages with DI words (synthetic redlines, Senate bills and a public redline
corpus, none from the frontier benchmark's test sets), the default finds 88.9% of struck words and
marks 203 of 111,955 live ones; `confidence_free()` finds 92.6% and marks 719. Every `ScanConfig`
constructor and preset except `confidence_free()` uses the DI calibration unless given
`confidence_gating=False`: with another engine, start from `ScanConfig.confidence_free(...)`, which
takes the same fields.

**RapidOCR defaults.** `rapidocr_backend()` turns RapidOCR's orientation classifier off, since it
turns struck and underlined lines upside down; turn upside-down pages upright first, or pass
`params={"Global.use_cls": True}`. The default model can read a small mark as a CJK character (`(a)`
as `(斗)`), so on a page with no run of CJK text it removes lone CJK characters, real ones included
(`100 円` loses the unit); `drop_lone_cjk=False` keeps them, and the CLI always removes them. Word
boxes past the page edge are clamped to it.

**Operating points.** `ScanConfig.recall_first()` catches more deletions at the cost of more review;
`ScanConfig.precision_first()` avoids dropping live text (RAG, indexing). Fit a threshold to your
own labeled data with `pdf_strikethrough.calibration` and pass it as `recall_first(cnn_p_hi=...)`.

**Degraded ruled forms (provisional).** On faint, heavily ruled scans, printed rules crossing text
read as strikes. On a private labeled corpus, `ScanConfig.ruled_forms(rescue_clean_chains=False)`
cuts false strikes from 133 to 26 (44 with `ruled_forms()` alone, 108 with
`rescue_clean_chains=False` alone), with recall unchanged. `rescue_clean_chains=False` does nothing
under `confidence_free()`. On a clean scan both cost recall, since a real strike there is also solid
and straight and leaves OCR undamaged. Provisional: pin exactly if you rely on them.

## Other inputs

```python
for w in st.strikethroughs_in_docx("contract.docx"):          # strike formatting and tracked
    print(w["para"], repr(w["chars"]), w["docx_change"])      # deletions, with author and date

res = st.detect_pdf("doc.pdf", words_by_page=st.words_from_textract(textract_response))
res = st.detect_pdf("doc.pdf", words_by_page=st.words_from_docai(docai_document))

st.provenance_text(res)   # 'The fee is [deleted: the old rate of 5%] billed monthly.'
st.save_overlays("contract.pdf", "out/")                      # page images with strikes boxed
```

`provenance_text` keeps each deletion as a marker, for a RAG index that must not retrieve deleted
text as live. Runnable scripts that build their own sample PDFs are in
[`examples/`](https://github.com/niles-liu/pdf-strikethrough-detect/tree/main/examples).

## CLI

```bash
pdf-strikethrough detect contract.pdf                       # native pages
pdf-strikethrough detect scan.pdf --ocr rapidocr            # scanned pages too (PDF or image)
pdf-strikethrough detect scan.pdf --di-result di.json       # or --textract-result, --docai-result
pdf-strikethrough detect contract.docx
pdf-strikethrough detect doc.pdf --json out.json            # records and evidence (- = stdout)
pdf-strikethrough detect doc.pdf --clean-text clean.txt     # or --markdown, --provenance, --overlay
pdf-strikethrough detect doc.pdf --method both --pages 1-5,12
pdf-strikethrough detect ./contracts/ --jobs 4 --jsonl out.jsonl   # batch: folders, globs, files
pdf-strikethrough detect 'docs/**/*.pdf' --fail-if-found    # CI gate: exit 3 if any strike
```

Batch mode writes one JSON line per file as each finishes and never stops on a bad file (its line
carries `error`); per-file outputs such as `--markdown` are ignored there. Exit codes: 0 ok, 1
usage or file error, 2 encrypted or OCR required, 3 `--fail-if-found` matched, 130 interrupted.
Every flag: `pdf-strikethrough detect -h`.

## Monitoring and models

**Telemetry.** `telemetry=True` on `detect_pdf` and `detect_image_file` (CLI `--telemetry FILE`)
adds a JSON report: package and model version and sha256, time per page and stage, candidates per
tier and a histogram of StrikeNet's scores. Compare runs to see when incoming scans change.
Diagnostics go to the `pdf_strikethrough` logger at DEBUG, silent by default.

**Your own model.** `--dump-crops DIR` (or `st.dump_crops`) exports every scored crop as a
labeling set; retrain with
[`training/train_strikenet.py`](https://github.com/niles-liu/pdf-strikethrough-detect/tree/main/training)
and load the weights with `PDF_STRIKETHROUGH_MODEL_DIR` or `st.cnn.set_model_dir`.
`st.get_model_meta()["sha256"]` identifies the weights loaded. The shipped weights cannot be
reproduced: their training data is not recorded
([model card](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/training/MODEL_CARD.md#provenance-and-training)).

**A model from a URL.** `st.ensure_model` downloads a model and its meta, checks both sha256s
before writing them, caches them and points the loader at them, so a new model needs no new
release. Call it once at startup, then `st.get_model_meta()` to check it loads. The shipped
StrikeNet is on the Hub:

```python
st.ensure_model(
    "https://huggingface.co/niles-liu/strikenet/resolve/main/strike_verdict_cnn.onnx",
    "fac2c51baaa75ee782196bdfe7452638cb48c7deddb21163b1ac6a0a72ae4457",
    meta_url="https://huggingface.co/niles-liu/strikenet/resolve/main/strike_verdict_cnn.meta.json",
    meta_sha256="4388b14715bfb1f51b56bb6c463d8f5c0847533316297890e70ac0b2234405d4",
)
```

- A cached file whose digest matches is used without a download, so a restart needs no network.
- A digest that does not match raises `ValueError`, writes nothing and keeps the current model.
- It does not load the model: one that downloads but cannot load has already replaced the old
  one, and fails at the first score; `st.cnn.set_model_dir(None)` goes back to the packaged
  weights. To fall back when the host is unreachable, catch the exception.
- Called while another thread is reading a document, the rest of that document can be scored by
  the new model against the old thresholds: call it before reading starts.
- It does not fetch external data files: for such a model, copy its folder and point
  `PDF_STRIKETHROUGH_MODEL_DIR` at it. The CLI takes that variable too, not a URL.

More: `help(st.ensure_model)`.

## Scope and known limits

Detection assumes horizontal, left-to-right text and a near-straight stroke through a word's
middle. Reading order is column-aware.

| Case | Status |
|---|---|
| Latin text, born-digital | validated (the 10-PDF benchmark above) |
| Printed scans | validated (the recovery table above) |
| Two-column pages, struck table rows, strikes across hyphenated line breaks | handled |
| CJK text-layer PDFs | handled: a strike is horizontal whatever the script |
| CJK scans | many false strikes: 13 to 29% of unstruck Korean, Chinese and Japanese words on synthetic 200 dpi sans-serif scans (a serif Chinese font more than doubled it), from horizontal strokes inside characters; check `markdown` before trusting `clean_text` |
| Handwritten strikes | mostly missed: under 5% of struck words found on BRESSAY's pages (a public dataset of handwriting) |
| Dashed strikes | often found only in part |
| Strikes steeper than 12° (PDF) or 25° (scan) | not found |
| Thin or light-colored strokes on scans | may be missed |
| Highlighted or shaded text | a colored background is whitened first (provisional); a uniformly dim scan is not handled: re-scan or normalize it |
| Vertical writing, right-to-left strike axes | out of scope |

RapidOCR's default model has no Hangul. For Korean, pass its Korean recognizer from Python (the
CLI uses the default model); it needs rapidocr 3.3 or later, takes the values as RapidOCR's enums,
not strings, and downloads 13 MB on first use:

```python
from rapidocr import LangRec, ModelType, OCRVersion
ocr = st.rapidocr_backend(params={"Rec.lang_type": LangRec.KOREAN,
                                  "Rec.ocr_version": OCRVersion.PPOCRV5,
                                  "Rec.model_type": ModelType.MOBILE})
```

A [failing-page report](https://github.com/niles-liu/pdf-strikethrough-detect/issues/new?template=failing-page.md)
with `--dump-crops` output is the fastest way to get a case fixed.

## How it works

- **Native:** vector strokes through each character's strike band, measured in ems above the
  baseline, so underlines, overlines, fraction bars and gridlines behind a filled box are not
  strikes. A bar just above x-height letters (`√x`) and a gridline through text with no box behind
  it still are.
- **Scanned geometry** (`lines.py`): morphological opening at each angle finds stroke fragments,
  which are stitched and filtered on fill, thickness against the text's stems, angle and length;
  a second pass adds thick strikes through bold text.
- **Attribution** (`scanned.py`): strokes are assigned to OCR words and measured against each
  word's glyph ink, so a word is fully struck only when the stroke reaches both ends.
- **StrikeNet** (`cnn.py`, 79k parameters, ONNX): decides what geometry cannot, such as a thin
  strike over a word with no ascenders, which looks like a chain of serifs.

Scans above 300 dpi are worked at 200, the resolution the geometry and model are calibrated at;
a page over 128 Mpix is also scaled down, with a warning ([`SECURITY.md`](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/SECURITY.md)).

## Against frontier models

On 45 US Senate bills published after the tested models' training cutoffs, asked only to transcribe
a page, Grok 4.7 left more struck text live than pdf-strikethrough-detect 0.12.1, Claude Opus 5.5
left less, and Gemini 4 Argon and GPT-6 Astra were not shown to differ; told in one line never to
present struck text as current, Claude Opus 5.5 and GPT-6 Astra were no worse than the package and
Gemini 4 Argon and Grok 4.7 were not shown to be. On the 6 bills introduced after the cutoff: Grok
4.7 left more struck text live on all 6, Claude Opus 5.5 left less on 1 and the same on 5, Gemini 4
Argon left less on 1, more on 2 as images only and the same on 3, and GPT-6 Astra more on 5 as PDFs
only (on one of them less as an image) and the same on 1; told in one line, Claude Opus 5.5 and
GPT-6 Astra were no worse on all 6, Gemini 4 Argon on 2 and Grok 4.7 on none. The full tables, the
pre-registration and the data:
[`benchmarks/frontier/RESULTS.md`](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/benchmarks/frontier/RESULTS.md).

## API surface

The top-level package (`import pdf_strikethrough as st`) exports these names, plus its submodules:

| Group | Names |
|---|---|
| High-level | `strikethroughs_in_pdf`, `clean_markdown`, `provenance_text`, `detect_pdf`, `detect_image_file`, `detect_scanned_image`, `strikethroughs_in_docx`, `open_pdf`, `render_page_gray`, `render_overlay`, `save_overlays` |
| Native detectors | `native_page_strikes`, `native_flag_strikes`, `native_annot_strikes`, `native_doc_strikes`, `page_strikes`, `native_markdown`, `strip_struck_markdown` |
| Scanned geometry + classifier | `strike_lines`, `ink_mask`, `to_gray_u8`, `analyze_scanned_page`, `ScanConfig`, `classify_page_source`, `apply_cnn_verdict` |
| CNN | `score_word`, `score_crops`, `std_crop`, `word_crop_px`, `verdict_of`, `get_model_meta`, `ensure_model` |
| Active learning + calibration | `dump_crops`, `pdf_strikethrough.calibration` (`threshold_for_recall`, `threshold_for_precision`, `conformal_threshold`, `pr_curve`) |
| OCR (backends + cloud-result adapters) | `Word`, `rapidocr_backend`, `tesseract_backend`, `words_from_azure_di`, `words_from_textract`, `words_from_docai` |
| Errors and warnings | `OcrRequiredError`, `EncryptedPdfError`, `GarbledTextWarning` |
| Types | `StruckWord`, `DetectResult`, `Passage`, `GarbledWord`, `Telemetry` (in `pdf_strikethrough.types`) |

Every `detect_pdf` and `detect_image_file` result, and every CLI `--json` and `--jsonl` payload,
carries `schema_version` (`pdf_strikethrough.types.SCHEMA_VERSION`, now `1`). It goes up when a
documented key is removed, renamed or changes meaning, never for an added key, so ignore keys you
don't recognize. The telemetry report has its own `telemetry_version` under the same rule.
`help(st.detect_pdf)` is the reference.

## Contributing, security, citation

- Development setup, tests, and training or exporting a model:
  [`CONTRIBUTING.md`](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/CONTRIBUTING.md).
- This package parses untrusted PDFs; how to report a problem:
  [`SECURITY.md`](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/SECURITY.md).
- Citing it: [`CITATION.cff`](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/CITATION.cff).

## License

MIT.
