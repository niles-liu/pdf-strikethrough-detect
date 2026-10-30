# pdf-strikethrough-detect

[![PyPI](https://img.shields.io/pypi/v/pdf-strikethrough-detect.svg)](https://pypi.org/project/pdf-strikethrough-detect/)
[![Python versions](https://img.shields.io/pypi/pyversions/pdf-strikethrough-detect.svg)](https://pypi.org/project/pdf-strikethrough-detect/)
[![CI](https://github.com/niles-liu/pdf-strikethrough-detect/actions/workflows/ci.yml/badge.svg)](https://github.com/niles-liu/pdf-strikethrough-detect/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/LICENSE)

Detect **struck-through (deleted) text** in PDFs and scanned document images.

It reads born-digital PDFs exactly, handles scans with a small CPU model, and makes no cloud calls.
Try it in the browser at
[huggingface.co/spaces/niles-liu/strikethrough-demo](https://huggingface.co/spaces/niles-liu/strikethrough-demo).

```python
import pdf_strikethrough as st

# born-digital PDF: exact, no OCR
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
pip install "pdf-strikethrough-detect[torch]"       # optional .pt CNN fallback (training: + onnx)
```

`[torch]` lets the loader fall back to a `strike_verdict_cnn.pt` checkpoint in
`PDF_STRIKETHROUGH_MODEL_DIR` when that folder has no ONNX model, and is what training needs;
training and exporting a model are covered in [`CONTRIBUTING.md`](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/CONTRIBUTING.md). On Python 3.11+
`[markdown]` installs PyMuPDF 2.0, which ships pymupdf4llm itself; to stay below PyMuPDF 2.0 there,
install `pymupdf4llm<2` instead of the extra.

## Native / born-digital PDFs: exact

In a born-digital PDF a strikethrough is a *vector drawing* (a line or thin rect over the text),
so detection is exact ground truth: no OCR, no model, no guessing. Both vector-rule and
filled-rect strikethrough styles are handled.

```python
print(st.clean_markdown("contract.pdf"))   # surviving text, deletions removed (needs [markdown])
```

Each native record: `{page, text, chars, char_span, partial, bbox_frac, tier, coverage, verdict,
final}` (`tier` is `"vector"`, `"flag"`, or `"annot"`), plus `garbled: True` when a broken
character map garbled the word ([below](#garbled-text-layers)). Vector records also carry
`stroke_color`/`stroke_width` (when a drawn stroke strikes them), the paint + thickness of the
dominant stroke (red = opposing counsel is evidence); annotation records carry
`annot_author`/`annot_created`/`annot_modified`/`annot_color`/`annot_id` (the redline's "who and
when"). Scanned records replace `coverage` with the evidence that decided them: `score`, `cnn_prob`,
`cnn_agrees`, `conf`, `geom_corroborated` (and `conf_veto: True` when the calibrated-confidence gate
downgrades a high-confidence word, or `line_support: True` when a short word is kept on its stroke's
confirmed neighbors). Their `tier` is `"auto"`/`"review"`. The keys are documented as `TypedDict`s
in `pdf_strikethrough.types` (listed under [API surface](#api-surface)); the package ships
`py.typed`, so type checkers see them. Partial strikes (`semi-` of `semi-monthly`) are resolved to
the characters the strike actually crosses, read from the PDF's own glyph boxes, and a word counts
as fully struck only when every character is, so a replacement printed straight after a deletion
(`~~December~~May`) survives in `clean_text`. `bbox_frac` is in fractions of the rendered page
(rotation-aware), so it maps directly onto a rendered pixmap.

Text can also be struck in the text layer itself, the way online strikethrough generators write it:
each character followed by U+0336 COMBINING LONG STROKE OVERLAY (`o̶f̶`). The `vector` detector
(and so `both`) reads that too. Each character an overlay follows is struck, and the record's
`text`, `markdown` and `clean_text` leave out the overlays on the characters it reports. A word
struck by overlays alone has no stroke, so its record carries no `stroke_color`/`stroke_width`.
The short overlays U+0335 and U+0337, which some generators emit, are not read. `flag` and `annot`
find nothing on overlays: an overlay is a character, not a drawn line or a strikeout annotation.

Three native detectors, all **base-PyMuPDF only** (no pymupdf4llm), selected by `method`:

```python
st.strikethroughs_in_pdf("contract.pdf", method="vector")  # stroke geometry (default):
                                                           #   precise partial-char spans + color/width
st.strikethroughs_in_pdf("contract.pdf", method="flag")    # MuPDF's FZ_STEXT_STRIKEOUT signal:
                                                           #   also catches font-attribute strikes
st.strikethroughs_in_pdf("contract.pdf", method="annot")   # explicit /StrikeOut annotations:
                                                           #   Acrobat/Preview redlines + forensics
st.strikethroughs_in_pdf("contract.pdf", method="both")    # union of all three: maximum recall
```

**Validated across domains.** On 10 public redline PDFs (federal regulatory redlines from the
Copyright Office, FDIC, CEQ and EPA; California privacy-regulation redlines; a municipal
development code; 55.2k struck words in all), **99.8% of vector detections are independently
confirmed by MuPDF's strikeout signal** (99.6–100% per document; 9 of 10 ≥99.9%), and the flag
method adds ~2.1% more words (font-attribute strikes and edge cases); use `method="both"` to
capture them.
MuPDF's flag can also take in the character a strike ends against when no space separates them
(`DecemberM` of `DecemberMay`), more often under PyMuPDF 2.0 than earlier versions; `vector` and
`both` stop at `December`.

## Any PDF: routed per page, scanned pages use OCR + CNN

```python
import pdf_strikethrough as st

res = st.detect_pdf("mixed.pdf", ocr=st.rapidocr_backend())  # OCR for the scanned pages

struck   = [w for w in res["words"] if w["final"]]          # struck words (boxes, char spans)
markdown = res["markdown"]                                  # deletions as ~~struck~~
clean    = res["clean_text"]                                # surviving text, deletions removed
passages = res["passages"]                                  # grouped deletion sections
```

`detect_pdf` classifies each page native-vs-scanned, runs the exact path on native pages
(`method="vector"|"flag"|"annot"|"both"`) and the geometry→OCR→CNN pipeline on scanned ones, and
assembles `markdown` / `clean_text` / `passages` for **both** page kinds from its own strike
decisions (so the text and the word records always agree, with no dependence on an external
markdown engine). Already have an Azure Document Intelligence result? Pass `di_result=...` (the
REST JSON dict, an `{'analyzeResult': ...}` envelope, or `sdk_result.as_dict()`) to skip re-OCR
and use DI's word boxes.

Scanned pages with no OCR backend raise `OcrRequiredError` by default; pass
`on_missing_ocr="skip"` to skip them (with a warning in `res["warnings"]`) and still get
everything from the native pages. Password-protected PDFs raise `EncryptedPdfError`.

> `clean_markdown()` is a separate **native-only** path that keeps pymupdf4llm's layout (headings,
> paragraphs); `detect_pdf`'s `markdown` is layout-plain but covers scanned pages too.
> `clean_markdown()` removes what MuPDF's strikeout flag marks, so, like `method="flag"`, it can
> also take the letter a strike ends against. Edge cases: `help(st.clean_markdown)`.

### Garbled text layers

Some PDFs map a font's glyphs to the wrong characters, so the text layer says something other than
the page. Equations Word exports in Cambria Math are the common case: `·(0.955) −` reads
`⋅ሺ0.955ሻെ`. `detect_pdf` lists such words in `res["garbled_words"]` (`page`, `text`,
`bbox_frac`; the CLI's `--json` and `--jsonl` carry it too), marks a struck record of one with
`garbled: True`, adds an entry to `res["warnings"]` for each page that holds any, and issues one
`GarbledTextWarning` naming those pages. `markdown` and `clean_text` keep them as extracted;
`on_garbled="drop"` (the CLI's `--on-garbled drop`) leaves them out. Real text in the scripts these
maps fall into (Ethiopic, the Indic scripts, Arabic, Hebrew, Thai and others), digits and Latin
abbreviations included, is not flagged. Not caught: a broken map that only gives letters of one such
script, in a font that is not a math font, since it reads like real text, and one that gives Latin,
Greek or Cyrillic letters or U+FFFD.

### Run telemetry

`telemetry=True` (on `detect_pdf` and `detect_image_file`) adds `res["telemetry"]`, a JSON-ready
report for monitoring a deployment. It holds:
- the package version, and the scoring model's `version` and `sha256` (`None` when no model was
  loaded);
- the seconds the call took;
- per page: its source, the seconds it spent rendering, in OCR and in detection, its candidates per
  tier, and how many words were reported struck. A scanned page also gives its dpi and OCR word
  count, and its tiers count the geometry stage's candidates, including the `weak` ones the CNN
  never sees;
- a ten-bin histogram of the scanned records' `cnn_prob`, with counts of crops too small to score
  and of values outside 0 to 1 (a broken model's).

Compare the tier mix and the histogram across runs to see when incoming scans start to differ.
Everything else in the result is unchanged. The CLI's `--telemetry FILE` appends one report per
document to FILE as a JSON line, with its `source`. In batch mode a file that fails gets a line with
its `error`.

### Choosing an OCR backend

The geometry + CNN carry the detection and are **OCR-independent**; OCR only supplies word boxes
to attribute strikes to, plus a confidence prior. Benchmarked by rasterizing born-digital redline
pages into image-only "scans" and taking the native detector's strikes on the born-digital
originals as ground truth (3 documents, 24 pages, 2,170 known strikes; reproduce with
[`benchmarks/scanned_recovery.py`](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/benchmarks/scanned_recovery.py)):

| Backend | Setup | Strike recovery | Word granularity |
|---|---|---|---|
| Azure Document Intelligence | cloud, paid | **96%** of known strikes with `confidence_free()`; 91% with the default DI calibration (see below) | exact word boxes |
| **RapidOCR** | `pip`, no binary | **96%** of known strikes | ~4× coarser (phrase-level) |
| Tesseract | needs system binary | not benchmarked here | genuine word-level |

`detect_pdf` runs confidence-free (`ScanConfig.confidence_free()`) when the words come from an
`ocr` backend or `words_by_page`, since RapidOCR's confidences cluster near 1.0 whether a word is
struck or not. With `di_result` it uses the Azure DI calibration (`ScanConfig()`), since DI's struck
words drop to 0.43–0.94. `detect_image_file`, `detect_scanned_image` and `analyze_scanned_page`
run confidence-free unless given a config. The DI calibration's confidence gates were built for
degraded ruled forms and drop some crisply struck words that DI still reads with high confidence,
hence its 91%; on a clean scan of a printed redline, pass `scan_config=ScanConfig.confidence_free()`
with `di_result`.

`rapidocr_backend()` builds RapidOCR with its text-line orientation classifier off
(`params={"Global.use_cls": False}`), since the classifier turns struck and underlined lines upside
down. With it off, a page scanned upside down reads as stray letters: turn the page upright first,
or call `rapidocr_backend(params={"Global.use_cls": True})`. An engine passed as `engine=` is used
as is; build it with `RapidOCR(params={"Global.use_cls": False})`. Word boxes past the edge of the
page are clamped to it.

RapidOCR's default recognizer also reads Chinese and Japanese, and on an English page it can read a
small mark as a CJK character (a footnote marker `(a)` as `(斗)`). On a page where no two CJK
characters come one after the other in reading order, the backend removes each one, real CJK
standing alone included (`100 円` loses the unit); a page with runs of CJK text is left as read.
`rapidocr_backend(drop_lone_cjk=False)` keeps them; the CLI's `--ocr rapidocr` always removes
them. The exact rule: `help(st.rapidocr_backend)`.

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

## Beyond PDFs: images, Word docs, cloud OCR

The same detection reaches inputs that never were a born-digital PDF:

```python
# Raster image files (.png/.jpg/.tiff, incl. multi-page TIFF): photos, faxes, scans.
# Every frame is a scanned page, so it needs OCR; DPI comes from image metadata (else 200; a
# placeholder 72/96 counts as none), and a phone photo's EXIF rotation is applied before OCR.
res = st.detect_image_file("scan.tiff", ocr=st.rapidocr_backend())

# Word .docx: strike formatting (w:strike/w:dstrike, set directly or by a style; text boxes too)
# + tracked deletions (w:del, with author/date).
# No OCR, no geometry: records use tier="docx" and a `para` index. Stdlib-only, no extra needed.
for w in st.strikethroughs_in_docx("contract.docx"):
    print(w["para"], repr(w["chars"]), w["docx_change"], w.get("docx_author"))

# AWS Textract / Google Document AI: neither flags strikethrough; add the layer on top.
# Convert a pre-fetched result to {page: [Word]} and feed it as words_by_page (works on images too).
# A word box past the edge of the page is clamped to it, here and in an Azure DI di_result.
res = st.detect_pdf("doc.pdf", words_by_page=st.words_from_textract(textract_response))
res = st.detect_pdf("doc.pdf", words_by_page=st.words_from_docai(docai_document))
```

**Provenance mode for RAG.** Struck text entering a vector index is a real failure mode: the
retriever surfaces a sentence the document *deleted*. `clean_text` removes it; `provenance_text`
instead keeps each deletion as a marker so it's recorded-as-deleted, not silently dropped:

```python
res = st.detect_pdf("contract.pdf")
res["clean_text"]           # 'The fee is billed monthly.'
st.provenance_text(res)     # 'The fee is [deleted: the old rate of 5%] billed monthly.'
```

## Low-level building blocks

```python
gray = st.render_page_gray(doc[0], dpi=200)     # always returns a HxW uint8 grayscale array

lines = st.strike_lines(gray, dpi=200)          # OCR-free stroke geometry (strike/underline/rule)
                                                # pass the dpi the image was rendered/scanned at

# word boxes are PAGE FRACTIONS in [0,1], origin top-left, not pixels
p = st.score_word(gray, (0.12, 0.34, 0.38, 0.36))   # CNN strike probability (0..1)

from pdf_strikethrough.ocr import Word
# detect_scanned_image accepts your own image; RGB / float arrays are coerced to grayscale uint8.
# It runs confidence-free unless told otherwise: pass config=st.ScanConfig.azure_di() for DI words
recs = st.detect_scanned_image(gray, [Word("foo", (0.12, 0.34, 0.38, 0.36), 0.6)])

# visual overlay: render each struck page with the strikes boxed (red=full, orange=partial)
for pg in st.render_overlay("contract.pdf", dpi=150):
    pg["image"].save(f"overlay-p{pg['page']}.png")     # or st.save_overlays(src, "out/")
```

**Logging.** Diagnostics (page routing, native method + record counts, OCR/CNN timings) are
emitted at `DEBUG` under the `pdf_strikethrough` logger, silent by default (a `NullHandler` is
attached). Opt in with `logging.getLogger("pdf_strikethrough").setLevel(logging.DEBUG)` and a
handler; `warnings` stay reserved for caller-facing hazards.

## CLI

```bash
pdf-strikethrough detect contract.pdf                       # native pages (scanned pages are
                                                            #   skipped with a warning)
pdf-strikethrough detect scan.pdf --ocr rapidocr            # include scanned pages
pdf-strikethrough detect scan.tiff --ocr rapidocr           # a raster image file (or multi-page TIFF)
pdf-strikethrough detect contract.docx                      # a Word doc (strike + tracked deletions)
pdf-strikethrough detect scan.pdf --di-result di.json       # use a pre-fetched Azure DI result
pdf-strikethrough detect scan.pdf --textract-result t.json  # ...or AWS Textract / Google DocAI
pdf-strikethrough detect doc.pdf --provenance prov.txt      # deletions kept as [deleted: ...] markers
pdf-strikethrough detect doc.pdf --method both              # max-recall native detection
pdf-strikethrough detect doc.pdf --method annot             # explicit /StrikeOut annotations only
pdf-strikethrough detect doc.pdf --pages 1-5,12             # only these pages (1-based)
pdf-strikethrough detect doc.pdf --json out.json            # full struck words + passages + evidence
pdf-strikethrough detect doc.pdf --json -                   # ...or stream JSON to stdout
pdf-strikethrough detect doc.pdf --clean-text clean.txt     # surviving text, deletions removed
pdf-strikethrough detect doc.pdf --markdown marked.md       # deletions as ~~struck~~
pdf-strikethrough detect doc.pdf --overlay out/             # page images with strikes boxed
pdf-strikethrough detect scan.pdf --ocr rapidocr --dump-crops crops/  # export scored crops to label
pdf-strikethrough detect doc.pdf --fail-if-found            # exit 3 if any strike found (CI gate)
cat doc.pdf | pdf-strikethrough detect -                    # read the PDF from stdin
pdf-strikethrough --version
```

**Batch mode.** Pass several files, a directory, or a glob (or ask for `--jsonl`) and each is
processed independently: JSONL output (one result object per line, written as each file
finishes), `--jobs N` to spread files across processes, and one bad file never aborts the run (its
line carries an `error` key). The mode follows the arguments, not how many files they match, so a
folder holding one file is still a batch. A directory means its top-level documents; a glob with
`**` recurses:

```bash
pdf-strikethrough detect ./contracts/ --jsonl out.jsonl        # every document in a directory
pdf-strikethrough detect *.pdf --jobs 4 --jsonl -              # glob, 4 workers, stream to stdout
pdf-strikethrough detect 'contracts/**/*.pdf' --fail-if-found  # CI gate over a whole tree (exit 3)
```

Per-file flags (`--markdown`, `--clean-text`, `--provenance`, `--overlay`, `--dump-crops`,
`--on-garbled`, cloud results, `--pages`) are ignored in batch mode, with a warning; batch results
go to `--jsonl`.

Other flags: `--dpi` (raster DPI for scanned pages, default 200; read from image metadata for image
files; very-high-dpi input is normalized to 200 internally, see below), `--overlay-dpi` (overlay
render DPI, default 150), `--limit` (max words in plain output), `--jobs` / `--jsonl` (batch),
`--scan-config auto|azure-di|confidence-free`, `--on-garbled keep|drop` (see
[Garbled text layers](#garbled-text-layers)), `--telemetry FILE` (append a JSON line per document,
see [Run telemetry](#run-telemetry)), `--docai-result`. Exit codes: `0` ok,
`1` usage/file error (batch: ≥1 file errored), `2` encrypted / OCR required, `3` `--fail-if-found`
matched, `130` interrupted (a batch keeps the lines already written). Results go to stdout (UTF-8
when written to `-`) and status lines to stderr, so `--json -` and `--markdown -` pipe cleanly.
Full help: `pdf-strikethrough detect -h`.

## Examples

Runnable, dependency-light scripts are in [`examples/`](https://github.com/niles-liu/pdf-strikethrough-detect/tree/main/examples): they generate their own
sample PDFs (no assets to download) so every snippet is copy-paste runnable:

```bash
python examples/native_quickstart.py     # build a redline PDF, detect strikes, print clean text
python examples/scanned_quickstart.py    # rasterize it to a "scan", run OCR + CNN  ([rapidocr])
python examples/overlay_quickstart.py     # print strike evidence + write a before/after overlay
python examples/rag_provenance.py         # clean_text vs provenance_text ([deleted: ...] markers)
```

## How it works

- **Native**: merged horizontal vector strokes through a word's letters. A stroke counts only inside
  a character's strike band, measured in ems above the glyph baseline (or the middle of the box when
  the text layer gives no usable baseline), so underlines and overlines are not strikes, whatever the
  font. A rule beside or behind the text is not a strike either: a fraction bar between a numerator
  and a denominator, a radical's bar over its radicand (though one just above x-height letters, as
  in `√x`, can still read as a strike), or a gridline behind a legend's box. A character is struck
  when strokes through its own band cover ≥ 50% of its box width or a U+0336 overlay follows it (a
  radical sign, and a small sub- or superscript beside a struck character, go with their neighbor);
  a word is fully struck only when every character is.
- **Scanned geometry** (`lines.py`): per-angle morphological opening extracts stroke *fragments*,
  collinear fragments are stitched, then strict filters (spine fill, stroke thickness against the
  stems of the text it crosses, angle/length) separate real strikes from bold crossbars and
  serif-glyph chains. A second pass adds a thick strike through bold or large text where its ink
  runs on unbroken across the glyphs, without changing the lines the first pass found.
- **CNN** (`cnn.py`, StrikeNet, 79k params): resolves pixel-ambiguous cases: a thin strike over
  an ascender-less word is pixel-identical to a glyph chain, and only a learned model tells them
  apart. Ships as ONNX; set `PDF_STRIKETHROUGH_MODEL_DIR` to use your own weights, or load them
  from a URL (below).
- **Attribution** (`scanned.py`): assigns strokes to OCR words with char spans, measured against
  the word's glyph ink and the stroke's own ink rather than the OCR box and the detected line's
  ends; a word is fully struck only when the stroke reaches its first and last character. A
  visual-row "orphan" pass then picks up words the detector's stroke evidence missed.

**Resolution and resource limits.** The geometry and the CNN's crop are calibrated at 200 dpi, so
scanned rasters above ~300 dpi are worked at 200. A page over 128 Mpix is downsampled with a
warning ([`SECURITY.md`](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/SECURITY.md)). Output boxes are page fractions either way.

**Operating points.** The CNN's struck/clean decision threshold is an explicit choice, not a
buried constant. `ScanConfig.recall_first()` biases toward catching every deletion (legal or audit
review: never miss a struck word); `ScanConfig.precision_first()` biases toward never dropping
live text as if deleted (RAG / indexing). Calibrate the threshold from your own labeled data with
`pdf_strikethrough.calibration` (`threshold_for_recall`, `threshold_for_precision`,
`conformal_threshold` for a distribution-free recall floor) and pass it as
`recall_first(cnn_p_hi=…)`.

**Underlines on table rules.** A line below a word's middle band counts as a strike only with glyph
ink on both sides of it, or with glyph ink between it and a second stroke above it, their middle in
the band (a double strike's lower line). The glyph ink is measured without the page's horizontal
runs of 0.25 in or more (table rules, underlines; `lines.rule_mask` finds them), so an underlined
number sitting on a table rule is not read as struck. `ScanConfig(glyph_ink=False)` measures on
all ink, with no double-strike test.

**Short words.** A struck word of three characters or fewer, not counting surrounding punctuation,
fills little of the CNN's input and often scores below the struck threshold.
`ScanConfig.line_support` (on by default; off in `precision_first()`) keeps such a word when the
line through it strikes confirmed words on both sides and the word has a stroke of its own: one
unbroken run of ink at the line's height across at least 65% of the word, with more ink there than
just above or below it. On the calibrated-confidence path a word read above `max_clean_conf` is
never kept.

**Degraded ruled forms (provisional).** On faint, heavily ruled scans (printed forms, dense tables)
the classifier over-fires on printed rules crossing text. Two opt-in switches cut false positives
on the private ruled-forms corpus from **113** by default to **38** with `ScanConfig.ruled_forms()`
(a veto on solid or dead-straight lines), **96** with `rescue_clean_chains=False` (which stops the
chain gate passing a clean-OCR glyph chain to the CNN on a page that looks pen-edited), or **25**
with both (`ScanConfig.ruled_forms(rescue_clean_chains=False)`), with **recall unchanged in every
configuration**. The veto keeps confidence gating on, which assumes Azure DI confidences (add
`confidence_gating=False` for other engines), and the chain switch does nothing under
`confidence_free()`. Each costs recall elsewhere: on a clean scan a real strike is also solid and
straight, and can leave OCR undamaged. Both are **provisional and
outside the v1.0 stability contract**, expected to go once the model handles ruled forms natively.
Pin exactly if you depend on them.

**Shaded and highlighted blocks.** A grey or coloured ground behind text (a highlight, a shaded
table row, a call-out frame) is flattened to white before the page is binarized and inside each CNN
crop, so a highlighted word is judged on its ink; ordinary paper is untouched. The constants that
decide when a ground counts as shaded rest on one measured document and are provisional. A
uniformly **dim** scan (no colour, paper around 110) is a different failure this does not fix:
re-scan or pre-normalize such a page first.

**Improving the model.** `--dump-crops DIR` (or `st.dump_crops`) exports every crop the pipeline
scored, with its verdict, as a labeling set; label it, retrain with
[`training/train_strikenet.py`](https://github.com/niles-liu/pdf-strikethrough-detect/tree/main/training), and load your weights via
`PDF_STRIKETHROUGH_MODEL_DIR`, `cnn.set_model_dir`, or `cnn.ensure_model` (below).
`st.get_model_meta()["sha256"]`, with `external_data` for weights kept beside the model, tells
retrained models apart in a log. The shipped weights cannot be reproduced: their training data is
not recorded ([model card](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/training/MODEL_CARD.md#provenance-and-training)).
A ["contribute a failing page"](https://github.com/niles-liu/pdf-strikethrough-detect/issues/new?template=failing-page.md) report feeds the same
loop, and [`demo/`](https://github.com/niles-liu/pdf-strikethrough-detect/tree/main/demo) is the Gradio app behind the live demo.

**Loading a model from a URL.** A model can be deployed without a new release of the package:
host `strike_verdict_cnn.onnx` and its `strike_verdict_cnn.meta.json` at an `https`, `http` or
`file` URL (the Hugging Face Hub, a storage bucket, a signed URL), and call `st.ensure_model` once
at startup, before the first word is scored. It downloads both files, checks each against its
sha256 before writing it, and points the loader at them, so a tampered host can't swap the model or
its thresholds. The shipped StrikeNet is published this way on the Hub
([`niles-liu/strikenet`](https://huggingface.co/niles-liu/strikenet)), and the demo loads it like
this:

```python
import pdf_strikethrough as st
st.ensure_model(
    "https://huggingface.co/niles-liu/strikenet/resolve/main/strike_verdict_cnn.onnx",
    "fac2c51baaa75ee782196bdfe7452638cb48c7deddb21163b1ac6a0a72ae4457",
    meta_url="https://huggingface.co/niles-liu/strikenet/resolve/main/strike_verdict_cnn.meta.json",
    meta_sha256="4388b14715bfb1f51b56bb6c463d8f5c0847533316297890e70ac0b2234405d4",
)
```

- The files are cached in `~/.cache/pdf_strikethrough/models/` under the first 12 characters of
  the model's sha256, or in `cache_dir=`. A cached file whose digest matches is used without a
  download, so once a model is cached a restart needs no network.
- A later call may leave out `meta_url` and `meta` to reuse the cached meta. Each call that passes
  one records which meta was verified with which model, and a call without one raises
  `ValueError` unless that record pairs this model with the meta as it is now. A folder from
  before the record is trusted if it is the model's default folder or already holds the model.
- A file whose digest does not match raises `ValueError` and is not written, and the loader keeps
  the model it had. With `meta_url` the meta's digest is required too, since its thresholds decide
  what counts as struck; or pass the meta itself as `meta={...}`. To fall back to the packaged
  weights when the host can't be reached, catch the exception, as the demo does.
- It does not load the model: one that downloads but cannot load (a meta that disagrees with the
  code) fails at the first score, the previous model already dropped. Call `st.get_model_meta()`
  right after it to find out at startup, and `cnn.set_model_dir(None)` to go back to
  `PDF_STRIKETHROUGH_MODEL_DIR` or the packaged weights.
- Called while another thread is reading a document, it takes effect from the next document: the
  one being read can finish with the new model's scores judged by the old model's thresholds.
- A model that keeps its weights in external data files, as torch's newer exporter writes by
  default, needs each file hosted beside the `.onnx` at the location the model gives, and its
  sha256 in the meta's `external_data`, listed as `st.get_model_meta()` reports it for the model
  loaded from its folder. Each file is checked like the model before anything is written, and a
  file the meta does not list raises `ValueError`. The files' URLs are the model's with its file
  name replaced by their location, so a model URL with a query string, such as a signed URL, is
  refused. The loader refuses a folder whose files do not match its meta's `external_data`, so
  when a write fails partway, the weights are not scored under another model's meta. Retrained
  weights under the same graph file share its sha256 and so its default cache folder: give each
  its own `cache_dir`.
- The CLI has no URL option: set `PDF_STRIKETHROUGH_MODEL_DIR` to the folder `ensure_model`
  returns, or to one you downloaded the files to.

**Scope: validated scripts and layouts.** Detection assumes **horizontal, left-to-right** text; a
strike is a near-horizontal stroke through a word's middle band. Reading-order assembly is
column-aware (two-column pages read down each column; tables read across rows).

| Case | Status |
|---|---|
| Latin / Western, horizontal | validated (10-PDF redline benchmark, 55.2k struck words) |
| CJK, **horizontal** set, text-layer PDFs | detected: strikes are horizontal regardless of script (regression-tested) |
| CJK, **horizontal** set, scans | strikes found, but many false strikes (below) |
| Two-column / newspaper layout, struck table rows, hyphenated-line-break strikes | handled |
| **Vertical** writing modes (CJK/Mongolian), RTL strike axes | out of scope (roadmap) |
| Handwritten / freehand strikes | untested (geometry assumes near-straight strokes) |

RapidOCR's default model (the recommended scanned backend) reads Chinese and Japanese. It has no
Hangul, so for Korean pass RapidOCR's Korean recognizer, from Python (the CLI's `--ocr rapidocr`
uses the default model). It needs rapidocr 3.3 or later, takes the values as RapidOCR's enums
(strings are refused), and downloads its 13 MB model on first use. These params are added to the
package's defaults, so the orientation classifier stays off:

```python
from rapidocr import LangRec, ModelType, OCRVersion
ocr = st.rapidocr_backend(params={"Rec.lang_type": LangRec.KOREAN,
                                  "Rec.ocr_version": OCRVersion.PPOCRV5,
                                  "Rec.model_type": ModelType.MOBILE})
res = st.detect_pdf("scan.pdf", ocr=ocr)
```

On CJK scans the scanned path takes the horizontal stroke inside many characters for a strike:
Hangul syllables built on a horizontal vowel (을, 등, 총) and Han characters such as 王 and 三.
StrikeNet scores those strokes about as high as real strikes, so no threshold removes them. On
synthetic 200-dpi scans set at 12 pt in sans-serif fonts, it marked 13% to 28% of the unstruck
Korean, Chinese and Japanese words struck (and none of the English). The rate changes with font,
size and resolution; a serif Chinese font more than doubled it. So check a CJK scan's struck words
(`markdown`, or the overlay) before trusting its `clean_text`. PDFs with a text layer take the
native path, which never looks at glyph shapes. Rendered as text-layer PDFs, the same pages had
every strike found and none false.

**Handwritten strikes** are the main known gap: the geometry gate looks for a near-straight,
near-horizontal stroke, and nothing records that the CNN saw handwritten strikes in training, so a
wavy pen scribble or a heavy cross-out may be missed. If you hit one, a
[failing-page report](https://github.com/niles-liu/pdf-strikethrough-detect/issues/new?template=failing-page.md) with `--dump-crops` output is
the fastest way to turn it into a fix: it becomes both a regression test and training data.

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

Every `detect_pdf` / `detect_image_file` result and every CLI `--json` / `--jsonl` payload carries
`schema_version` (`pdf_strikethrough.types.SCHEMA_VERSION`, currently `1`). It is bumped when a
documented key is removed or renamed or changes meaning, never for an added key, so ignore keys you
don't recognize. The `telemetry` report carries its own `telemetry_version` under the same rule.

Everything is docstringed; `help(st.detect_pdf)` is the reference.

## Contributing, security, citation

- Development setup, running the tests, and training or exporting a model:
  [`CONTRIBUTING.md`](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/CONTRIBUTING.md).
- This package parses untrusted PDFs; reporting policy in [`SECURITY.md`](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/SECURITY.md).
- Citing it in research: [`CITATION.cff`](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/CITATION.cff).

## License

MIT.
