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
pip install "pdf-strikethrough-detect[torch]"       # optional .pt CNN fallback (training: + onnx)
```

The CNN ships as ONNX and needs nothing extra. `[torch]` is only for development — the loader
prefers `strike_verdict_cnn.onnx`, falling back to a `strike_verdict_cnn.pt` checkpoint if you
point `PDF_STRIKETHROUGH_MODEL_DIR` at one. Train a replacement with
[`training/train_strikenet.py`](training/), or export an existing checkpoint to ONNX with
[`tools/export_model.py`](tools/export_model.py); both also need `onnx` (`pip install onnx`).

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
final}` (`tier` is `"vector"`, `"flag"`, or `"annot"`), plus `garbled: True` when a broken
character map garbled the word ([below](#garbled-text-layers)). Vector records also carry
`stroke_color`/`stroke_width` (when a drawn stroke strikes them), the paint + thickness of the
dominant stroke (red = opposing counsel is evidence); annotation records carry
`annot_author`/`annot_created`/`annot_modified`/`annot_color`/`annot_id` (the redline's "who and
when"). Scanned records replace `coverage` with the evidence that decided them: `score`, `cnn_prob`,
`cnn_agrees`, `conf`, `geom_corroborated` (and `conf_veto: True` when the calibrated-confidence gate
downgrades a high-confidence word, or `line_support: True` when a short word is kept on its stroke's
confirmed neighbors). Their `tier` is `"auto"`/`"review"`. The keys are documented as `TypedDict`s
in `pdf_strikethrough.types` (`StruckWord`, `DetectResult`, `Passage`); the package ships
`py.typed`, so type checkers see them. Partial strikes (`semi-` of `semi-monthly`) are resolved to
the characters the strike actually crosses, read from the PDF's own glyph boxes, and a word counts
as fully struck only when every character is — so a replacement printed straight after a deletion
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
st.strikethroughs_in_pdf("contract.pdf", method="vector")  # stroke geometry (default) —
                                                           #   precise partial-char spans + color/width
st.strikethroughs_in_pdf("contract.pdf", method="flag")    # MuPDF's FZ_STEXT_STRIKEOUT signal —
                                                           #   also catches font-attribute strikes
st.strikethroughs_in_pdf("contract.pdf", method="annot")   # explicit /StrikeOut annotations —
                                                           #   Acrobat/Preview redlines + forensics
st.strikethroughs_in_pdf("contract.pdf", method="both")    # union of all three — maximum recall
```

**Validated across domains.** On 10 public redline PDFs (federal regulatory redlines from the
Copyright Office, FDIC, CEQ and EPA; California privacy-regulation redlines; a municipal
development code — 55.2k struck words), **99.8% of vector detections are independently confirmed
by MuPDF's strikeout signal** (99.6–100% per document; 9 of 10 ≥99.9%), and the flag method adds
~2.1% more words (font-attribute strikes and edge cases) — use `method="both"` to capture them.
MuPDF's flag can also take in the character a strike ends against when no space separates them
(`DecemberM` of `DecemberMay`), more often under PyMuPDF 2.0 than earlier versions; `vector` and
`both` stop at `December`.
`pymupdf4llm` is not used for detection at all; it is only an optional `[markdown]` extra for
richer layout in `clean_markdown()`. On Python 3.11+ the extra installs PyMuPDF 2.0, which ships
pymupdf4llm itself; to stay on PyMuPDF below 2.0 there, install `pymupdf4llm<2` instead of the
extra.

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
> uniformly on scanned pages too. `clean_markdown()` removes what pymupdf4llm marks `~~`, which is
> MuPDF's strikeout flag, so, like `method="flag"`, it can also remove the letter a strike ends
> against. It drops U+0336 overlay strikes itself, code blocks included, but keeps text the flag
> marks where pymupdf4llm sets it as a code block, such as a paragraph set entirely in a monospaced
> font.

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

### Choosing an OCR backend

The geometry + CNN carry the detection and are **OCR-independent**; OCR only supplies word boxes
to attribute strikes to, plus a confidence prior. Benchmarked by rasterizing born-digital redline
pages into image-only "scans" and taking the native detector's strikes on the born-digital
originals as ground truth (3 documents, 24 pages, 2,170 known strikes; reproduce with
[`benchmarks/scanned_recovery.py`](benchmarks/scanned_recovery.py)):

| Backend | Setup | Strike recovery | Word granularity |
|---|---|---|---|
| Azure Document Intelligence | cloud, paid | **96%** of known strikes with `confidence_free()`; 91% with the default DI calibration (see below) | exact word boxes |
| **RapidOCR** | `pip`, no binary | **96%** of known strikes | ~4× coarser (phrase-level) |
| Tesseract | needs system binary | not benchmarked here | genuine word-level |

`detect_pdf` runs confidence-free (`ScanConfig.confidence_free()`) whenever the words come from an
`ocr` backend or `words_by_page` — RapidOCR's confidences cluster near 1.0 and don't separate struck
from clean text — and uses the Azure DI calibration (`ScanConfig()`), whose struck words drop to
0.43–0.94, with `di_result`. Across backends the scanned path recovers **96% of the exact native
strike set** — the geometry + CNN, not the OCR engine, carry the detection. The DI calibration
keeps 91%: its confidence gates, built for degraded ruled forms, drop some crisply struck words that
DI still reads with high confidence. On a clean scan of a printed redline, pass
`scan_config=ScanConfig.confidence_free()` with `di_result`.

`rapidocr_backend()` passes its other keyword arguments to `RapidOCR(...)` and builds the engine
with RapidOCR's text-line orientation classifier off (`params={"Global.use_cls": False}`): RapidOCR
turns it on by default, and it turns struck and underlined lines upside down, which then read as
stray letters. Your `params` keys override that default, so `params={"Global.use_cls": True}` turns
the classifier back on. An engine passed as `engine=` is used as is; build it with
`RapidOCR(params={"Global.use_cls": False})`. With the classifier off, a page scanned upside down
reads as stray letters: turn it upright first, or turn the classifier back on. Word boxes RapidOCR
places past the edge of the page are clamped to it.

RapidOCR's default recognizer reads Chinese and Japanese as well as English, and on an English page
it can read a small mark as a CJK character (a footnote marker `(a)` as `(斗)`). On a page where no
two CJK characters come one right after the other in reading order (inside a word, or the last of
one word and the first of the next, across a line break too; a digit in between breaks the run),
the backend removes each one. Real CJK goes too when it stands alone like that: `100 円`, `5 個` and
`7 元` lose the unit, and `Article 1 (第1条)` reads `(1)`. A page with CJK text is left as read. Pass
`rapidocr_backend(drop_lone_cjk=False)` to keep them. The CLI's `--ocr rapidocr` always removes
them; to keep them, call `detect_pdf(path, ocr=rapidocr_backend(drop_lone_cjk=False))` from Python.

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

## Beyond PDFs — images, Word docs, cloud OCR

The same detection reaches inputs that never were a born-digital PDF:

```python
# Raster image files (.png/.jpg/.tiff, incl. multi-page TIFF) — photos, faxes, scans.
# Every frame is a scanned page, so it needs OCR; DPI comes from image metadata (else 200 — a
# placeholder 72/96 counts as none), and a phone photo's EXIF rotation is applied before OCR.
res = st.detect_image_file("scan.tiff", ocr=rapidocr_backend())

# Word .docx — strike formatting (w:strike/w:dstrike, set directly or by a style; text boxes too)
# + tracked deletions (w:del, with author/date).
# No OCR, no geometry: records use tier="docx" and a `para` index. Stdlib-only, no extra needed.
for w in st.strikethroughs_in_docx("contract.docx"):
    print(w["para"], repr(w["chars"]), w["docx_change"], w.get("docx_author"))

# AWS Textract / Google Document AI — neither flags strikethrough; add the layer on top.
# Convert a pre-fetched result to {page: [Word]} and feed it as words_by_page (works on images too).
# A word box past the edge of the page is clamped to it, here and in an Azure DI di_result.
res = st.detect_pdf("doc.pdf", words_by_page=st.words_from_textract(textract_response))
res = st.detect_pdf("doc.pdf", words_by_page=st.words_from_docai(docai_document))
```

**Provenance mode for RAG.** Struck text entering a vector index is a real failure mode — the
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
processed independently — JSONL output (one result object per line, written as each file
finishes), `--jobs N` to spread files across processes, and one bad file never aborts the run (its
line carries an `error` key). The mode follows the arguments, not how many files they match, so a
folder holding one file is still a batch. A directory means its top-level documents; a glob with
`**` recurses:

```bash
pdf-strikethrough detect ./contracts/ --jsonl out.jsonl        # every document in a directory
pdf-strikethrough detect *.pdf --jobs 4 --jsonl -              # glob, 4 workers, stream to stdout
pdf-strikethrough detect 'contracts/**/*.pdf' --fail-if-found  # CI gate over a whole tree (exit 3)
```

Per-file output flags (`--markdown`/`--overlay`/`--clean-text`/`--provenance`/cloud results/
`--pages`) are single-file only; in batch mode use `--jsonl`.

Other flags: `--dpi` (raster DPI for scanned pages, default 200; read from image metadata for image
files — very-high-dpi input is normalized to 200 internally, see below), `--overlay-dpi` (overlay
render DPI, default 150), `--limit` (max words in plain output), `--jobs` / `--jsonl` (batch),
`--scan-config auto|azure-di|confidence-free`, `--on-garbled keep|drop` (drop leaves garbled words
out of the text outputs; single file only, see [Garbled text layers](#garbled-text-layers)),
`--markdown`, `--docai-result`. Exit codes: `0` ok,
`1` usage/file error (batch: ≥1 file errored), `2` encrypted / OCR required, `3` `--fail-if-found`
matched, `130` interrupted (a batch keeps the lines already written). Results go to stdout (UTF-8
when written to `-`) and status lines to stderr, so `--json -` and `--markdown -` pipe cleanly.
Full help: `pdf-strikethrough detect -h`.

## Examples

Runnable, dependency-light scripts are in [`examples/`](examples/) — they generate their own
sample PDFs (no assets to download) so every snippet is copy-paste runnable:

```bash
python examples/native_quickstart.py     # build a redline PDF, detect strikes, print clean text
python examples/scanned_quickstart.py    # rasterize it to a "scan", run OCR + CNN  ([rapidocr])
python examples/overlay_quickstart.py     # print strike evidence + write a before/after overlay
python examples/rag_provenance.py         # clean_text vs provenance_text ([deleted: ...] markers)
```

## How it works

- **Native**: merged horizontal vector strokes through a word's letters, measured from the glyph
  baseline in ems of the text's height, so an underline at or under the baseline and an overline
  above the capitals are not strikes, whatever the font (a word whose text layer gives no usable
  baseline is measured against the middle share of its box instead); a character is struck when a
  stroke crosses ≥ 50% of its glyph box or a U+0336 overlay character follows it, and a word is
  fully struck only when every character is.
- **Scanned geometry** (`lines.py`): per-angle morphological opening extracts stroke *fragments*,
  collinear fragments are stitched, then strict filters (spine fill, stroke thickness against the
  stems of the text it crosses, angle/length) separate real strikes from bold crossbars and
  serif-glyph chains.
- **CNN** (`cnn.py`, StrikeNet, 79k params): resolves pixel-ambiguous cases — a thin strike over
  an ascender-less word is pixel-identical to a glyph chain, and only a learned model tells them
  apart. Ships as ONNX; set `PDF_STRIKETHROUGH_MODEL_DIR` to use your own weights.
- **Attribution** (`scanned.py`): assigns strokes to OCR words with char spans, measured against
  the word's glyph ink and the stroke's own ink rather than the OCR box and the detected line's
  ends; a word is fully struck only when the stroke reaches its first and last character. A
  visual-row "orphan" pass then picks up words the detector's stroke evidence missed.

**Resolution & resource limits.** The geometry tunables and the CNN's crop are calibrated at
200 dpi; scanned rasters above ~300 dpi are downsampled to 200 internally (accuracy-neutral — extra
resolution buys nothing but cost). Output boxes are page fractions, so this is invisible to results.
A hard raster budget caps a single page at 128 Mpix, so a hostile huge-mediabox page is
auto-downsampled with a warning instead of OOMing the process (see [`SECURITY.md`](SECURITY.md)).

**Operating points.** The CNN's struck/clean decision threshold is an explicit choice, not a
buried constant. `ScanConfig.recall_first()` biases toward catching every deletion (legal / audit
review — never miss a struck word); `ScanConfig.precision_first()` biases toward never dropping
live text as if deleted (RAG / indexing). Calibrate the threshold from your own labeled data with
`pdf_strikethrough.calibration` (`threshold_for_recall`, `threshold_for_precision`,
`conformal_threshold` for a distribution-free recall floor) and pass it as
`recall_first(cnn_p_hi=…)`.

**Underlines on table rules.** A line below a word's middle band counts as a strike only with glyph
ink on both sides of it. That ink is measured with the page's horizontal runs of 0.25 in or more
(table rules, underlines) left out, so an underlined number sitting on a table rule is not read as
struck, and ruled text is judged like unruled text; `pdf_strikethrough.lines.rule_mask` finds those
runs. `ScanConfig(glyph_ink=False)` measures on all ink, as earlier releases did.

**Short words.** A word of three characters or fewer, not counting surrounding punctuation, fills
little of the CNN's input, so a struck "of" or "a" often scores below the struck threshold even when
the words around it are confirmed. `ScanConfig.line_support` (on by default) keeps such a word when
the line through it strikes confirmed words on both sides of it and the word has a stroke of its
own: one unbroken stretch of ink at the line's height across at least 65% of the word, with more ink
there than just above or below it. A live word between two words struck one by one, which the
detector can join into one line, has glyph ink at that height but no such stroke, so it is normally
not kept. On the calibrated-confidence path a word read above `max_clean_conf` is never kept.
`precision_first()` turns it off, since it reports CNN-confirmed strikes only.

**Degraded ruled forms (provisional).** On faint, heavily-ruled scans — printed forms, dense
tables — a printed rule crossing text is hard to tell from a pen strike, and the classifier
over-fires on it. Two switches address it, both **off by default** and opted into separately:

- `ScanConfig.ruled_forms()` — a geometry-only veto discarding solid or dead-straight lines as
  printed furniture. Works with any OCR engine.
- `rescue_clean_chains=False` — stops the chain gate handing a clean-OCR glyph chain to the CNN
  because the page *looks* pen-edited; on degraded scans that signal reads scan quality, not edits.
  Needs calibrated confidences, so it does nothing under `confidence_free()`.

False positives on the private ruled-forms corpus, **recall unchanged in every configuration**:
**113** default → **38** veto only → **96** chain switch only → **25** with both, i.e.
`ScanConfig.ruled_forms(rescue_clean_chains=False)`. Not bundled, because each trades away recall on
its own terms — on a clean scan a real strike is also solid and straight, and a real strike can leave
OCR undamaged. Treat both as **provisional and outside the v1.0 stability contract**, expected to be
removed once the model handles ruled forms natively. Pin exactly if you depend on them.

**Shaded and highlighted blocks.** A grey or coloured ground behind text — a highlight block, a
shaded table row, a call-out frame — is flattened back to white before the page is binarized and
again inside each CNN crop, so a highlighted word is judged on its ink and not on its background
(issue #15). Ordinary paper is untouched, and pages with no shaded region produce byte-identical
results. Two caveats: the constants that decide *when* a ground counts as shaded rest on a single
measured document and are marked provisional in the source; and a uniformly **dim** scan (no
colour, paper around 110) is a different failure that this does not address — re-scan or
pre-normalize such a page before running it.

**Improving the model.** `--dump-crops DIR` (or `st.dump_crops`) exports every crop the pipeline
scored plus its verdict as a labeling set; label it, retrain with
[`training/train_strikenet.py`](training/), and load your weights via
`PDF_STRIKETHROUGH_MODEL_DIR`, `cnn.set_model_dir`, or `cnn.ensure_model(url, sha256, …)`
(digest-verified download). The shipped weights cannot be reproduced: their training data is not
recorded ([model card](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/training/MODEL_CARD.md#provenance-and-training)).
A model the script trains records a digest of its labeled set and its settings in its meta. A
["contribute a failing page"](.github/ISSUE_TEMPLATE/failing-page.md)
report feeds the same loop, and [`demo/`](demo/) is a drag-and-drop Gradio app for trying it — live
at [**huggingface.co/spaces/niles-liu/strikethrough-demo**](https://huggingface.co/spaces/niles-liu/strikethrough-demo).

**Hosted weights.** The shipped StrikeNet is also published on the Hugging Face Hub
([`niles-liu/strikenet`](https://huggingface.co/niles-liu/strikenet)); the digests are verified
before the graph is ever loaded, so a tampered host can't swap the model or its thresholds:

```python
import pdf_strikethrough as st
st.ensure_model(
    "https://huggingface.co/niles-liu/strikenet/resolve/main/strike_verdict_cnn.onnx",
    "fac2c51baaa75ee782196bdfe7452638cb48c7deddb21163b1ac6a0a72ae4457",
    meta_url="https://huggingface.co/niles-liu/strikenet/resolve/main/strike_verdict_cnn.meta.json",
    meta_sha256="4388b14715bfb1f51b56bb6c463d8f5c0847533316297890e70ac0b2234405d4",
)
```

**Scope — validated scripts & layouts.** Detection assumes **horizontal, left-to-right** text; a
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
every strike found and none false. Full vertical-text support is future work.

**Handwritten strikes** are the main known gap: the geometry gate looks for a near-straight,
near-horizontal stroke, and nothing records that the CNN saw handwritten strikes in training, so a
wavy pen scribble or a heavy cross-out may be missed. If you hit one, a
[failing-page report](.github/ISSUE_TEMPLATE/failing-page.md) with `--dump-crops` output is the
fastest way to turn it into a fix — it becomes both a regression test and training data.

## API surface

The top-level package (`import pdf_strikethrough as st`) exports ~37 names; the most-used:

| Group | Names |
|---|---|
| High-level | `strikethroughs_in_pdf`, `clean_markdown`, `provenance_text`, `detect_pdf`, `detect_image_file`, `detect_scanned_image`, `strikethroughs_in_docx`, `open_pdf`, `render_page_gray`, `render_overlay`, `save_overlays` |
| Native detectors | `native_page_strikes`, `native_flag_strikes`, `native_annot_strikes`, `native_doc_strikes`, `page_strikes`, `native_markdown`, `strip_struck_markdown` |
| Scanned geometry + classifier | `strike_lines`, `ink_mask`, `to_gray_u8`, `analyze_scanned_page`, `ScanConfig`, `classify_page_source`, `apply_cnn_verdict` |
| CNN | `score_word`, `score_crops`, `std_crop`, `word_crop_px`, `verdict_of`, `get_model_meta`, `ensure_model` |
| Active learning + calibration | `dump_crops`, `pdf_strikethrough.calibration` (`threshold_for_recall`, `threshold_for_precision`, `conformal_threshold`, `pr_curve`) |
| OCR (backends + cloud-result adapters) | `Word`, `rapidocr_backend`, `tesseract_backend`, `words_from_azure_di`, `words_from_textract`, `words_from_docai` |
| Errors and warnings | `OcrRequiredError`, `EncryptedPdfError`, `GarbledTextWarning` |
| Types | `StruckWord`, `DetectResult`, `Passage`, `GarbledWord` (in `pdf_strikethrough.types`) |

Every `detect_pdf` / `detect_image_file` result and every CLI `--json` / `--jsonl` payload carries
`schema_version` (`pdf_strikethrough.types.SCHEMA_VERSION`, currently `1`). It is bumped when a
documented key is removed or renamed or changes meaning, never for an added key, so ignore keys you
don't recognize.

Everything is docstringed; `help(st.detect_pdf)` is the reference.

## Contributing, security, citation

- Development setup, running the tests, and training or exporting a model: [`CONTRIBUTING.md`](CONTRIBUTING.md).
- This package parses untrusted PDFs — reporting policy in [`SECURITY.md`](SECURITY.md).
- Citing it in research: [`CITATION.cff`](CITATION.cff).

## License

MIT.
