# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Detection output changes on scans in two cases, both under Changed: words on light grey paper
(paper at 240 to 254 of 255), and colour images (image files other than greyscale, and RGB(A)
arrays), even on white paper. Every other change is to `training/train_strikenet.py`, to the
files `dump_crops` writes, or to docs.

### Changed
- **Scans on light grey paper:** StrikeNet's scores, and the scanned path's verdicts with them,
  change for words on paper at 240 to 254 of 255: light grey or light-coloured paper, or paper
  under a light highlighter. `cnn.std_crop` now takes such paper to white, as it already took a
  shaded ground below 240; a PDF or greyscale image with no word on such paper gives the same
  output. On the faint grey ground 0.14.0 left, the shipped model calls live words struck from
  240 to about 245 and misses strikes from about 245. On 472 pages of printed text rendered with
  their paper greyed to 240, 245 or 250, it now finds about as many struck words, and marks about
  as many live ones, as on white paper; no scan of a printed page was measured. On 218 pages of
  the training split and 103 of the validation split of BRESSAY (a public dataset of
  handwriting), where most words sit on such paper, the shipped model now marks 7.4% and 4.6%
  more live words and finds 3 and 1 more struck words with RapidOCR words. With Azure DI words it
  marks 14 live words against 9 on 20 of the training pages and 31% more on 50 of the validation
  pages, and finds no more struck words. `cnn.PAPER_WHITE_FROM = 240` restores 0.14.0. (#107)
- **Colour images:** an image file other than greyscale (colour, palette, bilevel or with alpha),
  or an RGB(A) array given to `detect_scanned_image`, `analyze_scanned_page` or `score_word`, has
  its white paper read as 254, as before. Such paper is now taken to white too, so the output
  comes closer to the same image's in greyscale: read as colour images, 431 of the 472 pages on
  white paper give the same output as in greyscale, against 328 with 0.14.0. (#107)
- **Retraining:** crops dumped with 0.14.0 keep the faint grey ground of paper at 240 to 254, so
  dump scans again before training, into a new directory: a dump replaces the `crops.jsonl` it
  writes to, labels included. A model you trained on 0.14.0 dumps saw that ground, and the loader
  does not check the `std_crop` settings a meta records: use such a model with
  `cnn.PAPER_WHITE_FROM = 240`, or train a new one on a new dump. (#107, #108)
- `training/train_strikenet.py` matches a retrained model's thresholds to the model the package
  loads (the shipped model, unless `PDF_STRIKETHROUGH_MODEL_DIR` or `cnn.set_model_dir` points
  at another): `p_hi` is the highest threshold at which the new model calls at least as many
  struck validation crops struck as that model does at its own `p_hi`, in each `source` that rows
  name (a new, optional column); `p_lo` is the lowest at which it calls at least as many clean
  ones clean, capped at `p_hi`. The split-conformal pair (`--alpha`) no longer sets the
  thresholds: the meta's `training.calibration` reports it, with how the thresholds were set. The
  script stops before training if that model does not load, gives a validation crop no finite
  score or calls none of the struck validation crops struck, and the validation split needs at
  least 19 struck crops at any `--alpha` (more below 0.05). (#106)
- `export()` in `training/train_strikenet.py` takes a `thresholds` callback instead of `p_hi=`
  and `p_lo=`, refuses thresholds the loader would refuse or that are not rounded to 3 decimals,
  and keeps the old model when replacing it fails. (#105)

### Fixed
- `training/train_strikenet.py` held out random crops, so a page's crops fell on both sides of the
  split and the thresholds came from pages the model had trained on. It now holds out whole pages,
  or whole groups that rows name in a `group` column, so the same dump and seed hold out other
  crops than 0.14.0 did; a dataset with neither splits as before. It stops before training when
  the training split would lack a struck or a clean crop, or when no split by group can both train
  and calibrate (a one-page dump, say). It prints the split and records in the meta how it
  grouped the rows and the numpy version. (#104)
- `training/train_strikenet.py` set `p_hi` on torch's validation probabilities and wrote it
  rounded to 4 decimals, while the detector compares the exported model's score rounded to 3, so
  a validation strike near `p_hi` could be dropped. Both thresholds are now set on the exported
  model's scores as the detector reads them and written as they are. The split-conformal `p_lo`
  the meta reports is the finite-sample-corrected mirror of its `p_hi`, and the script no longer
  accepts a validation split whose split-conformal `p_hi` would be 0.0, as it did at an alpha
  such as 1/161. (#105)
- `dump_crops` wrote each crop PNG up to a grey level darker than the crop StrikeNet scored (it
  truncated; it now rounds). Each `crops.jsonl` row now records the `std_crop` settings its crop
  was made with, and `training/train_strikenet.py` trains on such a crop exactly as the detector
  scored it, rather than standardizing it again. The meta's `paper_white_from` and
  `shade_white_frac` record the settings the crops were made with (the script's own for a crop it
  standardizes), and the script stops when the labeled crops were made with different settings.
  A crop whose row records no settings is standardized as before, so a dump by 0.14.0 or earlier
  still trains, with a note to dump its scans again. The dataset digest marks each crop that
  trains as dumped. (#108)
- `training/train_strikenet.py` crashed on a `crops.jsonl` that starts with a byte-order mark
  (PowerShell 5.1 writes one with `-Encoding UTF8`), on a U+2028 in OCR text (`dump_crops` writes
  it unescaped) and on a label such as `1`, `true` or NaN. It now reads them, and reads UTF-16 too
  (as PowerShell 5.1's `>` and `Out-File` write it). A NaN label counts as none, and a label other
  than struck or clean, such as `1`, is skipped and named in a note of its own, apart from the
  unlabeled count. It stops, naming the line, on a row it cannot read, a crop PNG it cannot open
  or other text that is not UTF-8. (#108)
- The docstrings of `calibration` and `calibration.conformal_threshold` no longer call a
  calibrated threshold's recall a guarantee: the conformal bound holds on average over calibration
  sets, for struck words exchangeable with the calibration words, and words from another writer,
  scanner or kind of document can fall short of it. `cnn.std_crop`'s docstring now says that a
  crop more than about 5 times as wide as it is tall loses its ends, so a strike on only one end
  of a long word can be cut off, and that an integer crop of a type wider than 8 bits is scaled
  so that the type's maximum is white: pass 0 to 255 values as `uint8`. (#109)

## [0.14.0] - 2026-11-02

Minor, not patch: it adds public API (`InvalidPdfError`, `LimitExceededError`, `page_words=` with
`render_text`, `max_pages=` and `max_mpix=`). Detection output is unchanged: with the new options
off, every result 0.13.1 gave on a valid PDF, or on an image through `detect_image_file`, is the
same. What changed is the handling of inputs 0.13.1 failed on, of files that are not readable
PDFs, and of damaged PDFs (below). Results keep `schema_version` 1.

### Added
- `page_words=True` on `detect_pdf` and `detect_image_file` adds `res["page_words"]`, every word
  each page's text is built from (`types.PageWords`). Given a copy of `res["words"]` with a
  reviewer's changes to `final`, `render_text(res, words)` rebuilds `markdown`, `clean_text` and
  `passages` as reviewed, also from a result saved as JSON. (#101)
- `max_pages=` and `max_mpix=` on `detect_pdf` and `detect_image_file` raise `LimitExceededError`
  (a `ValueError`) before any page is detected. Toward `max_mpix`, a scanned PDF page counts as
  rendered, an image frame as stored and a born-digital page not at all; an image file's frames
  are read only up to the first one over a limit. (#99)
- `InvalidPdfError`, for a file that is not a readable PDF: a `ValueError`, and still a
  `pymupdf.FileDataError`, the error a file MuPDF cannot open raised before. (#96)
- The README says what a record's `verdict` means. (#97)

### Changed
- The PDF functions (`detect_pdf`, `open_pdf`, `strikethroughs_in_pdf`, `clean_markdown`,
  `render_overlay`, `save_overlays`) refuse a file MuPDF reads as another format, such as an
  image or an EPUB, with `InvalidPdfError`; most of them used to read it. Use `detect_image_file`
  for an image. A document you opened yourself is not refused for its format, and the CLI reads
  any document MuPDF opens, as before. (#96)
- MuPDF's own errors for a file the package cannot read arrive as `InvalidPdfError`, so code
  that caught `pymupdf.mupdf.FzErrorFormat` (a broken page tree) or `pymupdf.EmptyFileError`
  (an empty file) should catch `InvalidPdfError`. (#96)
- `open_pdf` loads every page once to check the page tree, and `detect_pdf` loads page 0 and the
  pages it will process. On a damaged PDF that MuPDF repairs as it reads, that can move the
  repair earlier, so a page can read differently. (#96)
- `rapidocr_backend()` sets RapidOCR's `Global.log_level` to `"error"` on rapidocr 3.4 and later;
  `params` you pass take precedence. (#98)

### Fixed
- **Broken page trees:** a PDF whose page tree holds fewer pages than it claims raised
  `IndexError` at the first missing page; it is now read as the pages it holds, and detection
  warns with both counts. A page tree MuPDF cannot walk (a cycle, a missing `/Kids`, a `/Count`
  above the file's object count) raises `InvalidPdfError` before any page is detected, and the
  CLI reports it without a traceback. (#96)
- **Long, narrow scans:** a page or image frame RapidOCR cannot resize (over 2000 px long and
  more than about 118 times as long as it is wide) is skipped with a warning; RapidOCR's
  `ResizeImgError` used to stop the document. (#100)
- On Windows, a PDF that failed to open, or whose page tree MuPDF cannot walk, stayed locked for
  as long as the caller kept the error; the error no longer holds the file. (#96)

## [0.13.1] - 2026-10-31

Patch: one CLI fix. Detection output is identical to 0.13.0's.

### Fixed
- **Batch `--jobs` on Linux:** worker processes are spawned, as on macOS and Windows, instead of
  forked. Importing PyMuPDF 2.0 starts native threads, and a worker forked while they ran could
  hang. (#92)

## [0.13.0] - 2026-10-31

Minor, not patch: it adds public API (telemetry, garbled-word flags, `ScanConfig.line_support`,
`lines.rule_mask`, the model's sha256). Results keep `schema_version` 1. Each PR linked below has
the measurements behind its change. What it still gets wrong (CJK scans, handwriting, dashed and
steep strikes) is in the README's scope table.

### Highlights
- **Scans find more strikes:** thick 2 to 3 pt strikes, double strikes, strikes whose detected line
  runs a few px off the stroke, and short struck words between confirmed ones. Underlined text on
  a table rule is no longer read as struck, and a stroke over most of a word no longer strikes all
  of it.
- **Born-digital PDFs:** strike lines leaning up to 12 degrees, strikes of 3 to 6 pt and U+0336
  overlay text are found; baseline underlines, fraction bars, radical bars and gridlines behind a
  legend are no longer strikes; words from a broken font map are flagged.
- **Azure DI words:** the default calibration finds more strikes on printed scans (95.2% of the
  scanned-recovery benchmark's known strikes, 91% in 0.12.1).
- **RapidOCR:** the orientation classifier, which turned struck lines upside down, is off; stray
  CJK characters on English pages are removed.
- **Monitoring:** opt-in run telemetry, and `get_model_meta()` gives the model's sha256.
- **PyMuPDF 2.0** is supported.

### Added
- `telemetry=True` on `detect_pdf` and `detect_image_file`, and the CLI's `--telemetry FILE`: per
  page timing, candidates per tier and a `cnn_prob` histogram (`types.Telemetry`). (#73, #85)
- `res["garbled_words"]`, `garbled: True` on struck records, a warning per affected page and one
  `GarbledTextWarning`, for words a broken character map garbled; `on_garbled="drop"` (CLI
  `--on-garbled drop`) leaves them out of the text. (#62, #64, #85)
- `get_model_meta()["sha256"]`, and `external_data` for ONNX weights kept in separate files. (#71)
- `ScanConfig.line_support` (on by default) and the `line_support: True` record key. (#61)
- `lines.rule_mask()`: horizontal ruling in an ink mask. (#59)
- `benchmarks/frontier/`: a pre-registered benchmark against frontier models, with its results,
  harness and scorer. (#39 to #48, and commits on `main`)

### Changed
- The Azure DI calibration's confidence veto spares a word whose line strikes another word with
  strike geometry, and line support no longer reads confidences: DI reads a word under a crisp
  printed strike as clean text. On 472 printed pages with DI words, struck words found rose from
  82.2% to 88.9% against the release before this change, with 203 live words marked where 199
  were. (#87)
- On the private ruled-forms corpus, false strikes under the default DI calibration went from 113
  in 0.12.1 to 133 across this release (123 before #87), and from 25 to 26 with
  `ruled_forms(rescue_clean_chains=False)`; recall is unchanged.
- `ScanConfig.azure_di()` and `ScanConfig.confidence_free()` accept any other `ScanConfig` field as
  a keyword, as in `confidence_free(line_support=False)`; `ScanConfig(...)` has the DI
  calibration's confidence gating on unless given `confidence_gating=False`. (#87)
- `detect_pdf` with a `di_result` chooses the calibration per page: DI's on the pages the result
  covers, confidence-free on pages an `ocr` backend reads. (#86)
- `detect_scanned_image`, `analyze_scanned_page` and `scanned.classify_lines` default to
  `ScanConfig.confidence_free()`, like every other entry point without a `di_result`. For Azure DI
  words pass `config=ScanConfig.azure_di()`. (#67)
- On scans, a word is fully struck only when the stroke reaches its first and last character.
  Spans and full/partial can change; which words are struck does not. (#60)
- A vector record's `text` leaves out the U+0336 overlays on the characters it reports. (#56)
- `rapidocr_backend()` turns RapidOCR's orientation classifier off
  (`params={"Global.use_cls": True}` turns it back on) and removes lone CJK characters from pages
  without CJK text (`drop_lone_cjk=False` keeps them). (#53, #54)
- `ensure_model` reuses a cached meta only for the model it was verified with: it could pair a
  cached meta with another model, and a call with neither `meta_url` nor `meta` can now raise
  `ValueError`. (#80)
- New floors: `onnxruntime>=1.18`, `pillow>=10.1`. On Python 3.11+ `[markdown]` installs PyMuPDF
  2.0. (#49, #51)
- Only `v*` release tags publish to PyPI; a manual `publish.yml` run is a dry run. (#37, #39)

### Fixed
- **Scanned path:** strikes thicker than about 1.4 pt were dropped (#58), as were thick strikes
  through bold or large text (#69) and over words with few ascenders (#75), double strikes where
  only one line was found (#76), and strikes whose line ran a few px off the stroke (#70).
  Underlined text on a table rule read as struck (#59). With Azure DI words, the confidence veto
  could drop a double or thick strike the confidence rescue had passed (#83, #86). Words with no
  OCR confidence scored at most 0.60 under the DI calibration, `apply_cnn_verdict` could drop an
  `annot` record, and a caller-built line without `fill` could lose its words (#67).
- **Vector detector, missed:** strike lines leaning more than 2 degrees and strokes under 6 pt
  (#55), and U+0336 overlay text (#56, #85). **Read as strikes:** an underline through hidden text
  laid over an inserted letter (#55), baseline underlines in Courier and similar fonts (#57), and
  fraction bars, radical bars and gridlines behind a legend (#68).
- **OCR adapters:** one word box off the page from RapidOCR, Azure DI, Textract or Document AI
  lost the whole document (#53, #63); a blank Textract page refused the scan (#63); a page missing
  from `words_by_page` gave the wrong error (#67).
- **Model loading and tools:** `tools/export_model.py` could leave a new model beside an old meta,
  and `training/train_strikenet.py` checked its validation split only after training (#40).
- **PyMuPDF 2.0:** the flag detector's version guard could not read `2.0`; `[markdown]` could
  install without a usable pymupdf4llm; `clean_markdown()` printed to stdout. (#49, #51)
- **Docs:** the model card's thresholds, `SECURITY.md`'s network claim, the failing-page template,
  and the README on CJK scans and MuPDF's strikeout flag. (#40, #65, #66, #74)

### Removed
- `benchmarks/di_parity.py` and `benchmarks/ocr_backend_table.py`, which ran on nothing. (#72)
- The TestPyPI rehearsal. (#37)

### Restoring 0.12.1 behavior
Module constants are process-wide and outside the stability contract: they are here to compare
outputs, and may go.

| Change | Setting |
|---|---|
| Short-word line support (#61) | `line_support=False` on the preset, as `ScanConfig.confidence_free(line_support=False)` |
| Glyph-ink test (#59) | `glyph_ink=False` on the preset |
| Thick-strike cap and stem test (#58) | `lines.STEM_RATIO = 0`, `lines.MAX_STROKE_RUN_PX = 4`, `lines.RUN_SLACK_PX = 0` |
| Thick strikes through bold text (#69) | `lines.SOLID_GLYPHS = 0` |
| Stroke-row retry (#70) | `scanned.STROKE_SNAP_PX = 0` |
| Thick-bar retry (#75) | `scanned.INK_GAP_THICK_PX = 2` |
| Double-strike pairs (#76) | `scanned.PAIR_REACH = 0` |
| Rescued words get the retries (#83) | `scanned.RESCUE_RETRIES = False` |
| RapidOCR orientation classifier (#53) | `rapidocr_backend(params={"Global.use_cls": True})` |
| Lone CJK removal (#54) | `rapidocr_backend(drop_lone_cjk=False)` |
| The DI calibration in `detect_scanned_image` and `analyze_scanned_page` (#67) | `config=ScanConfig.azure_di()` |
| One calibration for every page with a partial `di_result` (#86) | `scan_config=ScanConfig.azure_di()` |

## [0.12.1] - 2026-10-08

Patch: the shipped StrikeNet's docs and the tooling around it. Nothing under `src/` changes except
the version, so detection output is identical to 0.12.0's.

### Fixed
- **The shipped StrikeNet weights were described as reproducible; they are not.** They are
  byte-identical in every release since 0.4.0 and predate `training/train_strikenet.py`; their
  training set, split and settings, and how their thresholds (0.85 / 0.15) were chosen, are not
  recorded in this repository or any release. The model card, `training/README.md` and the script
  now say so, and the model card says its scanned-path figures measure the whole pipeline, not the
  model alone. This corrects 0.9.0, which called the training script reproducible.
- The training script's `1 - alpha` recall floor holds only for crops exchangeable with the
  validation crops (the split is over crops, not documents); the docs called it distribution-free
  or guaranteed.
- `training/train_strikenet.py` could not export on torch 2.9 or later, and could write a model
  that calls every candidate struck when the validation split had too few struck crops. It now
  uses the TorchScript exporter, checks its dependencies before training, stops when the split
  has too few crops, and replaces the output model only with an export whose logits match the
  trained net's.
- A fresh clone could not fetch four of the ten corpus PDFs; `scanned_recovery.py` did not report
  the Azure DI figure for `confidence_free()`, and an entry without a DI result turned its totals
  into NaN.
- `CONTRIBUTING.md` named a `--checkpoint` flag (it is `--ckpt`), and under uv an editable install
  kept the version it was installed at.

### Added
- `training/train_strikenet.py` records what a model was trained on in the exported meta: a digest
  of the labeled set, the counts, the hyperparameters and seed, and the package and torch versions.
  `--batch`, `--lr` and `--seed` are flags.
- The scanned-recovery benchmark needs no Azure account: its cached Azure DI results are hosted,
  sha256-pinned, in the
  [`niles-liu/strikethrough-benchmark`](https://huggingface.co/datasets/niles-liu/strikethrough-benchmark)
  dataset.

### Changed
- The Gradio demo requires 0.12.0, and its deploy steps are updated.

### Security
- `.env`, where `benchmarks/prep_scanned_di.py` reads the Azure DI key, is git-ignored.

## [0.12.0] - 2026-10-08

A correctness release from an audit of the whole package; each fix has a regression test that fails
on 0.11.0. Several fixes change output, so read **Changed** first.

### Changed
- **Native strike spans are character-exact.** A strike maps onto the characters it crosses, read
  from the PDF's glyph boxes, and a word is a full strike only when every character is struck (it
  was 70% of its width). On the 10-document benchmark corpus (55.2k vector records) this removes
  41 records, adds 464 and corrects 1,140; the flag detector changes similarly. 99.8% of vector
  detections are still confirmed by the flag signal, and the per-document floor rose from 92.5% to
  99.6%.
- `detect_pdf(ocr=...)` defaults to `ScanConfig.confidence_free()`, like `detect_image_file`,
  `words_by_page=` and the CLI. `di_result=` keeps the Azure DI calibration;
  `scan_config=ScanConfig()` restores the old default.
- **The Azure DI calibration keeps crisp strikes.** Glyph ink on both sides of an in-band line now
  counts as strike geometry for 0.9.1's confidence veto. The default `di_result=` path keeps 91% of
  the scanned-recovery benchmark's known strikes, up from 28% (`confidence_free()`: 96%). On the
  private ruled-forms corpus false positives go from 109 to 113 by default; recall is unchanged.
- Arguments that were misread now raise at the call: a string, bool or float `pages=`; a
  `words_by_page` with keys outside the document, a flat list, or `di_result` alongside it;
  `ScanConfig` values outside [0, 1], NaN or strings; a non-finite `Word` box; calibration labels
  other than struck/clean, True/False or 1/0. `method`, `on_missing_ocr` and `dpi` are checked even
  when no page uses them, and `provenance_text` raises on a result without `markdown`.
- **CLI:** batch mode follows the form of the arguments, not the number of files matched; status
  lines go to stderr; usage errors exit 1, since argparse's 2 collided with "encrypted / OCR
  required"; an interrupted run exits 130; `--dpi`, `--jobs` and `--overlay-dpi` must be positive;
  plain output numbers pages from 1.

### Added
- `schema_version` on every `detect_pdf` and `detect_image_file` result and on the CLI's `--jsonl`
  error records (`types.SCHEMA_VERSION`, still 1).
- An open binary file is accepted wherever a path is.
- A warning on PDFs MuPDF had to repair to open.

### Fixed
- **Native path:** text inserted straight after a deletion (`~~December~~May`) was deleted with
  it; the edge of a filled shape (a watermark) and highlighter-wide lines read as strikes; a
  slightly slanted strike line was ignored (lines leaning up to 2 degrees are now tested); content
  turned upright by `/Rotate` was misread, and some scans were routed to the native path; passages
  joined across live text; a word straddling the page edge vanished; annotation colours came
  through as 1- or 4-tuples, and `/NoView` annotations were reported.
- **Scanned path and images:** image metadata of 72 or 96 dpi made strikes vanish (under 100 dpi
  now reads as absent); image frames skipped the luminance conversion, lost transparency and
  ignored EXIF rotation; wide integer arrays read as black; `cnn_prob` was missing instead of
  `None` on a crop too small to score; a concurrent `set_model_dir` or `ensure_model` could crash
  scoring; `score_struck` and `dump_crops` crashed on caller-built input.
- **Calibration:** manifest labels were all read as struck, and `threshold_for_precision` and
  `threshold_for_recall` could miss their targets on tied or rounded probabilities.
- **Cloud OCR adapters:** an Azure DI page range or Document AI shard put words on the wrong pages,
  and some word boxes were dropped silently (a page with words but no usable box now raises
  `ValueError`).
- **Word documents:** strikes from styles and document defaults, in text boxes and in Strict OOXML
  files were missed; a malformed file raises the documented `ValueError`.
- **CLI:** output to `-` was not UTF-8 on Windows; a batch held every result until the end, so an
  interrupt or a crashed worker lost the run (each line is now written as its file finishes, and a
  file that crashes its worker becomes one error line); several inputs ended in a traceback. A
  directory is read at its top level only, as the README now says; a `**` glob recurses, a glob
  keeps only documents, and `-` cannot be mixed with other inputs.
- `rapidocr_backend()` crashed on a page with no text, and the Gradio demo could not read scans.
- **Benchmarks:** `scanned_recovery.py` could score Azure DI words against the wrong pages.
  Re-measured: RapidOCR recovers 96% of the 2,170 known strikes, Azure DI 96% with
  `confidence_free()` and 91% by default. Ruled-forms false positives are 113 / 38 / 96 / 25
  (default / `ruled_forms()` / `rescue_clean_chains=False` / both).

### Security
- `ensure_model` could load unverified content: an empty digest skipped the check, and `meta_url`
  did not need `meta_sha256`. `meta_sha256` is now required with `meta_url`, every digest must be
  64 hex characters, and a model whose thresholds are not `0 <= p_lo <= p_hi <= 1` is refused.

### Known limits at release
The `~~` marker on partial strikes in strict CommonMark; vector detection of rotated content; the
flag detector's boundary character (`DecemberM`); struck words under 3 pt; baseline underlines in
Courier-like fonts and a strike over a long word's first letters (both fixed in 0.13.0); tracked
moves in Word.

## [0.11.0] - 2026-09-04

Minor, not patch: adds `ScanConfig.rescue_clean_chains`. Scanned output changes only on pages with
a coloured or grey ground (issue #15); predictions on the recovery benchmark are identical to
0.10.0's.

### Added
- `ScanConfig.rescue_clean_chains` (provisional, default `True`): `False` turns off the escape that
  passed a line whose words all OCR cleanly on to the CNN when the page looked pen-edited. It is
  separate from `ruled_forms()` and acts only with `confidence_gating` on. Ruled-forms false
  positives, recall 2/3 throughout: 109 by default, 38 with `ruled_forms()`, 92 with this
  switch, 25 with both.
- `benchmarks/confidence_veto.py --switches` scores all four combinations;
  `tests/test_corpus_flag_detector.py` runs the flag detector over the benchmark corpus when it is
  present.

### Fixed
- **A word on a grey or coloured highlight block read as struck (issue #15).** `std_crop` flattens
  a shaded ground to white, `to_gray_u8` uses luminance instead of a channel mean, and `ink_mask`
  falls back to a block-wise flatten when one threshold reads a whole block as ink. Issue #15's
  page: 41 false strikes to 2 by default. Recall on manufactured colour grounds rose from 0% to
  between 82% and 92%.
- **PyMuPDF 1.26.3 to 1.26.5 segfault in the flag detector** on ordinary PDFs. The floor is now
  `pymupdf>=1.26.6`, and on an older install the flag detector raises `RuntimeError` instead.

### Performance
- `lines.strike_lines` is about 1.5x faster, with output verified identical.

### Notes
- Issue #7 is closed with its residual false positives accepted. On degraded ruled forms, use the
  native path and keep the scanned path off, or treat its output as a review queue under
  `ScanConfig.ruled_forms(rescue_clean_chains=False)`.

## [0.10.0] - 2026-07-29

Minor, not patch: adds `ScanConfig.ruled_forms()`, `veto_printed_rules` and the `straightness` line
field. Nothing changes by default.

### Added
- `ScanConfig.ruled_forms()` (provisional, off by default) for degraded, heavily ruled scans (issue
  #7): drops detected lines that look like printed rules, solid or dead-straight. A real strike on
  a clean scan is solid and straight too, so it costs recall there. Ruled-forms false positives:
  113 to 42, recall unchanged.
- The `straightness` line field (`None` when not measurable) and
  `benchmarks/confidence_veto.py --ab`.

### Fixed
- Hand-built lines without `fill` passed to `classify_lines` were all vetoed, and an unmeasurable
  wobble read as dead-straight.
- `strike_lines` is 1.7 to 2.3x faster.

### Stability
`ruled_forms()`, `veto_printed_rules` and `straightness` are provisional and outside the v1.0
stability contract; pin exactly if you depend on them.

## [0.9.1] - 2026-07-26

Patch for scanned, tightly ruled tables read with Azure DI (issue #4): clean printed text was
flagged as struck. On a private 8-document ruled-table benchmark, high-confidence over-flags fell
from 95 to 22 with no real strike lost; the confidence-free (RapidOCR) path is unchanged.

### Fixed
- A solid full-width table rule could pass as a strike. The through-glyph ink test runs on every
  hit and rejects a long, solid in-band line with ink on one side only.
- **DI confidence veto:** `apply_cnn_verdict` turns a struck candidate OCR'd above
  `ScanConfig.max_clean_conf` that lacks strike geometry into `final=False`, `verdict="unsure"`,
  `conf_veto: True`. It applies only under the DI calibration, and never on confidence alone.

### Added
- `geom_corroborated` and `conf_veto` on scanned records; `benchmarks/confidence_veto.py`.

## [0.9.0] - 2026-07-06

Evidence and the model: operating points, calibration, crop export for labeling, a
digest-verified model loader, a training script, a demo and a public benchmark corpus. StrikeNet's
weights and model card are on Hugging Face
([`niles-liu/strikenet`](https://huggingface.co/niles-liu/strikenet)) and the demo is a Space
([`niles-liu/strikethrough-demo`](https://huggingface.co/spaces/niles-liu/strikethrough-demo)).

### Added
- `ScanConfig.recall_first()` and `ScanConfig.precision_first()`, and the `cnn_p_hi` / `cnn_p_lo`
  fields.
- `pdf_strikethrough.calibration`: `threshold_for_recall`, `threshold_for_precision`,
  `conformal_threshold` and `pr_curve`.
- `dump_crops` and the CLI's `--dump-crops DIR`: every scored crop as a PNG plus a `crops.jsonl`
  manifest, for labeling; a "contribute a failing page" issue template.
- `cnn.ensure_model(url, sha256, ...)`: downloads a model, verifies its sha256 before loading it,
  and points the loader at it.
- `training/train_strikenet.py` trains StrikeNet from a labeled crop set and exports ONNX plus its
  meta. It does not reproduce the shipped weights (see 0.12.1).
- `demo/app.py`, the Gradio demo.
- `benchmarks/fetch_corpus.py` and a public corpus of 10 regulatory redline PDFs (54.7k struck
  words, sha256-pinned), on which 99.8% of vector detections are confirmed by the flag signal.
  `benchmarks/scanned_recovery.py` scores the scanned path on rasterized redline pages: 95% (Azure
  DI) and 97% (RapidOCR) of 2,170 known strikes.

### Fixed
- `rapidocr_backend()` rejected valid RapidOCR 3.x installs; the demo crashed at startup; `.env`
  files were not git-ignored.

### Changed
- The README's figures now come from the reproducible corpus and benchmark above, replacing
  figures that could not be reproduced.

## [0.8.0] - 2026-07-06

Scale and armor.

### Added
- Batch CLI: several files, a directory or a glob, JSONL output, and `--jobs N` worker processes.
  An unreadable file gives an `error` line instead of ending the run.
- Dashed, dotted and flat bezier strikes are found in born-digital PDFs.
- Malformed-PDF regression tests, and tests for horizontal CJK strikes.

### Changed
- Scans above about 300 dpi are processed at 200 dpi, and a page is capped at 128 Mpix
  (downsampled with a warning).
- `markdown`, `clean_text` and `passages` read a two-column page column by column.

## [0.7.0] - 2026-07-05

Beyond PDFs.

### Added
- `detect_image_file`, and CLI support for PNG, JPEG and multi-page TIFF.
- `words_from_textract` and `words_from_docai`, and `detect_pdf(words_by_page=...)` for any
  provider's words, which defaults to `ScanConfig.confidence_free()`; the CLI's
  `--textract-result` and `--docai-result`.
- `strikethroughs_in_docx`: strike formatting and tracked deletions in Word documents
  (`tier="docx"`).
- `provenance_text` and `markdown.mark_provenance`: struck spans kept as `[deleted: ...]`
  markers, for RAG; the CLI's `--provenance`.

### Changed
- The CLI picks the input kind from the file extension, and `--dpi` defaults to an image's
  metadata.

## [0.6.0] - 2026-07-05

Annotations and evidence.

### Added
- `method="annot"` and `native_annot_strikes` (`tier="annot"`): `/StrikeOut` annotations, with
  their author, dates, colour and id (`annot_*`); `method="both"` includes them.
- `stroke_color` and `stroke_width` on vector records.
- `render_overlay`, `save_overlays` and the CLI's `--overlay`: struck pages with the strikes boxed.
- A `pdf_strikethrough` logger, silent by default, with DEBUG diagnostics.

### Changed
- `detect_pdf(native_method=...)` is now `method=`; the old name is a deprecated alias, and
  passing both with different values raises `ValueError`.
- CI adds weekly newest-dependency, lowest-bounds and torch-fallback jobs.

## [0.5.0] - 2026-07-05

Reachable and credible.

### Added
- `pages=` and `progress=` on `detect_pdf`: the result gains a `pages` key, and an out-of-range
  page raises `IndexError`.
- CLI parity with the API: `--version`, `--pages`, `--di-result`, `--scan-config`,
  `--fail-if-found` (exit 3), `-` for stdin and stdout, and JSON output with `schema_version`,
  `warnings` and evidence fields.
- Typed results (`pdf_strikethrough.types`, `py.typed`).
- `strikethroughs_in_pdf` still returns `[]` for a scanned page, but now warns.
- Runnable examples, a benchmark harness, `CONTRIBUTING.md`, `SECURITY.md` and `CITATION.cff`.

### Changed
- The version is single-sourced from `__version__`.
- A scanned word the CNN votes down no longer keeps `verdict="struck"`: `verdict == "struck"`
  implies `final is True`.
- CI lints, tests the built wheel, and runs Linux, macOS and Windows on Python 3.10 to 3.14;
  `publish.yml` tests the wheel before publishing.
- Native detection reads each page's words once: `native_page_strikes`, `native_flag_strikes` and
  `page_strikes` take an optional `words=`.

## [0.4.1] - 2026-07-05

### Fixed
- An encrypted PDF was refused even after `authenticate()`, and `open_pdf` leaked the document it
  refused.
- The CLI crashed writing output its console encoding could not hold.
- Page-source classification over-counted repeated images, and a page classified as scanned that
  has text but no OCR backend falls back to the native detector with a warning.
- Sloped scanned strikes are attributed to the right words.
- The flag path reported partials where a strike overshot into the next word, and invisible
  strokes (white, background-coloured or transparent) counted as strikes.
- Now raise: pixel-coordinate `Word` boxes, RapidOCR older than 3.2, and Azure DI results missing
  page dimensions.
- Model loading: `PDF_STRIKETHROUGH_MODEL_DIR` is read at load time, `torch.load` uses
  `weights_only=True`, and the meta's crop geometry is validated.
- Thin strikes were lost on 72 to 100 dpi scans, and 16-bit rasters were misread.

### Changed
- `import fitz` is replaced with `import pymupdf`.

## [0.4.0] - first public release

Strike detection in born-digital and scanned PDFs, with struck-aware markdown, clean text and
grouped passages, through a Python API and a CLI. Versions 0.1 to 0.3 were internal and never
published.
