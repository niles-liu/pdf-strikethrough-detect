# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] — 2026-07-06

Prove it: the evidence & model program. This release ships the *machinery* — operating points,
calibration, an active-learning export, a digest-verified model loader, a reproducible training
script, and a demo — plus the **populated public benchmark corpus** those numbers stand on. All
pure-code, no new runtime dependencies. Asset-gated follow-ups that need external accounts (the
StrikeNet model card + weights on Hugging Face, a hosted Space) and the Azure-DI parity number are
published separately and are not part of this code drop.

### Added
- **Selectable operating points** (R-cal) — `ScanConfig.recall_first()` (legal / audit: never miss
  a deletion) and `ScanConfig.precision_first()` (RAG / indexing: never drop live text) set the
  CNN's struck/clean decision threshold, overridable via the new `cnn_p_hi` / `cnn_p_lo` fields.
- **Threshold calibration** (R-cal) — new `pdf_strikethrough.calibration`:
  `threshold_for_recall` / `threshold_for_precision` (extreme threshold meeting a target on a
  labeled set), `conformal_threshold` (split-conformal, distribution-free recall floor of
  `1 - alpha`), and `pr_curve`. Picks an operating point *from data* instead of a magic number.
- **Active-learning crop export** (R-active) — `dump_crops` / CLI `--dump-crops DIR` writes every
  crop the scanned pipeline scored (as a PNG) plus its verdict and geometry evidence to a
  `crops.jsonl` manifest for labeling; the labeled set feeds retraining and calibration. A
  "contribute a failing page" issue template routes bug reports into the same format.
- **Digest-verified model download** (R-hf) — `cnn.ensure_model(url, sha256, …)` downloads a
  StrikeNet model to a local cache, verifies its sha256 before anything is loaded (a tampered host
  can't swap the graph), and points the loader at it.
- **Reproducible training script** (R-hf) — `training/train_strikenet.py` trains StrikeNet from a
  `dump_crops` labeling set, calibrates `p_hi` / `p_lo` conformally, and exports ONNX + meta with
  the crop geometry recorded (so preprocessing can't drift). Closes the model reproducibility hole.
- **Gradio demo** (R-space) — `demo/app.py` (+ requirements/README): drag in a PDF or scan, see the
  strikes boxed plus the struck-aware markdown and surviving text. Deployable as a Hugging Face Space.
- **Corpus fetcher + populated benchmark corpus** (R-bench) — `benchmarks/fetch_corpus.py`
  downloads the manifest's PDFs into `corpus/` and verifies each sha256. `benchmarks/manifest.json`
  now lists **10 public regulatory redline PDFs** (US Copyright Office, FDIC, CEQ ×2, EPA ×3,
  California CCPA/CPPA, Gretna LA development code — 54.7k struck words, each sha256-pinned). The
  benchmark is one command to reproduce: `confirmation_rate.py` prints **99.8% of vector detections
  independently confirmed by the flag signal** (92.5–100% per document, 8 of 10 ≥99.9%).
- **Reproducible scanned-path benchmark** (R-bench) — `benchmarks/scanned_recovery.py` rasterizes
  the born-digital redline pages into image-only "scans" and scores the scanned path (OCR words +
  geometry + CNN) against the native detector's *known* strikes on the originals. On 3 documents /
  24 pages / 2,170 known strikes the scanned path recovers **95% (Azure DI) / 97% (RapidOCR)** of the
  strike set. `prep_scanned_di.py` is the one-time asset generator (rasterize + one Azure DI call,
  cached). This supersedes the DI-vs-original-pipeline `di_parity.py` / `ocr_backend_table.py`, which
  needed a vanished reference pipeline; both are kept as legacy for a genuinely scanned corpus.

### Fixed
- **RapidOCR ≥ 3.2 version gate** — `rapidocr_backend()` read `rapidocr.__version__`, which the
  `rapidocr` 3.x wheels don't set, so a valid install (3.9.1) was wrongly rejected as "too old". It
  now falls back to `importlib.metadata.version("rapidocr")`.

### Changed
- **README** documents operating points, the calibration surface, the model-improvement loop
  (dump-crops → label → train → load), and expands the handwritten-strike scope note (the main known
  gap) with the failing-page contribution path.
- **Headline confirmation-rate figures reconciled to the reproducible corpus** — the README and
  `native.py` "99.9–100% on 12 public redline PDFs (33k struck words)" (from the original private
  dev corpus) is now the measured "99.8% on 10 public redline PDFs (54.7k struck words; 92.5–100%
  per doc)". The domain description was corrected to the documents actually in the corpus (federal
  & state regulatory redlines + a municipal code; the previous list named court rules / procurement
  clauses / university policy, which the published corpus does not contain).
- **OCR-backend / DI-parity claims reconciled to the reproducible benchmark** — the README's
  "RapidOCR 100% covered / ~99%" table and "reproduces the original Azure-DI pipeline to 99.5%
  (1477 vs 1484)" (both from the original private validation) are now the measured "95% (DI) / 97%
  (RapidOCR) of the native strike set recovered", produced by `scanned_recovery.py`.

## [0.8.0] — 2026-07-06

Scale & armor: robustness on hostile and high-resolution input, dashed/curve-drawn strikes,
column-aware reading order, and a batch CLI. All pure-code — no new dependencies or shipped assets.

### Added
- **Batch / directory CLI mode** — `pdf-strikethrough detect` accepts several files, a directory
  (its top-level PDFs/images/.docx), or a glob (`*.pdf`, expanded internally so it works on shells
  that don't glob). Output is JSONL (`--jsonl PATH`, one result object per line; `--json` behaves as
  JSONL here); `--jobs N` spreads files across worker processes. A single unreadable file yields an
  `error` line instead of aborting the run. Per-file output flags (`--markdown`/`--overlay`/cloud
  results/`--pages`) stay single-file. The picklable worker lives in the new `_batch` module so it
  resolves under both the console script and `python -m`.
- **Dashed & curve-drawn native strikes** (R-dash) — `native.horiz_strokes` now chains collinear
  sub-`MIN_STROKE_LEN` segments (dashes/dots) into runs before the length gate, and recognizes flat
  (near-horizontal) cubic-bezier `"c"` path items as strike segments. A dashed or bezier strike is
  detected like a solid line; an isolated short tick still isn't.
- **Malformed-PDF regression suite** (R-hostile) — truncated / header-only / not-a-PDF / byte-flip
  fuzz inputs are asserted to fail cleanly (catchable error) or recover, never hang or crash.
- **Validated-scripts scope** (R-cjk slice 1) — horizontal CJK strikes are regression-tested
  (native + scanned); the README documents the validated script/layout matrix. Vertical writing
  modes remain out of scope.

### Changed
- **High-DPI normalization + pixel-budget guard** (R-highdpi / R-guard) — scanned rasters above
  ~300 dpi are worked at the 200-dpi calibration point (accuracy-neutral; extra resolution is pure
  cost), and a single page is capped at 128 Mpix — a hostile huge-mediabox page auto-downsamples
  with a warning instead of OOMing. Output boxes are page fractions, so both are invisible to
  results. This resolution normalization keeps CNN crops on-distribution by construction, superseding
  the previously-planned dpi-proportional-pad + ONNX re-export. Documented in `SECURITY.md`.
- **Column-aware reading order** (R-layout) — `markdown` / `clean_text` / `passages` now read
  down each column of a two-column page instead of interleaving across the gutter; narrow-column
  tables are left as rows (a struck table row stays one passage) and strikes across a hyphenated
  line break still group into one passage.

## [0.7.0] — 2026-07-05

Beyond PDFs: the same strike detection now reaches raster images, Word documents, and the AWS/Google
OCR ecosystems, plus an audit-preserving text mode for RAG. All pure-code — no new dependencies.

### Added
- **Image-file input** — `detect_image_file(source, ocr=..., dpi=None)` and CLI support for
  `.png/.jpg/.tiff` (incl. multi-page TIFF), for photos/faxes/scans that never were a PDF. Every
  frame is treated as a scanned page (the pipeline is already image-native); DPI is taken from an
  explicit `dpi=`, else the image metadata, else 200. Returns the same result shape as `detect_pdf`.
- **Cloud-OCR adapters** — `words_from_textract(result)` (AWS Textract) and `words_from_docai(document)`
  (Google Document AI), mirroring `words_from_azure_di`: each converts a pre-fetched result into
  `{0-based page: [Word]}`. Neither cloud flags strikethrough natively — feed the result to
  `detect_pdf(pdf, words_by_page=...)` (new provider-neutral parameter) or `detect_image_file`. CLI
  gains `--textract-result` / `--docai-result`. These confidences aren't calibrated to the scanned
  classifier, so `words_by_page` defaults `scan_config` to `ScanConfig.confidence_free()`.
- **DOCX detection** — `strikethroughs_in_docx(source)` reads strike character formatting
  (`w:strike`/`w:dstrike`) and tracked deletions (`w:del`, carrying `docx_author`/`docx_date`) from
  a Word document. Records use `tier="docx"` with a `para` index instead of `bbox_frac`/`page`.
  Stdlib-only (a .docx is a zip of XML) — no new dependency. The CLI routes `.docx` inputs here.
- **Provenance text for RAG** — `provenance_text(result, template="[deleted: {}]")` and
  `markdown.mark_provenance`: keep struck spans as `[deleted: …]` markers instead of removing them
  (contrast `clean_text`), so struck text entering a vector index is recorded-as-deleted rather than
  silently surfaced. CLI gains `--provenance PATH`; `examples/rag_provenance.py` demonstrates it.

### Changed
- CLI `detect` infers the input kind (PDF / image / .docx) from the file extension; `--dpi` now
  defaults to the image metadata for image files (still 200 for PDFs). `--json` carries the new
  `para`/`docx_*` evidence fields; `--overlay` warns and is skipped for non-PDF input.
- **Fixed a clobbered `## [0.5.0]` heading** in this changelog (the 0.6.0 edit dropped it, orphaning
  the 0.5.0 notes under the 0.6.0 section).

## [0.6.0] — 2026-07-05

Annotations & evidence: complete the *evidence story* for the signals already detected — an
explicit annotation pass with redline forensics, stroke color/width on vector records, a visual
overlay, a library logger, and CLI/API naming unification. Code-only, no new dependencies or
shipped assets.

### Added
- **Explicit `/StrikeOut` annotation pass** (`native_annot_strikes`, `method="annot"`, folded into
  `method="both"`). Reads the redlines Acrobat/Preview/editors write as annotation objects
  (QuadPoints over the struck text) — distinct from the vector drawings the geometry path reads and
  the font-attribute flag path — and snaps them onto the page words (partial spans supported).
  Records carry `tier="annot"` plus **flattened-redline forensics no extractor exposes**:
  `annot_author` (/T), `annot_created`, `annot_modified` (the "who struck this, and when"),
  `annot_color`, and `annot_id`. Hidden annotations paint no ink and are skipped.
- **Stroke color/width on vector records** (`stroke_color`, `stroke_width`). `get_drawings()`
  already carries them; the native geometry path now reports the dominant contributing stroke's
  paint (RGB in [0, 1]) and thickness (pt). Pen-color conventions (red = opposing counsel) are
  evidence in legal review. Both keys are surfaced in the CLI `--json` output alongside the
  annotation forensics.
- **Visual overlay** — `render_overlay(source, result=None, dpi=150, pages=None)` renders each
  struck page to a PIL image with strike boxes drawn (red = full, orange = partial); `save_overlays`
  writes them to disk; and the CLI gains `--overlay PATH` (+ `--overlay-dpi`). Useful for debugging,
  `ScanConfig` tuning, and the before/after documentation figure. No new dependency — PyMuPDF
  renders, Pillow draws (both already core).
- **Library logger.** The package attaches a `NullHandler` to the `pdf_strikethrough` logger (silent
  by default) and logs pipeline diagnostics at DEBUG — page routing, native detector method + record
  counts, OCR and geometry+CNN timings. `warnings` stays reserved for caller-facing hazards.

### Changed
- **`detect_pdf` native-page selector renamed `native_method` → `method`**, matching
  `strikethroughs_in_pdf`, `page_strikes`, and the CLI `--method`. `native_method` remains as a
  deprecated alias (emits a `DeprecationWarning`; passing both with different values raises
  `ValueError`) so 0.5.x callers keep working. The CLI is unchanged (`--method` already matched).
- CLI `--method` gains the `annot` choice; `--json` evidence now includes `stroke_color`/
  `stroke_width`/`annot_*`.
- **CI early-warning legs** (`.github/workflows/ci.yml`): a weekly `deps-latest` job against
  newest + pre-release deps (the `fitz`→`pymupdf` rename bit once); a `lowest-bounds` job that
  installs the exact `>=` floors on Python 3.10 to prove they're real; and a `torch-fallback` job
  that removes onnxruntime and loads a `.pt` through the CNN's torch fallback — a shipped code path
  CI never touched.

## [0.5.0] — 2026-07-05

Reachable & credible: new public API/CLI surface, inline typing, runnable examples, a reproducible
benchmark harness, and release/CI hardening.

### Added
- **`pages=` and `progress=` on `detect_pdf`.** `pages=` restricts work to a subset of 0-based
  page indices (negatives index from the end; out-of-range raises `IndexError`), so a 300-page
  scan no longer has to OCR every page or nothing; the result gains a `pages` key aligned to
  `page_sources`. `progress=` is a `progress(completed, total, page_index)` callback fired after
  each page, so long OCR+CNN runs aren't silent.
- **CLI feature parity with the API.** `--version`; `--pages 1-5,12` (1-based); `--di-result PATH`
  (use a pre-fetched Azure DI result from the CLI); `--scan-config auto|azure-di|confidence-free`;
  `--fail-if-found` (exit 3 for CI gating); `-` for stdin PDF and for stdout on
  `--json`/`--markdown`/`--clean-text`; per-page stderr progress on a TTY;
  `ArgumentDefaultsHelpFormatter` and documented exit codes. JSON output now carries
  `schema_version`, a `warnings` list, evidence fields (`coverage`/`score`/`cnn_prob`/`cnn_agrees`/
  `conf`) alongside each word, and is written with `ensure_ascii=False`.
- **Typed public surface.** New `pdf_strikethrough.types` module with `StruckWord`, `Passage`, and
  `DetectResult` TypedDicts (tier-dependent keys documented); ships `py.typed` so downstream type
  checkers see the annotations. Re-exported from the top-level package.
- **`strikethroughs_in_pdf` warns on scanned input.** A scanned page has no vector strikes, so the
  function still returns `[]` for it — but now emits a `UserWarning` naming the scanned pages
  (previously a silent `[]`, the package's most dangerous confusion).

- **Examples.** `examples/native_quickstart.py` and `examples/scanned_quickstart.py` — each builds
  its own sample PDF, so every quick-start snippet is copy-paste runnable with no assets to fetch.
- **Benchmark harness.** `benchmarks/` with a manifest-driven, sha256-verified corpus loader and
  three reproducible scripts: `confirmation_rate.py` (vector↔flag agreement — the headline native
  claim), `ocr_backend_table.py` (per-backend struck-region coverage vs an Azure DI reference), and
  `di_parity.py` (the "1477 vs 1484" DI-pipeline parity). The corpus (public PDFs) is downloaded
  locally, not committed.
- **Project meta.** `CONTRIBUTING.md`, `SECURITY.md` (the package parses untrusted PDFs),
  `CITATION.cff`, an API-surface table in the README, and a `[torch]` install extra documented.

### Changed
- **Version is single-sourced** from `pdf_strikethrough.__version__` via `dynamic = ["version"]`
  (no more lockstep bump of `pyproject.toml` + `__init__.py` at release time).
- **Scanned auto-tier verdict is now consistent with the ship decision.** A geometrically-"auto"
  word the CNN votes down (`final=False`) no longer keeps a misleading `verdict="struck"`; it
  reports the CNN's read (`clean`/`unsure`). Kept words are unchanged. Invariant: `verdict=="struck"`
  ⇒ `final is True`.
- **CI hardening.** Added a lint job (ruff), a build job that installs the built wheel and runs the
  tests against it, and an extras import-smoke job; the test matrix now spans Linux/macOS/Windows
  and Python 3.10–3.14 (3.14 + `Development Status`/`OS Independent`/`3 :: Only` classifiers added).
  `publish.yml` now tests the built wheel before publishing and adds a `workflow_dispatch` →
  TestPyPI rehearsal with explicit attestations. Added Dependabot for actions + pip.
- **Docs accuracy.** README record schema now lists `tier` and the scanned-only evidence fields;
  the CLI section documents every flag and exit code; the "RGB/float coerced" note moved to the
  function that actually coerces; `render_page_gray` documented as always-uint8.
- `MANIFEST.in`: dropped the dead `recursive-exclude test_docs` line; added `py.typed`.
- The `98%+` confirmation figure in `native.py` reconciled to the README's `99.9-100%` (12 PDFs,
  33k words) and pointed at `benchmarks/confirmation_rate.py`.
- **Native detection extracts each page's word list once.** Under `native_method="both"` a native
  page ran `get_text("words")` up to four times (page-source classification + vector detector +
  flag detector + the markdown word match). The detectors now take an optional `words=` argument
  and `detect_pdf` threads a single per-page extraction through all of them (down to two: the
  classifier's own probe plus one shared pass). `native_page_strikes`/`native_flag_strikes`/
  `page_strikes` gain the `words=` parameter; behavior is unchanged when it's omitted.

### Documented (not yet fixed)
- CNN crop geometry (`PAD_X`/`PAD_Y`) is fixed-pixel and calibrated at 200 dpi; it drifts
  off-distribution at other resolutions. The dpi-proportional fix needs a model re-export and is
  deferred to be done with the high-DPI normalization work (roadmap R-highdpi).
- Native detection is **horizontal (left-to-right) text only** — vertical writing modes and
  non-Latin scripts whose strikes run along a different axis are out of scope (`native.py` module
  docstring). Full support is roadmap R-cjk.

## [0.4.1] — 2026-07-05

### Fixed
- **Encryption gate now recognizes authenticated documents.** Detection gated on
  `needs_pass`, which stays `True` even after a successful `doc.authenticate(password)`, so the
  exact workflow the error message recommends still failed. Both entry points now gate on
  `is_encrypted` through a single shared helper, and `open_pdf` closes the freshly-opened
  document before raising `EncryptedPdfError` (previously it leaked, locking the file on Windows).
- **CLI no longer crashes writing non-cp1252 output.** All markdown/clean-text/JSON writes open
  with `encoding="utf-8"`, `sys.stdout`/`sys.stderr` are reconfigured with `errors="replace"`,
  and the `RuntimeError` open-failure handler is scoped to the open step only (mid-run
  onnxruntime/PyMuPDF errors are no longer mislabeled "cannot open FILE").
- **Page-source classification.** Clipped image rectangles are now unioned on a coarse boolean
  grid instead of summed (the same image placed three times no longer reads as 3× its coverage);
  a page is routed to "scanned" only when heavy image coverage coincides with invisible text or
  no content-stream drawings. A "scanned"-classified page that still has extractable text and no
  OCR backend falls back to the native detector with a warning instead of raising. A partial
  `di_result` now honors `on_missing_ocr="skip"`.
- **Sloped scanned strikes attribute to the right words.** The stroke-y is interpolated at each
  word's x-midpoint from the line endpoints instead of using one global bbox-center for the whole
  line, so a multi-word pen strike at a few degrees of slope no longer misses every word.
- **Native detection.** The flag path now applies the vector path's grazing guard after merging
  spans (no more spurious partials from overshoot into a neighbor word); invisible strokes
  (white / background-colored / zero-opacity lines) are skipped instead of confirmed as strikes.
- **OCR input safety.** Pixel-coordinate `Word` boxes raise instead of silently reporting
  all-clean; the RapidOCR adapter checks for the `>=3.2` result shape and raises a clear error on
  older installs; `words_from_azure_di` raises on missing/zero page dimensions.
- **Model loading.** `PDF_STRIKETHROUGH_MODEL_DIR` is read at load time (the documented
  "set it then import" override no longer no-ops); `torch.load` uses `weights_only=True`; the
  shipped `meta.json` crop/pad geometry is validated against the code constants.
- **Low-DPI and wide-dtype scans.** Scaled stroke-run/stitch thresholds are floored so 72–100 dpi
  scans keep thin strikes; 16-bit and other wide integer rasters are rescaled to 8-bit instead of
  saturating to all-white (`to_gray_u8`) or wrapping mod-256 (`std_crop`).

### Changed
- `import fitz` replaced with `import pymupdf` throughout (the `fitz` alias is deprecated upstream
  and collides with the abandoned `fitz` PyPI package).

## [0.4.0] — first public release

- Detect struck-through (deleted) text in born-digital PDFs (exact vector/flag detection) and scanned pages (stroke geometry + OCR + ONNX CNN), with `~~struck~~` markdown, clean text, and grouped passages via a Python API and CLI.

> Versions 0.1–0.3 were internal pre-release iterations and were never published to PyPI; 0.4.0 is
> the first public release.
