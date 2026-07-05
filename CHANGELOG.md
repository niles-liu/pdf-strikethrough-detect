# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
