# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## 0.4.0 — first public release

- Detect struck-through (deleted) text in born-digital PDFs (exact vector/flag detection) and scanned pages (stroke geometry + OCR + ONNX CNN), with `~~struck~~` markdown, clean text, and grouped passages via a Python API and CLI.
