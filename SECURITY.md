# Security Policy

## Scope

`pdf-strikethrough-detect` parses **untrusted PDFs and images**. PDF and image parsing is done by
third-party libraries (PyMuPDF, Pillow, ONNX Runtime, and, for scanned pages, an OCR backend you
choose). Most parser-level vulnerabilities therefore live in those dependencies; keep them current.

This package makes **no network calls** except `cnn.ensure_model` (below), which fetches only a URL
you pass, and executes no code from the documents it reads. It does not accept or store
credentials. Password-protected PDFs are rejected with `EncryptedPdfError` rather than being
decrypted.

A few defensive choices worth knowing:

- The torch checkpoint loader (only reached if you point `PDF_STRIKETHROUGH_MODEL_DIR` at a `.pt`
  file) uses `torch.load(..., weights_only=True)`, so a malicious checkpoint cannot execute
  arbitrary pickle code.
- Any crop/pad geometry a model records (in its ONNX meta or `.pt` checkpoint) is checked against
  the code constants at load time.
- `cnn.ensure_model(url, sha256, ...)` (opt-in, for loading a model off the network) verifies the
  downloaded bytes against the caller-supplied sha256 *before* the model is written or loaded; a
  mismatch raises and nothing is kept, so a tampered host or a man-in-the-middle cannot swap in a
  different ONNX graph. It fetches only `http`/`https`/`file` URLs you pass explicitly.

## Resource limits

Rendering a scanned page allocates a raster of `page_size × dpi²` pixels, so a hostile
document (a huge mediabox, or an absurd requested DPI) could otherwise exhaust memory. Four
guards bound this:

- **Per-page raster budget (128 Mpix).** A page that would exceed it is rendered at a
  proportionally lower DPI (auto-downsampled) with a warning, rather than allocating the full
  raster. Output coordinates are page fractions, so results are unaffected.
- **High-DPI normalization.** Rasters above ~300 dpi are worked at 200 dpi internally (the
  detector's calibration point); extra resolution adds cost without accuracy.
- **Limits you set.** `max_pages=` and `max_mpix=` on `detect_pdf` and `detect_image_file`
  refuse a document with more pages to read, or more megapixels in all, with
  `LimitExceededError`, before any page is detected: a scanned PDF page counts as rendered, an
  image frame as stored. An image file's frames are read only up to the first one over a
  limit, so `max_pages=` bounds how many frame headers are read; with `max_mpix=` alone, a file
  of many small frames is read to its last header.
- **One TIFF frame at a time.** A multi-page TIFF file (a path, bytes, or an open file that can
  seek, read in place) is decoded one frame at a time, as detection reaches it, so its memory
  grows with its largest frame, not with its frame count. Other images, a PIL image you pass, and
  a TIFF with a frame over the decompression-bomb limit below are decoded whole first.

Image files are additionally subject to Pillow's decompression-bomb guard
(`PIL.Image.MAX_IMAGE_PIXELS`), which raises on a maliciously large image before it is decoded.
These limits target memory exhaustion, not a formal DoS guarantee; batch mode isolates per-file
failures so one malformed input cannot abort a run.

## Supported versions

Fixes are released against the latest published version on PyPI. Please upgrade before reporting.

## Reporting a vulnerability

Please report suspected vulnerabilities privately via
[GitHub Security Advisories](https://github.com/niles-liu/pdf-strikethrough-detect/security/advisories/new)
rather than opening a public issue. Include the PDF/image that triggers the problem (if it can be
shared), the package version, and the traceback. If the root cause is in a dependency, please also
report it upstream.

We aim to acknowledge reports within a few days and to coordinate disclosure once a fix is available.
