# Security Policy

## Scope

`pdf-strikethrough-detect` parses **untrusted PDFs and images**. PDF and image parsing is done by
third-party libraries (PyMuPDF, Pillow, ONNX Runtime, and — for scanned pages — an OCR backend you
choose). Most parser-level vulnerabilities therefore live in those dependencies; keep them current.

This package makes **no network calls** and executes no code from the documents it reads. It does
not accept or store credentials. Password-protected PDFs are rejected with `EncryptedPdfError`
rather than being decrypted.

A few defensive choices worth knowing:

- The torch checkpoint loader (only reached if you point `PDF_STRIKETHROUGH_MODEL_DIR` at a `.pt`
  file) uses `torch.load(..., weights_only=True)`, so a malicious checkpoint cannot execute
  arbitrary pickle code.
- The shipped model's crop/pad geometry is validated against the code constants at load time.

## Resource limits

Rendering a scanned page allocates a raster of `page_size × dpi²` pixels, so a hostile
document — a huge mediabox, or an absurd requested DPI — could otherwise exhaust memory. Two
guards bound this:

- **Per-page raster budget (128 Mpix).** A page that would exceed it is rendered at a
  proportionally lower DPI (auto-downsampled) with a warning, rather than allocating the full
  raster. Output coordinates are page fractions, so results are unaffected.
- **High-DPI normalization.** Rasters above ~300 dpi are worked at 200 dpi internally (the
  detector's calibration point); extra resolution adds cost without accuracy.

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
