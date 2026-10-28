# Contributing

Thanks for your interest in `pdf-strikethrough-detect`. Bug reports, redline test documents, and
PRs are all welcome.

## Dev setup

```bash
git clone https://github.com/niles-liu/pdf-strikethrough-detect
cd pdf-strikethrough-detect
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,markdown]"
```

## Running the tests

```bash
pytest tests/ -q
```

The suite is self-contained: it synthesizes its own PDFs and images, so no fixtures to download.
It runs in well under a minute on CPU.

## Linting

```bash
ruff check .
```

Config lives in `pyproject.toml` under `[tool.ruff]`. The project uses a compact style; the config
documents which default rules it opts out of and why.

## The CNN model

The strike-verdict CNN (StrikeNet, ~79k params) ships as ONNX inside the package
(`src/pdf_strikethrough/strike_verdict_cnn.onnx` + `.meta.json`). The shipped weights cannot be
reproduced: their training data is not recorded
([model card](training/MODEL_CARD.md#provenance-and-training)). To train a replacement on a labeled
crop set and export it, see [`training/README.md`](training/README.md); to export an existing
PyTorch checkpoint to ONNX:

```bash
pip install -e ".[torch]" onnx       # torch's ONNX exporter imports onnx
python tools/export_model.py --ckpt path/to/strike_verdict_cnn.pt
```

## Adding a test document

The most useful contribution is a real redline PDF the detector gets wrong. Open a
["contribute a failing page"](.github/ISSUE_TEMPLATE/failing-page.md) issue (public documents
only, please); it also says what to send when the file can't be shared. For scanned samples,
`examples/scanned_quickstart.py` shows how a born-digital PDF is rasterized into a synthetic scan.

## PRs

- Keep changes focused; one concern per PR.
- Add a regression test for any bug fix or new behavior (the suite is where the project's
  correctness guarantees live).
- Run `pytest` and `ruff check .` before opening the PR. CI runs ruff once, and pytest on Linux
  (Python 3.10 to 3.14) and on Windows and macOS (3.10 and 3.13).
- Update `CHANGELOG.md` under the `## [Unreleased]` heading.
- **Before blaming a red check on your diff, look at which step failed.** A job that dies in
  `Set up job` at `Getting action download info` (`Service Unavailable`, `Bad Gateway` or
  `Internal Server Error`) never reached your code: it is a GitHub Actions outage. Re-run the
  failed jobs rather than reverting.
