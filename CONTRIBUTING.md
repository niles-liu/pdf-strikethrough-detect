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

The suite is self-contained — it synthesizes its own PDFs and images, so no fixtures to download.
It runs in a couple of seconds on CPU.

## Linting

```bash
ruff check .
```

Config lives in `pyproject.toml` under `[tool.ruff]`. The project uses a compact style; the config
documents which default rules it opts out of and why.

## The CNN model

The strike-verdict CNN (StrikeNet, ~79k params) ships as ONNX inside the package
(`src/pdf_strikethrough/strike_verdict_cnn.onnx` + `.meta.json`). To regenerate the ONNX from a
trained PyTorch checkpoint:

```bash
pip install -e ".[torch]"
python tools/export_model.py --checkpoint path/to/strike_verdict_cnn.pt
```

The exported `meta.json` records the crop/pad geometry the model was trained with; the loader
asserts it matches the code constants in `cnn.py`, so a mismatched re-export fails loudly rather
than silently feeding off-distribution crops.

## Adding a test document

The most useful contribution is a real redline PDF the detector gets wrong. If you can share one
(public documents only, please), open an issue with the PDF and a note on which words are struck.
For scanned samples, `examples/scanned_quickstart.py` shows how a born-digital PDF is rasterized
into a synthetic scan for testing.

## PRs

- Keep changes focused; one concern per PR.
- Add a regression test for any bug fix or new behavior (the suite is where the project's
  correctness guarantees live).
- Run `pytest` and `ruff check .` before opening the PR — CI runs both across Linux/macOS/Windows
  and Python 3.10–3.14.
- Update `CHANGELOG.md` under the `## [Unreleased]` heading.
