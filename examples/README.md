# Examples

Runnable, self-contained scripts. Each one **generates its own sample PDF** in memory, so there is
nothing to download and every snippet is copy-paste runnable.

| Script | What it shows | Needs |
|---|---|---|
| [`native_quickstart.py`](native_quickstart.py) | Build a redline PDF, run the exact native detector, print struck words + clean text + markdown | base install |
| [`scanned_quickstart.py`](scanned_quickstart.py) | Rasterize that PDF into a synthetic "scan", run the geometry → OCR → CNN pipeline | `[rapidocr]` extra |

```bash
python examples/native_quickstart.py
pip install "pdf-strikethrough-detect[rapidocr]" && python examples/scanned_quickstart.py
```
