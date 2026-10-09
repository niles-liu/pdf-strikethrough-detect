# Examples

Runnable, self-contained scripts. Each one **generates its own sample PDF** in memory, so there is
nothing to download and every snippet is copy-paste runnable.

| Script | What it shows | Needs |
|---|---|---|
| [`native_quickstart.py`](native_quickstart.py) | Build a redline PDF, run the exact native detector, print struck words + clean text + markdown | base install |
| [`scanned_quickstart.py`](scanned_quickstart.py) | Rasterize that PDF into a synthetic "scan", run the geometry → OCR → CNN pipeline | `[rapidocr]` extra |
| [`overlay_quickstart.py`](overlay_quickstart.py) | Print each record's forensics (stroke colour and width, annotation author) and write an overlay image of each struck page | base install |
| [`rag_provenance.py`](rag_provenance.py) | Keep deleted text visible as `[deleted: …]` markers with `provenance_text` instead of dropping it | base install |

```bash
python examples/native_quickstart.py
pip install "pdf-strikethrough-detect[rapidocr]" && python examples/scanned_quickstart.py
```
