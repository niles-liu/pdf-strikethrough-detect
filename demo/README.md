# Demo — Gradio app

A drag-and-drop web UI: drop in a PDF or a scan and see the detected strikethroughs boxed on the
page, plus the struck-aware markdown and the surviving clean text. It's the adoption front door —
the fastest way for someone to see what the package does without writing any code.

## Run locally

```bash
pip install -r demo/requirements.txt
python demo/app.py          # opens http://127.0.0.1:7860
```

Native (born-digital) PDFs work with no extra setup. Scanned PDFs and image files use RapidOCR
(pulled in by `requirements.txt`); if it isn't installed the demo still runs and just skips scanned
pages with a note.

## Deploy as a Hugging Face Space

1. Create a new **Gradio** Space.
2. Add `demo/app.py` as the Space's `app.py`, and `demo/requirements.txt` as its `requirements.txt`.
3. Push — the Space builds and serves the same UI.

Launch it alongside the StrikeNet model card (see `training/`) so the demo and the model land
together.

## Notes

- The hosted demo caps rendering at the first few pages (`MAX_PAGES` in `app.py`) to stay
  responsive; raise it for local use.
- Overlay colors match the CLI's `--overlay`: **red** = full strike, **orange** = partial.
