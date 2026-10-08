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
(pulled in by `requirements.txt` through the package's `rapidocr` extra). The demo builds the OCR
engine once at startup; if that fails, the page says so at the top and the demo runs on born-digital
PDFs only, skipping scanned pages and declining image uploads with a note.

## Deploy as a Hugging Face Space

1. Create a new **Gradio** Space.
2. Add `demo/app.py` as the Space's `app.py`, and `demo/requirements.txt` as its `requirements.txt`.
3. Push — the Space builds and serves the same UI.
4. Check the startup log for `OCR unavailable`. If it names `libGL.so.1`, add a `packages.txt`
   holding `libgl1` to the Space (RapidOCR imports OpenCV, which needs it on Linux) and push again.

Launch it alongside the StrikeNet model card (see `training/`) so the demo and the model land
together.

## Notes

- The hosted demo reads at most `MAX_PAGES` pages of a PDF (the first ones) and declines images
  with more frames than that, to stay responsive; raise it in `app.py` for local use.
- Overlay colors match the CLI's `--overlay`: **red** = full strike, **orange** = partial.
