"""Gradio demo for pdf-strikethrough-detect — drag in a PDF or scan, see what's struck.

Renders the first page(s) with detected strikes boxed (red = full, orange = partial) and shows the
struck-aware markdown and the surviving clean text. Native (born-digital) PDFs need nothing;
scanned PDFs and image files run RapidOCR when it's installed.

Run locally:
    pip install -r demo/requirements.txt
    python demo/app.py

Deploy as a Hugging Face Space: create a Gradio Space, add this file as `app.py` and
`demo/requirements.txt` as `requirements.txt` at the repo root of the Space.
"""
from __future__ import annotations

import os

import gradio as gr

import pdf_strikethrough as st

try:
    from pdf_strikethrough.ocr import rapidocr_backend
    _OCR = rapidocr_backend()
except Exception:                                    # noqa: BLE001 - OCR is optional in the demo
    _OCR = None

MAX_PAGES = 5                                        # keep the hosted demo responsive


def _detect(path, is_image):
    scan_config = st.ScanConfig.confidence_free()
    if is_image:
        return st.detect_image_file(path, ocr=_OCR, scan_config=scan_config)
    pages = list(range(MAX_PAGES))
    try:
        return st.detect_pdf(path, ocr=_OCR, scan_config=scan_config,
                             on_missing_ocr="skip", pages=pages)
    except IndexError:                               # fewer than MAX_PAGES pages
        return st.detect_pdf(path, ocr=_OCR, scan_config=scan_config, on_missing_ocr="skip")


def analyze(file):
    """Gradio handler: a file path in -> (overlay images, markdown, clean text, summary)."""
    if not file:
        return [], "", "", "Upload a PDF or image to begin."
    path = file.name if hasattr(file, "name") else file
    ext = os.path.splitext(path)[1].lower()
    is_image = ext in st.detect.IMAGE_SUFFIXES

    res = _detect(path, is_image)
    n_final = sum(1 for w in res["words"] if w.get("final"))
    warns = "\n".join(f"warning: {w}" for w in res.get("warnings", []))

    images = []
    if not is_image:
        for pg in st.render_overlay(path, result=res, dpi=130):
            images.append(pg["image"])
    summary = (f"{n_final} struck word(s) in {len(res.get('passages', []))} passage(s) across "
               f"{res['page_count']} page(s) [{', '.join(sorted(set(res['page_sources'])))}]."
               + (f"\n{warns}" if warns else ""))
    if _OCR is None and "scanned" in res.get("page_sources", []):
        summary += "\n(RapidOCR not installed — scanned pages were skipped.)"
    return images, res.get("markdown", ""), res.get("clean_text", ""), summary


def build():
    with gr.Blocks(title="pdf-strikethrough-detect") as demo:
        gr.Markdown("# pdf-strikethrough-detect\n"
                    "Detect struck-through (deleted) text in PDFs and scanned images. "
                    "Boxes: **red** = full strike, **orange** = partial.")
        with gr.Row():
            inp = gr.File(label="PDF or image", file_types=[".pdf", ".png", ".jpg", ".jpeg",
                                                             ".tif", ".tiff"])
            summary = gr.Textbox(label="Summary", lines=4)
        gallery = gr.Gallery(label="Detected strikes (overlay)", columns=1, height=520)
        with gr.Row():
            md = gr.Textbox(label="Struck-aware markdown (~~deleted~~)", lines=14)
            clean = gr.Textbox(label="Surviving clean text", lines=14)
        inp.change(analyze, inputs=inp, outputs=[gallery, md, clean, summary])
    return demo


if __name__ == "__main__":
    _ = tempfile  # reserved for future crop-download support
    build().launch()
