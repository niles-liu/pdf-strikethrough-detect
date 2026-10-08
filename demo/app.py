"""Gradio demo for pdf-strikethrough-detect — drag in a PDF or scan, see what's struck.

Renders the first page(s) with detected strikes boxed (red = full, orange = partial) and shows the
struck-aware markdown and the surviving clean text. Native (born-digital) PDFs need nothing;
scanned PDFs and image files run RapidOCR when it's installed.

Run locally:
    pip install -r demo/requirements.txt
    python demo/app.py

Deploy as a Hugging Face Space: create a Gradio Space, add this file as `app.py` and
`demo/requirements.txt` as `requirements.txt` at the repo root of the Space. Live instance:
https://huggingface.co/spaces/niles-liu/strikethrough-demo
"""
from __future__ import annotations

import os
import sys

import gradio as gr
import pymupdf
from PIL import Image

import pdf_strikethrough as st


# Build the OCR engine at startup, where a failure can be reported on the page, not on the first
# scanned upload (rapidocr_backend() imports lazily).
try:
    from rapidocr import RapidOCR

    from pdf_strikethrough.ocr import rapidocr_backend
    _OCR = rapidocr_backend(engine=RapidOCR())
except Exception as e:                               # noqa: BLE001 - OCR is optional in the demo
    print(f"OCR unavailable, scanned pages will be skipped: {type(e).__name__}: {e}",
          file=sys.stderr)
    _OCR = None
_NO_OCR = ("OCR is unavailable on this instance, so only born-digital PDFs work here: scanned "
           "pages and image files need RapidOCR "
           "(pip install 'pdf-strikethrough-detect[rapidocr]').")

# Keep the Space and the published model in lockstep: pull the HF-hosted StrikeNet (digest-verified
# before it is loaded) at startup, falling back to the packaged weights if the Hub is unreachable.
_MODEL_BASE = "https://huggingface.co/niles-liu/strikenet/resolve/main"
try:
    st.ensure_model(
        f"{_MODEL_BASE}/strike_verdict_cnn.onnx",
        "fac2c51baaa75ee782196bdfe7452638cb48c7deddb21163b1ac6a0a72ae4457",
        meta_url=f"{_MODEL_BASE}/strike_verdict_cnn.meta.json",
        meta_sha256="4388b14715bfb1f51b56bb6c463d8f5c0847533316297890e70ac0b2234405d4",
    )
except Exception:                                    # noqa: BLE001 - packaged model is the fallback
    pass

MAX_PAGES = 5                                        # keep the hosted demo responsive


def _detect(path, is_image):
    """Detect on at most MAX_PAGES PDF pages or image frames, so one upload can't monopolize the
    hosted demo."""
    scan_config = st.ScanConfig.confidence_free()
    if is_image:
        with Image.open(path) as img:
            frames = getattr(img, "n_frames", 1)
        if frames > MAX_PAGES:
            raise ValueError(f"the image has {frames} frames; this demo reads at most {MAX_PAGES}")
        return st.detect_image_file(path, ocr=_OCR, scan_config=scan_config)
    with pymupdf.open(path) as doc:
        pages = range(min(MAX_PAGES, doc.page_count))
    return st.detect_pdf(path, ocr=_OCR, scan_config=scan_config, on_missing_ocr="skip",
                         pages=pages)


def analyze(file):
    """Gradio handler: a file path in -> (overlay images, markdown, clean text, summary)."""
    if not file:
        return [], "", "", "Upload a PDF or image to begin."
    path = file.name if hasattr(file, "name") else file
    ext = os.path.splitext(path)[1].lower()
    is_image = ext in st.detect.IMAGE_SUFFIXES
    if is_image and _OCR is None:
        return [], "", "", _NO_OCR

    try:
        res = _detect(path, is_image)
    except Exception as e:                           # noqa: BLE001 - say why, not a bare "Error"
        return [], "", "", f"Detection failed: {type(e).__name__}: {e}"
    n_final = sum(1 for w in res["words"] if w.get("final"))
    warns = "\n".join(f"warning: {w}" for w in res.get("warnings", []))

    images = []
    if not is_image:
        for pg in st.render_overlay(path, result=res, dpi=130):
            images.append(pg["image"])
    read = len(res["page_sources"])                  # pages actually processed
    summary = (f"{n_final} struck word(s) in {len(res.get('passages', []))} passage(s) across "
               f"{read} page(s) [{', '.join(sorted(set(res['page_sources'])))}]."
               + (f" Only the first {read} of {res['page_count']} pages are read here."
                  if read < res["page_count"] else "")
               + (f"\n{warns}" if warns else ""))
    if _OCR is None and "scanned" in res.get("page_sources", []):
        summary += f"\n{_NO_OCR}"
    return images, res.get("markdown", ""), res.get("clean_text", ""), summary


def build():
    with gr.Blocks(title="pdf-strikethrough-detect") as demo:
        gr.Markdown("# pdf-strikethrough-detect\n"
                    "Detect struck-through (deleted) text in PDFs and scanned images. "
                    "Boxes: **red** = full strike, **orange** = partial."
                    + (f"\n\n**{_NO_OCR}**" if _OCR is None else ""))
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
    build().launch()
