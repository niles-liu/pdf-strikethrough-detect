"""Provider-neutral OCR interface for the scanned word-level path.

The scanned classifier needs, per word: a box, the recognized text, and (ideally) an OCR
confidence. Different engines report these differently, so everything funnels through one small
type — ``Word`` — with adapters that convert each engine's native output into a list of them.

    Word.bbox is (x0, y0, x1, y1) in PAGE FRACTIONS (0..1), origin top-left.
    Word.confidence is 0..1 or None.

A "backend" is just a callable ``(image_ndarray) -> list[Word]``. Build one with
``rapidocr_backend()`` / ``tesseract_backend()``, or convert a pre-fetched Azure DI page with
``words_from_azure_di()``.

Confidence caveat: the scanned classifier's confidence thresholds are calibrated to Azure
Document Intelligence, whose struck words drop to 0.43-0.94 while clean text sits at 0.976-1.0.
Other engines score differently (RapidOCR clusters near 1.0). Use ``ScanConfig.confidence_free()``
when your engine's confidence is unreliable — it disables every confidence-dependent decision
(chain rejection, ink-fail rescue, and the confidence term of the evidence score), so detection
rests on geometry + the CNN alone.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    """One recognized word. bbox in page fractions (x0, y0, x1, y1); confidence 0..1 or None."""
    text: str
    bbox: tuple
    confidence: float | None = None

    def __post_init__(self):
        # bboxes are used as dict keys downstream — coerce lists/ndarrays to a hashable tuple
        object.__setattr__(self, "bbox", tuple(float(v) for v in self.bbox))


def _bbox_from_points(points, w, h):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h)


# --------------------------------------------------------------------------- Azure Document Intelligence

def words_from_azure_di(di_page) -> list[Word]:
    """Convert one Azure DI ``pages[i]`` dict (prebuilt-layout / read) into Words. DI word
    polygons are 8 numbers in the page's own units; page width/height give the fractions."""
    pw = di_page.get("width") or 1.0
    ph = di_page.get("height") or 1.0
    out = []
    for w in di_page.get("words", []):
        text = w.get("content", "")
        if not text.strip():
            continue
        poly = w.get("polygon") or []
        if len(poly) < 8:
            continue
        xs, ys = poly[0::2], poly[1::2]
        bbox = (min(xs) / pw, min(ys) / ph, max(xs) / pw, max(ys) / ph)
        out.append(Word(text, bbox, w.get("confidence")))
    return out


# --------------------------------------------------------------------------- RapidOCR (free, pip-only)

def rapidocr_backend(engine=None, **engine_kwargs):
    """Return an OCR backend using RapidOCR (ONNX, no system binary). Requires the `rapidocr`
    extra. `engine` may be a preconstructed ``rapidocr.RapidOCR``; otherwise one is built lazily.

    RapidOCR detection is phrase/line-grained, so its word boxes run coarser than a true
    word-level engine — good for locating struck regions, weaker for per-word char spans."""
    holder = {"eng": engine}

    def backend(image) -> list[Word]:
        if holder["eng"] is None:
            from rapidocr import RapidOCR
            holder["eng"] = RapidOCR(**engine_kwargs)
        h, w = image.shape[:2]
        res = holder["eng"](image, return_word_box=True)
        out = []
        for line in (res.word_results or []):
            for (text, score, box) in line:
                if text and str(text).strip():
                    out.append(Word(str(text), _bbox_from_points(box, w, h),
                                    float(score) if score is not None else None))
        return out

    return backend


# --------------------------------------------------------------------------- Tesseract (word-level boxes)

def tesseract_backend(lang="eng", config="", min_conf=0.0):
    """Return an OCR backend using Tesseract via pytesseract (needs the tesseract binary too;
    `pip install pdf-strikethrough-detect[tesseract]` then install tesseract). Gives genuine
    word-level boxes and per-word confidence (0..1) — closest to Azure DI granularity."""
    def backend(image) -> list[Word]:
        import pytesseract
        from pytesseract import Output
        h, w = image.shape[:2]
        data = pytesseract.image_to_data(image, lang=lang, config=config,
                                         output_type=Output.DICT)
        out = []
        for i, text in enumerate(data["text"]):
            if not text or not text.strip():
                continue
            conf = float(data["conf"][i])
            if conf < 0:                       # -1 = no confidence (layout blocks)
                conf = None
            else:
                conf /= 100.0
                if conf < min_conf:
                    continue
            x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            out.append(Word(text, (x / w, y / h, (x + bw) / w, (y + bh) / h), conf))
        return out

    return backend
