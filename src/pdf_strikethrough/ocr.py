"""Provider-neutral OCR interface for the scanned word-level path.

The scanned classifier needs, per word: a box, the recognized text, and (ideally) an OCR
confidence. Different engines report these differently, so everything funnels through one small
type — ``Word`` — with adapters that convert each engine's native output into a list of them.

    Word.bbox is (x0, y0, x1, y1) in PAGE FRACTIONS (0..1), origin top-left.
    Word.confidence is 0..1 or None.

A "backend" is just a callable ``(image_ndarray) -> list[Word]``. Build one with
``rapidocr_backend()`` / ``tesseract_backend()``, or convert a pre-fetched cloud-OCR result with
``words_from_azure_di()`` (one page) / ``words_from_textract()`` / ``words_from_docai()`` (whole
document → ``{page: [Word]}`` for ``detect_pdf(..., words_by_page=...)``). None of the three
clouds flag strikethrough natively; these adapters add that layer on top.

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
        if len(self.bbox) != 4:
            raise ValueError(f"Word.bbox must be (x0, y0, x1, y1); got {self.bbox!r}")
        # a box in pixel coordinates would sail through the whole scanned pipeline and quietly
        # report every word clean — reject it at construction (mirrors cnn.word_crop_px's >1.5 gate)
        if max(abs(v) for v in self.bbox) > 1.5:
            raise ValueError(
                f"Word.bbox must be normalized page fractions in [0,1], got {self.bbox!r} "
                "(these look like pixel coordinates — divide by the image width/height)")


def _bbox_from_points(points, w, h):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h)


# --------------------------------------------------------------------------- Azure Document Intelligence

def words_from_azure_di(di_page) -> list[Word]:
    """Convert one Azure DI ``pages[i]`` dict (prebuilt-layout / read) into Words. DI word
    polygons are 8 numbers in the page's own units; page width/height give the fractions."""
    pw = di_page.get("width")
    ph = di_page.get("height")
    if not pw or not ph:
        # falling back to 1.0 turned the polygons (in inches) into inch-unit "fractions" far
        # outside [0,1] that silently detect nothing — fail loudly instead.
        raise ValueError(
            "Azure DI page is missing a non-zero 'width'/'height'; cannot normalize word "
            f"polygons to page fractions (got width={pw!r}, height={ph!r})")
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


# --------------------------------------------------------------------------- AWS Textract

def _as_dict(obj):
    """Coerce an SDK result object to a plain dict (boto3 already returns dicts; the DocAI SDK
    exposes ``.to_dict()``/``as_dict()``). Pass through anything already dict-like."""
    for attr in ("to_dict", "as_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            return fn()
    return obj


def words_from_textract(result) -> "dict[int, list[Word]]":
    """Convert an AWS Textract ``AnalyzeDocument``/``DetectDocumentText`` result into per-page
    Words: ``{0-based page: [Word, ...]}``. Textract ``WORD`` blocks already carry a normalized
    ``Geometry.BoundingBox`` (Left/Top/Width/Height in [0,1]) and a 0..100 ``Confidence`` (scaled
    to 0..1 here). Blocks tag their 1-based ``Page`` on multi-page input (absent ⇒ page 1).

    Textract does not flag strikethrough; feed the result to ``detect_pdf(pdf,
    words_by_page=...)`` (or ``detect_image_file(img, words_by_page=...)``) to add the strike
    layer. Its confidences aren't calibrated to the scanned classifier, so pass
    ``ScanConfig.confidence_free()`` (``detect_pdf`` defaults to it for ``words_by_page``)."""
    result = _as_dict(result)
    by_page: dict[int, list[Word]] = {}
    for block in result.get("Blocks", []):
        if block.get("BlockType") != "WORD":
            continue
        text = block.get("Text", "")
        if not text.strip():
            continue
        box = (block.get("Geometry") or {}).get("BoundingBox") or {}
        left, top = box.get("Left"), box.get("Top")
        width, height = box.get("Width"), box.get("Height")
        if None in (left, top, width, height):
            continue
        conf = block.get("Confidence")
        conf = float(conf) / 100.0 if conf is not None else None
        page = int(block.get("Page", 1)) - 1
        by_page.setdefault(page, []).append(
            Word(text, (left, top, left + width, top + height), conf))
    return by_page


# --------------------------------------------------------------------------- Google Document AI

def _docai_key(d, *names):
    """First present of the given keys — DocAI is camelCase over REST/JSON but snake_case once a
    proto goes through the Python SDK's ``Document.to_dict()``."""
    for n in names:
        if n in d:
            return d[n]
    return None


def _docai_token_text(full_text, layout):
    anchor = _docai_key(layout, "textAnchor", "text_anchor") or {}
    segs = _docai_key(anchor, "textSegments", "text_segments") or []
    parts = []
    for seg in segs:
        start = int(_docai_key(seg, "startIndex", "start_index") or 0)
        end = int(_docai_key(seg, "endIndex", "end_index") or 0)
        parts.append(full_text[start:end])
    return "".join(parts)


def _docai_bbox(layout, pw, ph):
    poly = _docai_key(layout, "boundingPoly", "bounding_poly") or {}
    norm = _docai_key(poly, "normalizedVertices", "normalized_vertices")
    if norm:
        xs = [float(v.get("x", 0.0)) for v in norm]
        ys = [float(v.get("y", 0.0)) for v in norm]
        return (min(xs), min(ys), max(xs), max(ys))
    verts = poly.get("vertices")
    if verts and pw and ph:                      # pixel vertices — normalize by page dimension
        xs = [float(v.get("x", 0.0)) / pw for v in verts]
        ys = [float(v.get("y", 0.0)) / ph for v in verts]
        return (min(xs), min(ys), max(xs), max(ys))
    return None


def words_from_docai(document) -> "dict[int, list[Word]]":
    """Convert a Google Document AI ``Document`` (REST JSON or ``document.to_dict()``) into
    per-page Words: ``{0-based page: [Word, ...]}``. Each page's ``tokens`` carry a ``layout``
    with a ``textAnchor`` (offsets into the document ``text``) and a ``boundingPoly``; normalized
    vertices are used directly, pixel vertices are divided by the page ``dimension``.

    Like Textract, DocAI does not flag strikethrough — feed the result to ``detect_pdf(pdf,
    words_by_page=...)`` and run confidence-free (its ``layout.confidence`` isn't calibrated to
    the scanned classifier)."""
    document = _as_dict(document)
    full_text = document.get("text", "") or ""
    by_page: dict[int, list[Word]] = {}
    for i, page in enumerate(document.get("pages", [])):
        dim = _docai_key(page, "dimension") or {}
        pw, ph = dim.get("width"), dim.get("height")
        words = []
        for tok in _docai_key(page, "tokens") or []:
            layout = _docai_key(tok, "layout") or {}
            text = _docai_token_text(full_text, layout)
            if not text.strip():
                continue
            bbox = _docai_bbox(layout, pw, ph)
            if bbox is None:
                continue
            conf = _docai_key(layout, "confidence")
            words.append(Word(text, bbox, float(conf) if conf is not None else None))
        by_page[i] = words
    return by_page


# --------------------------------------------------------------------------- RapidOCR (free, pip-only)

def _ver_tuple(v):
    """Best-effort (major, minor, patch) from a version string; non-numeric parts -> 0."""
    parts = []
    for p in str(v).split(".")[:3]:
        num = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts)


def _require_rapidocr_3_2(version):
    """Raise a clear error on rapidocr < 3.2, whose result object predates the ``word_results``
    nested shape this adapter reads (older installs throw opaque unpack errors or emit garbage)."""
    if _ver_tuple(version) < (3, 2):
        raise RuntimeError(
            f"rapidocr {version} is too old for this adapter — the word-box result shape "
            "changed in 3.2. Upgrade: pip install 'rapidocr>=3.2'")


def rapidocr_backend(engine=None, **engine_kwargs):
    """Return an OCR backend using RapidOCR (ONNX, no system binary). Requires the `rapidocr`
    extra (``>=3.2``). `engine` may be a preconstructed ``rapidocr.RapidOCR``; otherwise one is
    built lazily.

    RapidOCR detection is phrase/line-grained, so its word boxes run coarser than a true
    word-level engine — good for locating struck regions, weaker for per-word char spans."""
    holder = {"eng": engine}

    def backend(image) -> list[Word]:
        if holder["eng"] is None:
            import rapidocr
            _require_rapidocr_3_2(getattr(rapidocr, "__version__", "0"))
            from rapidocr import RapidOCR
            holder["eng"] = RapidOCR(**engine_kwargs)
        h, w = image.shape[:2]
        res = holder["eng"](image, return_word_box=True)
        if not hasattr(res, "word_results"):
            raise RuntimeError(
                "RapidOCR returned a result without 'word_results'; this adapter needs "
                "rapidocr>=3.2 (pip install 'rapidocr>=3.2')")
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
