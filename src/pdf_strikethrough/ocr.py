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
Other engines score differently (RapidOCR clusters near 1.0), so the functions that take a
``ScanConfig`` default to ``ScanConfig.confidence_free()``, except ``detect_pdf`` with a
``di_result``. That config disables every confidence-dependent decision (chain rejection,
ink-fail rescue, the DI-confidence veto, line support's clean-text check, and the confidence term
of the evidence score), so detection rests on geometry + the CNN alone.
Pass ``ScanConfig.azure_di()`` when you give Azure DI words to ``detect_scanned_image``,
``analyze_scanned_page`` or ``scanned.classify_lines``.
"""
from __future__ import annotations

import math
import re
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
        if not all(math.isfinite(v) for v in self.bbox):
            raise ValueError(f"Word.bbox must be finite page fractions, got {self.bbox!r}")
        # a box in pixel coordinates would sail through the whole scanned pipeline and quietly
        # report every word clean — reject it at construction (mirrors cnn.word_crop_px's >1.5 gate)
        if max(abs(v) for v in self.bbox) > 1.5:
            raise ValueError(
                f"Word.bbox must be normalized page fractions in [0,1], got {self.bbox!r} "
                "(these look like pixel coordinates — divide by the image width/height)")
        x0, y0, x1, y1 = self.bbox             # corners put in order: a swapped box would match nothing
        object.__setattr__(self, "bbox", (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
        if self.confidence is not None:        # a plain float, so dump_crops can json.dump it
            object.__setattr__(self, "confidence", float(self.confidence))


def _bbox_from_points(points, w, h):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h)


def _clip_to_page(bbox):
    """A word box (page fractions) clamped to the page, or None when it is not finite. A box wholly
    past an edge becomes a zero-width or zero-height box on that edge, so the word's text, which
    the OCR engine read from the page, is kept. `rapidocr_backend()` and the cloud adapters (through
    `_page_words`) clamp the boxes an engine reports, since `Word` refuses a box past 1.5 and one
    refused box loses the whole document; a `Word` built directly keeps that check."""
    if not all(math.isfinite(v) for v in bbox):
        return None
    return tuple(min(max(v, 0.0), 1.0) for v in bbox)


def _page_words(items):
    """Words from one page's (text, box, confidence) triples, each box in page fractions as a cloud
    service gave it: clamped to the page, or dropped when it is not finite. When most of the page's
    boxes lie past 1.5 they are left as given, so `Word` refuses them with its pixel-coordinates
    error."""
    items = [it for it in items if all(math.isfinite(v) for v in it[1])]
    # A stray box past the edge is one word among a page's (RapidOCR's ran to x 1.57). A box in the
    # wrong units, pixels or inches where page fractions belong, puts nearly every word on the page
    # past 1.5, so that page still fails loudly.
    clamp = 2 * sum(max(abs(v) for v in box) > 1.5 for _, box, _ in items) <= len(items)
    return [Word(text, _clip_to_page(box) if clamp else box, conf) for text, box, conf in items]


# --------------------------------------------------------------------- Azure Document Intelligence

def words_from_azure_di(di_page) -> list[Word]:
    """Convert one Azure DI ``pages[i]`` dict (prebuilt-layout / read) into Words. DI word
    polygons are in the page's own units; page width/height give the fractions. Polygons are read
    in the shapes the REST API and the Python SDKs emit; a page whose words carry no usable polygon
    raises ValueError. A word box that runs past the edge of the page is clamped to it; a word
    placed wholly off the page keeps its text, on a zero-width box at the edge. A page where most
    word boxes lie past 1.5 page widths or heights, as with polygons in the wrong units, still
    raises ValueError.

    Only ``detect_pdf(di_result=...)`` applies the DI calibration by default: pass
    ``config=ScanConfig.azure_di()`` when you give these words to ``detect_scanned_image``,
    ``analyze_scanned_page`` or ``scanned.classify_lines``."""
    pw = di_page.get("width")
    ph = di_page.get("height")
    if not pw or not ph:
        # a fallback of 1.0 would leave the polygons in inches, "fractions" far outside [0, 1]
        # that silently detect nothing: fail loudly instead
        raise ValueError(
            "Azure DI page is missing a non-zero 'width'/'height'; cannot normalize word "
            f"polygons to page fractions (got width={pw!r}, height={ph!r})")
    items, unreadable = [], 0
    for w in di_page.get("words", []):
        text = w.get("content", "")
        if not text.strip():
            continue
        poly = w.get("polygon") or []
        if all(isinstance(v, (int, float)) for v in poly):     # REST JSON: flat numbers
            pts = list(zip(poly[0::2], poly[1::2])) if len(poly) % 2 == 0 else []
        else:                                                  # SDKs: {x, y} or [x, y] points
            pts = []
            for p in poly:
                xy = (p.get("x"), p.get("y")) if hasattr(p, "get") else p
                if not isinstance(xy, (list, tuple)) or len(xy) != 2 or None in xy:
                    pts = []
                    break
                pts.append(tuple(xy))
        if len(pts) < 4:
            unreadable += 1
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        items.append((text, (min(xs) / pw, min(ys) / ph, max(xs) / pw, max(ys) / ph),
                      w.get("confidence")))
    out = _page_words(items)
    unreadable += len(items) - len(out)                       # a box that is not finite
    if unreadable and not out:
        raise ValueError(f"none of the {unreadable} Azure DI word(s) on this page has a polygon "
                         "this adapter can read (expected 8 numbers or 4 {x, y} points)")
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
    to 0..1 here). Blocks tag their 1-based ``Page`` on multi-page input (absent ⇒ page 1). Each
    page's ``PAGE`` block makes it a key, so a page with no usable words, such as a blank scanned
    page, maps to an empty list, as in ``words_from_docai()``. A box that runs past the edge of the
    page is clamped to it, and a page where most boxes lie past 1.5 raises ValueError, as in
    ``words_from_azure_di()``.

    Textract does not flag strikethrough; feed the result to ``detect_pdf(pdf,
    words_by_page=...)`` (or ``detect_image_file(img, words_by_page=...)``) to add the strike
    layer. Its confidences aren't calibrated to the scanned classifier, so pass
    ``ScanConfig.confidence_free()`` (``detect_pdf`` defaults to it for ``words_by_page``)."""
    result = _as_dict(result)
    items_by_page: dict[int, list] = {}
    for block in result.get("Blocks", []):
        kind = block.get("BlockType")
        if kind not in ("PAGE", "WORD"):
            continue
        # a blank page has a PAGE block and no WORD blocks; left out, detect_pdf would take it
        # for a page the OCR skipped
        items = items_by_page.setdefault(int(block.get("Page", 1)) - 1, [])
        if kind == "PAGE":
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
        items.append((text, (left, top, left + width, top + height), conf))
    return {page: _page_words(items) for page, items in items_by_page.items()}


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
    if verts:                                    # pixel vertices — normalize by page dimension
        if not pw or not ph:
            raise ValueError(
                "Document AI token has pixel vertices but its page has no 'dimension' "
                f"width/height to normalize them by (got width={pw!r}, height={ph!r})")
        xs = [float(v.get("x", 0.0)) / pw for v in verts]
        ys = [float(v.get("y", 0.0)) / ph for v in verts]
        return (min(xs), min(ys), max(xs), max(ys))
    return None


def words_from_docai(document) -> "dict[int, list[Word]]":
    """Convert a Google Document AI ``Document`` (REST JSON or ``document.to_dict()``) into
    per-page Words: ``{0-based page: [Word, ...]}``. Each page's ``tokens`` carry a ``layout``
    with a ``textAnchor`` (offsets into the document ``text``) and a ``boundingPoly``; normalized
    vertices are used directly, pixel vertices are divided by the page ``dimension`` (ValueError
    without one). Pages are keyed by their 1-based ``pageNumber`` when present, else by position.
    A box that runs past the edge of the page is clamped to it, and a page where most boxes lie
    past 1.5 raises ValueError, as in ``words_from_azure_di()``.

    Like Textract, DocAI does not flag strikethrough — feed the result to ``detect_pdf(pdf,
    words_by_page=...)`` and run confidence-free (its ``layout.confidence`` isn't calibrated to
    the scanned classifier)."""
    document = _as_dict(document)
    full_text = document.get("text", "") or ""
    by_page: dict[int, list[Word]] = {}
    for pos, page in enumerate(document.get("pages", [])):
        num = _docai_key(page, "pageNumber", "page_number")
        try:
            i = int(num) - 1 if num is not None and int(num) >= 1 else pos
        except (TypeError, ValueError):
            i = pos
        dim = _docai_key(page, "dimension") or {}
        pw, ph = dim.get("width"), dim.get("height")
        items = []
        for tok in _docai_key(page, "tokens") or []:
            layout = _docai_key(tok, "layout") or {}
            text = _docai_token_text(full_text, layout)
            if not text.strip():
                continue
            bbox = _docai_bbox(layout, pw, ph)
            if bbox is None:
                continue
            conf = _docai_key(layout, "confidence")
            items.append((text, bbox, float(conf) if conf is not None else None))
        by_page[i] = _page_words(items)
    return by_page


# ----------------------------------------------------------------------- RapidOCR (free, pip-only)

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


# Merged under the caller's `params` by rapidocr_backend(). RapidOCR's orientation classifier (on
# by default) turns struck, underlined and some plain lines upside down, and they then read as stray
# letters or not at all.
_RAPIDOCR_DEFAULT_PARAMS = {"Global.use_cls": False}


# CJK letters and ideographs. CJK punctuation (U+3000-303F but for U+3005), the katakana double
# hyphen and middle dot (U+30A0, U+30FB) and fullwidth forms (U+FF00-FF65) are left out: a misread
# there stands for punctuation or a Latin letter.
_CJK_RANGES = (
    (0x1100, 0x11FF),       # Hangul jamo
    (0x2E80, 0x2FDF),       # CJK and Kangxi radicals
    (0x3005, 0x3005),       # the ideographic iteration mark
    (0x3040, 0x309F),       # hiragana
    (0x30A1, 0x30FA),       # katakana
    (0x30FC, 0x30FF),       # katakana length and iteration marks
    (0x3100, 0x318F),       # bopomofo, Hangul compatibility jamo
    (0x31A0, 0x31FF),       # bopomofo extended, CJK strokes, katakana extensions
    (0x3400, 0x4DBF),       # CJK Extension A
    (0x4E00, 0x9FFF),       # CJK Unified Ideographs
    (0xA960, 0xA97F),       # Hangul jamo extended A
    (0xAC00, 0xD7FF),       # Hangul syllables, jamo extended B
    (0xF900, 0xFAFF),       # CJK compatibility ideographs
    (0xFF66, 0xFFDC),       # halfwidth katakana and Hangul
    (0x20000, 0x3FFFF),     # CJK Extensions B and later
)
_CJK_CLASS = "[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _CJK_RANGES) + "]"
_CJK = re.compile(_CJK_CLASS)
_CJK_PAIR = re.compile(_CJK_CLASS + "{2}")


def _has_cjk_text(words):
    """True when two CJK characters stand next to each other, in one word or across two words in
    reading order. RapidOCR returns each Han character as a word of its own, so CJK text reads as
    runs of one-character words, while its misreads on a Latin page stand alone."""
    prev_cjk = False
    for w in words:
        text = w.text.strip()
        if _CJK_PAIR.search(text) or (prev_cjk and text and _CJK.match(text)):
            return True
        prev_cjk = bool(text) and _CJK.match(text[-1]) is not None
    return False


def _drop_lone_cjk(words):
    """Words with every CJK character removed, when no two CJK characters on the page stand next
    to each other. One inside a word becomes a space, which keeps the word's characters where they
    sit in its box; a word left with no text is dropped. A page with CJK text is returned as is."""
    if not any(_CJK.search(w.text) for w in words) or _has_cjk_text(words):
        return words
    out = []
    for w in words:
        if not _CJK.search(w.text):
            out.append(w)
        elif text := _CJK.sub(" ", w.text).strip():
            out.append(Word(text, w.bbox, w.confidence))
    return out


def rapidocr_backend(engine=None, *, drop_lone_cjk=True, **engine_kwargs):
    """Return an OCR backend using RapidOCR (ONNX, no system binary). Requires the `rapidocr`
    extra (``>=3.2``). The engine is built lazily as ``RapidOCR(**engine_kwargs)``, with
    ``params={"Global.use_cls": False}`` merged under any ``params`` you pass: RapidOCR's
    text-line orientation classifier is off, since it turns struck and underlined lines upside
    down. Your ``params`` keys win, so ``params={"Global.use_cls": True}`` turns it back on. The
    default goes in as a param, so it also overrides ``use_cls`` in a ``config_path`` file. With
    the classifier off, a page scanned upside down reads as stray letters: turn it upright first,
    or turn the classifier back on.

    `engine` may be a preconstructed ``rapidocr.RapidOCR``, used as is: the default above does not
    apply to it, so build it with ``RapidOCR(params={"Global.use_cls": False})``.

    RapidOCR's default recognizer reads Chinese and Japanese as well as English, and on a Latin
    page it can read a small mark as a CJK character (a table's footnote marker ``(a)`` as ``(斗)``,
    an ``=`` as ``二``). So on a page where no two CJK characters (Han, kana, Hangul, bopomofo) come
    one right after the other in reading order (inside a word, or the last of one word and the
    first of the next, across a line break too; a digit in between breaks the run), each one is
    removed: a word left empty is dropped, and one inside a word becomes a space. Real CJK goes too
    when it stands alone like that: ``100 円``, ``5 個`` and ``7 元`` lose the unit, and
    ``Article 1 (第1条)`` reads ``( 1 )``. A page with CJK text, where the characters come in runs, is
    returned as read. CJK punctuation and fullwidth forms are kept. Pass ``drop_lone_cjk=False`` to
    keep every character; it applies to an `engine` you pass too. The CLI's ``--ocr rapidocr``
    always removes them.

    A word box that runs past the edge of the page is clamped to it; a word placed wholly off the
    page keeps its text, on a zero-width box at the edge. RapidOCR detection is phrase/line-grained,
    so its word boxes run coarser than a true word-level engine: good for locating struck regions,
    weaker for per-word char spans."""
    if engine is None:
        engine_kwargs["params"] = {**_RAPIDOCR_DEFAULT_PARAMS,
                                   **(engine_kwargs.get("params") or {})}
    holder = {"eng": engine}

    def backend(image) -> list[Word]:
        if holder["eng"] is None:
            import rapidocr
            version = getattr(rapidocr, "__version__", None)
            if not version:                        # rapidocr 3.x doesn't set a module __version__
                import importlib.metadata as _md
                try:
                    version = _md.version("rapidocr")
                except _md.PackageNotFoundError:
                    version = "0"
            _require_rapidocr_3_2(version)
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
            if line and isinstance(line[0], str):     # a bare triple: rapidocr's "no text"
                line = (line,)
            for (text, score, box) in line:
                if not (text and str(text).strip()):
                    continue
                # RapidOCR can place a word partly or wholly off the page (seen at x 0.78-1.57);
                # Word would refuse that box and, with it, the whole document
                bbox = _clip_to_page(_bbox_from_points(box, w, h))
                if bbox is not None:
                    out.append(Word(str(text), bbox,
                                    float(score) if score is not None else None))
        return _drop_lone_cjk(out) if drop_lone_cjk else out

    return backend


# -------------------------------------------------------------------- Tesseract (word-level boxes)

def tesseract_backend(lang="eng", config="", min_conf=0.0):
    """Return an OCR backend using Tesseract via pytesseract (needs the tesseract binary too;
    `pip install "pdf-strikethrough-detect[tesseract]"`, then install tesseract). Gives genuine
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
