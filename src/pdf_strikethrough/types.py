"""Typed shapes for the public record dicts, so ``StruckWord``/``DetectResult`` can be imported
and annotated against. These are the *documented* keys — the dicts are plain ``dict`` at runtime
(no validation, no cost), and ``total=False`` throughout because the key set is tier-dependent:

  native  (tier 'vector' / 'flag' / 'annot')  ->  carries ``coverage``; no CNN fields
  scanned (tier 'auto' / 'review')  ->  carries ``score`` + ``cnn_prob`` (+ ``cnn_agrees`` on
                                        auto records); ``conf`` is the OCR confidence

``page`` is present on every record that flows through ``detect_pdf``; the low-level
per-page detectors (``native_page_strikes`` etc.) already stamp it too. Consumers should treat
tier-specific keys as optional and branch on ``tier`` — or use ``.get(...)``.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, TypedDict

# Version of the documented output shape (the keys below and their meaning), carried as
# ``schema_version`` on every ``DetectResult`` and CLI --json / --jsonl payload. Bumped when a key
# is removed, renamed or changes meaning; not for an added key (ignore unknown keys) or for
# detections.
SCHEMA_VERSION = 1

# (x0, y0, x1, y1) as page fractions in [0, 1], origin top-left.
BBoxFrac = Tuple[float, float, float, float]
# (start, end) character offsets into ``text`` that the strike covers.
CharSpan = Tuple[int, int]
Tier = Literal["vector", "flag", "annot", "auto", "review", "weak", "docx"]
Verdict = Literal["struck", "clean", "unsure"]


class StruckWord(TypedDict, total=False):
    """One struck-word record. Keys common to all tiers first, then tier-specific ones."""
    page: int
    text: str                    # the full OCR/extracted word ('vector': less the U+0336
                                 # overlays on the characters `char_span` reports)
    chars: str                   # just the struck substring (== text on a full strike)
    char_span: CharSpan          # offsets of `chars` within `text`
    partial: bool                # True when only some characters are struck
    bbox_frac: BBoxFrac
    tier: Tier
    verdict: Verdict
    final: bool                  # the ship decision: is this word reported as struck?
    # native only
    coverage: float              # vector: fraction of the word's width the stroke spans (with
                                 # U+0336 overlays, that its struck characters' boxes span);
                                 # flag/annot: fraction of its characters that are struck
    garbled: bool                # present (True) when a broken character map garbled the word:
                                 # `text` and `chars` are not what the page shows
    # native 'vector' only: forensics from the dominant contributing stroke (absent when
    # U+0336 overlays alone strike the word)
    stroke_color: Tuple[float, float, float]   # RGB in [0, 1]; unset paint reads as black
    stroke_width: float          # stroke line width (pt), or bar height for a filled-rect strike
    # native 'annot' only — /StrikeOut annotation forensics (present when the annotation supplies)
    annot_author: Optional[str]     # /T   "who struck this"
    annot_created: Optional[str]    # /CreationDate (PDF date string)
    annot_modified: Optional[str]   # /M           (PDF date string)  "...and when"
    annot_color: Optional[Tuple[float, float, float]]    # annotation stroke color, RGB in [0, 1]
    annot_id: Optional[str]         # /NM annotation name/id
    # docx only (tier 'docx') — no bbox/page; ``para`` is the paragraph index
    para: int
    docx_change: Literal["format", "deletion"]   # strike formatting vs a tracked w:del deletion
    docx_double: bool            # True for double-strike (w:dstrike)
    docx_author: Optional[str]   # tracked-deletion author (w:del/@w:author)
    docx_date: Optional[str]     # tracked-deletion timestamp (w:del/@w:date)
    docx_id: Optional[str]       # tracked-deletion id (w:del/@w:id)
    # scanned only
    score: float                 # layer-1 geometric score
    cnn_prob: Optional[float]    # StrikeNet probability (None if the crop was too small to score)
    cnn_agrees: Optional[bool]   # 'auto' records: did the CNN confirm at p_hi? (None if unscored)
    conf: Optional[float]        # OCR word confidence, if the backend supplied one
    geom_corroborated: bool      # in-band line with glyph ink on both sides (solid or shattered),
                                 # or, for a word only the pair test passed, a double strike's two
                                 # strokes: real-strike geometry, which the DI-confidence veto
                                 # spares and ScanConfig.line_support requires
    conf_veto: bool              # True when the DI-confidence veto downgraded this record
                                 # (OCR conf > max_clean_conf AND no corroborating geometry)
    line_support: bool           # True when ScanConfig.line_support kept this short word: the CNN
                                 # left it unconfirmed, but it has a stroke of its own and its line
                                 # strikes confirmed words on both sides of it


class Passage(TypedDict, total=False):
    """A contiguous run of struck words grouped into one deletion passage."""
    page: int
    text: str
    n_words: int
    bbox_frac: List[float]       # union box of the passage's words, page fractions (x0, y0, x1, y1)


class GarbledWord(TypedDict, total=False):
    """A word, struck or not, whose text a broken character map garbled: a font that maps its
    glyphs to the wrong characters, so ``text`` is not what the page shows."""
    page: int
    text: str                    # the word as extracted
    bbox_frac: BBoxFrac


class PageTelemetry(TypedDict, total=False):
    """One processed page (or image frame) in :class:`Telemetry`."""
    page: int
    source: str                  # as in page_sources
    seconds: Dict[str, float]    # "render", "ocr", "detect": those the page ran
    dpi: float                   # scanned: the resolution it was detected at
    ocr_words: int               # scanned: the words its OCR or word source gave (absent when
                                 # they came as an iterator with no length)
    tiers: Dict[str, int]        # candidates per tier: layer 1's on a scanned page ("weak"
                                 # included, which the CNN pass drops), the records on a native one
    final: int                   # its words reported struck
    skipped: bool                # present (True) when on_missing_ocr="skip" skipped it


class CnnProbHistogram(TypedDict):
    """The scanned records' ``cnn_prob``: ``counts[i]`` holds ``edges[i] <= p < edges[i + 1]``,
    the last bin ``p == 1.0`` too."""
    edges: List[float]
    counts: List[int]
    unscored: int                # crops too small to score (cnn_prob None)
    invalid: int                 # NaN or outside [0, 1]: a broken model's output


class Telemetry(TypedDict, total=False):
    """``telemetry=True``'s report of one detect_pdf / detect_image_file call. JSON-ready."""
    telemetry_version: int       # bumped when a key is removed, renamed or changes meaning
    package: str                 # the package version
    model: Optional[Dict[str, object]]   # get_model_meta()'s version and sha256 (+ external_data)
                                         # of the model that scored; None when none was loaded
    seconds: float               # the whole call
    pages: List[PageTelemetry]
    tiers: Dict[str, int]        # the pages' tiers, summed
    final: int
    cnn_prob: CnnProbHistogram


class DetectResult(TypedDict, total=False):
    """The dict returned by :func:`pdf_strikethrough.detect_pdf`. The ``markdown``,
    ``clean_text`` and ``passages`` keys are present only when ``include_markdown=True`` (the
    default); ``pages`` only when a ``pages=`` subset was requested; ``telemetry`` only with
    ``telemetry=True``."""
    schema_version: int          # == SCHEMA_VERSION of the release that produced it
    source: Optional[str]
    page_count: int
    page_sources: List[str]      # "native" | "scanned" | "blank", aligned to processed pages
    pages: List[int]             # processed 0-based page indices (only when a subset was requested)
    words: List[StruckWord]
    n_struck_final: int
    warnings: List[str]
    garbled_words: List[GarbledWord]   # every word a broken character map garbled, page order
    markdown: str
    clean_text: str
    passages: List[Passage]
    telemetry: Telemetry


# Re-exported so ``from pdf_strikethrough.types import ...`` covers the whole surface.
__all__ = [
    "StruckWord", "Passage", "GarbledWord", "DetectResult", "SCHEMA_VERSION",
    "Telemetry", "PageTelemetry", "CnnProbHistogram",
    "BBoxFrac", "CharSpan", "Tier", "Verdict",
]
