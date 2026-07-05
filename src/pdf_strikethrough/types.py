"""Typed shapes for the public record dicts, so ``StruckWord``/``DetectResult`` can be imported
and annotated against. These are the *documented* keys — the dicts are plain ``dict`` at runtime
(no validation, no cost), and ``total=False`` throughout because the key set is tier-dependent:

  native  (tier 'vector' / 'flag')  ->  carries ``coverage``; no CNN fields
  scanned (tier 'auto' / 'review')  ->  carries ``score`` + ``cnn_prob`` (+ ``cnn_agrees`` on
                                        auto records); ``conf`` is the OCR confidence

``page`` is present on every record that flows through ``detect_pdf``; the low-level
per-page detectors (``native_page_strikes`` etc.) already stamp it too. Consumers should treat
tier-specific keys as optional and branch on ``tier`` — or use ``.get(...)``.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Tuple, TypedDict

# (x0, y0, x1, y1) as page fractions in [0, 1], origin top-left.
BBoxFrac = Tuple[float, float, float, float]
# (start, end) character offsets into ``text`` that the strike covers.
CharSpan = Tuple[int, int]
Tier = Literal["vector", "flag", "auto", "review", "weak"]
Verdict = Literal["struck", "clean", "unsure"]


class StruckWord(TypedDict, total=False):
    """One struck-word record. Keys common to all tiers first, then tier-specific ones."""
    page: int
    text: str                    # the full OCR/extracted word
    chars: str                   # just the struck substring (== text on a full strike)
    char_span: CharSpan          # offsets of `chars` within `text`
    partial: bool                # True when only some characters are struck
    bbox_frac: BBoxFrac
    tier: Tier
    verdict: Verdict
    final: bool                  # the ship decision: is this word reported as struck?
    # native only
    coverage: float              # fraction of the word's width the stroke spans
    # scanned only
    score: float                 # layer-1 geometric score
    cnn_prob: Optional[float]    # StrikeNet probability (None if the crop was too small to score)
    cnn_agrees: Optional[bool]   # on 'auto' records: did the CNN confirm at p_hi? (None if unscored)
    conf: Optional[float]        # OCR word confidence, if the backend supplied one


class Passage(TypedDict, total=False):
    """A contiguous run of struck words grouped into one deletion passage."""
    page: int
    text: str
    n_words: int


class DetectResult(TypedDict, total=False):
    """The dict returned by :func:`pdf_strikethrough.detect_pdf`. The last three keys are present
    only when ``include_markdown=True`` (the default); ``pages`` is present only when a ``pages=``
    subset was requested."""
    source: Optional[str]
    page_count: int
    page_sources: List[str]      # "native" | "scanned" | "blank", aligned to processed pages
    pages: List[int]             # processed 0-based page indices (only when a subset was requested)
    words: List[StruckWord]
    n_struck_final: int
    warnings: List[str]
    markdown: str
    clean_text: str
    passages: List[Passage]


# Re-exported so ``from pdf_strikethrough.types import ...`` covers the whole surface.
__all__ = [
    "StruckWord", "Passage", "DetectResult",
    "BBoxFrac", "CharSpan", "Tier", "Verdict", "Any",
]
