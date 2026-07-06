"""Strikethrough detection for Word ``.docx`` files — the redline sibling of the PDF path.

A .docx has no geometry (pagination is a render-time concern), so its struck text is read from
the markup, not from ink: a run carrying the ``w:strike`` / ``w:dstrike`` character format, and
tracked deletions (``w:del``, which move the text into ``w:delText`` and record who deleted it and
when — the same "who struck this, and when" forensics as a PDF ``/StrikeOut`` annotation).

Records share the package's struck-word schema but with ``tier="docx"`` and no ``bbox_frac`` /
``page`` (there is none); the paragraph index is reported as ``para`` instead. This is stdlib-only
(a .docx is a zip of XML) — no new dependency.

    import pdf_strikethrough as st
    for w in st.strikethroughs_in_docx("contract.docx"):
        print(w["para"], repr(w["chars"]), w["docx_change"], w.get("docx_author"))
"""
from __future__ import annotations

import io
import zipfile
from xml.etree import ElementTree as ET

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_OFF = {"false", "0", "off"}       # w:val values that turn a boolean run property OFF


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _on(el):
    """A boolean run property (w:strike/w:dstrike) is on when present unless w:val disables it."""
    return el is not None and (el.get(f"{_W}val") or "true").lower() not in _OFF


def _run_text(run):
    """All literal text under a run — ``w:t`` (live) and ``w:delText`` (tracked-deleted)."""
    return "".join(n.text or "" for n in run.iter() if _local(n.tag) in ("t", "delText"))


def _strike_format(run):
    """'strike' / 'dstrike' if the run carries that character format, else None."""
    rpr = run.find(f"{_W}rPr")
    if rpr is None:
        return None
    for kind in ("strike", "dstrike"):
        if _on(rpr.find(f"{_W}{kind}")):
            return kind
    return None


def _run_record(run, para, del_info):
    """A struck-word record for one run, or None if it is neither deletion nor strike-formatted."""
    text = _run_text(run)
    if not text.strip():
        return None
    fmt = _strike_format(run)
    if del_info is None and fmt is None:
        return None
    rec = {"para": para, "text": text, "chars": text, "char_span": (0, len(text)),
           "partial": False, "tier": "docx", "verdict": "struck", "final": True,
           "docx_change": "deletion" if del_info is not None else "format"}
    if fmt == "dstrike":
        rec["docx_double"] = True
    if del_info is not None:
        author, date, ident = del_info
        if author:
            rec["docx_author"] = author
        if date:
            rec["docx_date"] = date
        if ident:
            rec["docx_id"] = ident
    return rec


def _collect(elem, state, del_info, out):
    """Depth-first walk in document order, tracking the paragraph index and whether we are inside
    a tracked deletion (``w:del``, whose author/date apply to the runs it wraps)."""
    tag = _local(elem.tag)
    if tag == "p":
        state["para"] += 1
    elif tag == "del":
        del_info = (elem.get(f"{_W}author"), elem.get(f"{_W}date"), elem.get(f"{_W}id"))
    elif tag == "r":
        rec = _run_record(elem, state["para"], del_info)
        if rec is not None:
            out.append(rec)
        return                     # runs don't nest meaningfully; stop descending
    for child in elem:
        _collect(child, state, del_info, out)


def strikethroughs_in_docx(source) -> "list[dict]":
    """Struck-run records for a Word ``.docx`` (path or bytes), in document order.

    Catches both strike character formatting (``w:strike``/``w:dstrike``) and tracked deletions
    (``w:del`` — carrying ``docx_author``/``docx_date``). Each record has ``tier="docx"``, a
    ``para`` index (no page/geometry), ``docx_change`` ('format' | 'deletion'), and ``chars`` ==
    ``text`` (a run is struck as a whole). Reads only the main document body."""
    zf_source = io.BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray)) else source
    label = source if isinstance(source, str) else "<bytes>"
    try:
        with zipfile.ZipFile(zf_source) as zf:
            xml = zf.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as e:
        raise ValueError(f"not a readable .docx (no word/document.xml): {label}") from e
    out: list[dict] = []
    _collect(ET.fromstring(xml), {"para": -1}, None, out)
    return out
