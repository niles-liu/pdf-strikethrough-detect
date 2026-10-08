"""Strikethrough detection for Word ``.docx`` files — the redline sibling of the PDF path.

A .docx has no geometry (pagination is a render-time concern), so its struck text is read from
the markup, not from ink: a run carrying the ``w:strike`` / ``w:dstrike`` character format —
directly, or through its character style, its paragraph style or the document defaults — and
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
import os
import zipfile
import zlib
from xml.etree import ElementTree as ET

_OFF = {"false", "0", "off"}       # w:val values that turn a boolean run property OFF
_KINDS = ("strike", "dstrike")


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _attr(el, name):
    """An attribute by local name, so Strict OOXML (another namespace) reads like transitional."""
    for key, value in el.attrib.items():
        if _local(key) == name:
            return value
    return None


def _child(el, name):
    """The first child element with this local name, or None."""
    if el is None:
        return None
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _toggles(rpr):
    """``{kind: True/False}`` for the strike toggles a run-properties element sets explicitly; a
    property it leaves out is absent from the dict (inherit), unlike one set off."""
    out = {}
    for kind in _KINDS:
        el = _child(rpr, kind)
        if el is not None:
            out[kind] = (_attr(el, "val") or "true").lower() not in _OFF
    return out


def _resolve(styles, style_id):
    """``{kind: bool}`` a style sets, nearest definition along its ``basedOn`` chain first."""
    out, seen = {}, set()
    while style_id is not None and style_id not in seen:
        seen.add(style_id)
        for kind, on in styles["by_id"].get(style_id, {}).items():
            out.setdefault(kind, on)
        style_id = styles["based_on"].get(style_id)
    return out


def _run_text(run):
    """The run's own literal text — ``w:t`` (live) and ``w:delText`` (tracked-deleted) children.
    A text box anchored in the run is NOT its text; its paragraphs are walked on their own."""
    return "".join(n.text or "" for n in run if _local(n.tag) in ("t", "delText"))


def _run_record(run, para, del_info, styles, para_style):
    """A struck-word record for one run, or None if it is neither deletion nor strike-formatted."""
    text = _run_text(run)
    if not text.strip():
        return None
    rpr = _child(run, "rPr")
    rstyle = _child(rpr, "rStyle")
    direct = _toggles(rpr)
    # Word takes strike from the run itself, else from its paragraph and character styles (which
    # toggle one another), else from the document defaults (ECMA-376 §17.7.3)
    in_para = _resolve(styles, para_style if para_style is not None else styles["default_para"])
    in_char = _resolve(styles, _attr(rstyle, "val") if rstyle is not None else None)
    fmt = None
    for kind in _KINDS:
        if kind in direct:
            on = direct[kind]
        elif kind in in_para or kind in in_char:
            on = in_para.get(kind, False) != in_char.get(kind, False)
        else:
            on = styles["defaults"].get(kind, False)
        if on:
            fmt = kind
            break
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


def _collect(elem, state, para, del_info, para_style, styles, out, in_box=False):
    """Depth-first walk in document order, tracking the enclosing paragraph's index and style and
    whether we are inside a tracked deletion (``w:del``, whose author/date apply to the runs it
    wraps). A run in a text box reports the body paragraph the box is anchored in. Of an
    ``mc:AlternateContent`` only the first choice is walked: Word writes a text box twice."""
    tag = _local(elem.tag)
    if tag == "p":
        if not in_box:
            state["para"] += 1
            para = state["para"]
        style = _child(_child(elem, "pPr"), "pStyle")
        para_style = _attr(style, "val") if style is not None else None
    elif tag == "txbxContent":
        in_box = True
    elif tag == "del":
        del_info = (_attr(elem, "author"), _attr(elem, "date"), _attr(elem, "id"))
    elif tag == "r":
        rec = _run_record(elem, para, del_info, styles, para_style)
        if rec is not None:
            out.append(rec)
    elif tag == "AlternateContent":
        branches = [c for c in elem if _local(c.tag) in ("Choice", "Fallback")]
        for child in branches[:1]:
            _collect(child, state, para, del_info, para_style, styles, out, in_box)
        return
    for child in elem:
        _collect(child, state, para, del_info, para_style, styles, out, in_box)


def strikethroughs_in_docx(source) -> "list[dict]":
    """Struck-run records for a Word ``.docx`` (path or bytes), in document order.

    Catches both strike character formatting (``w:strike``/``w:dstrike``, set on the run or
    inherited from its character style, its paragraph style or the document defaults) and tracked
    deletions (``w:del`` — carrying ``docx_author``/``docx_date``). Each record has
    ``tier="docx"``, a ``para`` index (no page/geometry), ``docx_change`` ('format' |
    'deletion'), and ``chars`` == ``text`` (a run is struck as a whole). Reads the main document
    body, text boxes included; transitional and Strict OOXML. A file that is not a readable
    .docx raises ValueError."""
    zf_source = io.BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray)) else source
    label = os.fspath(source) if isinstance(source, (str, os.PathLike)) else "<bytes>"
    try:
        with zipfile.ZipFile(zf_source) as zf:
            xml = zf.read("word/document.xml")
            names = set(zf.namelist())
            styles_xml = zf.read("word/styles.xml") if "word/styles.xml" in names else None
    except KeyError as e:
        raise ValueError(f"not a readable .docx (no word/document.xml): {label}") from e
    except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError, RuntimeError) as e:
        # a damaged, truncated, encrypted or oddly compressed archive
        raise ValueError(f"not a readable .docx ({type(e).__name__}: {e}): {label}") from e
    # strike toggles per style (with its basedOn parent) and the document defaults
    styles = {"by_id": {}, "based_on": {}, "default_para": None, "defaults": {}}
    try:
        root = ET.fromstring(xml)
        for el in ET.fromstring(styles_xml) if styles_xml is not None else ():
            tag = _local(el.tag)
            if tag == "docDefaults":
                styles["defaults"] = _toggles(_child(_child(el, "rPrDefault"), "rPr"))
            elif tag == "style" and _attr(el, "styleId") is not None:
                sid = _attr(el, "styleId")
                styles["by_id"][sid] = _toggles(_child(el, "rPr"))
                base = _child(el, "basedOn")
                if base is not None:
                    styles["based_on"][sid] = _attr(base, "val")
                if (_attr(el, "type") == "paragraph"
                        and (_attr(el, "default") or "").lower() in ("1", "true", "on")):
                    styles["default_para"] = sid
    except ET.ParseError as e:
        raise ValueError(f"not a readable .docx (malformed XML: {e}): {label}") from e
    out: list[dict] = []
    _collect(root, {"para": -1}, -1, None, None, styles, out)
    return out
