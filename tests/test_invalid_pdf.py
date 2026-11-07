"""A file MuPDF cannot read as a PDF raises InvalidPdfError before any page is detected: it does not
open, it opens as another format, or its page tree cannot be walked. The PDFs are written by hand,
since pymupdf will not save a broken page tree."""
import io
import json
import re

import pymupdf
import pytest
from PIL import Image

import pdf_strikethrough as st
from pdf_strikethrough import __main__ as cli
from pdf_strikethrough import detect


def _raw_pdf(objects):
    """A PDF of these objects, numbered from 1, the first being the catalog."""
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    return out + f"startxref\n{xref}\n%%EOF\n".encode()


CATALOG = "<< /Type /Catalog /Pages 2 0 R >>"
PAGE = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 100] >>"


def _claims(count, padding=0):
    """One page under a Pages node claiming `count`, the file padded with `padding` null objects."""
    return _raw_pdf([CATALOG, f"<< /Type /Pages /Kids [3 0 R] /Count {count} >>", PAGE]
                    + ["null"] * padding)


CYCLE = _raw_pdf([CATALOG, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                  "<< /Type /Pages /Kids [2 0 R] /Count 1 /Parent 2 0 R >>"])
NO_KIDS = _raw_pdf([CATALOG, "<< /Type /Pages /Count 1 >>"])
# page 0 reads; page 1's branch loops back to the root
BROKEN_AFTER_FIRST_PAGE = _raw_pdf([
    CATALOG, "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>", PAGE,
    "<< /Type /Pages /Kids [2 0 R] /Count 1 /Parent 2 0 R >>"])
def _count_past_the_file_is_refused():
    """Whether this MuPDF refuses a /Count above the file's object count outright (page_count
    raises; 1.28 and 2.0 do). PyMuPDF 1.26 counts the pages the tree holds instead."""
    try:
        pymupdf.open("pdf", _claims(9)).page_count
    except RuntimeError:
        return True
    return False


COUNT_PAST_REFUSED = _count_past_the_file_is_refused()
UNWALKABLE = {
    **({"count past the file's objects": _claims(9)} if COUNT_PAST_REFUSED else {}),
    "cycle": CYCLE,
    "no kids": NO_KIDS,
    "broken after the first page": BROKEN_AFTER_FIRST_PAGE,
}


def _png():
    doc = pymupdf.open()
    doc.new_page(width=200, height=100).insert_text((20, 50), "hello", fontsize=20)
    return doc[0].get_pixmap(dpi=72).tobytes("png")


@pytest.mark.parametrize("data", UNWALKABLE.values(), ids=UNWALKABLE.keys())
def test_a_page_tree_that_cannot_be_walked_is_an_invalid_pdf(data):
    done = []
    with pytest.raises(st.InvalidPdfError, match="page tree"):
        st.detect_pdf(data, on_missing_ocr="skip", progress=lambda *a: done.append(a))
    assert done == []                      # refused before any page was detected


@pytest.mark.parametrize("data", UNWALKABLE.values(), ids=UNWALKABLE.keys())
def test_every_call_that_opens_a_pdf_refuses_a_broken_page_tree(data, tmp_path):
    for call in (st.open_pdf, st.strikethroughs_in_pdf):
        with pytest.raises(st.InvalidPdfError):
            call(data)
    with pytest.raises(st.InvalidPdfError):
        st.save_overlays(data, tmp_path)


@pytest.mark.parametrize("data", [*UNWALKABLE.values(), b"not a pdf", _png()],
                         ids=[*UNWALKABLE.keys(), "unopenable", "an image"])
def test_the_file_is_released_while_the_error_is_kept(data, tmp_path):
    # Windows keeps an open file locked: a document left open, or held by the error's
    # traceback, would make the unlink fail
    path = tmp_path / "bad.pdf"
    path.write_bytes(data)
    with pytest.raises(st.InvalidPdfError) as raised:
        st.detect_pdf(path, on_missing_ocr="skip")
    path.unlink()
    assert raised.value is not None


@pytest.mark.parametrize("data", UNWALKABLE.values(), ids=UNWALKABLE.keys())
def test_a_document_opened_for_the_call_is_closed_on_the_error(data, monkeypatch):
    opened = []
    real = pymupdf.open
    monkeypatch.setattr(pymupdf, "open", lambda *a, **k: opened.append(real(*a, **k)) or opened[-1])
    with pytest.raises(st.InvalidPdfError):
        st.detect_pdf(data)
    assert len(opened) == 1 and opened[0].is_closed


def test_invalid_pdf_error_is_a_value_error_and_still_a_file_data_error():
    # ValueError like the package's other input errors; FileDataError (a RuntimeError) like the
    # error opening a broken file raised before, so handlers written for that still catch it
    with pytest.raises(st.InvalidPdfError) as raised:
        st.detect_pdf(b"not a pdf")
    assert isinstance(raised.value, ValueError)
    assert isinstance(raised.value, pymupdf.FileDataError)
    assert isinstance(raised.value.__cause__, pymupdf.FileDataError)


@pytest.mark.parametrize("data", [b"", b"not a pdf", b"%PDF-1.7\n"])
def test_a_file_mupdf_cannot_open_is_an_invalid_pdf(data):
    with pytest.raises(st.InvalidPdfError):
        st.open_pdf(data)


def test_image_bytes_are_not_a_pdf():
    # MuPDF opens PNG bytes as a one-page image document even when asked for a PDF
    with pytest.raises(st.InvalidPdfError, match="not a PDF"):
        st.detect_pdf(_png(), on_missing_ocr="skip")
    with pytest.raises(st.InvalidPdfError, match="not a PDF"):
        st.open_pdf(_png())


def test_an_image_path_is_not_a_pdf(tmp_path):
    path = tmp_path / "scan.png"
    path.write_bytes(_png())
    with pytest.raises(st.InvalidPdfError, match="detect_image_file"):
        st.detect_pdf(path, on_missing_ocr="skip")


def test_a_document_the_caller_opened_is_used_as_it_is():
    # the caller chose the document; an image MuPDF opened for them still reads as a scanned page
    res = st.detect_pdf(pymupdf.open(stream=_png()), on_missing_ocr="skip")
    assert res["page_count"] == 1 and res["page_sources"] == ["scanned"]


@pytest.mark.parametrize("name, data, dump", [
    ("scan.gif", None, 2),                # --dump-crops: a scan with no OCR, as for a PDF
    ("page.html", b"<p>keep <s>this</s> text</p>", 0),
], ids=["gif", "html"])
def test_the_cli_reads_any_document_mupdf_opens_as_before(name, data, dump, tmp_path, capsys):
    # the CLI has always sent a file that is not a .docx or an image through MuPDF, which reads
    # GIF, HTML, EPUB, XPS and more; the package's PDF functions take only PDFs
    if data is None:
        buf = io.BytesIO()
        Image.open(io.BytesIO(_png())).save(buf, "GIF")
        data = buf.getvalue()
    path = tmp_path / name
    path.write_bytes(data)
    out, lines = tmp_path / "res.json", tmp_path / "res.jsonl"
    assert cli.main(["detect", str(path), "--json", str(out),
                     "--overlay", str(tmp_path / "overlays")]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["page_count"] == 1
    assert cli.main(["detect", str(path), "--jsonl", str(lines)]) == 0
    assert "error" not in json.loads(lines.read_text(encoding="utf-8"))
    assert cli.main(["detect", str(path), "--dump-crops", str(tmp_path / "crops")]) == dump
    err = capsys.readouterr().err
    assert "Traceback" not in err and "not a PDF" not in err


def test_only_an_image_is_pointed_to_detect_image_file(tmp_path):
    path = tmp_path / "page.html"
    path.write_bytes(b"<p>hello</p>")
    with pytest.raises(st.InvalidPdfError, match="not a PDF") as raised:
        st.detect_pdf(path)
    assert "detect_image_file" not in str(raised.value)


def test_a_closed_document_is_refused_as_before():
    doc = pymupdf.open()
    doc.new_page()
    doc.close()
    for call in (st.detect_pdf, st.open_pdf, st.strikethroughs_in_pdf):
        with pytest.raises(ValueError, match="document closed"):
            call(doc)


def test_an_overlay_of_a_page_the_document_lacks_is_an_index_error():
    doc = pymupdf.open()
    doc.new_page()
    words = [{"page": 4, "final": True, "partial": False, "bbox_frac": (0.1, 0.1, 0.2, 0.2)}]
    with pytest.raises(IndexError):
        st.render_overlay(doc.tobytes(), result={"words": words})


def test_the_other_pdf_functions_warn_about_a_short_page_tree():
    with pytest.warns(UserWarning, match="holds 1"):
        assert st.strikethroughs_in_pdf(_claims(3)) == []
    pytest.importorskip("pymupdf4llm")
    with pytest.warns(UserWarning, match="holds 1"):
        st.clean_markdown(_claims(3))


def test_cli_reports_a_broken_page_tree_without_a_traceback(tmp_path, capsys):
    path = tmp_path / "cycle.pdf"
    path.write_bytes(CYCLE)
    assert cli.main(["detect", str(path)]) == 1
    err = capsys.readouterr().err
    assert "cannot open" in err and "Traceback" not in err


NESTED = _raw_pdf([CATALOG, "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 4 >>",
                   "<< /Type /Pages /Kids [5 0 R] /Count 3 /Parent 2 0 R >>", PAGE,
                   "<< /Type /Page /Parent 3 0 R /MediaBox [0 0 300 100] >>"])


@pytest.mark.parametrize("data, claimed, found", [
    (_claims(3), 3, 1),
    (_claims(9, padding=9), 9, 1),        # a /Count the file's object count allows
    (NESTED, 4, 2),                       # an inner /Count lies too
], ids=["/Count 3", "/Count 9 in a larger file", "nested"])
def test_a_page_tree_holding_fewer_pages_than_it_claims_is_read_with_a_warning(data, claimed,
                                                                                found):
    # before, the page loop was sized by the claim and the first missing page raised IndexError
    note = f"the PDF says it has {claimed} pages but its page tree holds {found}"
    with pytest.warns(UserWarning, match=note):
        res = st.detect_pdf(data)
    assert res["page_count"] == found and len(res["page_sources"]) == found
    assert len(res["warnings"]) == 1 and res["warnings"][0].startswith(note)
    with pytest.warns(UserWarning, match=note):
        assert len(st.strikethroughs_in_pdf(data)) == 0


def test_the_warning_survives_open_pdf_then_detect_pdf(recwarn):
    # MuPDF rewrites /Count in memory once it has walked the tree, as open_pdf does (so does
    # the CLI, which opens the document first); open_pdf itself does not warn
    doc = st.open_pdf(_claims(3))
    assert len(recwarn) == 0
    res = st.detect_pdf(doc)
    assert res["page_count"] == 1 and "holds 1" in res["warnings"][0]
    assert st.detect_pdf(doc, pages=[0])["pages"] == [0]
    with pytest.raises(IndexError):
        st.detect_pdf(doc, pages=[1])


def test_only_the_pages_asked_for_are_checked(monkeypatch):
    # a branch broken past page 0 reads page 0 when only page 0 is asked for, as before, and is
    # refused before detection when the broken page is; a long valid document loads the first
    # page (to count the pages) and the ones asked for
    res = st.detect_pdf(BROKEN_AFTER_FIRST_PAGE, pages=[0])
    assert res["pages"] == [0] and res["page_sources"] == ["blank"]
    done = []
    with pytest.raises(st.InvalidPdfError):
        st.detect_pdf(BROKEN_AFTER_FIRST_PAGE, pages=[1], progress=lambda *a: done.append(a))
    assert done == []
    doc = pymupdf.open()
    for _ in range(40):
        doc.new_page(width=300, height=100)
    data = doc.tobytes()
    loaded = []
    real = pymupdf.Document.load_page
    monkeypatch.setattr(pymupdf.Document, "load_page",
                        lambda self, *a, **k: loaded.append(a[0]) or real(self, *a, **k))
    st.detect_pdf(data, pages=[7])
    assert set(loaded) == {0, 7}


# page 0's branch loops back to the root; page 1 reads
BROKEN_AT_FIRST_PAGE = _raw_pdf([
    CATALOG, "<< /Type /Pages /Kids [4 0 R 3 0 R] /Count 2 >>", PAGE,
    "<< /Type /Pages /Kids [2 0 R] /Count 1 /Parent 2 0 R >>"])


def test_a_broken_first_page_matters_only_when_it_is_read(tmp_path, capsys):
    assert st.detect_pdf(BROKEN_AT_FIRST_PAGE, pages=[1])["page_sources"] == ["blank"]
    for call in (st.detect_pdf, st.open_pdf):
        with pytest.raises(st.InvalidPdfError):
            call(BROKEN_AT_FIRST_PAGE)
    assert [o["page"] for o in st.render_overlay(BROKEN_AT_FIRST_PAGE, pages=[1])] == [1]
    assert [o["page"] for o in st.render_overlay(BROKEN_AFTER_FIRST_PAGE, pages=[0])] == [0]
    path = tmp_path / "broken.pdf"
    path.write_bytes(BROKEN_AFTER_FIRST_PAGE)
    assert cli.main(["detect", str(path), "--pages", "1"]) == 0
    assert cli.main(["detect", str(path)]) == 1
    assert "cannot open" in capsys.readouterr().err


@pytest.mark.skipif(not COUNT_PAST_REFUSED, reason="this MuPDF counts the pages it holds")
def test_a_document_the_caller_opened_whose_page_count_raises():
    for call in (st.detect_pdf, st.open_pdf):
        with pytest.raises(st.InvalidPdfError):
            call(pymupdf.open("pdf", _claims(9)))


def test_a_count_past_the_files_objects():
    # refused where MuPDF refuses it (and so in UNWALKABLE); where it counts the pages the tree
    # holds, read as any short page tree is
    if COUNT_PAST_REFUSED:
        with pytest.raises(st.InvalidPdfError, match="page tree"):
            st.detect_pdf(_claims(9))
    else:
        with pytest.warns(UserWarning, match="says it has 9 pages but its page tree holds 1"):
            assert st.detect_pdf(_claims(9))["page_count"] == 1


def _callers_own_step():
    order_id = 4711                       # noqa: F841  (what the caller's error report shows)
    raise KeyError("the caller's own error")


@pytest.mark.parametrize("data", [CYCLE, b"not a pdf"], ids=["broken tree", "unopenable"])
def test_an_error_the_caller_is_handling_keeps_its_frames(data):
    try:
        _callers_own_step()
    except KeyError as own:
        with pytest.raises(st.InvalidPdfError):
            st.detect_pdf(data)
        assert own.__traceback__.tb_next.tb_frame.f_locals == {"order_id": 4711}


def test_a_page_tree_holding_no_page(monkeypatch):
    # MuPDF's repair of a damaged file can leave a tree that claims pages and holds none
    monkeypatch.setattr(detect, "_claimed_pages", lambda doc: 3)
    data = _raw_pdf([CATALOG, "<< /Type /Pages /Kids [] /Count 0 >>"])
    with pytest.warns(UserWarning, match="holds 0; no page was read"):
        res = st.detect_pdf(data)
    assert res["page_count"] == 0 and res["words"] == []


def test_the_warning_names_the_pages_read():
    with pytest.warns(UserWarning) as one:
        st.detect_pdf(_claims(3))
    with pytest.warns(UserWarning) as two:
        st.detect_pdf(NESTED)
    assert str(one[0].message) == (
        "the PDF says it has 3 pages but its page tree holds 1; only that page was read, so "
        "strikes on the missing pages are not reported")
    assert str(two[0].message) == (
        "the PDF says it has 4 pages but its page tree holds 2; only those pages were read, so "
        "strikes on the missing pages are not reported")


def test_no_warning_once_the_caller_changes_the_page_count(recwarn):
    doc = st.open_pdf(_claims(3))
    for _ in range(4):
        doc.new_page()
    res = st.detect_pdf(doc)
    assert res["page_count"] == 5 and res["warnings"] == [] and len(recwarn) == 0


def test_cli_reads_the_pages_a_short_page_tree_holds(tmp_path, capsys):
    path = tmp_path / "short.pdf"
    path.write_bytes(_claims(3))
    assert cli.main(["detect", str(path)]) == 0
    assert capsys.readouterr().err.count("holds 1") == 1     # once, as the result's warning line


def test_a_valid_pdf_opens_as_before():
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page(width=300, height=100).insert_text((20, 50), "hello", fontsize=20)
    res = st.detect_pdf(doc.tobytes())
    assert res["page_count"] == 3 and res["warnings"] == []


# ----------------------------------------------------------------------- pages MuPDF does not read

def _pdf(objects, tail=b"", broken=None):
    """A PDF of `objects` ({number: body}, 1 the catalog), `tail` written after them with no xref
    entry, and object `broken`'s header misspelt, so MuPDF rebuilds the xref when it reads it."""
    out, offsets = b"%PDF-1.4\n", {}
    for number in sorted(objects):
        offsets[number] = len(out)
        word = "ojb" if number == broken else "obj"
        out += f"{number} 0 {word}\n{objects[number]}\nendobj\n".encode()
    out += tail
    xref, size = len(out), max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    out += b"".join((f"{offsets[n]:010d} 00000 n \n" if n in offsets else
                     "0000000000 00000 f \n").encode() for n in range(1, size))
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\n".encode()
    return out + f"startxref\n{xref}\n%%EOF\n".encode()


def _leaf(parent, n=None, extra=""):
    """A page under `parent`; with `n`, it shows "page<n>" from content object 20 + n."""
    shows = f" /Contents {20 + n} 0 R /Resources << /Font << /F1 9 0 R >> >>" if n is not None else ""
    return f"<< /Type /Page /Parent {parent} 0 R /MediaBox [0 0 300 100]{shows}{extra} >>"


def _text_pages(n):
    """The font and the content objects of pages showing "page0" .. "page<n-1>"."""
    objects = {9: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"}
    for i in range(n):
        s = f"BT /F1 12 Tf 20 50 Td (page{i}) Tj ET"
        objects[20 + i] = f"<< /Length {len(s)} >>\nstream\n{s}\nendstream"
    return objects


# The root's /Count says 2 and MuPDF reads pages 0 and 1; the tree holds 3
LOW_ROOT = _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 2 >>",
                 3: _leaf(2, 0), 4: _leaf(2, 1), 5: _leaf(2, 2), **_text_pages(3)})
# An inner node says 1 for its 2 pages, and the root agrees: MuPDF reads pages 0 and 2
LOW_INNER = _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [6 0 R 5 0 R] /Count 2 >>",
                  6: "<< /Type /Pages /Parent 2 0 R /Kids [3 0 R 4 0 R] /Count 1 >>",
                  3: _leaf(6, 0), 4: _leaf(6, 1), 5: _leaf(2, 2), **_text_pages(3)})
# The second kid is object 7, which the xref lists as free, or object 40, past its end
MISSING_KID = {
    "a free object": _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [3 0 R 7 0 R 4 0 R] /Count 3 >>",
                           3: _leaf(2, 0), 4: _leaf(2, 1), 8: "null", **_text_pages(2)}),
    "past the xref": _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [3 0 R 40 0 R 4 0 R] /Count 3 >>",
                           3: _leaf(2, 0), 4: _leaf(2, 1), **_text_pages(2)}),
}
INDIRECT_COUNT = _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 8 0 R >>",
                       3: _leaf(2), 4: _leaf(2), 8: "5"})


def _mupdf_reads(data):
    """The pages this MuPDF reads in `data` once it has loaded the first, and their text."""
    doc = pymupdf.open("pdf", data)
    doc.load_page(0)
    return doc.page_count, [doc[p].get_text().strip() for p in range(doc.page_count)]


@pytest.mark.parametrize("data, read", [(LOW_ROOT, ["page0", "page1"]),
                                        (LOW_INNER, ["page0", "page2"])],
                         ids=["root /Count", "inner /Count"])
def test_pages_the_page_tree_holds_and_mupdf_does_not_read_are_named(data, read):
    found, texts = _mupdf_reads(data)
    if found == 3:
        pytest.skip("this MuPDF reads every page")
    assert texts == read                                 # the tree's /Count entries steer MuPDF
    note = ("the PDF's page tree holds 3 pages, but MuPDF reads only 2; strikes on the 1 page it "
            "does not read are not reported")
    with pytest.warns(UserWarning, match=note):
        res = st.detect_pdf(data)
    assert res["page_count"] == found and note in res["warnings"]
    with pytest.warns(UserWarning, match=note):
        st.strikethroughs_in_pdf(data)


@pytest.mark.parametrize("data", MISSING_KID.values(), ids=MISSING_KID.keys())
def test_a_kid_that_names_no_object_is_named(data):
    found, texts = _mupdf_reads(data)
    assert found == 3 and texts == ["page0", "", "page1"]  # MuPDF reads it as a blank page
    note = "page 1 of the PDF's page tree names no object in the file and is read as a blank page"
    with pytest.warns(UserWarning, match=note):
        res = st.detect_pdf(data)
    assert res["page_count"] == 3 and res["warnings"] == [note]


def test_an_indirect_count_is_the_claim():
    with pytest.warns(UserWarning, match="the PDF says it has 5 pages but its page tree holds 2"):
        assert st.detect_pdf(INDIRECT_COUNT)["page_count"] == 2


def test_mupdf_running_out_of_memory_reading_the_claim_is_not_an_invalid_pdf():
    doc = pymupdf.open("pdf", LOW_ROOT)

    def no_memory(*_):
        raise pymupdf.mupdf.FzErrorSystem("code=2: realloc (28346448 bytes) failed")

    doc.xref_get_key = no_memory                       # as MuPDF raises it, under a memory cap
    with pytest.raises(pymupdf.mupdf.FzErrorSystem):
        st.detect_pdf(doc)


@pytest.mark.parametrize("data", [
    _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [6 0 R 5 0 R] /Count 3 >>",
          6: "<< /Type /Pages /Parent 2 0 R /Kids [3 0 R 4 0 R] /Count 2 >>",
          3: _leaf(6, 0), 4: _leaf(6, 1), 5: _leaf(2, 2), **_text_pages(3)}),
    _pdf({1: CATALOG, 2: "<< /Kids [3 0 R 4 0 R] /Count 2 >>", 3: _leaf(2), 4: _leaf(2)}),
    _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 8 0 R >>",
          3: _leaf(2), 4: _leaf(2), 8: "2"}),
    _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [3 0 R 3 0 R] /Count 2 >>", 3: _leaf(2)}),
], ids=["nested", "no /Type on the root", "indirect /Count", "a page listed twice"])
def test_page_trees_mupdf_reads_whole_give_no_warning(data, recwarn):
    res = st.detect_pdf(data)
    assert res["warnings"] == [] and len(recwarn) == 0
    assert res["page_count"] == _mupdf_reads(data)[0]


def test_a_long_pymupdf_document_gives_no_warning(recwarn):
    doc = pymupdf.open()
    for _ in range(300):
        doc.new_page(width=200, height=100)
    data = doc.tobytes(garbage=3, use_objstms=1)
    assert st.detect_pdf(data, pages=[0])["warnings"] == [] and len(recwarn) == 0


# The xref names a /Pages node holding 3 pages; a stale copy of it after the objects holds 1, and
# a repair, which rebuilds the xref from the file, takes the last copy
STALE = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
SHRINKS = {
    # page 1's annotation has a misspelt header: loading page 1 repairs the file
    "on loading a page": _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>",
                               3: _leaf(2, 0), 4: _leaf(2, 1, " /Annots [12 0 R]"), 5: _leaf(2, 2),
                               12: "<< /Type /Annot /Subtype /Square /Rect [10 10 50 50] >>",
                               **_text_pages(3)}, STALE, broken=12),
    # page 1's content has a misspelt header: reading page 1's text repairs the file
    "on reading a page": _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>",
                               3: _leaf(2, 0), 4: _leaf(2, 1), 5: _leaf(2, 2), **_text_pages(3)},
                              STALE, broken=21),
}


def _mupdf_shrinks(data):
    """Whether this MuPDF's page count falls as it loads the pages of `data` and reads them."""
    doc = pymupdf.open("pdf", data)
    for p in range(3):
        if p >= doc.page_count:
            return True
        doc.load_page(p).get_text()
    return doc.page_count < 3


@pytest.mark.parametrize("data", SHRINKS.values(), ids=SHRINKS.keys())
def test_a_page_tree_that_shrinks_while_mupdf_reads_it_is_an_invalid_pdf(data, tmp_path, capsys):
    # before, the pages it lost raised IndexError, or open_pdf returned a document of fewer pages
    if not _mupdf_shrinks(data):
        pytest.skip("this MuPDF keeps the page tree it opened")
    done = []
    with pytest.raises(st.InvalidPdfError, match="page tree changed from 3 to 1 pages"):
        st.detect_pdf(data, on_missing_ocr="skip", progress=lambda *a: done.append(a))
    assert done == []                                    # refused before any page was detected
    for call in (st.strikethroughs_in_pdf, st.render_overlay):
        with pytest.raises(st.InvalidPdfError, match="page tree changed"):
            call(data)
    path = tmp_path / "shrinks.pdf"
    path.write_bytes(data)
    assert cli.main(["detect", str(path)]) == 1
    err = capsys.readouterr().err
    assert "page tree changed" in err and "Traceback" not in err


def test_open_pdf_refuses_a_page_tree_that_shrinks_as_it_loads_the_pages():
    data = SHRINKS["on loading a page"]
    if not _mupdf_shrinks(data):
        pytest.skip("this MuPDF keeps the page tree it opened")
    with pytest.raises(st.InvalidPdfError, match="page tree changed from 3 to 1 pages"):
        st.open_pdf(data)


# ------------------------------------------------------- the walk, and repairs after the first page

def _repairs_on_reading(data, number):
    """Whether this MuPDF's page count changes when it reads object `number` of `data` after
    loading page 0, as the walk of a tree it did not map reads it."""
    doc = pymupdf.open("pdf", data)
    doc.load_page(0)
    before = doc.page_count
    doc.xref_object(number)
    return doc.page_count != before


# An untyped root: MuPDF does not map the tree, and its first load reads only page 0. Page 2's
# object has a misspelt header, so the walk's read of it repairs the file to the copy after it
WALK_REPAIRS = {
    "shrinks": (_pdf({1: CATALOG, 2: "<< /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>", 3: _leaf(2, 0),
                      4: _leaf(2, 1), 5: _leaf(2, 2), **_text_pages(3)}, STALE, broken=5),
                "3 to 1"),
    "grows": (_pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 1 >>",
                    3: _leaf(2, 0), 4: _leaf(2, 1), 5: _leaf(2, 2), 6: _leaf(2, 2),
                    **_text_pages(3)},
                   b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R 6 0 R] /Count 3 >>\nendobj\n",
                   broken=5), "1 to 3"),
}


@pytest.mark.parametrize("data, change", WALK_REPAIRS.values(), ids=WALK_REPAIRS.keys())
def test_a_walk_that_sets_off_a_repair_is_an_invalid_pdf(data, change, tmp_path, capsys):
    # before, these files raised IndexError or gave a result with no warning
    if not _repairs_on_reading(data, 5):
        pytest.skip("this MuPDF keeps the page tree it opened")
    for call in (st.detect_pdf, lambda d: st.detect_pdf(d, pages=[0]), st.open_pdf,
                 st.strikethroughs_in_pdf):
        with pytest.raises(st.InvalidPdfError, match=f"page tree changed from {change} pages"):
            call(data)
    path = tmp_path / "walk.pdf"
    path.write_bytes(data)
    assert cli.main(["detect", str(path)]) == 1
    err = capsys.readouterr().err
    assert "page tree changed" in err and "Traceback" not in err


def test_a_walk_that_sets_off_a_repair_keeping_the_count_walks_the_repaired_file():
    # root /Count 1 over 3 kids, the third's header misspelt: the walk's read of it repairs the
    # file, the third is then no object, and the walk of the repaired file finds 1 page lost
    data = _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 1 >>",
                 3: _leaf(2, 0), 4: _leaf(2, 1), 5: _leaf(2, 2), **_text_pages(3)}, broken=5)
    doc = pymupdf.open("pdf", data)
    doc.load_page(0)
    doc.xref_object(5)
    if doc.page_count != 1 or not doc.is_repaired:
        pytest.skip("this MuPDF reads the tree another way")
    note = ("the PDF's page tree holds 2 pages, but MuPDF reads only 1; strikes on the 1 page it "
            "does not read are not reported")
    with pytest.warns(UserWarning) as rec:
        res = st.detect_pdf(data)
    assert res["page_count"] == 1 and res["warnings"][0] == note and len(res["warnings"]) == 2
    assert [str(w.message) for w in rec] == res["warnings"]      # the note, then the repair


@pytest.mark.parametrize("count", ["/Count 0", ""], ids=["/Count 0", "no /Count"])
def test_a_root_that_counts_no_page_names_the_pages_it_holds(count):
    data = _pdf({1: CATALOG, 2: f"<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] {count} >>",
                 3: _leaf(2, 0), 4: _leaf(2, 1), 5: _leaf(2, 2), **_text_pages(3)})
    if pymupdf.open("pdf", data).page_count:
        pytest.skip("this MuPDF counts the pages")
    note = ("the PDF's page tree holds 3 pages, but MuPDF reads none; strikes on the 3 pages it "
            "does not read are not reported")
    with pytest.warns(UserWarning, match=note):
        res = st.detect_pdf(data)
    assert res["page_count"] == 0 and res["warnings"] == [note]
    with pytest.warns(UserWarning, match=note):
        assert st.strikethroughs_in_pdf(data) == []


def test_a_tree_of_one_page_mupdf_does_not_read_says_page():
    data = _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [3 0 R] /Count 0 >>", 3: _leaf(2, 0),
                 **_text_pages(1)})
    if pymupdf.open("pdf", data).page_count:
        pytest.skip("this MuPDF counts the page")
    with pytest.warns(UserWarning,
                      match="holds 1 page, but MuPDF reads none; strikes on the 1 page it"):
        st.detect_pdf(data)


def _struck_on(on):
    """Three pages; page `on` has a /StrikeOut whose /QuadPoints object (50) has a misspelt header,
    which only the annotation detector reads, and a stale 1-page copy of the root follows."""
    s = "0 0 1 RG 0 7 m 42 7 l S"
    return _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>",
                 **{3 + i: _leaf(2, i, " /Annots [12 0 R]" if i == on else "") for i in range(3)},
                 12: "<< /Type /Annot /Subtype /StrikeOut /Rect [18 45 60 60] /QuadPoints 50 0 R "
                     "/C [1 0 0] /AP << /N 13 0 R >> >>",
                 13: f"<< /Type /XObject /Subtype /Form /BBox [0 0 42 15] /Length {len(s)} >>\n"
                     f"stream\n{s}\nendstream",
                 50: "[18 60 60 60 18 45 60 45]", **_text_pages(3)}, STALE, broken=50)


def _quad_read_repairs(data, on):
    """Whether this MuPDF keeps 3 pages as it loads and reads them, and has fewer once it reads
    the annotation's /QuadPoints on page `on`."""
    doc = pymupdf.open("pdf", data)
    for p in range(3):
        doc.load_page(p).get_text()
    if doc.page_count != 3:
        return False
    for annot in doc.load_page(on).annots():
        annot.vertices
    return doc.page_count != 3


@pytest.mark.parametrize("on", [1, 2], ids=["a middle page", "the last page"])
def test_a_repair_while_a_page_is_detected_is_an_invalid_pdf(on):
    # before: a bare IndexError at the next page, or after the last a 1-page result with 3 sources
    data = _struck_on(on)
    if not _quad_read_repairs(data, on):
        pytest.skip("this MuPDF keeps the page tree it opened")
    done = []
    with pytest.raises(st.InvalidPdfError, match="page tree changed from 3 to 1 pages"):
        st.detect_pdf(data, method="annot", progress=lambda *a: done.append(a[2]))
    assert done == list(range(on + 1))
    with pytest.raises(st.InvalidPdfError, match="page tree changed from 3 to 1 pages"):
        st.strikethroughs_in_pdf(data, method="annot")


def test_rendering_or_markdown_that_repairs_the_file_is_an_invalid_pdf():
    data = SHRINKS["on reading a page"]
    if not _mupdf_shrinks(data):
        pytest.skip("this MuPDF keeps the page tree it opened")
    with pytest.raises(st.InvalidPdfError, match="page tree changed from 3 to 1 pages"):
        st.render_overlay(data, {"words": []}, pages=[0, 1, 2])
    pytest.importorskip("pymupdf4llm")
    with pytest.raises(st.InvalidPdfError, match="page tree changed from 3 to 1 pages"):
        st.clean_markdown(data)


# Reading page 0's text repairs the file to a root whose /Count is past the file's objects
COUNT_PAST_ON_REPAIR = _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>",
                             3: _leaf(2, 0), 4: _leaf(2, 1), 5: _leaf(2, 2), **_text_pages(3)},
                            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 99999999 >>\nendobj\n",
                            broken=20)


def test_a_page_count_mupdf_cannot_read_after_a_repair_is_an_invalid_pdf(tmp_path, capsys):
    # before: RuntimeError "Invalid number of pages", or a MuPDF FzErrorFormat on PyMuPDF 1.26
    for call in (st.detect_pdf, st.strikethroughs_in_pdf,
                 lambda d: st.render_overlay(d, {"words": []}, pages=[0, 1, 2])):
        with pytest.raises(st.InvalidPdfError):
            call(COUNT_PAST_ON_REPAIR)
    path = tmp_path / "count.pdf"
    path.write_bytes(COUNT_PAST_ON_REPAIR)
    assert cli.main(["detect", str(path)]) == 1
    assert "Traceback" not in capsys.readouterr().err


def test_a_kid_past_the_pages_mupdf_reads_that_names_no_object_gives_no_warning(recwarn):
    data = _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [3 0 R 4 0 R 40 0 R] /Count 2 >>",
                 3: _leaf(2, 0), 4: _leaf(2, 1), **_text_pages(2)})
    assert _mupdf_reads(data) == (2, ["page0", "page1"])        # MuPDF loses no page
    res = st.detect_pdf(data)
    assert res["warnings"] == [] and len(recwarn) == 0


def test_the_kids_that_name_no_object_are_listed():
    kids = " ".join(["3 0 R", *(f"{40 + i} 0 R" for i in range(12)), "4 0 R"])
    data = _pdf({1: CATALOG, 2: f"<< /Type /Pages /Kids [{kids}] /Count 14 >>", 3: _leaf(2, 0),
                 4: _leaf(2, 1), 60: "null", **_text_pages(2)})
    if _mupdf_reads(data)[0] != 14:
        pytest.skip("this MuPDF does not read the 14 pages")
    note = ("pages 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (+2 more) of the PDF's page tree name no object "
            "in the file and are read as blank pages")
    with pytest.warns(UserWarning, match=re.escape(note)):
        assert st.detect_pdf(data)["warnings"] == [note]


def test_a_blank_page_is_named_only_where_mupdf_reads_it():
    # the inner node says 1 for its 2 pages, so MuPDF reads the kid that names no object as
    # page 1, not page 2 where the walk meets it, and loses page1
    data = _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [6 0 R 40 0 R] /Count 3 >>",
                 6: "<< /Type /Pages /Parent 2 0 R /Kids [3 0 R 4 0 R] /Count 1 >>",
                 3: _leaf(6, 0), 4: _leaf(6, 1), **_text_pages(2)})
    with pytest.warns(UserWarning, match="an entry of the PDF's page tree names no object"):
        res = st.detect_pdf(data, pages=[0, 1])
    assert not any("page 2" in w for w in res["warnings"])
    assert ("an entry of the PDF's page tree names no object in the file; MuPDF reads it as a "
            "blank page") in res["warnings"]


def test_the_walk_stops_past_the_pages_mupdf_reads():
    # root /Count 1 over 1,500 pages: MuPDF reads 1, and the walk reads 1,000 kids past it
    kids = " ".join(f"{100 + i} 0 R" for i in range(1500))
    data = _pdf({1: CATALOG, 2: f"<< /Type /Pages /Kids [{kids}] /Count 1 >>",
                 **{100 + i: _leaf(2) for i in range(1500)}})
    if _mupdf_reads(data)[0] != 1:
        pytest.skip("this MuPDF reads every page")
    note = ("the PDF's page tree holds at least 1001 pages, but MuPDF reads only 1; strikes on "
            "the 1000 or more pages it does not read are not reported")
    with pytest.warns(UserWarning, match=note):
        assert st.detect_pdf(data, pages=[0])["warnings"] == [note]


def test_a_tree_mupdf_maps_is_not_walked():
    doc = pymupdf.open()
    for _ in range(300):
        doc.new_page(width=200, height=100)
    doc = pymupdf.open("pdf", doc.tobytes())
    doc.load_page(0)
    if not detect._page_map_loaded(doc):
        pytest.skip("this PyMuPDF does not show MuPDF's page map")
    assert detect._tree_pages(doc, doc.page_count) is None      # MuPDF read the whole tree
    low = pymupdf.open("pdf", LOW_ROOT)
    low.load_page(0)
    assert not detect._page_map_loaded(low) and detect._tree_pages(low, low.page_count)[0] == 1


@pytest.mark.filterwarnings("ignore:the PDF is damaged")
@pytest.mark.parametrize("repaired", [False, True], ids=["a valid file", "a file MuPDF repaired"])
def test_a_page_the_caller_adds_while_detecting_is_not_a_repair(repaired):
    # repaired: page 1's content has a misspelt header, so the caller's read repairs the file
    data = _pdf({1: CATALOG, 2: "<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>",
                 3: _leaf(2, 0), 4: _leaf(2, 1), 5: _leaf(2, 2), **_text_pages(3)},
                broken=21 if repaired else None)
    doc = pymupdf.open("pdf", data)
    for page in doc:
        page.get_text()
    if doc.is_repaired != repaired or doc.page_count != 3:
        pytest.skip("this MuPDF reads the file another way")
    res = st.detect_pdf(doc, progress=lambda done, *_: doc.new_page() if done == 1 else None)
    assert len(res["page_sources"]) == 3 and res["page_count"] == 4     # as before


def test_an_overlay_whose_rendering_repairs_the_file_is_an_error_not_a_traceback(
        tmp_path, capsys, monkeypatch):
    # the overlay step renders the file again, and MuPDF can repair it then
    from pdf_strikethrough import overlay

    def repaired(*a, **k):
        raise st.InvalidPdfError("not a readable PDF: its page tree changed from 3 to 1 pages")

    monkeypatch.setattr(overlay, "save_overlays", repaired)
    doc = pymupdf.open()
    doc.new_page(width=300, height=100)
    path = tmp_path / "ok.pdf"
    path.write_bytes(doc.tobytes())
    assert cli.main(["detect", str(path), "--overlay", str(tmp_path / "ov")]) == 1
    err = capsys.readouterr().err
    assert "cannot render the overlay" in err and "Traceback" not in err
