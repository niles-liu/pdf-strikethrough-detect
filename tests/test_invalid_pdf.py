"""A file MuPDF cannot read as a PDF raises InvalidPdfError before any page is detected: it does not
open, it opens as another format, or its page tree cannot be walked. The PDFs are written by hand,
since pymupdf will not save a broken page tree."""
import io
import json

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
UNWALKABLE = {
    "count past the file's objects": _claims(9),        # PyMuPDF's page_count itself raises
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


def test_a_document_the_caller_opened_whose_page_count_raises():
    for call in (st.detect_pdf, st.open_pdf):
        with pytest.raises(st.InvalidPdfError):
            call(pymupdf.open("pdf", _claims(9)))


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
    with pytest.warns(UserWarning):
        assert cli.main(["detect", str(path)]) == 0
    assert "holds 1" in capsys.readouterr().err


def test_a_valid_pdf_opens_as_before():
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page(width=300, height=100).insert_text((20, 50), "hello", fontsize=20)
    res = st.detect_pdf(doc.tobytes())
    assert res["page_count"] == 3 and res["warnings"] == []
