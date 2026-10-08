"""Word documents: strike formatting that arrives through styles, text boxes, and Strict OOXML, and
the documented ValueError for a damaged file."""
import io
import zipfile

import pytest

import pdf_strikethrough as st

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
STRICT = "http://purl.oclc.org/ooxml/wordprocessingml/main"


def _docx(body, styles=None, ns=W):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", f'<?xml version="1.0"?><w:document xmlns:w="{ns}">'
                                        f'<w:body>{body}</w:body></w:document>')
        if styles is not None:
            z.writestr("word/styles.xml", f'<?xml version="1.0"?><w:styles xmlns:w="{ns}">'
                                          f'{styles}</w:styles>')
    return buf.getvalue()


def _struck(data):
    return [(r["chars"], r["para"]) for r in st.strikethroughs_in_docx(data)]


_STYLES = ('<w:style w:type="character" w:styleId="Deleted"><w:rPr><w:strike/></w:rPr></w:style>'
           '<w:style w:type="character" w:styleId="DeletedRed"><w:basedOn w:val="Deleted"/>'
           '</w:style>'
           '<w:style w:type="paragraph" w:styleId="Repealed"><w:rPr><w:strike/></w:rPr></w:style>')


def _run(text, rpr=""):
    return f"<w:r><w:rPr>{rpr}</w:rPr><w:t>{text}</w:t></w:r>"


def test_strike_from_a_character_style_and_its_base():
    gone = _run("gone", '<w:rStyle w:val="Deleted"/>')
    also = _run("also", '<w:rStyle w:val="DeletedRed"/>')
    body = f"<w:p>{_run('keep')}{gone}{also}</w:p>"
    assert _struck(_docx(body, _STYLES)) == [("gone", 0), ("also", 0)]


def test_strike_from_a_paragraph_style_and_direct_override():
    kept = _run("kept", '<w:strike w:val="0"/>')
    body = f'<w:p><w:pPr><w:pStyle w:val="Repealed"/></w:pPr>{_run("old")}{kept}</w:p>'
    assert _struck(_docx(body, _STYLES)) == [("old", 0)]


def test_paragraph_and_character_style_strikes_toggle_off():
    restored = _run("restored", '<w:rStyle w:val="Deleted"/>')
    body = f'<w:p><w:pPr><w:pStyle w:val="Repealed"/></w:pPr>{restored}</w:p>'
    assert _struck(_docx(body, _STYLES)) == []


def test_strike_from_document_defaults():
    styles = ("<w:docDefaults><w:rPrDefault><w:rPr><w:strike/></w:rPr></w:rPrDefault>"
              "</w:docDefaults>")
    assert _struck(_docx(f"<w:p>{_run('everything')}</w:p>", styles)) == [("everything", 0)]


def test_struck_run_in_a_text_box_is_read_once():
    box = f'<w:txbxContent><w:p>{_run("boxed", "<w:strike/>")}</w:p></w:txbxContent>'
    body = ('<w:p><w:r><mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/'
            'markup-compatibility/2006"><mc:Choice Requires="wps"><w:drawing>'
            f'{box}</w:drawing></mc:Choice><mc:Fallback><w:pict>{box}</w:pict></mc:Fallback>'
            f'</mc:AlternateContent></w:r>{_run("after", "<w:strike/>")}</w:p>')
    assert _struck(_docx(body)) == [("boxed", 0), ("after", 0)]        # the anchoring paragraph


def test_text_box_paragraphs_do_not_renumber_the_body():
    box = f'<w:txbxContent><w:p>{_run("inside")}</w:p><w:p>{_run("too")}</w:p></w:txbxContent>'
    body = (f'<w:p><w:r><w:drawing>{box}</w:drawing></w:r></w:p>'
            f'<w:p>{_run("second")}</w:p><w:p>{_run("repealed", "<w:strike/>")}</w:p>')
    assert _struck(_docx(body)) == [("repealed", 2)]                # as in a body with no box


def test_strict_ooxml_strike_and_deletion():
    body = (f'<w:p>{_run("struck", "<w:strike/>")}'
            '<w:del w:author="Ed"><w:r><w:delText>deleted</w:delText></w:r></w:del></w:p>')
    recs = st.strikethroughs_in_docx(_docx(body, ns=STRICT))
    assert [(r["chars"], r["docx_change"]) for r in recs] == [("struck", "format"),
                                                              ("deleted", "deletion")]
    assert recs[1]["docx_author"] == "Ed"


def test_malformed_document_xml_is_a_value_error(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", "<w:document><unclosed></w:document>")
    path = tmp_path / "broken.docx"
    path.write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="broken.docx"):
        st.strikethroughs_in_docx(path)                  # a pathlib.Path, named in the error


def test_damaged_archive_is_a_value_error():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        body = "<w:p><w:r><w:t>hello world</w:t></w:r></w:p>" * 200
        z.writestr("word/document.xml",
                   f'<w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>')
    data = bytearray(buf.getvalue())
    start = data.index(b"word/document.xml") + len("word/document.xml")
    for i in range(start + 5, start + 15):                # flip bits inside the deflate stream
        data[i] ^= 0xFF
    with pytest.raises(ValueError, match="not a readable .docx"):
        st.strikethroughs_in_docx(bytes(data))
