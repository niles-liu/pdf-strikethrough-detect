"""The public API's argument contract and the cloud-OCR adapters: inputs that used to be accepted
and then misread (a string as a page list, 1-based word keys, a page-range Azure DI result, a
sharded Document AI result) now either line up or fail at the call, with a message."""
import io
import warnings

import pymupdf
import pytest

import pdf_strikethrough as st
from pdf_strikethrough import detect as D
from pdf_strikethrough.ocr import Word, words_from_azure_di, words_from_docai


def _native_pdf(n=2):
    doc = pymupdf.open()
    for i in range(n):
        doc.new_page(width=300, height=200).insert_text((40, 80), f"page {i} text", fontsize=12)
    return doc.tobytes()


def _scanned_pdf(n=2):
    doc = pymupdf.open()
    for _ in range(n):
        page = doc.new_page(width=300, height=200)
        pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 60, 40), False)
        pix.set_rect(pix.irect, (250,))
        page.insert_image(page.rect, pixmap=pix)
    return doc.tobytes()


def _spy(monkeypatch):
    """Record the config and the word texts each scanned page reaches the classifier with."""
    calls = []

    def fake(gray, words, config=None, meta=None, dpi=200, crop_sink=None):
        calls.append((config, [w.text for w in words]))
        return []
    monkeypatch.setattr(D, "detect_scanned_image", fake)
    return calls


# ----------------------------------------------------------------------- argument validation

@pytest.mark.parametrize("pages, exc", [("12", TypeError), (True, TypeError), ([1.9], TypeError),
                                        ([5], IndexError)])
def test_bad_pages_values_fail_at_the_call(pages, exc):
    with pytest.raises(exc):
        st.detect_pdf(_native_pdf(), pages=pages)


def test_numpy_and_empty_page_selections_still_work():
    import numpy as np
    assert st.detect_pdf(_native_pdf(), pages=np.int64(1))["pages"] == [1]
    assert st.detect_pdf(_native_pdf(), pages=[])["page_sources"] == []


@pytest.mark.parametrize("kw", [{"on_missing_ocr": "Skip"}, {"method": "vectr"}, {"dpi": 0},
                                {"dpi": -200}])
def test_bad_options_fail_even_when_no_page_would_use_them(kw):
    with pytest.raises(ValueError):
        st.detect_pdf(_native_pdf(), **kw)


def test_words_by_page_shape_and_range_are_checked():
    words = [Word("hello", (0.1, 0.1, 0.3, 0.2))]
    with pytest.raises(TypeError, match="flat list"):
        st.detect_pdf(_scanned_pdf(), words_by_page=words)
    with pytest.raises(ValueError, match=r"key\(s\) \[2\]"):
        st.detect_pdf(_scanned_pdf(), words_by_page={1: words, 2: words})     # 1-based keys
    with pytest.raises(ValueError, match="only one of"):
        st.detect_pdf(_scanned_pdf(), words_by_page={0: words},
                      di_result={"pages": [{"width": 1, "height": 1, "words": []}]})


def test_ocr_backend_defaults_to_confidence_free(monkeypatch):
    calls = _spy(monkeypatch)
    st.detect_pdf(_scanned_pdf(1), ocr=lambda img: [Word("w", (0.1, 0.1, 0.2, 0.2))])
    assert calls and calls[0][0].confidence_gating is False


def test_di_result_still_defaults_to_its_calibration(monkeypatch):
    calls = _spy(monkeypatch)
    di = {"pages": [{"pageNumber": 1, "width": 1, "height": 1, "words": [
        {"content": "w", "polygon": [0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2]}]}]}
    st.detect_pdf(_scanned_pdf(1), di_result=di)
    assert calls[0][0].confidence_gating is True


def test_provenance_text_needs_markdown():
    res = st.detect_pdf(_native_pdf(1), include_markdown=False)
    with pytest.raises(ValueError, match="markdown"):
        st.provenance_text(res)


def test_open_binary_file_is_a_pdf_source():
    res = st.detect_pdf(io.BytesIO(_native_pdf(1)))
    assert res["page_count"] == 1 and res["source"] is None


def test_repaired_pdf_says_so():
    data = _native_pdf(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.detect_pdf(data[: int(len(data) * 0.75)])
    assert any("repaired" in w for w in res["warnings"])
    assert not any("repaired" in w for w in st.detect_pdf(data)["warnings"])


# ----------------------------------------------------------------------- Azure DI

def _di_page(number, word):
    return {"pageNumber": number, "width": 8.5, "height": 11.0, "unit": "inch",
            "words": [{"content": word, "confidence": 0.99,
                       "polygon": [1.0, 1.0, 2.0, 1.0, 2.0, 1.2, 1.0, 1.2]}]}


def test_di_page_range_result_lines_up_by_page_number(monkeypatch):
    calls = _spy(monkeypatch)
    res = st.detect_pdf(_scanned_pdf(2), di_result={"pages": [_di_page(2, "second")]},
                        on_missing_ocr="skip")
    assert [c[1] for c in calls] == [["second"]]                # went to page 1, not page 0
    assert any(w.startswith("page 0 is scanned") for w in res["warnings"])


def test_di_point_polygons_read_like_flat_ones():
    flat = words_from_azure_di(_di_page(1, "w"))
    pts = _di_page(1, "w")
    pts["words"][0]["polygon"] = [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0},
                                  {"x": 2.0, "y": 1.2}, {"x": 1.0, "y": 1.2}]
    assert words_from_azure_di(pts) == flat


def test_di_page_with_no_readable_polygon_raises():
    page = _di_page(1, "w")
    page["words"][0]["polygon"] = "garbage"
    with pytest.raises(ValueError, match="polygon"):
        words_from_azure_di(page)


# ----------------------------------------------------------------------- Document AI

def _docai(page_number, vertices_key="normalizedVertices", dimension=True):
    verts = [{"x": 0.1, "y": 0.1}, {"x": 0.3, "y": 0.1}, {"x": 0.3, "y": 0.2}, {"x": 0.1, "y": 0.2}]
    if vertices_key == "vertices":
        verts = [{"x": v["x"] * 1000, "y": v["y"] * 1000} for v in verts]
    page = {"pageNumber": page_number, "tokens": [{"layout": {
        "textAnchor": {"textSegments": [{"startIndex": 0, "endIndex": 5}]},
        "boundingPoly": {vertices_key: verts}}}]}
    if dimension:
        page["dimension"] = {"width": 1000, "height": 1000}
    return {"text": "hello", "pages": [page]}


def test_docai_shard_keys_by_its_page_number():
    assert list(words_from_docai(_docai(16))) == [15]


def test_docai_pixel_vertices_without_dimension_raise():
    assert words_from_docai(_docai(1, "vertices"))[0][0].text == "hello"
    with pytest.raises(ValueError, match="dimension"):
        words_from_docai(_docai(1, "vertices", dimension=False))


# ----------------------------------------------------------------------- documented types

def test_passage_type_documents_its_box():
    assert "bbox_frac" in st.Passage.__annotations__
