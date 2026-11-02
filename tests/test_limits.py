"""max_pages and max_mpix refuse a document before any page is read, sized the way detection reads
it: the pages it would process, and the pixels of the rasters it would hold (a scanned PDF page as
rendered, an image frame as stored, since it is decoded whole)."""
import copy
import io
import pickle
import struct

import pymupdf
import pytest
from PIL import Image

import pdf_strikethrough as st
from pdf_strikethrough import detect
from _synth import _scanned_pdf


def _native(n=3):
    doc = pymupdf.open()
    for _ in range(n):
        doc.new_page(width=300, height=100).insert_text((20, 50), "hello", fontsize=20)
    return doc.tobytes()


def _rendered_mpix(data, dpi):
    doc = pymupdf.open("pdf", data)
    return sum(p.get_pixmap(dpi=dpi).width * p.get_pixmap(dpi=dpi).height for p in doc) / 1e6


def _tiff(sizes):
    buf = io.BytesIO()
    first, *rest = [Image.new("L", s, 255) for s in sizes]
    first.save(buf, "TIFF", save_all=True, append_images=rest)
    return buf.getvalue()


def _never(*args):
    raise AssertionError("a refused document reached OCR")


def test_max_pages_counts_the_pages_to_process():
    with pytest.raises(st.LimitExceededError) as e:
        st.detect_pdf(_native(3), max_pages=2, progress=_never)
    assert (e.value.limit, e.value.value, e.value.maximum) == ("max_pages", 3, 2)
    assert isinstance(e.value, ValueError)
    assert st.detect_pdf(_native(3), max_pages=3)["page_count"] == 3
    assert st.detect_pdf(_native(3), max_pages=2, pages=[0, 2])["pages"] == [0, 2]


def test_max_mpix_sums_the_scanned_pages_as_rendered():
    data = _scanned_pdf(2)
    mpix = _rendered_mpix(data, 200)
    with pytest.raises(st.LimitExceededError) as e:
        st.detect_pdf(data, ocr=_never, max_mpix=mpix - 1e-6)
    assert e.value.limit == "max_mpix" and e.value.value == pytest.approx(mpix, abs=1e-9)
    res = st.detect_pdf(data, words_by_page={0: [], 1: []}, max_mpix=mpix)
    assert res["page_sources"] == ["scanned", "scanned"]


@pytest.mark.parametrize("dpi, rendered", [(100, 100), (300, 300), (400, 200)])
def test_max_mpix_follows_the_resolution_detection_renders_at(dpi, rendered):
    # above 300 dpi a scan is worked at 200, the resolution detection is calibrated at
    data = _scanned_pdf(1)
    with pytest.raises(st.LimitExceededError) as e:
        st.detect_pdf(data, ocr=_never, dpi=dpi, max_mpix=1e-6)
    assert e.value.value == pytest.approx(_rendered_mpix(data, rendered), abs=1e-9)


def test_born_digital_pages_are_not_rendered_and_count_nothing():
    assert st.detect_pdf(_native(3), max_mpix=1e-6)["page_count"] == 3


def test_an_image_is_refused_before_any_frame_is_decoded(monkeypatch):
    decoded = []
    real = detect._frame_gray
    monkeypatch.setattr(detect, "_frame_gray", lambda f: decoded.append(f) or real(f))
    tiff = _tiff([(400, 200)] * 3)
    with pytest.raises(st.LimitExceededError) as e:
        st.detect_image_file(tiff, ocr=_never, max_pages=2)
    assert (e.value.value, decoded) == (3, [])
    with pytest.raises(st.LimitExceededError) as e:
        st.detect_image_file(tiff, ocr=_never, max_mpix=0.2)
    assert (e.value.limit, e.value.value, decoded) == ("max_mpix", 0.24, [])
    res = st.detect_image_file(tiff, words_by_page={0: [], 1: [], 2: []}, dpi=200, max_pages=3,
                               max_mpix=0.24)
    assert res["page_count"] == 3 and len(decoded) == 3


@pytest.mark.parametrize("kwargs", [{"max_pages": 0}, {"max_pages": -1}, {"max_pages": True},
                                    {"max_pages": 1.5}, {"max_mpix": 0}, {"max_mpix": -2},
                                    {"max_mpix": "60"}])
def test_limits_must_be_positive(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        st.detect_pdf(_native(1), **kwargs)
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        st.detect_image_file(_tiff([(40, 20)]), words=[], **kwargs)


def test_the_error_survives_pickle_and_copy():
    # so it comes back from a worker process
    for e in (st.LimitExceededError("max_pages", 3, 2),
              st.LimitExceededError("max_mpix", 0.5, 0.1, at_least=True)):
        for back in (pickle.loads(pickle.dumps(e)), copy.copy(e)):
            assert type(back) is st.LimitExceededError and str(back) == str(e)
            assert vars(back) == vars(e)
    e = st.LimitExceededError("max_pages", 3, 2)
    e.seen_by = "worker 3"                        # as add_note() or a caller's handler sets
    assert pickle.loads(pickle.dumps(e)).seen_by == "worker 3"


def test_an_image_file_is_read_only_up_to_the_first_frame_over_a_limit():
    tiff = _tiff([(40, 20)] * 50)
    with pytest.raises(st.LimitExceededError, match="at least 3 pages") as e:
        st.detect_image_file(tiff, ocr=_never, max_pages=2)
    assert e.value.value == 3 and e.value.at_least
    with pytest.raises(st.LimitExceededError, match="at least 0.004 megapixels to read") as e:
        st.detect_image_file(io.BytesIO(tiff), ocr=_never, max_mpix=0.0035)
    assert e.value.value == 0.004                 # 5 frames of 800 px


def test_a_pil_image_is_read_as_given_then_checked():
    # its frames as they are in memory, edits included, as without a limit
    im = Image.open(io.BytesIO(_tiff([(40, 20)] * 2)))
    im.load()
    im.paste(0, (0, 0, 40, 20))                    # frame 0, black in memory only
    seen = []
    st.detect_image_file(im, ocr=lambda rgb: seen.append(int(rgb.mean())) or [], dpi=200,
                         max_pages=2)
    assert seen == [0, 255]
    with pytest.raises(st.LimitExceededError) as e:
        st.detect_image_file(im, ocr=_never, max_pages=1)
    assert e.value.value == 2 and not e.value.at_least


def test_an_open_file_is_read_as_without_a_limit(tmp_path):
    path = tmp_path / "scan.tif"
    path.write_bytes(_tiff([(400, 200)] * 2))
    out = []
    for kw in ({}, {"max_pages": 2}):
        with open(path, "rb") as f:
            f.seek(10)                            # Image.open reads from the start
            res = st.detect_image_file(f, words_by_page={0: [], 1: []}, dpi=200, **kw)
            out.append((res, f.closed))
    assert out[0] == out[1]
    with open(path, "rb") as f:
        f.seek(10)
        with pytest.raises(st.LimitExceededError) as e:
            st.detect_image_file(f, ocr=_never, max_pages=1)
    assert e.value.at_least                       # refused by the first pass, before decoding


def _damaged_tiff():
    """Two frames: frame 0's compressed data broken, frame 1's header without a width."""
    buf = io.BytesIO()
    Image.new("L", (64, 32), 0).save(buf, "TIFF", save_all=True, compression="tiff_adobe_deflate",
                                     append_images=[Image.new("L", (64, 32), 255)])
    data = bytearray(buf.getvalue())
    ifd, tags = struct.unpack_from("<I", data, 4)[0], []
    while ifd:
        n = struct.unpack_from("<H", data, ifd)[0]
        tags.append({struct.unpack_from("<H", data, ifd + 2 + 12 * i)[0]: ifd + 2 + 12 * i
                     for i in range(n)})
        ifd = struct.unpack_from("<I", data, ifd + 2 + 12 * n)[0]
    strip = struct.unpack_from("<I", data, tags[0][273] + 8)[0]      # frame 0's one strip
    data[strip:strip + 8] = b"\xff" * 8
    struct.pack_into("<H", data, tags[1][256], 0xFFF0)             # ImageWidth: an unknown tag
    return bytes(data)


def test_a_damaged_image_raises_what_it_raises_without_a_limit():
    data = _damaged_tiff()
    with pytest.raises(OSError):
        st.detect_image_file(data, words_by_page={0: [], 1: []})
    with pytest.raises(OSError):
        st.detect_image_file(data, words_by_page={0: [], 1: []}, max_pages=5)


def test_max_mpix_measures_a_fractional_dpi_as_detection_rounds_it():
    data = _scanned_pdf(1)
    with pytest.raises(st.LimitExceededError) as e:
        st.detect_pdf(data, ocr=_never, dpi=150.4, max_mpix=1e-6)
    assert e.value.value == pytest.approx(_rendered_mpix(data, 150), abs=1e-9)


def test_max_mpix_measures_a_page_the_raster_budget_scales_down(monkeypatch):
    data = _scanned_pdf(1)
    budget = _rendered_mpix(data, 200) / 4               # the page fits at about 100 dpi
    monkeypatch.setattr(detect, "MAX_RENDER_MPIX", budget)
    rect = pymupdf.open("pdf", data)[0].rect
    capped = int(round((budget * 1e6 / (rect.width / 72 * rect.height / 72)) ** 0.5))
    assert 90 < capped < 110
    with pytest.raises(st.LimitExceededError) as e:
        st.detect_pdf(data, ocr=_never, max_mpix=1e-6)
    assert e.value.value == pytest.approx(_rendered_mpix(data, capped), abs=1e-9)


def test_blank_pages_count_nothing():
    doc = pymupdf.open("pdf", _scanned_pdf(1))
    doc.new_page()
    mpix = _rendered_mpix(_scanned_pdf(1), 200)
    res = st.detect_pdf(doc.tobytes(), words_by_page={0: []}, max_mpix=mpix)
    assert res["page_sources"] == ["scanned", "blank"]


def test_a_long_pdf_is_refused_before_its_pages_are_loaded(monkeypatch):
    # a small file can list millions of pages; at most the first is loaded, to count them
    data = _native(40)
    loaded = []
    real = pymupdf.Document.load_page
    monkeypatch.setattr(pymupdf.Document, "load_page",
                        lambda self, *a, **k: loaded.append(a[0]) or real(self, *a, **k))
    with pytest.raises(st.LimitExceededError):
        st.detect_pdf(data, max_pages=10)
    assert set(loaded) <= {0}


def test_no_limit_by_default():
    assert st.detect_pdf(_native(3))["page_count"] == 3
