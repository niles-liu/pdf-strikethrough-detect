"""A multi-page TIFF file is decoded one frame at a time, as detection reaches it, not every frame
up front, so its memory grows with its largest frame, not its frame count. The output is what
decoding every frame up front gives and a damaged TIFF raises what it raised. Every other image is
decoded whole, as before: a PIL image you pass (its frames may live only in memory), a GIF or APNG
(Pillow decodes each frame to reach the next), any single-frame file, and a TIFF with a frame over
Pillow's decompression-bomb limit."""
import io

import numpy as np
import pytest
from PIL import Image

import pdf_strikethrough as st
from pdf_strikethrough import detect
from _synth import _redline_image_png


def _frame():
    png, words = _redline_image_png()
    frame = Image.open(io.BytesIO(png))
    frame.load()
    return frame, words


def _tiff(n, **save):
    frame, words = _frame()
    buf = io.BytesIO()
    frame.save(buf, "TIFF", save_all=True, append_images=[frame.copy() for _ in range(n - 1)],
               dpi=(200, 200), **save)
    return buf.getvalue(), words


@pytest.fixture
def log(monkeypatch):
    calls = []
    real = detect._frame_gray
    monkeypatch.setattr(detect, "_frame_gray", lambda f: calls.append("decode") or real(f))
    return calls


def _ocr(log, words):
    return lambda rgb: log.append("ocr") or list(words)


def test_each_frame_is_decoded_when_detection_reaches_it(log):
    tiff, words = _tiff(4)
    res = st.detect_image_file(tiff, ocr=_ocr(log, words))
    assert log == ["decode", "ocr"] * 4
    assert res["page_count"] == 4
    assert [w["page"] for w in res["words"] if w["final"]] == [0, 1, 2, 3]


def test_a_tiff_path_or_open_file_is_read_the_same_way(tmp_path, log):
    tiff, words = _tiff(3)
    path = tmp_path / "scan.tif"
    path.write_bytes(tiff)
    want = st.detect_image_file(tiff, words_by_page={i: words for i in range(3)})
    log.clear()
    with open(path, "rb") as f:
        for source in (path, str(path), f):
            got = st.detect_image_file(source, ocr=_ocr(log, words))
            assert log == ["decode", "ocr"] * 3
            log.clear()
            assert got["words"] == want["words"]


def test_a_frame_without_words_raises_once_the_later_frames_are_decoded(log):
    tiff, words = _tiff(3)              # decoded, so a damaged later frame raises its own error
    with pytest.raises(st.OcrRequiredError, match="frame 1"):
        st.detect_image_file(tiff, words_by_page={0: words, 2: words})
    assert log == ["decode"] * 3


def test_every_frame_gives_what_it_gives_alone():
    tiff, words = _tiff(3)
    png, _ = _redline_image_png()
    alone = st.detect_image_file(png, words=words, dpi=200)["words"]
    res = st.detect_image_file(tiff, words_by_page={i: words for i in range(3)})
    for pno in range(3):
        assert [{**w, "page": 0} for w in res["words"] if w["page"] == pno] == alone


def _tiffs():
    """TIFFs of several kinds of frame: colour, rotated, compressed, 16-bit, frames of two sizes."""
    frame, words = _frame()
    out = {}
    for name, kw in [("plain", {}), ("lzw", {"compression": "tiff_lzw"}),
                     ("deflate", {"compression": "tiff_adobe_deflate"})]:
        buf = io.BytesIO()
        frame.save(buf, "TIFF", save_all=True, append_images=[frame.copy()] * 2, dpi=(200, 200),
                   **kw)
        out[name] = buf.getvalue()
    rgb, small = frame.convert("RGB"), frame.resize((frame.width // 2, frame.height // 2))
    buf = io.BytesIO()
    rgb.save(buf, "TIFF", save_all=True, append_images=[small, frame.rotate(90, expand=True)],
             dpi=(150, 150))
    out["mixed"] = buf.getvalue()
    deep = Image.fromarray(np.asarray(frame.convert("L")).astype(np.uint16) * 257)
    buf = io.BytesIO()
    deep.save(buf, "TIFF", save_all=True, append_images=[deep.copy()], dpi=(200, 200))
    out["16-bit"] = buf.getvalue()
    exif = frame.getexif()
    exif[0x0112] = 6                                              # stored rotated
    buf = io.BytesIO()
    frame.save(buf, "TIFF", save_all=True, append_images=[frame.copy()], exif=exif,
               dpi=(200, 200))
    out["oriented"] = buf.getvalue()
    no_unit, inch = io.BytesIO(), io.BytesIO()     # a resolution with no unit, then 300 dpi
    frame.save(no_unit, "TIFF", tiffinfo={296: 1, 282: 300, 283: 300})
    frame.save(inch, "TIFF", dpi=(300, 300))
    buf = io.BytesIO()
    Image.open(no_unit).save(buf, "TIFF", save_all=True, append_images=[Image.open(inch)])
    out["no-unit"] = buf.getvalue()
    return out, words


def _as_tuples(frames):
    return [(g.shape, g.tobytes(), None if u is None else (u.shape, u.tobytes()), d)
            for g, u, d in frames]


def test_the_output_is_what_decoding_every_frame_first_gives(monkeypatch, tmp_path):
    tiffs, words = _tiffs()
    for name, data in tiffs.items():
        lazy = st.detect_image_file(data, ocr=lambda rgb: list(words))
        with monkeypatch.context() as m:
            m.setattr(detect, "_frames", lambda source, *a: detect._image_frames(source, *a))
            whole = st.detect_image_file(data, ocr=lambda rgb: list(words))
        assert lazy == whole, name
        want = _as_tuples(detect._image_frames(data))
        path = tmp_path / f"{name}.tif"
        path.write_bytes(data)
        with open(path, "rb") as f:
            for source in (data, path, f):
                frames = detect._frames(source)
                assert isinstance(frames, detect._TiffFrames), name
                assert _as_tuples(frames) == want, (name, source)


def _damaged():
    """A 3-frame TIFF whose last frame is cut short: its headers walk, its pixels do not decode."""
    tiff, words = _tiff(3)
    data = tiff[:int(len(tiff) * 0.8)]
    assert isinstance(detect._frames(data), detect._TiffFrames)
    return data, words


def _fails(rgb):
    raise RuntimeError("the OCR backend failed")


@pytest.mark.parametrize("call", [
    {"words_by_page": {i: [] for i in range(3)}},
    {"words_by_page": {0: []}},                                   # frames without words
    {"words_by_page": {i: [] for i in range(4)}},                 # a frame past the last
    {"words": []},                                                # words= on a multi-page file
    {"ocr": _fails}], ids=["all", "missing", "past", "words", "ocr-fails"])
def test_a_damaged_tiff_raises_what_decoding_every_frame_first_raises(call):
    data, _ = _damaged()
    with pytest.raises(Exception) as whole:
        detect._image_frames(data)
    with pytest.raises(type(whole.value)) as lazy:
        st.detect_image_file(data, **call)
    assert str(lazy.value) == str(whole.value)


def test_an_unreadable_file_raises_its_own_error_alone():
    with pytest.raises(Exception) as unread:
        st.detect_image_file(b"not an image", ocr=lambda rgb: [])
    assert unread.value.__context__ is None and unread.value.__cause__ is None


def test_a_pil_image_is_decoded_whole_with_the_frames_it_holds(log):
    tiff, words = _tiff(2)
    img = Image.open(io.BytesIO(tiff))
    res = st.detect_image_file(img, ocr=_ocr(log, words))
    assert log == ["decode", "decode", "ocr", "ocr"] and res["page_count"] == 2
    made = Image.new("L", (400, 160), 255)                       # frames that live in memory
    assert st.detect_image_file(made, words=[], dpi=200)["page_count"] == 1


@pytest.mark.parametrize("fmt", ["GIF", "PNG"])
def test_an_animated_image_is_decoded_whole_as_before(log, fmt):
    frame, words = _frame()
    buf = io.BytesIO()
    frame.convert("L").save(buf, fmt, save_all=True,
                            append_images=[frame.convert("L").rotate(5 * i) for i in (1, 2)])
    if Image.open(io.BytesIO(buf.getvalue())).n_frames != 3:
        pytest.skip(f"this Pillow writes no animated {fmt}")
    st.detect_image_file(buf.getvalue(), ocr=_ocr(log, words))
    assert log == ["decode"] * 3 + ["ocr"] * 3


def test_a_tiff_over_a_limit_is_refused_before_any_frame_is_decoded(log):
    tiff, words = _tiff(3)
    with pytest.raises(st.LimitExceededError, match="at least 2 pages"):
        st.detect_image_file(tiff, ocr=_ocr(log, words), max_pages=1)
    w, h = Image.open(io.BytesIO(tiff)).size
    with pytest.raises(st.LimitExceededError, match="max_mpix"):
        st.detect_image_file(tiff, ocr=_ocr(log, words), max_mpix=1.5 * w * h / 1e6)
    assert log == []
    assert st.detect_image_file(tiff, ocr=_ocr(log, words), max_pages=3)["page_count"] == 3


def test_the_image_is_closed_after_detection_and_after_an_error(monkeypatch):
    tiff, words = _tiff(2)
    opened = []
    real = Image.open
    monkeypatch.setattr(Image, "open", lambda *a, **k: opened.append(real(*a, **k)) or opened[-1])
    st.detect_image_file(tiff, words_by_page={0: words, 1: words})
    with pytest.raises(st.OcrRequiredError) as stopped:
        st.detect_image_file(tiff, words_by_page={0: words})
    with pytest.raises(st.LimitExceededError):
        st.detect_image_file(tiff, words_by_page={0: words, 1: words}, max_pages=1)
    assert opened and all(getattr(im, "fp", None) is None for im in opened)
    assert stopped.value                                          # even while the error is kept


class _Reads(io.BytesIO):
    """An open file that records the size each read() asks for."""

    def __init__(self, data):
        super().__init__(data)
        self.sizes = []

    def read(self, size=-1):
        self.sizes.append(size)
        return super().read(size)


@pytest.mark.parametrize("limit", [{}, {"max_pages": 10}, {"max_mpix": 100}],
                         ids=["no-limit", "max_pages", "max_mpix"])
def test_an_open_tiff_is_read_in_place_and_closed_as_any_image_is(limit):
    tiff, words = _tiff(3)
    f = _Reads(tiff)
    f.seek(7)
    assert st.detect_image_file(f, ocr=lambda rgb: list(words), **limit)["page_count"] == 3
    assert f.sizes and all(s is not None and 0 <= s < len(tiff) for s in f.sizes)
    png = _Reads(_redline_image_png()[0])
    st.detect_image_file(png, ocr=lambda rgb: list(words), **limit)
    assert f.closed and png.closed                 # Pillow closes the file it decodes


def test_a_refused_open_file_is_left_open_at_its_start(tmp_path):
    tiff, _ = _tiff(3)
    png, _ = _redline_image_png()
    for name, data, limit in [("scan.tif", tiff, {"max_pages": 1}),
                              ("scan.png", png, {"max_mpix": 0.01})]:
        path = tmp_path / name
        path.write_bytes(data)
        with open(path, "rb") as f:
            f.seek(7)
            with pytest.raises(st.LimitExceededError):
                st.detect_image_file(f, ocr=lambda rgb: [], **limit)
            assert not f.closed and f.tell() == 0, name


def test_a_tiff_with_a_frame_over_the_bomb_limit_is_decoded_whole(log, monkeypatch):
    """So Pillow's decompression-bomb check warns or raises before any frame is detected."""
    frame, words = _frame()
    buf = io.BytesIO()
    frame.save(buf, "TIFF", save_all=True, dpi=(200, 200),
               append_images=[frame.copy(), frame.resize((frame.width * 3, frame.height))])
    px = frame.width * frame.height
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 2 * px)        # Pillow warns on the last frame
    with pytest.warns(Image.DecompressionBombWarning):
        st.detect_image_file(buf.getvalue(), ocr=_ocr(log, words))
    assert log == ["decode"] * 3 + ["ocr"] * 3
    log.clear()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 6 * px // 5)   # and raises on it
    with pytest.raises(Image.DecompressionBombError):
        st.detect_image_file(buf.getvalue(), ocr=_ocr(log, words))
    assert log == ["decode", "decode"]
