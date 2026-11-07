"""A damaged image file that Pillow opens but cannot decode raises OSError naming the frame, as
an unreadable file does, rather than Pillow's own TypeError, KeyError, SyntaxError or ValueError
(or a numpy IndexError on a frame with no pixels); the CLI then exits 1 with the error. A PIL
image or file the caller closed still raises Pillow's ValueError."""
import io
import struct
import zlib

import pytest
from PIL import Image

import pdf_strikethrough as st
from pdf_strikethrough import __main__ as cli


def _tiff(frames=2, size=(40, 20), **info):
    buf = io.BytesIO()
    ims = [Image.new("L", size, 255) for _ in range(frames)]
    ims[0].save(buf, "TIFF", save_all=True, append_images=ims[1:], dpi=(200, 200), **info)
    return bytearray(buf.getvalue())


def _patch(data, ifd, tag, value=None, new_tag=None, new_type=None):
    """`data` (a little-endian TIFF) with one entry of frame `ifd` changed: its value, its tag or
    its type."""
    off = struct.unpack_from("<I", data, 4)[0]
    for _ in range(ifd):
        entries = struct.unpack_from("<H", data, off)[0]
        off = struct.unpack_from("<I", data, off + 2 + 12 * entries)[0]
    for i in range(struct.unpack_from("<H", data, off)[0]):
        e = off + 2 + 12 * i
        t, typ = struct.unpack_from("<HH", data, e)
        if t == tag:
            if new_tag is not None:
                struct.pack_into("<H", data, e, new_tag)
            elif new_type is not None:
                struct.pack_into("<H", data, e + 2, new_type)
            else:
                struct.pack_into("<H" if typ == 3 else "<I", data, e + 8, value)
            return bytes(data)
    raise KeyError(tag)


def _chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def _png_broken_mid_data():
    """A PNG whose pixel data runs on into a chunk with a type that is not letters."""
    z = zlib.compress(b"".join(b"\0" + b"\xff" * 40 for _ in range(20)))
    half = len(z) // 2
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", 40, 20, 8, 0, 0, 0, 0))
            + _chunk(b"IDAT", z[:half]) + b"\0\0\0\x10\x00\xc7\xf1s" + z[half:] + b"\0" * 8
            + _chunk(b"IEND", b""))


DAMAGED = {
    "TIFF frame with no width": (_patch(_tiff(), 1, 256, new_tag=0x9000), 1),     # TypeError
    "TIFF frame of an unknown compression": (_patch(_tiff(), 1, 259, 9), 1),       # KeyError 9
    "TIFF frame whose width is text": (_patch(_tiff(), 1, 256, new_type=2), 1),    # ValueError
    "PNG broken in its pixel data": (_png_broken_mid_data(), 0),                   # SyntaxError
    "TIFF frame 0 px wide": (_patch(_tiff(3, (400, 150)), 1, 256, 0), 1),          # no pixels
}
HOW = ["bytes", "path", "file", "stream", "image"]


class _Stream:
    """A binary stream with no ``closed`` attribute, which Pillow reads as it reads a file."""

    def __init__(self, data):
        b = io.BytesIO(data)
        self.read, self.seek, self.tell, self.close = b.read, b.seek, b.tell, b.close


def _detect(data, how, tmp_path):
    """detect_image_file on `data` given as bytes, a path, an open file, a stream or a PIL
    image."""
    path = tmp_path / ("d.png" if data.startswith(b"\x89PNG") else "d.tif")
    path.write_bytes(data)
    words_by_page = {i: [] for i in range(3)}
    if how == "stream":
        return st.detect_image_file(_Stream(data), words_by_page=words_by_page)
    if how == "file":
        with open(path, "rb") as f:
            return st.detect_image_file(f, words_by_page=words_by_page)
    if how == "image":
        with Image.open(path) as img:
            return st.detect_image_file(img, words_by_page=words_by_page)
    return st.detect_image_file(data if how == "bytes" else path, words_by_page=words_by_page)


@pytest.mark.parametrize("name", DAMAGED)
@pytest.mark.parametrize("how", HOW)
def test_a_damaged_image_raises_oserror_naming_the_frame(name, how, tmp_path):
    data, frame = DAMAGED[name]
    with pytest.raises(OSError, match=rf"^frame {frame} "):
        _detect(data, how, tmp_path)


def test_the_error_says_what_pillow_raised():
    data, _ = DAMAGED["TIFF frame of an unknown compression"]
    with pytest.raises(OSError) as e:
        st.detect_image_file(data, words_by_page={0: [], 1: []})
    assert str(e.value) == "frame 1 cannot be decoded (KeyError: 9)"
    assert isinstance(e.value.__cause__, KeyError)


@pytest.mark.parametrize("tag, info", [(256, {}), (257, {}), (256, {"tiffinfo": {274: 6}})],
                         ids=["0 px wide", "0 px tall", "0 px wide, stored turned"])
@pytest.mark.parametrize("how", HOW)
def test_a_frame_with_no_pixels_is_named(tag, info, how, tmp_path):
    # it is not decoded: Pillow's own error for it differs with the codec and the source, and
    # the order of its size on a turned TIFF with the Pillow version
    with pytest.raises(OSError, match=r"^frame 1 has no pixels$"):
        _detect(_patch(_tiff(3, (400, 150), **info), 1, tag, 0), how, tmp_path)


def test_a_closed_image_or_file_is_not_called_damaged(tmp_path):
    path = tmp_path / "a.png"
    Image.new("L", (40, 20), 255).save(path)
    img = Image.open(path)
    img.close()
    with pytest.raises(ValueError, match="closed image"):
        st.detect_image_file(img, words=[])
    with open(path, "rb") as f:
        img = Image.open(f)
    with pytest.raises(ValueError, match="closed file"):
        st.detect_image_file(img, words=[])
    tif = tmp_path / "a.tif"
    tif.write_bytes(bytes(_tiff()))
    with open(tif, "rb") as f:
        img = Image.open(f)
        img.seek(1)                             # so reading frame 0 again needs its file
    with pytest.raises(ValueError, match="closed file"):
        st.detect_image_file(img, words_by_page={0: [], 1: []})


@pytest.mark.parametrize("name", DAMAGED)
def test_the_cli_exits_1_with_the_error(name, tmp_path, capsys):
    data, frame = DAMAGED[name]
    path = tmp_path / ("d.png" if data.startswith(b"\x89PNG") else "d.tif")
    path.write_bytes(data)
    # with no words, the damaged frame's error still comes first: exit 1, not 2 (no OCR words)
    assert cli.main(["detect", str(path)]) == 1
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1 and lines[0].startswith(f"error: cannot read {path}: frame {frame} ")
