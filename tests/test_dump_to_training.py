"""The dump -> train loop. `dump_crops` writes the crop StrikeNet scored for each word as a PNG
and a manifest row, and `training/train_strikenet.py` reads the two back as a training input and
its label. 0.14.0 wrote each PNG up to a grey level darker (it truncated), and the trainer
standardized the PNG again, which can change a crop the detector scored. Now the PNG is rounded,
the row records the `std_crop` settings the crop was made with, the trainer takes such a crop
as dumped, and the meta records the settings the crops were made with. The manifest reader also
takes a byte-order mark, UTF-16, a U+2028 in OCR text and labels that are not text, and names the
line of a row it cannot read. The training script is not part of the package, so its tests run
from a source checkout and skip against a wheel."""
import hashlib
import importlib.util
import io
import json
import pathlib
import re

import numpy as np
import pymupdf
import pytest
from PIL import Image

import pdf_strikethrough as st
from pdf_strikethrough import active, cnn
from pdf_strikethrough.ocr import Word
from _synth import _redline_image_png

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "training" / "train_strikenet.py"
BOM = b"\xef\xbb\xbf"


@pytest.fixture(scope="module")
def trainer():
    if not SCRIPT.is_file():
        pytest.skip("running against an installed package, not a source checkout")
    spec = importlib.util.spec_from_file_location("train_strikenet", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _settings(**changed):
    return {"paper_white_from": cnn.PAPER_WHITE_FROM, "shade_white_frac": cnn.SHADE_WHITE_FRAC,
            **changed}


def _manifest(root, rows, prefix=b""):
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (root / "crops.jsonl").write_bytes(prefix + text.encode("utf-8"))


def _level(k, shape=(cnn.CROP_H, cnn.CROP_W)):
    """Grey level k as a standardized crop holds it, computed as std_crop computes it."""
    return 1.0 - np.full(shape, k, np.uint8).astype(np.float32) / 255.0


def _faint_ground_crop():
    """A standardized crop that keeps a faint grey ground at 245, as the detector leaves noisy
    near-white paper whose commonest light pixel is 255: std_crop run again on its PNG, whose
    commonest light pixel is 245, would take that ground to white."""
    std = _level(245)
    for x in range(20, 140, 12):
        std[10:22, x:x + 5] = _level(26)[0, 0]
    return std


def _dumped(root, std, name="c.png"):
    (root / "crops").mkdir(exist_ok=True)
    active._save_crop_png(std, root / "crops" / name)
    return f"crops/{name}"


def _digest(root, *lines):
    return hashlib.sha256("".join(
        f"{hashlib.sha256((root / crop).read_bytes()).hexdigest()} {rest}\n"
        for crop, rest in lines).encode("utf-8")).hexdigest()


def _scored_dump(root, monkeypatch, png, words):
    """dump_crops on an image, with the crops StrikeNet scored, the rows relabeled struck."""
    scored, save = [], active._save_crop_png
    monkeypatch.setattr(active, "_save_crop_png",
                        lambda std, path: (scored.append(np.array(std)), save(std, path)))
    st.dump_crops(png, str(root), words_by_page={0: words}, image=True)
    rows = [json.loads(s) for s in (root / "crops.jsonl").read_text("utf-8").splitlines()]
    assert rows and len(rows) == len(scored)
    _manifest(root, [{**r, "label": "struck"} for r in rows])
    return rows, np.stack(scored)


def _noisy_long_words_png(sigma=6, seed=0):
    """PNG bytes of long struck words on noisy paper, and their Word boxes. The first word is
    wider than CROP_W once scaled, so its crop has no white side padding, and std_crop's resize
    smooths the noise to a light ground below 255: std_crop run again on its PNG changes it."""
    doc = pymupdf.open()
    page = doc.new_page(width=560, height=120)
    page.insert_text((20, 60), "acknowledgements responsibilities", fontsize=17)
    found = page.get_text("words")
    for w in found:
        r = pymupdf.Rect(w[:4])
        y = (r.y0 + r.y1) / 2 + 0.5
        page.draw_line(pymupdf.Point(r.x0, y), pymupdf.Point(r.x1, y), width=1.6)
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    W, H = page.rect.width, page.rect.height
    words = [Word(w[4], (w[0] / W, w[1] / H, w[2] / W, w[3] / H)) for w in found]
    doc.close()
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    noisy = gray + np.random.default_rng(seed).normal(0, sigma, gray.shape)
    buf = io.BytesIO()
    Image.fromarray(np.rint(np.clip(noisy, 0, 255)).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue(), words


def test_a_dump_png_holds_each_grey_level_exactly(tmp_path):
    """Every grey level a standardized crop can hold comes back from its PNG as itself (0.14.0
    truncated, writing 95 of the 256 levels one darker)."""
    for k in range(256):
        active._save_crop_png(_level(k), tmp_path / "c.png")
        assert np.all(np.asarray(Image.open(tmp_path / "c.png")) == k), k


def test_a_dumped_crop_trains_as_the_detector_scored_it(tmp_path, monkeypatch, trainer, capsys):
    """Each row records the std_crop settings its crop was made with, and once the rows are
    labeled, each crop reaches training exactly as the detector scored it."""
    monkeypatch.setattr(cnn, "PAPER_WHITE_FROM", 250)     # values no default or literal gives:
    monkeypatch.setattr(cnn, "SHADE_WHITE_FRAC", 0.83)    # recorded as set
    rows, scored = _scored_dump(tmp_path, monkeypatch, *_redline_image_png())
    assert all(r["std_crop"] == {"paper_white_from": 250, "shade_white_frac": 0.83} for r in rows)
    assert np.array_equal(trainer.load_dataset(tmp_path)[0], scored)
    assert "std_crop settings" not in capsys.readouterr().out


def test_a_dumped_crop_a_second_std_crop_would_change_trains_as_scored(tmp_path, monkeypatch,
                                                                         trainer):
    rows, scored = _scored_dump(tmp_path, monkeypatch, *_noisy_long_words_png())
    again = [cnn.std_crop(np.asarray(Image.open(tmp_path / r["crop"]))) for r in rows]
    # a second std_crop changes one of these crops, so the test tells the two reads apart
    assert any(not np.array_equal(a, s) for a, s in zip(again, scored))
    assert np.array_equal(trainer.load_dataset(tmp_path)[0], scored)


def test_numpy_settings_are_recorded_as_numbers(tmp_path, monkeypatch, trainer):
    """A sweep over np.arange sets numpy scalars, which json cannot write: the dump records the
    numbers they hold, and the trainer takes the rows, whose settings are this run's, as dumped."""
    monkeypatch.setattr(cnn, "PAPER_WHITE_FROM", np.int64(250))
    monkeypatch.setattr(cnn, "SHADE_WHITE_FRAC", np.float32(0.83))
    rows, scored = _scored_dump(tmp_path, monkeypatch, *_redline_image_png())
    assert all(r["std_crop"] == {"paper_white_from": 250,
                                 "shade_white_frac": float(np.float32(0.83))} for r in rows)
    assert np.array_equal(trainer.load_dataset(tmp_path)[0], scored)


def test_a_crop_dumped_with_a_faint_ground_trains_with_it(tmp_path, trainer):
    std = _faint_ground_crop()
    name = _dumped(tmp_path, std)
    _manifest(tmp_path, [{"crop": name, "label": "clean", "std_crop": _settings()}])
    x, _, digest = trainer.load_dataset(tmp_path)
    assert np.array_equal(x[0], std)
    assert digest == _digest(tmp_path, (name, "clean as dumped"))


@pytest.mark.parametrize("settings", [{}, {"std_crop": None}, {"std_crop": {}}])
def test_a_row_without_the_settings_is_standardized_again_and_counted(tmp_path, trainer, capsys,
                                                                      settings):
    """A row from a dump made by 0.14.0 or earlier, or from elsewhere, carries no settings: its
    crop is standardized as before, and a note counts such rows."""
    name = _dumped(tmp_path, _faint_ground_crop())
    _manifest(tmp_path, [{"crop": name, "label": "clean", **settings}])
    x, _, digest = trainer.load_dataset(tmp_path)
    assert np.array_equal(x[0], cnn.std_crop(np.asarray(Image.open(tmp_path / name))))
    assert digest == _digest(tmp_path, (name, "clean"))
    assert "note: 1 row(s) carry no std_crop settings" in capsys.readouterr().out


def test_a_crop_dumped_with_other_settings_is_taken_as_dumped_and_counted(tmp_path, trainer,
                                                                           capsys):
    std = _faint_ground_crop()
    name = _dumped(tmp_path, std)
    _manifest(tmp_path, [{"crop": name, "label": "clean",
                          "std_crop": _settings(paper_white_from=240)}])
    x, _, digest = trainer.load_dataset(tmp_path)
    assert np.array_equal(x[0], std) and digest == _digest(tmp_path, (name, "clean as dumped"))
    assert ('note: 1 row(s) were dumped with std_crop settings {"paper_white_from": 240, '
            '"shade_white_frac": 0.85}, not this run'
            in capsys.readouterr().out)


def test_rows_made_with_this_runs_settings_train_together(tmp_path, trainer):
    """An old row without settings next to new rows dumped with this run's: the same settings,
    however a row writes them (255.0 for 255, keys in another order)."""
    std = _faint_ground_crop()
    marked = [_settings(), {"shade_white_frac": cnn.SHADE_WHITE_FRAC,
                            "paper_white_from": float(cnn.PAPER_WHITE_FROM)}, None]
    _manifest(tmp_path, [{"crop": _dumped(tmp_path, std, f"c{i}.png"), "label": "clean",
                          "std_crop": m} for i, m in enumerate(marked)])
    assert len(trainer.load_dataset(tmp_path)[0]) == 3


@pytest.mark.parametrize("second", ["none", "default"])
def test_crops_made_with_two_settings_stop_the_script(tmp_path, trainer, second):
    """A model's meta records a single paper_white_from and shade_white_frac, and a row without
    settings counts as made with this run's, since its crop is standardized here."""
    rows = [{"crop": _dumped(tmp_path, _faint_ground_crop(), f"c{i}.png"), "label": "clean",
             "std_crop": _settings(paper_white_from=240)} for i in range(2)]
    rows[1]["std_crop"] = None if second == "none" else _settings()
    _manifest(tmp_path, rows)
    with pytest.raises(SystemExit, match=re.escape(
            "made with different std_crop settings, and a model's meta records a single "
            'paper_white_from and shade_white_frac: 1 row(s) with {"paper_white_from": 240')):
        trainer.load_dataset(tmp_path)


@pytest.mark.parametrize("value", [
    False, 0, "no", [0], [255, 0.85], {"paper_white_from": 255},
    {"paper_white_from": "255", "shade_white_frac": 0.85},
    {"paper_white_from": True, "shade_white_frac": 0.85},
    {"paper_white_from": float("nan"), "shade_white_frac": 0.85},
    {"paper_white_from": 255, "shade_white_frac": 0.85, "light_from": 128}])
def test_a_std_crop_that_is_not_the_settings_stops_the_script(tmp_path, trainer, value):
    """Only the settings dump_crops writes mark a crop as dumped: a row whose "std_crop" holds
    anything else stops the script, naming its line (null, {} and blank text mark none)."""
    rows = [{"crop": _dumped(tmp_path, _faint_ground_crop(), f"c{i}.png"), "label": "clean",
             "std_crop": _settings()} for i in range(2)]
    rows[1]["std_crop"] = value
    _manifest(tmp_path, rows)
    with pytest.raises(SystemExit, match=r'crops\.jsonl:2: "std_crop" is .*, not the settings '
                                         r"dump_crops writes"):
        trainer.load_dataset(tmp_path)


def test_a_marked_crop_of_another_size_stops_the_script(tmp_path, trainer):
    (tmp_path / "crops").mkdir()
    Image.fromarray(np.full((24, 96), 255, np.uint8)).save(tmp_path / "crops" / "c.png")
    _manifest(tmp_path, [{"crop": "crops/c.png", "label": "clean", "std_crop": _settings()}])
    with pytest.raises(SystemExit, match=r"crops/c\.png is 96 x 24 px"):
        trainer.load_dataset(tmp_path)


def test_the_manifest_reader_takes_what_editors_and_ocr_write(tmp_path, trainer, capsys):
    """A byte-order mark (PowerShell 5.1 writes one with -Encoding UTF8), a U+2028 inside OCR text
    (dump_crops writes text unescaped), an empty spreadsheet cell's NaN, blank text, [] or {}, and
    labels that are not struck or clean, which are named rather than read as unlabeled."""
    std = _faint_ground_crop()
    labels = ["struck", float("nan"), 1, True, " Stuck", None, "clean", 1, "  ", [], {}]
    rows = [{"crop": _dumped(tmp_path, std, f"c{i}.png"), "label": lab, "std_crop": _settings(),
             "text": f"str{chr(0x2028)}uck"} for i, lab in enumerate(labels)]
    _manifest(tmp_path, rows, prefix=BOM)
    x, y, _ = trainer.load_dataset(tmp_path)
    assert y.tolist() == [1.0, 0.0] and len(x) == 2
    out = capsys.readouterr().out
    assert "note: skipped 5 unlabeled row(s)" in out
    assert 'note: skipped 4 row(s) labeled neither struck nor clean: 1, true, "stuck"\n' in out


def test_utf16_and_merged_manifests_are_read(tmp_path, trainer):
    """PowerShell 5.1's > and Out-File write UTF-16 with a byte-order mark and CRLF; manifests
    merged from files written with -Encoding UTF8 carry a byte-order mark per file."""
    rows = [{"crop": _dumped(tmp_path, _faint_ground_crop(), f"c{i}.png"), "label": "clean",
             "std_crop": _settings(), "text": "caf" + chr(0xE9)} for i in range(2)]
    _manifest(tmp_path, rows)
    want = trainer.load_dataset(tmp_path)
    lines = [json.dumps(r, ensure_ascii=False) + "\r\n" for r in rows]
    for data in ("".join(lines).encode("utf-16"),
                 b"\xfe\xff" + "".join(lines).encode("utf-16-be"),
                 BOM + lines[0].encode("utf-8") + BOM + lines[1].encode("utf-8")):
        (tmp_path / "crops.jsonl").write_bytes(data)
        got = trainer.load_dataset(tmp_path)
        assert np.array_equal(got[0], want[0]) and got[2] == want[2]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_a_manifest_not_in_utf8_is_named_by_its_line(tmp_path, trainer, newline):
    """PowerShell 5.1's Set-Content writes the system code page (cp1252 here)."""
    rows = [{"crop": _dumped(tmp_path, _faint_ground_crop()), "label": "clean"},
            {"crop": "crops/c.png", "label": "clean", "text": "caf" + chr(0xE9)}]
    (tmp_path / "crops.jsonl").write_bytes(
        "".join(json.dumps(r, ensure_ascii=False) + newline for r in rows).encode("cp1252"))
    with pytest.raises(SystemExit, match=r"crops\.jsonl:2: not UTF-8 text"):
        trainer.load_dataset(tmp_path)


@pytest.mark.parametrize("others, shown", [("abcdefa", '"a", "b", "c", "d", "e", ...'),
                                           ("abcde", '"a", "b", "c", "d", "e"')])
def test_the_note_names_five_other_labels_at_most(tmp_path, trainer, capsys, others, shown):
    labels = ["clean", *others]
    _manifest(tmp_path, [{"crop": _dumped(tmp_path, _faint_ground_crop(), f"c{i}.png"),
                          "label": lab, "std_crop": _settings()} for i, lab in enumerate(labels)])
    trainer.load_dataset(tmp_path)
    assert (f"note: skipped {len(others)} row(s) labeled neither struck nor clean: {shown}\n"
            in capsys.readouterr().out)


def test_the_note_shows_a_label_in_ascii_and_cuts_a_long_one(tmp_path, trainer, capsys):
    """A label is shown as ASCII JSON, so the note prints to a console in any code page, and one
    longer than 40 characters is cut."""
    labels = ["clean", "durchgestrichen " + chr(0x2713), "x" * 50]
    _manifest(tmp_path, [{"crop": _dumped(tmp_path, _faint_ground_crop(), f"c{i}.png"),
                          "label": lab, "std_crop": _settings()} for i, lab in enumerate(labels)])
    trainer.load_dataset(tmp_path)
    out = capsys.readouterr().out
    assert out.isascii() and "durchgestrichen" in out
    assert ', "' + "x" * 36 + "...\n" in out


@pytest.mark.parametrize("bad, says", [
    ('{"crop": "crops/c0.png", "label": ', "not a JSON row: Expecting value at column 35"),
    ('{"crop": "crops/c0.png", "label": "cl' + chr(1) + 'ean"}',
     "not a JSON row: Invalid control character at column"),
    ("null", "a row is a JSON object"), ("[1, 2]", "a row is a JSON object"),
    ('{"label": "clean"}', 'a labeled row needs a "crop"')])
def test_a_row_it_cannot_read_is_named_by_its_line(tmp_path, trainer, bad, says):
    rows = [{"crop": _dumped(tmp_path, _faint_ground_crop(), f"c{i}.png"), "label": "clean",
             "std_crop": _settings()} for i in range(2)]
    _manifest(tmp_path, rows)
    with open(tmp_path / "crops.jsonl", "a", encoding="utf-8", newline="") as f:
        f.write("\n" + bad + "\n")                      # a blank line still counts as a line
    with pytest.raises(SystemExit, match=re.escape(f"crops.jsonl:4: {says}")):
        trainer.load_dataset(tmp_path)


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_line_numbers_count_every_line_end(tmp_path, trainer, newline):
    row = {"crop": _dumped(tmp_path, _faint_ground_crop()), "label": "clean"}
    text = json.dumps(row) + newline + newline + "null" + newline
    (tmp_path / "crops.jsonl").write_bytes(text.encode("utf-8"))
    with pytest.raises(SystemExit, match=r"crops\.jsonl:3: a row is a JSON object"):
        trainer.load_dataset(tmp_path)


def _resaved(gray, mode):
    """A crop's grey levels as an image editor can save them: 16-bit, or with its paper
    transparent over black (grey+alpha, RGBA, or a palette with a transparent black entry)."""
    paper = gray == 255
    if mode == "I;16":
        return Image.fromarray(gray.astype(np.uint16) * 257)
    if mode == "I;16 0-255":                                   # 8-bit levels held in 16 bits
        return Image.fromarray(gray.astype(np.uint16))
    if mode == "LA":
        return Image.fromarray(np.stack([np.where(paper, 0, gray), np.where(paper, 0, 255)], -1)
                               .astype(np.uint8), "LA")
    if mode == "RGBA":
        ink = np.where(paper, 0, gray)
        return Image.fromarray(np.stack([ink, ink, ink, np.where(paper, 0, 255)], -1)
                               .astype(np.uint8), "RGBA")
    levels = sorted(set(gray[~paper].tolist()))                 # index 0: transparent black
    img = Image.fromarray(np.where(paper, 0, np.searchsorted(levels, gray) + 1).astype(np.uint8),
                          "P")
    img.putpalette([0, 0, 0] + [v for k in levels for v in (k, k, k)])
    img.info["transparency"] = 0
    return img


@pytest.mark.parametrize("mode", ["I;16", "I;16 0-255", "LA", "RGBA", "P"])
@pytest.mark.parametrize("marked", [True, False])
def test_a_crop_saved_16_bit_or_transparent_reads_as_its_grey_levels(tmp_path, trainer, mode,
                                                                    marked):
    """A crop re-saved as 16-bit read as blank paper, and one with its paper transparent over
    black read as solid ink: each now reads as the 8-bit crop it holds."""
    std = _level(255)
    std[8:24, 20:140:9] = _level(26)[0, 0]
    std[12:20, 30:60] = _level(128)[0, 0]
    std[12:20, 70:100] = _level(200)[0, 0]                     # 200 x 257 / 256 rounds to 201
    rows = [{"crop": _dumped(tmp_path, std, "a.png"), "label": "struck"},
            {"crop": "crops/b.png", "label": "struck"}]
    gray = np.asarray(Image.open(tmp_path / rows[0]["crop"]))
    _resaved(gray, mode).save(tmp_path / rows[1]["crop"])
    with Image.open(tmp_path / rows[1]["crop"]) as img:     # Pillow 10.1 opens 16-bit grey as I
        assert img.mode in ({"I;16", "I"} if mode.startswith("I;16") else {mode})
    for row in rows:
        if marked:
            row["std_crop"] = _settings()
    _manifest(tmp_path, rows)
    x = trainer.load_dataset(tmp_path)[0]
    assert np.array_equal(x[1], x[0]) and x[0].max() > 0.8


@pytest.mark.parametrize("content", [None, b"not a png"])
def test_a_labeled_row_whose_png_cannot_be_read_is_named_by_its_line(tmp_path, trainer, content):
    (tmp_path / "crops").mkdir()
    if content is not None:
        (tmp_path / "crops" / "c.png").write_bytes(content)
    _manifest(tmp_path, [{"crop": "crops/c.png", "label": "clean"}])
    with pytest.raises(SystemExit, match=r"crops\.jsonl:1: cannot read the crop crops/c\.png: "):
        trainer.load_dataset(tmp_path)
