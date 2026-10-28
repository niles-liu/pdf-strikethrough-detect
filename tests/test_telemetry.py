"""Run telemetry: ``telemetry=True`` on detect_pdf / detect_image_file and the CLI's --telemetry.
It reports per-page timing, candidates per tier (layer 1's on a scanned page, the weak ones the
CNN pass drops included) and a cnn_prob histogram, and changes nothing else in the result."""
import io
import json
import os
import sys
import time
import weakref
import zipfile

import numpy as np
import pymupdf
import pytest
from PIL import Image

import pdf_strikethrough as st
from pdf_strikethrough import __main__ as cli
from pdf_strikethrough import _batch, cnn, detect
from pdf_strikethrough.ocr import Word

from _synth import _mixed_native_scanned_pdf, _redline_image_png, _scanned_pdf, _synthetic_native_pdf


def _page_words(page):
    W, H = page.rect.width, page.rect.height
    return [Word(w[4], (w[0] / W, w[1] / H, w[2] / W, w[3] / H)) for w in page.get_text("words")]


def _mixed():
    """The mixed PDF's bytes and words for its scanned page 1 (an image of page 0)."""
    doc = _mixed_native_scanned_pdf()
    return doc.tobytes(), {1: _page_words(doc[0])}


def _layer1(monkeypatch, tiers, probs=None):
    """Layer 1 replaced by one candidate per tier in `tiers`, side by side, and StrikeNet by
    `probs` in candidate order (default: all 1.0). A tier given as ("review", None) gets a box too
    small to crop, so its cnn_prob is None."""
    recs = []
    for i, t in enumerate(tiers):
        small = isinstance(t, tuple)
        box = (0.0, 0.0, 0.0, 0.0) if small else (0.05 + 0.1 * i, 0.4, 0.13 + 0.1 * i, 0.5)
        recs.append({"text": f"w{i}", "chars": f"w{i}", "char_span": (0, 2), "partial": False,
                     "bbox_frac": box, "tier": t[0] if small else t, "score": 0.5})
    monkeypatch.setattr(detect, "analyze_scanned_page",
                        lambda gray, words, **kw: ([], [dict(r) for r in recs]))
    seq = iter(probs) if probs is not None else None
    monkeypatch.setattr(cnn, "score_crops",
                        lambda crops, **kw: np.array([next(seq) if seq else 1.0 for _ in crops]))


def _blank_png():
    pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 600, 200), False)
    pix.set_rect(pix.irect, (255,))
    return pix.tobytes("png")


_W = [Word("w", (0.1, 0.4, 0.2, 0.5))]


# ----------------------------------------------------------------------------- the result key

def test_off_by_default_and_changes_nothing_else():
    pdf, wbp = _mixed()
    plain = st.detect_pdf(pdf, words_by_page=wbp)
    with_tel = st.detect_pdf(pdf, words_by_page=wbp, telemetry=True)
    assert "telemetry" not in plain
    tel = with_tel.pop("telemetry")
    assert with_tel == plain
    assert json.loads(json.dumps(tel)) == tel                       # JSON-ready as it stands
    assert tel["telemetry_version"] == detect.TELEMETRY_VERSION == 1
    assert tel["package"] == st.__version__
    assert [p["source"] for p in tel["pages"]] == plain["page_sources"]
    assert tel["final"] == sum(p["final"] for p in tel["pages"]) == plain["n_struck_final"]


def test_native_page_counts_its_records_by_tier_and_runs_no_model():
    res = st.detect_pdf(_synthetic_native_pdf().tobytes(), telemetry=True)
    tel = res["telemetry"]
    (page,) = tel["pages"]
    assert page["source"] == "native" and set(page["seconds"]) == {"detect"}
    assert page["tiers"] == tel["tiers"] == {"vector": 2} and page["final"] == 2
    assert "dpi" not in page and "ocr_words" not in page
    assert tel["model"] is None                                     # nothing reached StrikeNet
    assert tel["cnn_prob"] == {"edges": list(detect.CNN_PROB_EDGES), "counts": [0] * 10,
                               "unscored": 0, "invalid": 0}


def test_a_scanned_page_read_by_the_native_detector_counts_its_records():
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 80, 40), False)
    pix.set_rect(pix.irect, (250,))
    page.insert_image(page.rect, pixmap=pix)                        # a scan ...
    page.insert_text((50, 100), "keep deleted text", fontsize=12, render_mode=3)   # ... with OCR
    r = {w[4]: w[:4] for w in page.get_text("words")}["deleted"]
    page.draw_line((r[0], (r[1] + r[3]) / 2), (r[2], (r[1] + r[3]) / 2), width=1.0)
    with pytest.warns(UserWarning, match="extractable text"):
        tel = st.detect_pdf(doc.tobytes(), telemetry=True)["telemetry"]
    (pg,) = tel["pages"]
    assert pg["source"] == "scanned" and pg["tiers"] == {"vector": 1} and pg["final"] == 1
    assert tel["model"] is None


def test_scanned_tiers_are_layer_1s_with_the_weak_ones(monkeypatch):
    _layer1(monkeypatch, ["auto", "review", "weak", "weak"])
    res = st.detect_image_file(_blank_png(), words=_W, dpi=200, telemetry=True)
    (page,) = res["telemetry"]["pages"]
    assert page["tiers"] == {"auto": 1, "review": 1, "weak": 2}
    assert sorted(w["tier"] for w in res["words"]) == ["auto", "review"]   # weak never reported
    assert page["final"] == 2 and page["ocr_words"] == 1
    assert detect._LAYER1_TIERS.get() is None                       # nothing left counting


def test_the_document_sums_its_pages_tiers(monkeypatch):
    _layer1(monkeypatch, ["auto", "weak"])
    buf = io.BytesIO()
    frames = [Image.new("L", (600, 200), 255) for _ in range(3)]
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
    tel = st.detect_image_file(buf.getvalue(), words_by_page={0: _W, 1: _W, 2: _W}, dpi=200,
                               telemetry=True)["telemetry"]
    assert [p["page"] for p in tel["pages"]] == [0, 1, 2]
    assert tel["tiers"] == {"auto": 3, "weak": 3} and tel["final"] == 3


def test_cnn_prob_histogram_bins_at_their_lower_edge_and_keeps_the_rest_apart(monkeypatch):
    _layer1(monkeypatch, ["review"] * 7 + [("review", None)],
            probs=[0.0, 0.1, 0.35, 0.9, 1.0, float("nan"), 1.5])
    res = st.detect_image_file(_blank_png(), words=_W, dpi=200, telemetry=True)
    hist = res["telemetry"]["cnn_prob"]
    assert hist["edges"] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert hist["counts"] == [1, 1, 0, 1, 0, 0, 0, 0, 0, 2]          # 0.1 opens bin 1; 1.0 is in 9
    assert hist["unscored"] == 1                                     # the box too small to crop
    assert hist["invalid"] == 2                                      # NaN, and 1.5


def test_seconds_name_the_steps_each_page_ran():
    pdf, wbp = _mixed()
    given = st.detect_pdf(pdf, words_by_page=wbp, telemetry=True)["telemetry"]["pages"]
    assert [set(p["seconds"]) for p in given] == [{"detect"}, {"render", "detect"}]
    ocr = lambda rgb: wbp[1]                                        # noqa: E731
    ran = st.detect_pdf(pdf, ocr=ocr, telemetry=True)["telemetry"]["pages"]
    assert set(ran[1]["seconds"]) == {"render", "ocr", "detect"}
    assert all(v >= 0 and round(v, 4) == v for p in ran for v in p["seconds"].values())
    png, words = _redline_image_png()
    img = st.detect_image_file(png, words=words, telemetry=True)["telemetry"]["pages"]
    assert set(img[0]["seconds"]) == {"detect"}
    img_ocr = st.detect_image_file(png, ocr=lambda rgb: words, telemetry=True)["telemetry"]
    assert set(img_ocr["pages"][0]["seconds"]) == {"ocr", "detect"}
    assert img_ocr["seconds"] >= sum(img_ocr["pages"][0]["seconds"].values())


def test_a_step_timed_twice_adds_up():
    pt = {"seconds": {}}
    with detect._clock(pt, "render"):
        time.sleep(0.01)
    first = pt["seconds"]["render"]
    with detect._clock(pt, "render"):
        time.sleep(0.01)
    assert pt["seconds"]["render"] >= first + 0.009
    with detect._clock(None, "render"):                             # telemetry off: nothing kept
        pass


def test_scanned_page_reports_its_dpi_words_and_the_model_that_scored():
    pdf, wbp = _mixed()
    tel = st.detect_pdf(pdf, words_by_page=wbp, dpi=150, telemetry=True)["telemetry"]
    page = tel["pages"][1]
    assert page["dpi"] == 150 and page["ocr_words"] == len(wbp[1])
    meta = st.get_model_meta()
    assert tel["model"] == {"version": meta["version"], "sha256": meta["sha256"]}
    png, words = _redline_image_png(dpi=300)
    img = st.detect_image_file(png, words=words, telemetry=True)["telemetry"]
    assert img["pages"][0]["dpi"] == 300 and img["pages"][0]["page"] == 0
    assert img["model"] == tel["model"]


def test_the_model_is_the_one_that_scored_and_the_report_never_loads_one(tmp_path):
    pdf, wbp = _mixed()
    meta = st.get_model_meta()
    def gone_after_the_last_page(done, total, pno):
        if done == total:                    # page 1, the scanned one, is the last
            cnn.set_model_dir(str(tmp_path))
    try:
        tel = st.detect_pdf(pdf, words_by_page=wbp, telemetry=True,
                            progress=gone_after_the_last_page)["telemetry"]
        assert tel["model"]["sha256"] == meta["sha256"]
        # nothing to score and a caller's own thresholds: no model is loaded, none is reported
        res = st.detect_image_file(_blank_png(), words=_W, dpi=200, telemetry=True,
                                   meta={"p_hi": 0.85, "p_lo": 0.15})
        assert res["telemetry"]["model"] is None
    finally:
        cnn.set_model_dir(None)


def test_skipped_and_unprocessed_pages():
    tel = st.detect_pdf(_scanned_pdf(2), on_missing_ocr="skip", telemetry=True)["telemetry"]
    assert [(p["page"], p.get("skipped")) for p in tel["pages"]] == [(0, True), (1, True)]
    assert tel["model"] is None and tel["final"] == 0
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page()
    tel = st.detect_pdf(doc.tobytes(), pages=[2, 0], telemetry=True)["telemetry"]
    assert [p["page"] for p in tel["pages"]] == [0, 2]               # the processed pages only


def test_words_given_as_an_iterator():
    png, words = _redline_image_png()
    plain = st.detect_image_file(png, words=iter(words))
    res = st.detect_image_file(png, words=iter(words), telemetry=True)
    assert "ocr_words" not in res.pop("telemetry")["pages"][0]      # its length is gone
    assert res == plain


def test_a_replaced_detect_scanned_image_still_works(monkeypatch):
    calls = []

    def old_style(gray, words, config=None, meta=None, dpi=200, crop_sink=None):
        calls.append(dpi)
        return []
    monkeypatch.setattr(detect, "detect_scanned_image", old_style)
    pdf, wbp = _mixed()
    st.detect_pdf(pdf, words_by_page=wbp)
    tel = st.detect_pdf(pdf, words_by_page=wbp, telemetry=True)["telemetry"]
    assert len(calls) == 2 and tel["pages"][1]["tiers"] == {}


def test_the_ocr_raster_is_freed_before_the_page_is_detected(monkeypatch):
    pdf, wbp = _mixed()
    refs, alive = [], []

    def ocr(rgb):
        refs.append(weakref.ref(rgb))
        return wbp[1]
    real = detect.detect_scanned_image

    def spy(*a, **kw):
        alive.append(refs[-1]() is not None)
        return real(*a, **kw)
    monkeypatch.setattr(detect, "detect_scanned_image", spy)
    for telemetry in (False, True):
        st.detect_pdf(pdf, ocr=ocr, telemetry=telemetry)
    assert alive == [False, False]


# ----------------------------------------------------------------------------------- the CLI

def _struck_pdf(path):
    path.write_bytes(_synthetic_native_pdf().tobytes())
    return path


def _docx(path):
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml",
                   f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{ns}"><w:body>'
                   f'<w:p><w:r><w:rPr><w:strike/></w:rPr><w:t>gone</w:t></w:r></w:p>'
                   f'</w:body></w:document>')
    path.write_bytes(buf.getvalue())
    return path


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fake_ocr(monkeypatch, words):
    """--ocr rapidocr answered by a callable returning `words`, in this process."""
    fake = lambda rgb: words                                        # noqa: E731
    for mod in (cli, _batch):
        monkeypatch.setattr(mod, "_check_ocr_available", lambda name: None)
    monkeypatch.setattr(cli, "_build_ocr", lambda name: fake)
    monkeypatch.setattr(_batch, "_get_ocr_cached", lambda name: fake)


def test_cli_appends_one_line_per_run(tmp_path, capsys):
    src, tel = _struck_pdf(tmp_path / "a.pdf"), tmp_path / "runs.jsonl"
    for _ in range(2):
        assert cli.main(["detect", str(src), "--telemetry", str(tel)]) == 0
    lines = _lines(tel)
    assert len(lines) == 2
    assert all(ln["source"] == str(src) and ln["telemetry_version"] == 1 for ln in lines)
    assert lines[0]["tiers"] == {"vector": 2} and lines[0]["pages"][0]["source"] == "native"
    assert "appended run telemetry" in capsys.readouterr().err
    docx, tel2 = _docx(tmp_path / "b.docx"), tmp_path / "docx.jsonl"
    assert cli.main(["detect", str(docx), "--telemetry", str(tel2)]) == 0
    (line,) = _lines(tel2)
    assert line["source"] == str(docx) and line["tiers"] == {"docx": 1} and line["final"] == 1
    assert line["pages"] == [] and line["model"] is None


def test_cli_image_input_single_and_batch(tmp_path, monkeypatch):
    png, words = _redline_image_png()
    (tmp_path / "p.png").write_bytes(png)
    _fake_ocr(monkeypatch, words)
    tel = tmp_path / "tel.jsonl"
    assert cli.main(["detect", str(tmp_path / "p.png"), "--ocr", "rapidocr",
                     "--telemetry", str(tel)]) == 0
    assert cli.main(["detect", str(tmp_path / "p.png"), "--ocr", "rapidocr",
                     "--jsonl", str(tmp_path / "o.jsonl"), "--telemetry", str(tel)]) == 0
    lines = _lines(tel)
    assert len(lines) == 2 and all("error" not in ln for ln in lines)
    assert all(ln["pages"][0]["ocr_words"] == len(words) and ln["model"] for ln in lines)


@pytest.mark.parametrize("jobs", ["1", "2"])
def test_cli_batch_writes_a_line_per_file_failures_included(tmp_path, jobs):
    d = tmp_path / "in"
    d.mkdir()
    _struck_pdf(d / "a.pdf")
    _docx(d / "b.docx")
    (d / "c.pdf").write_bytes(b"not a pdf")
    out, tel = tmp_path / "out.jsonl", tmp_path / "tel.jsonl"
    for _ in range(2):                                             # a second run appends
        assert cli.main(["detect", str(d), "--jsonl", str(out), "--telemetry", str(tel),
                         "--jobs", jobs]) == 1                      # c.pdf errors
    lines = _lines(tel)
    assert len(lines) == 6
    by_name = {os.path.basename(ln["source"]): ln for ln in lines}
    assert set(by_name) == {"a.pdf", "b.docx", "c.pdf"}
    assert by_name["a.pdf"]["final"] == 2 and by_name["b.docx"]["tiers"] == {"docx": 1}
    assert set(by_name["c.pdf"]) == {"telemetry_version", "source", "error"}
    assert all("telemetry" not in r for r in _lines(out))           # results stay as they were


def test_cli_batch_carries_on_when_telemetry_cannot_be_written(tmp_path, monkeypatch, capsys):
    d = tmp_path / "in"
    d.mkdir()
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        _struck_pdf(d / name)
    real, n = cli._telemetry_text, [0]

    def fails_second(line):
        n[0] += 1
        if n[0] == 2:
            raise OSError(28, "No space left on device")
        return real(line)
    monkeypatch.setattr(cli, "_telemetry_text", fails_second)
    out, tel = tmp_path / "out.jsonl", tmp_path / "tel.jsonl"
    assert cli.main(["detect", str(d), "--jsonl", str(out), "--telemetry", str(tel)]) == 0
    assert len(_lines(out)) == 3 and len(_lines(tel)) == 1
    assert "warning: --telemetry" in capsys.readouterr().err


def test_cli_ends_an_unfinished_last_line_and_escapes_a_bad_name(tmp_path):
    tel = tmp_path / "tel.jsonl"
    tel.write_bytes(b'{"telemetry_version": 1, "source": ')       # a run cut off mid-write
    cli._append_telemetry(str(tel), {"telemetry_version": 1, "source": "bad\udcff.pdf"})
    lines = tel.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["source"] == "bad\udcff.pdf"


def test_cli_checks_the_telemetry_file_before_any_work(tmp_path, capsys):
    src = _struck_pdf(tmp_path / "a.pdf")
    assert cli.main(["detect", str(src), "--json", str(tmp_path / "r.json"),
                     "--telemetry", str(tmp_path / "missing" / "t.jsonl")]) == 1
    assert "cannot write" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()


_CASE = pytest.mark.skipif(sys.platform != "win32", reason="case-blind paths are Windows'")


@pytest.mark.parametrize("flags", [
    ["--telemetry", "-"],
    ["--telemetry", "{d}/x.json", "--json", "{d}/x.json"],
    pytest.param(["--telemetry", "{d}/X.JSON", "--json", "{d}/x.json"], marks=_CASE),
    ["--telemetry", "x.json", "--json", "{d}/x.json"],
    ["--telemetry", "{d}/x.json", "--clean-text", "{d}/x.json"],
    ["--telemetry", "{d}/x.json", "--overlay", "{d}/x.json"],
    ["--telemetry", "{src}"],
])
def test_cli_telemetry_needs_a_file_of_its_own(tmp_path, capsys, monkeypatch, flags):
    monkeypatch.chdir(tmp_path)
    src = _struck_pdf(tmp_path / "a.pdf")
    size = src.stat().st_size
    argv = ["detect", str(src)] + [f.format(d=tmp_path, src=src) for f in flags]
    assert cli.main(argv) == 1
    assert "--telemetry" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists() and src.stat().st_size == size


def test_cli_telemetry_and_a_hard_link_to_another_output(tmp_path, capsys):
    src = _struck_pdf(tmp_path / "a.pdf")
    (tmp_path / "r.json").write_text("{}", encoding="utf-8")
    os.link(tmp_path / "r.json", tmp_path / "t.jsonl")
    assert cli.main(["detect", str(src), "--json", str(tmp_path / "r.json"),
                     "--telemetry", str(tmp_path / "t.jsonl")]) == 1
    assert "--telemetry" in capsys.readouterr().err
