"""v0.8.0 "scale & armor" regression tests.

Covers the pure-code groups of the release: high-DPI normalization + pixel-budget guard
(R-highdpi / R-guard), dashed & flat-bezier native strikes (R-dash), the batch / directory CLI
mode (R-batch), malformed-PDF robustness (R-hostile), and layout reading-order edge cases
(R-layout). All fixtures are synthesized in-process — no data files are checked in.
"""
import json

import numpy as np
import pymupdf as fitz
import pytest

import pdf_strikethrough as st
from pdf_strikethrough import __main__ as cli
from pdf_strikethrough import markdown as md
from pdf_strikethrough.detect import (MAX_RENDER_MPIX, RENDER_DPI, _downsample_gray, _working_dpi)


# --- helpers ---------------------------------------------------------------------------------
def _word_box(page, word):
    return {w[4]: w[:4] for w in page.get_text("words")}[word]


def _solid_strike(page, word, width=1.0):
    x0, y0, x1, y1 = _word_box(page, word)
    ym = (y0 + y1) / 2
    page.draw_line(fitz.Point(x0, ym), fitz.Point(x1, ym), width=width)


def _one_page(build, width=400, height=200):
    doc = fitz.open()
    build(doc.new_page(width=width, height=height))
    return doc


# ===== R-highdpi / R-guard ====================================================================
def test_working_dpi_normal_page_unchanged():
    dpi, note = _working_dpi(8.5, 11, 200)
    assert dpi == 200 and note is None


def test_working_dpi_high_dpi_normalized_silently():
    # >HIGH_DPI_CAP is normalized to the 200-dpi calibration point; accuracy-neutral -> no warning
    dpi, note = _working_dpi(8.5, 11, 600)
    assert dpi == RENDER_DPI and note is None


def test_working_dpi_pixel_budget_caps_hostile_page():
    dpi, note = _working_dpi(700, 700, 200)
    assert dpi < 200 and note is not None
    assert 700 * 700 * dpi * dpi <= MAX_RENDER_MPIX * 1_000_000 + 1   # stayed under the budget


def test_downsample_gray_reduces_high_dpi_raster():
    big = np.full((1500, 1500), 255, np.uint8)
    gray, dpi, _ = _downsample_gray(big, 600)
    assert dpi == RENDER_DPI and gray.shape[0] < 1500


def test_downsample_gray_noop_at_reference_dpi():
    gray, dpi, note = _downsample_gray(np.full((400, 400), 255, np.uint8), 200)
    assert dpi == 200 and gray.shape == (400, 400) and note is None


def test_huge_mediabox_page_does_not_oom():
    # a 40000x40000-pt page must be handled without allocating a gigapixel raster (it classifies
    # blank here — the point is that detect_pdf returns promptly instead of hanging / OOMing)
    doc = fitz.open()
    doc.new_page(width=40000, height=40000)
    res = st.detect_pdf(doc)
    assert res["n_struck_final"] == 0


# ===== R-dash (dashed / flat-bezier native strikes) ===========================================
def test_dashed_strike_is_chained_and_detected():
    doc = _one_page(lambda p: p.insert_text((50, 100), "keep deleted text here", fontsize=12))
    page = doc[0]
    x0, y0, x1, y1 = _word_box(page, "deleted")
    ym, x = (y0 + y1) / 2, x0
    while x < x1:                                    # 3-pt dashes with 2-pt gaps across the word
        page.draw_line(fitz.Point(x, ym), fitz.Point(min(x + 3, x1), ym), width=1.0)
        x += 5
    assert any(r["chars"] == "deleted" for r in st.native_page_strikes(page, 0))


def test_flat_bezier_strike_detected():
    doc = _one_page(lambda p: p.insert_text((50, 100), "keep deleted text here", fontsize=12))
    page = doc[0]
    x0, y0, x1, y1 = _word_box(page, "deleted")
    ym = (y0 + y1) / 2
    sh = page.new_shape()
    sh.draw_bezier(fitz.Point(x0, ym), fitz.Point(x0 + (x1 - x0) / 3, ym),
                   fitz.Point(x0 + 2 * (x1 - x0) / 3, ym), fitz.Point(x1, ym))
    sh.finish(width=1.2)
    sh.commit()
    assert any(r["chars"] == "deleted" for r in st.native_page_strikes(page, 0))


def test_isolated_short_tick_is_not_a_strike():
    # a lone sub-MIN_STROKE_LEN tick must not chain into (or be promoted to) a strike
    doc = _one_page(lambda p: p.insert_text((50, 100), "keep deleted text here", fontsize=12))
    page = doc[0]
    x0, y0, x1, y1 = _word_box(page, "deleted")
    ym = (y0 + y1) / 2
    page.draw_line(fitz.Point(x0, ym), fitz.Point(x0 + 3, ym), width=1.0)
    assert not any(r["chars"] == "deleted" for r in st.native_page_strikes(page, 0))


def test_chain_short_merges_within_gap_only():
    # three 3-pt dashes 2 pt apart chain into one >=6-pt run; a far tick stays its own short piece
    segs = [(0, 3, 50, (0, 0, 0), 1.0), (5, 8, 50, (0, 0, 0), 1.0), (10, 13, 50, (0, 0, 0), 1.0),
            (200, 203, 50, (0, 0, 0), 1.0)]
    chains = sorted(st.native._chain_short(segs))
    assert chains[0][0] == 0 and chains[0][1] == 13        # merged run 0..13
    assert chains[1][0] == 200 and chains[1][1] == 203     # isolated, still short


# ===== R-batch ================================================================================
def _strike_pdf_bytes(word):
    doc = _one_page(lambda p: (p.insert_text((50, 100), f"keep {word} text here", fontsize=12)))
    _solid_strike(doc[0], word)
    return doc.tobytes()


def test_expand_inputs_directory_glob_and_dedup(tmp_path):
    (tmp_path / "a.pdf").write_bytes(_strike_pdf_bytes("deleted"))
    (tmp_path / "b.pdf").write_bytes(_strike_pdf_bytes("removed"))
    (tmp_path / "note.txt").write_text("ignored")
    from_dir = cli._expand_inputs([str(tmp_path)])
    assert len(from_dir) == 2 and all(p.endswith(".pdf") for p in from_dir)
    deduped = cli._expand_inputs([str(tmp_path / "*.pdf"), str(tmp_path / "a.pdf")])
    assert len(deduped) == 2                              # glob + explicit repeat collapse


def test_expand_inputs_missing_returns_none():
    assert cli._expand_inputs(["definitely-not-here.pdf"]) is None


def test_batch_jsonl_resilient_to_bad_file(tmp_path):
    (tmp_path / "d0.pdf").write_bytes(_strike_pdf_bytes("deleted"))
    (tmp_path / "d1.pdf").write_bytes(_strike_pdf_bytes("removed"))
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 not a real pdf")
    out = tmp_path / "out.jsonl"
    code = cli.main(["detect", str(tmp_path), "--jsonl", str(out)])
    lines = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(lines) == 3
    assert sum(1 for L in lines if L.get("n_struck_final")) == 2
    assert sum(1 for L in lines if "error" in L) == 1
    assert code == 1                                     # >=1 file errored


def test_batch_fail_if_found_exit_code(tmp_path):
    (tmp_path / "d0.pdf").write_bytes(_strike_pdf_bytes("deleted"))
    (tmp_path / "d1.pdf").write_bytes(_strike_pdf_bytes("removed"))
    assert cli.main(["detect", str(tmp_path), "--fail-if-found", "--jsonl", "-"]) == 3


def test_batch_multiprocessing_jobs(tmp_path):
    # locks the picklability fix (worker lives in _batch, not __main__) under a real process pool
    for i, word in enumerate(("deleted", "removed", "gone")):
        (tmp_path / f"d{i}.pdf").write_bytes(_strike_pdf_bytes(word))
    out = tmp_path / "out.jsonl"
    code = cli.main(["detect", str(tmp_path), "--jobs", "2", "--jsonl", str(out)])
    lines = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert code == 0 and len(lines) == 3
    assert all(L["n_struck_final"] == 1 for L in lines)


# ===== R-hostile (malformed-PDF robustness) ===================================================
@pytest.mark.parametrize("data", [
    b"",                              # empty
    b"%PDF-1.4 total garbage here",   # header but no structure
    b"%PDF-1.7\n",                    # header only
    b"not even a pdf header",         # no header at all
])
def test_malformed_pdf_raises_cleanly(data):
    # pymupdf's FileDataError / EmptyFileError both subclass RuntimeError; the point is a clean,
    # catchable failure — never a hang, segfault, or bare crash.
    with pytest.raises(RuntimeError):
        st.detect_pdf(data)


def test_truncated_pdf_handled_cleanly():
    # pymupdf recovers some truncations and rejects others; either is fine — the armor property is
    # a well-formed result OR a catchable error, never a hang / crash.
    good = _strike_pdf_bytes("deleted")
    for frac in (0.2, 0.5, 0.75):
        try:
            res = st.detect_pdf(good[: int(len(good) * frac)])
            assert "words" in res
        except (RuntimeError, ValueError):
            pass


def test_byteflip_fuzz_never_hangs_or_crashes():
    good = _one_page(lambda p: p.insert_text((40, 60), "keep deleted words here now")).tobytes()
    _solid = bytearray(good)
    for off in range(30, len(good), max(1, len(good) // 50)):    # deterministic, evenly spaced
        data = bytearray(good)
        data[off] ^= 0xFF
        try:
            st.detect_pdf(bytes(data))                # either returns or raises a KNOWN error type
        except (RuntimeError, ValueError):
            pass                                      # clean handling; anything else fails the test


# ===== R-layout (reading-order edge cases) ====================================================
def _detect_native(build, width=595, height=842):
    doc = fitz.open()
    build(doc.new_page(width=width, height=height))
    return st.detect_pdf(doc)


def test_two_column_reads_down_each_column():
    def build(p):
        left = ["Left column first line alpha here", "Left column second line beta",
                "Left column third line gone now", "Left column fourth line delta end"]
        right = ["Right column first line echo here", "Right column second removed line",
                 "Right column third line golf now", "Right column fourth line hotel end"]
        y = 100
        for ln_l, ln_r in zip(left, right):
            p.insert_text((72, y), ln_l, fontsize=11)
            p.insert_text((330, y), ln_r, fontsize=11)
            y += 24
        _solid_strike(p, "gone")
        _solid_strike(p, "removed")
    res = _detect_native(build)
    lines = res["clean_text"].splitlines()
    # every Left line must precede every Right line (column order, not interleaved by row)
    last_left = max(i for i, ln in enumerate(lines) if ln.startswith("Left"))
    first_right = min(i for i, ln in enumerate(lines) if ln.startswith("Right"))
    assert last_left < first_right
    assert {w["chars"] for w in res["words"] if w["final"]} == {"gone", "removed"}


def test_struck_table_row_is_one_passage():
    def build(p):
        rows = [("Item", "Qty", "Price"), ("Widget", "10", "5.00"),
                ("Obsolete", "99", "1.00"), ("Gadget", "3", "9.00")]
        y, cellx = 100, [72, 240, 400]
        for row in rows:
            for x, cell in zip(cellx, row):
                p.insert_text((x, y), cell, fontsize=11)
            y += 24
        for word in ("Obsolete", "99", "1.00"):
            _solid_strike(p, word)
    res = _detect_native(build)
    # narrow table columns must NOT split; the struck row groups as a single deletion passage
    assert len(res["passages"]) == 1
    assert res["passages"][0]["n_words"] == 3


def test_strike_across_hyphenated_line_break_is_one_passage():
    def build(p):
        p.insert_text((72, 100), "the fee is semi-", fontsize=11)
        p.insert_text((72, 124), "monthly billed now", fontsize=11)
        _solid_strike(p, "semi-")
        _solid_strike(p, "monthly")
    res = _detect_native(build)
    assert len(res["passages"]) == 1 and res["passages"][0]["n_words"] == 2
    assert "semi-" in res["passages"][0]["text"] and "monthly" in res["passages"][0]["text"]


# ===== R-cjk (slice 1: horizontal CJK) ========================================================
def test_cjk_horizontal_strike_detected():
    # A horizontal strike over horizontally-set CJK text is detected by the vector path — strikes
    # stay horizontal regardless of script (vertical writing modes remain out of scope). Escapes:
    # 合同 = "contract", 已删除 = "deleted", 条款 = "clause".
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((50, 100), "合同 已删除 条款",
                     fontsize=16, fontname="china-s")
    x0, y0, x1, y1 = page.get_text("words")[1][:4]        # the middle token
    ym = (y0 + y1) / 2
    page.draw_line(fitz.Point(x0, ym), fitz.Point(x1, ym), width=1.2)
    recs = st.native_page_strikes(page, 0)
    assert len(recs) == 1 and recs[0]["coverage"] >= 0.7


def test_single_column_prose_is_not_split():
    items = [(f"word{i}", (0.1 + 0.08 * (i % 8), 0.1 + 0.03 * (i // 8),
                           0.16 + 0.08 * (i % 8), 0.12 + 0.03 * (i // 8)), None) for i in range(24)]
    assert len(md._column_partition(items)) == 1
