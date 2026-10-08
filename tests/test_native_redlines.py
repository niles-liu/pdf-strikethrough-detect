"""Native-path redline correctness: character-exact strike spans, page routing on rotated and
partly-imaged scans, and the paint/geometry rules that decide what counts as a strike stroke.
All fixtures are synthesized in-process; the corpus-level effect of these rules (records lost,
gained and corrected on the public benchmark set) is recorded in CHANGELOG.md."""
import io
import math

import numpy as np
import pymupdf
import pytest
from PIL import Image

import pdf_strikethrough as st
from pdf_strikethrough import detect, markdown as md

FS = 12


def _reopen(doc):
    return pymupdf.open("pdf", doc.tobytes())


def _redline(deleted, inserted, prefix="ends in ", suffix=" each year."):
    """One line where `deleted` (struck) is printed straight before `inserted` (underlined), with
    no space between them: get_text("words") returns the two as ONE word."""
    doc = pymupdf.open()
    page = doc.new_page()
    x, y = 72, 200
    page.insert_text((x, y), prefix, fontsize=FS)
    x += pymupdf.get_text_length(prefix, fontsize=FS)
    wd = pymupdf.get_text_length(deleted, fontsize=FS)
    page.insert_text((x, y), deleted, fontsize=FS, color=(1, 0, 0))
    page.draw_line((x, y - 0.3 * FS), (x + wd, y - 0.3 * FS), color=(1, 0, 0), width=0.8)
    x += wd
    wi = pymupdf.get_text_length(inserted, fontsize=FS)
    page.insert_text((x, y), inserted, fontsize=FS, color=(0, 0, 1))
    page.draw_line((x, y + 1.5), (x + wi, y + 1.5), color=(0, 0, 1), width=0.8)    # underline
    page.insert_text((x + wi, y), suffix, fontsize=FS)
    return _reopen(doc)


def _words(page):
    return {w[4]: pymupdf.Rect(w[:4]) for w in page.get_text("words")}


# ----------------------------------------------------------------------- character-exact spans

@pytest.mark.parametrize("deleted, inserted", [("December", "May"), ("shall", "may"),
                                               ("Policy", "To"), ("Seller's", "its")])
def test_inserted_text_after_a_deletion_survives(deleted, inserted):
    res = st.detect_pdf(_redline(deleted, inserted))
    (rec,) = res["words"]
    assert rec["chars"] == deleted and rec["partial"] and rec["char_span"] == (0, len(deleted))
    assert res["clean_text"] == f"ends in {inserted} each year."
    assert res["markdown"] == f"ends in ~~{deleted}~~{inserted} each year."


def test_relettered_label_strikes_the_one_character():
    # '(ed)': 'e' inserted, 'd' struck -- a single struck character inside a word is a deletion
    doc = pymupdf.open()
    page = doc.new_page()
    x, y = 72, 200
    for text, struck in (("(e", False), ("d", True), (") Notice", False)):
        w = pymupdf.get_text_length(text, fontsize=FS)
        page.insert_text((x, y), text, fontsize=FS)
        if struck:
            page.draw_line((x, y - 0.3 * FS), (x + w, y - 0.3 * FS), width=0.8)
        x += w
    res = st.detect_pdf(_reopen(doc))
    recs = [(r["text"], r["chars"], r["char_span"]) for r in res["words"]]
    assert recs == [("(ed)", "d", (2, 3))]
    assert res["clean_text"] == "(e) Notice"


def test_unstruck_trailing_punctuation_is_kept():
    res = st.detect_pdf(_redline("sale", ".", prefix="the ", suffix=" Next"))
    assert res["words"][0]["chars"] == "sale"
    assert res["clean_text"] == "the . Next"


def test_overshoot_from_a_struck_neighbor_does_not_strike_a_character():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 200), "keep gone lazy", fontsize=FS)
    w = _words(page)
    g, lz = w["gone"], w["lazy"]
    y = (g.y0 + g.y1) / 2
    page.draw_line((g.x0, y), (lz.x0 + 2.0, y), width=0.8)          # runs on into 'l' of 'lazy'
    recs = st.strikethroughs_in_pdf(_reopen(doc))
    assert [r["text"] for r in recs] == ["gone"]


@pytest.mark.parametrize("struck_part, chars", [(1.0, "CO2"), (0.5, "C")])
def test_struck_subscript_is_struck_with_its_word(struck_part, chars):
    # the strike stops short of the subscript '2', as redline tools draw it; the '2' joins only a
    # struck neighbor
    doc = pymupdf.open()
    page = doc.new_page()
    x, y = 72, 200
    w = pymupdf.get_text_length("CO", fontsize=FS)
    page.insert_text((x, y), "CO", fontsize=FS)
    page.insert_text((x + w, y + 2), "2", fontsize=7)
    page.draw_line((x + 0.5, y - 4), (x + struck_part * w - 0.5, y - 4), width=0.8)
    (rec,) = st.strikethroughs_in_pdf(_reopen(doc))
    assert (rec["text"], rec["chars"]) == ("CO2", chars)


def test_flag_and_annot_spans_snap_to_characters():
    doc = _redline("December", "May")
    flag = st.strikethroughs_in_pdf(doc, method="flag")[0]
    assert flag["chars"].startswith("December") and flag["partial"]   # MuPDF may add the 'M'
    page = doc[0]
    x = 72 + pymupdf.get_text_length("ends in ", fontsize=FS)
    wd = pymupdf.get_text_length("December", fontsize=FS)
    r = _words(page)["DecemberMay"]
    page.add_strikeout_annot(pymupdf.Rect(x, r.y0, x + wd, r.y1))
    (rec,) = st.strikethroughs_in_pdf(_reopen(doc), method="annot")
    assert rec["chars"] == "December" and rec["partial"]


# ----------------------------------------------------------------------- what counts as a stroke

def _line_with(build):
    doc = pymupdf.open()
    page = doc.new_page()
    build(page)
    page.insert_text((72, 200), "alpha bravo charlie", fontsize=FS)
    return _reopen(doc)


def _bravo_mid():
    tmp = pymupdf.open()
    page = tmp.new_page()
    page.insert_text((72, 200), "alpha bravo charlie", fontsize=FS)
    r = _words(page)["bravo"]
    return r, (r.y0 + r.y1) / 2


def test_fill_only_shape_edges_are_not_strokes():
    r, y = _bravo_mid()
    # a white polygon whose top edge crosses every word's middle band
    res = st.detect_pdf(_line_with(lambda p: p.draw_polyline(
        [(60, y), (300, y), (320, y + 60), (40, y + 60)], color=None, fill=(1, 1, 1),
        closePath=True)))
    assert res["words"] == [] and res["clean_text"] == "alpha bravo charlie"


def test_thin_fill_only_bar_is_a_strike_in_its_own_color():
    r, y = _bravo_mid()
    # a red bar, sheared so it is drawn as lines, not a rect
    (rec,) = st.strikethroughs_in_pdf(_line_with(lambda p: p.draw_polyline(
        [(r.x0, y - 0.6), (r.x1, y - 0.6), (r.x1 + 0.8, y + 0.6), (r.x0 + 0.8, y + 0.6)],
        color=None, fill=(1, 0, 0), closePath=True)))
    assert rec["text"] == "bravo" and rec["stroke_color"] == (1.0, 0.0, 0.0)


def test_highlighter_wide_line_is_not_a_strike():
    r, y = _bravo_mid()
    doc = _line_with(lambda p: p.draw_line((r.x0, y), (r.x1, y), color=(1, 1, 0), width=12))
    assert st.strikethroughs_in_pdf(doc) == []


@pytest.mark.parametrize("deg", [0.5, 1.0, 1.5])
def test_slightly_slanted_strike_strikes_every_word(deg):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 200), "the parties agree to pay ninety days after notice", fontsize=FS)
    ws = page.get_text("words")
    x0, x1 = ws[0][0], ws[-1][2]
    yc = (ws[0][1] + ws[0][3]) / 2 + 1
    dy = math.tan(math.radians(deg)) * (x1 - x0)
    page.draw_line((x0, yc - dy / 2), (x1, yc + dy / 2), width=0.8)
    assert len(st.strikethroughs_in_pdf(_reopen(doc))) == len(ws)


def test_steep_line_is_not_a_strike():
    r, y = _bravo_mid()
    doc = _line_with(lambda p: p.draw_line((r.x0, y - 3), (r.x1, y + 3), width=0.8))
    assert st.strikethroughs_in_pdf(doc) == []


# ----------------------------------------------------------------------- rotated content

def _landscape_page():
    """pdflscape / ps2pdf auto-rotate: text drawn rotated, turned upright by /Rotate 90. A red
    strike through 'gone' (vertical in text space) and a short tick across 'keep'."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((300, 500), "keep gone here", fontsize=14, rotate=90)
    w = _words(page)
    g = w["gone"]
    x = (g.x0 + g.x1) / 2 + 1
    page.draw_line((x, g.y0), (x, g.y1), color=(1, 0, 0), width=1.2)
    k = w["keep"]
    y = (k.y0 + k.y1) / 2
    page.draw_line((k.x0 - 2, y), (k.x1 + 2, y), color=(0, 0, 1), width=0.8)
    page.set_rotation(90)
    return _reopen(doc)


def test_rotated_content_no_vector_false_positive():
    assert st.strikethroughs_in_pdf(_landscape_page()) == []


def test_rotated_content_read_by_the_flag_detector():
    recs = st.strikethroughs_in_pdf(_landscape_page(), method="flag")
    assert [r["text"] for r in recs] == ["gone"]


# ----------------------------------------------------------------------- annotations

def test_annotation_gray_and_cmyk_colors_report_rgb():
    from pdf_strikethrough.native import _rgb
    assert _rgb((0.2,)) == (0.2, 0.2, 0.2)
    assert _rgb((0, 1, 1, 0)) == (1.0, 0.0, 0.0)
    assert _rgb(None) == (0.0, 0.0, 0.0)


def test_noview_annotation_is_skipped():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 200), "alpha bravo charlie", fontsize=FS)
    a = page.add_strikeout_annot(_words(page)["bravo"])
    a.set_flags(pymupdf.PDF_ANNOT_IS_NO_VIEW)
    a.update()
    assert st.strikethroughs_in_pdf(_reopen(doc), method="annot") == []


# ----------------------------------------------------------------------- page routing

def _png(w, h, value=245):
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), value, np.uint8)).save(buf, "PNG")
    return buf.getvalue()


@pytest.mark.parametrize("size", [(612, 792), (612, 1008), (792, 1224)])
def test_rotated_full_page_scan_routes_scanned(size):
    doc = pymupdf.open()
    page = doc.new_page(width=size[0], height=size[1])
    page.insert_image(page.rect, stream=_png(80, 100), keep_proportion=False)
    page.insert_text((72, 300), "ocr layer words", fontsize=14, render_mode=3)
    page.set_rotation(90)
    page = _reopen(doc)[0]
    assert detect._image_coverage(page) > 0.95
    assert st.classify_page_source(page) == "scanned"


def test_partial_scan_with_invisible_ocr_layer_routes_scanned():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(pymupdf.Rect(40, 40, 572, 520), stream=_png(532, 480, 255))     # ~53%
    page.insert_text((140, 245), "scanned struck words", fontsize=14, render_mode=3)
    assert st.classify_page_source(_reopen(doc)[0]) == "scanned"


def test_page_with_a_logo_and_visible_text_stays_native():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(pymupdf.Rect(40, 40, 300, 300), stream=_png(100, 100))
    page.insert_text((72, 400), "visible body text", fontsize=12)
    assert st.classify_page_source(_reopen(doc)[0]) == "native"


# ----------------------------------------------------------------------- markdown assembly

def _item(text, x0, rec=None):
    return (text, (x0, 0.10, x0 + 0.08, 0.12), rec)


def test_passage_stops_where_a_partial_strike_leaves_live_text():
    semi = {"final": True, "partial": True, "char_span": (0, 5), "chars": "semi-"}
    rate = {"final": True, "partial": False, "char_span": (0, 4), "chars": "rate"}
    items = [_item("semi-monthly", 0.10, semi), _item("rate", 0.20, rate)]
    assert [p["text"] for p in md.group_passages(items)] == ["semi-", "rate"]


def test_full_strikes_still_join_into_one_passage():
    full = {"final": True, "partial": False}
    items = [_item("the", 0.10, full), _item("old", 0.20, full), _item("rate", 0.30, full)]
    assert [p["text"] for p in md.group_passages(items)] == ["the old rate"]


def test_word_past_the_page_edge_keeps_its_column():
    left = [(f"l{i}", (0.05, 0.1 + i * 0.03, 0.40, 0.12 + i * 0.03), None) for i in range(6)]
    right = [(f"r{i}", (0.55, 0.1 + i * 0.03, 0.95, 0.12 + i * 0.03), None) for i in range(6)]
    edge = [("W", (0.99, 0.5, 1.02, 0.52), None)]                  # centered past 1.0
    text = md.page_clean_text(left + right + edge)
    assert "W" in text.split()


# ----------------------------------------------------------------------- review follow-ups

def test_compound_fill_path_is_judged_bar_by_bar():
    """One fill-only path holding two thin bars (outlined/Illustrator-style output) strikes the two
    words under them, not the live word between them that the union box would span."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 200), "alpha bravo charlie", fontsize=FS)
    w = _words(page)
    y = page.rect.height - (w["alpha"].y0 + w["alpha"].y1) / 2            # PDF user space

    def bar(r):                          # five vertices, so MuPDF keeps line items, not a rect
        return (f"{r.x0:.2f} {y - 0.6:.2f} m {r.x1:.2f} {y - 0.6:.2f} l {r.x1 + 0.3:.2f} {y:.2f} l "
                f"{r.x1:.2f} {y + 0.6:.2f} l {r.x0:.2f} {y + 0.6:.2f} l h ")
    xref = page.get_contents()[0]
    ops = "q 1 0 0 rg " + bar(w["alpha"]) + bar(w["charlie"]) + "f Q"
    doc.update_stream(xref, doc.xref_stream(xref) + b"\n" + ops.encode())
    res = st.detect_pdf(_reopen(doc))
    assert [r["text"] for r in res["words"]] == ["alpha", "charlie"]
    assert res["clean_text"] == "bravo"


def test_proportionate_strike_on_display_text_counts():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((40, 300), "Repealed", fontsize=72)
    r = pymupdf.Rect(page.get_text("words")[0][:4])
    page.draw_line((r.x0, (r.y0 + r.y1) / 2), (r.x1, (r.y0 + r.y1) / 2), width=4.3)
    assert [w["chars"] for w in st.strikethroughs_in_pdf(_reopen(doc))] == ["Repealed"]


def test_character_index_splits_words_where_pymupdf_does():
    from pdf_strikethrough.native import _WORD_DELIMITERS
    assert {" ", "\u00a0", "\u202e"} <= _WORD_DELIMITERS
    assert not {"\u202f", "\u2009"} & _WORD_DELIMITERS                     # no split there
