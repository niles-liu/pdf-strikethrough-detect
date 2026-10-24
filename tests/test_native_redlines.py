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
from pdf_strikethrough import detect, markdown as md, native

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


def test_short_strike_through_a_lowered_subscript_strikes_it():
    # the 2 of CO2 set 3 pt low, struck on its own by a 3.4 pt line through its middle
    doc = pymupdf.open()
    page = doc.new_page()
    x, y = 72, 200
    w = pymupdf.get_text_length("CO", fontsize=FS)
    page.insert_text((x, y), "CO", fontsize=FS)
    page.insert_text((x + w, y + 3), "2", fontsize=7)
    r = [c for b in page.get_text("rawdict")["blocks"] for ln in b["lines"]
         for s in ln["spans"] for c in s["chars"] if c["c"] == "2"][0]["bbox"]
    page.draw_line((r[0] + 0.2, (r[1] + r[3]) / 2), (r[0] + 3.6, (r[1] + r[3]) / 2), width=0.5)
    (rec,) = st.strikethroughs_in_pdf(_reopen(doc))
    assert (rec["text"], rec["chars"]) == ("CO2", "2")


def test_rest_of_a_ligature_is_struck_with_its_first_letter():
    # a PDF that maps an 'fi' ligature to the letters 'f' and 'i' extracts as an 'f' box spanning
    # the glyph and a zero-width 'i' at its right edge: a strike stopping short of that edge must
    # take the 'i' with the 'f', not leave 'ile'
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 200), "keep the file now", fontsize=FS)
    word = _words(page)["file"]
    f, i = native._PageChars(page).lookup((*word, "file")).boxes[:2]
    page.draw_line((f[0], (f[1] + f[3]) / 2), (i[2] - 0.6, (f[1] + f[3]) / 2), width=0.6)
    page = _reopen(doc)[0]
    chars = native._PageChars(page)
    real = chars.lookup

    def as_ligature(w):
        found = real(w)
        if w[4] == "file":
            f, i, *rest = found.boxes
            found = found._replace(boxes=[(f[0], f[1], i[2], f[3]), (i[2], i[1], i[2], i[3]),
                                          *rest])
        return found

    chars.lookup = as_ligature
    recs = native.native_page_strikes(page, 0, _chars=chars)
    assert [(r["text"], r["chars"]) for r in recs] == [("file", "fi")]


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


def _leaning(deg, slack=0.0):
    """'bravo' crossed through its middle by a line leaning `deg` degrees, its rise `slack` pt
    short of that angle's."""
    r, y = _bravo_mid()
    rise = math.tan(math.radians(deg)) * (r.x1 - r.x0) - slack
    return _line_with(lambda p: p.draw_line((r.x0, y + rise / 2), (r.x1, y - rise / 2), width=0.8))


@pytest.mark.parametrize("deg", [4, 8, 12])
def test_leaning_strike_strikes_the_word(deg):
    # 12 degrees is the cap: drawn a hair inside it, so rounding in the written PDF can't tip it over
    res = st.detect_pdf(_leaning(deg, slack=0.02))
    assert [(r["text"], r["chars"]) for r in res["words"]] == [("bravo", "bravo")]
    assert res["clean_text"] == "alpha charlie"


@pytest.mark.parametrize("deg", [13, 15, 25])
def test_line_leaning_past_the_cap_is_not_a_strike(deg):
    assert st.strikethroughs_in_pdf(_leaning(deg)) == []


def _digit_with(draw):
    """'5' (9 pt, 5 pt wide) with a mark `draw(page, x, y)` centred on its middle."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 200), "item 5 and 7 more", fontsize=9)
    r = _words(page)["5"]
    draw(page, (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
    return _reopen(doc)


def _tick(rise):
    """A 3.2 pt line rising `rise` pt."""
    return lambda p, x, y: p.draw_line((x - 1.6, y + rise / 2), (x + 1.6, y - rise / 2), width=0.6)


def test_short_strike_over_one_digit_is_a_strike():
    res = st.detect_pdf(_digit_with(_tick(0.0)))
    assert [(r["text"], r["chars"]) for r in res["words"]] == [("5", "5")]
    assert res["clean_text"] == "item and 7 more"


def test_short_steep_tick_over_one_digit_is_not_a_strike(monkeypatch):
    # rising 1.45 pt over 3.2 (24 degrees): inside the 1.5 pt allowance for a level line, so only
    # the slope cap keeps the digit live
    assert st.strikethroughs_in_pdf(_digit_with(_tick(1.45))) == []
    monkeypatch.setattr(native, "MAX_STROKE_SLOPE", math.tan(math.radians(30)))
    assert [r["text"] for r in st.strikethroughs_in_pdf(_digit_with(_tick(1.45)))] == ["5"]


def _curve_tick(p, x, y):
    a, b = pymupdf.Point(x - 1.6, y + 0.725), pymupdf.Point(x + 1.6, y - 0.725)
    sh = p.new_shape()
    sh.draw_bezier(a, a + (b - a) / 3, a + (b - a) * 2 / 3, b)
    sh.finish(width=0.6)
    sh.commit()


@pytest.mark.parametrize("draw", [
    _curve_tick,
    lambda p, x, y: p.draw_polyline([(x - 1.6, y + 0.725), (x + 1.6, y - 0.725),
                                     (x + 1.6, y - 0.125), (x - 1.6, y + 1.325)],
                                    color=None, fill=(0, 0, 0), closePath=True),
    lambda p, x, y: p.draw_rect((x - 1.6, y - 1.6, x + 1.6, y + 1.6), color=None, fill=(0, 0, 0)),
    lambda p, x, y: p.draw_rect((x - 1.7, y - 1.7, x + 1.7, y + 1.7), color=(0, 0, 0), width=0.4),
], ids=["curve-tick", "filled-tick", "filled-square", "stroked-box"])
def test_short_tick_or_box_of_any_kind_over_one_digit_is_not_a_strike(draw):
    # the cap covers curves too, and a bar under 6 pt wide must be as flat as a short line
    assert st.strikethroughs_in_pdf(_digit_with(draw)) == []


def test_fraction_bar_beside_a_full_size_character_is_not_a_strike():
    # TeX sets '(0, 3/2, 2]' with a small 3 over a small 2 and the comma at full size, and the
    # extractor reads '2,' as one word: the 4.5 pt bar crosses that word's middle band, but it sits
    # above the small 2's own middle band
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 200), "over (0, ", fontsize=10)
    x = 72 + pymupdf.get_text_length("over (0, ", fontsize=10)
    tw = pymupdf.TextWriter(page.rect)
    tw.append((x + 0.3, 195.8), "3", fontsize=7)
    tw.append((x + 0.3, 203.5), "2", fontsize=7)
    tw.append((x + 4.8, 200), ", 2] then", fontsize=10)
    tw.write_text(page)
    page.draw_line((x, 197), (x + 4.5, 197), width=0.4)
    doc = _reopen(doc)
    assert "2," in [w[4] for w in doc[0].get_text("words")]
    assert st.strikethroughs_in_pdf(doc) == []


def test_hand_drawn_strike_with_jitter_strikes_the_whole_word():
    # a pen stroke sampled every 1.5 pt, wobbling up to 0.45 pt: some of its pieces lean past the
    # cap, but they chain with the level ones into one run
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 200), "the parties shall indemnify", fontsize=12)
    r = _words(page)["indemnify"]
    y = (r.y0 + r.y1) / 2 + 1
    wobble = [0, 0.4, 0.1, -0.35, 0.05, 0.45, -0.1, -0.4, 0.2, 0.0, 0.35, -0.2, 0.25, -0.3, 0.1]
    n = math.ceil((r.x1 - r.x0 + 2) / 1.5) + 1
    page.draw_polyline([(r.x0 - 1 + 1.5 * i, y + wobble[i % len(wobble)]) for i in range(n)],
                       width=0.8, color=(1, 0, 0))
    res = st.detect_pdf(_reopen(doc))
    assert [(r["text"], r["chars"]) for r in res["words"]] == [("indemnify", "indemnify")]


@pytest.mark.parametrize("deg", [5, 8, 11])
def test_underline_of_slanted_text_is_not_a_strike(deg):
    # text set at a slant has an upright box taller than its line, so a line along its baseline
    # crosses the box's middle band
    doc = pymupdf.open()
    page = doc.new_page()
    org = pymupdf.Point(100, 300)
    page.insert_text(org, "Approved for release", fontsize=12, morph=(org, pymupdf.Matrix(-deg)))
    length = pymupdf.get_text_length("Approved for release", fontsize=12)
    m = pymupdf.Matrix(-deg)
    page.draw_line(org + pymupdf.Point(0, 1.5) * m, org + pymupdf.Point(length, 1.5) * m,
                   width=0.6, color=(0, 0, 1))
    assert st.strikethroughs_in_pdf(_reopen(doc)) == []


def _hidden_dash_over(text, letter, mark):
    """`text` with an invisible '-' laid 2 pt low over its `letter`, as some redline tools do,
    and a 3.3 pt line across that letter: through its middle (`mark='strike'`) or under it."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 200), text, fontsize=12)
    chars = page.get_text("rawdict")["blocks"][0]["lines"][0]["spans"][0]["chars"]
    c = pymupdf.Rect(next(ch["bbox"] for ch in chars if ch["c"] == letter))
    page.insert_text((c.x0, 202), "-", fontsize=12.4, render_mode=3)
    y = (c.y0 + c.y1) / 2 if mark == "strike" else 201
    page.draw_line((c.x0, y), (c.x0 + 3.3, y), width=0.6, color=(0, 0, 1))
    doc = _reopen(doc)
    assert "-" in [w[4] for w in doc[0].get_text("words")]
    return doc


def test_invisible_text_over_an_inserted_letter_is_not_struck_by_its_underline():
    # the underline of the inserted 't' runs through the lower, invisible '-' box's middle band
    assert st.strikethroughs_in_pdf(_hidden_dash_over("add Ttest now", "t", "underline")) == []


def test_invisible_text_over_a_struck_letter_is_struck_with_it():
    recs = st.strikethroughs_in_pdf(_hidden_dash_over("and Llight duty", "l", "strike"))
    assert sorted((r["text"], r["chars"]) for r in recs) == [("-", "-"), ("Llight", "l")]


def test_strike_over_an_invisible_only_text_layer_is_kept():
    # an OCR'd scan with its markup flattened to vectors: the text layer is all invisible
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 200), "the parties shall indemnify", fontsize=11, render_mode=3)
    r = _words(page)["shall"]
    page.draw_line((r.x0, (r.y0 + r.y1) / 2), (r.x1, (r.y0 + r.y1) / 2), width=0.8)
    assert [(r["text"], r["chars"]) for r in st.strikethroughs_in_pdf(_reopen(doc))] == [
        ("shall", "shall")]


# ----------------------------------------------------------------------- the strike band

def _marked_line(font, heights, size=FS, base=200, width=0.6):
    """'alpha bravo charlie' in `font`, with a line under or through 'bravo' at each height in
    `heights` (ems of the font size above the baseline; 'x' is the middle of the x-height)."""
    xmid = pymupdf.Font(font).glyph_bbox(ord("x")).y1 / 2
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, base), "alpha bravo charlie", fontname=font, fontsize=size)
    r = _words(page)["bravo"]
    for h in heights:
        y = base - size * (xmid if h == "x" else h)
        page.draw_line((r.x0, y), (r.x1, y), width=width)
    return _reopen(doc)


# Courier's word box, and bold Times' and Helvetica's, put the baseline inside the box-share band
@pytest.mark.parametrize("font", ["cour", "cobo", "tiro", "tibo", "helv", "hebo"])
@pytest.mark.parametrize("heights", [[0.0], [0.0, -0.12], [-0.12]],
                         ids=["at-baseline", "double", "under-baseline"])
def test_underline_is_not_a_strike_in_any_font(font, heights):
    assert st.strikethroughs_in_pdf(_marked_line(font, heights)) == []


@pytest.mark.parametrize("font", ["cour", "cobo", "tiro", "tibo", "helv", "hebo"])
@pytest.mark.parametrize("height", ["x", 0.15, 0.40], ids=["mid-x-height", "low", "high"])
def test_strike_through_the_letters_counts_in_any_font(font, height):
    (rec,) = st.strikethroughs_in_pdf(_marked_line(font, [height]))
    assert (rec["text"], rec["chars"], rec["partial"]) == ("bravo", "bravo", False)


def test_overline_above_the_capitals_is_not_a_strike():
    assert st.strikethroughs_in_pdf(_marked_line("cour", [0.9])) == []


def _superscript_word(strike_y=None, underline=False):
    """'see note' + a raised 7 pt '12' joined to 'note', in Courier: one word, 'note12'."""
    doc = pymupdf.open()
    page = doc.new_page()
    x, y = 72, 200
    page.insert_text((x, y), "see note", fontname="cour", fontsize=FS)
    x += pymupdf.get_text_length("see note", fontname="cour", fontsize=FS)
    page.insert_text((x, y - 4.5), "12", fontname="cour", fontsize=7)
    r = _words(page)["note12"]
    if underline:
        page.draw_line((r.x0, y), (r.x1, y), width=0.6)
    if strike_y is not None:
        page.draw_line((r.x0, y - strike_y), (r.x1, y - strike_y), width=0.6)
    return _reopen(doc)


def test_superscript_does_not_move_the_baseline():
    # an underline at the main baseline under a word ending in a superscript stays live, and a
    # strike through the main letters strikes the superscript with them
    assert st.strikethroughs_in_pdf(_superscript_word(underline=True)) == []
    (rec,) = st.strikethroughs_in_pdf(_superscript_word(strike_y=2.6, underline=True))
    assert (rec["text"], rec["chars"], rec["partial"]) == ("note12", "note12", False)


def test_strike_band_follows_text_set_upside_down():
    # text upside down in unrotated space (/Rotate 180): its letters run below the baseline there
    def doc_with(dy):
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((300, 400), "keep gone here", fontname="cour", fontsize=FS, rotate=180)
        g = _words(page)["gone"]
        page.draw_line((g.x0, 400 + dy), (g.x1, 400 + dy), width=0.6)
        page.set_rotation(180)
        return _reopen(doc)
    assert [r["text"] for r in st.strikethroughs_in_pdf(doc_with(2.6))] == ["gone"]
    assert st.strikethroughs_in_pdf(doc_with(0.0)) == []


def _content_page(ops):
    """A page whose content stream is `ops` (PDF space, y up: MuPDF's y = 200 is 642; /helv is
    Helvetica)."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_font(fontname="helv")
    xref = doc.get_new_xref()
    doc.update_object(xref, "<<>>")
    doc.update_stream(xref, ops.encode())
    page.set_contents(xref)
    return _reopen(doc)


def _struck_with_line(doc, target, y):
    page = doc[0]
    r = _words(page)[target]
    page.draw_line((r.x0, y), (r.x1, y), width=0.6)
    return [w["text"] for w in st.strikethroughs_in_pdf(_reopen(doc))]


@pytest.fixture
def quad_corrections(request):
    """PyMuPDF's quad corrections on (or off, given False) for one test, restored after: importing
    pymupdf4llm, as an earlier test may have, turns them off for the whole process."""
    prev = pymupdf.TOOLS.unset_quad_corrections()
    pymupdf.TOOLS.unset_quad_corrections(not getattr(request, "param", True))
    yield
    pymupdf.TOOLS.unset_quad_corrections(prev)


# Tz stretches text sideways only: the band is set by the glyphs' height, not the span's size
@pytest.mark.parametrize("quad_corrections", [True, False], indirect=True, ids=["qc-on", "qc-off"])
@pytest.mark.parametrize("tz, height, struck", [(25, 0.36, True), (25, 0.0, False),
                                                (200, 0.30, True), (200, 0.82, False)])
def test_band_follows_the_height_of_stretched_text(quad_corrections, tz, height, struck):
    doc = _content_page(f"BT /helv 12 Tf {tz} Tz 72 642 Td (alpha BRAVO charlie) Tj ET")
    assert _struck_with_line(doc, "BRAVO", 200 - 12 * height) == (["BRAVO"] if struck else [])


# mirrored left to right (letters stand above the baseline) and upside down (they hang below it)
@pytest.mark.parametrize("matrix, up", [("-12 0 0 12 400 642", 1), ("12 0 0 -12 72 642", -1)])
@pytest.mark.parametrize("height, struck", [(0.22, True), (0.0, False), (-0.1, False)])
def test_band_follows_mirrored_text(matrix, up, height, struck):
    doc = _content_page(f"BT /helv 1 Tf {matrix} Tm (alpha bravo charlie) Tj ET")
    assert _struck_with_line(doc, "bravo", 200 - up * 12 * height) == (["bravo"] if struck else [])


def test_output_is_the_same_with_quad_corrections_off():
    # importing pymupdf4llm turns them off for the process; the character pass turns them back on
    # for itself and restores the caller's setting
    stretched = _content_page("BT /helv 12 Tf 25 Tz 72 642 Td (alpha BRAVO charlie) Tj ET")
    r = _words(stretched[0])["BRAVO"]
    stretched[0].draw_line((r.x0, 200 - 12 * 0.5), (r.x1, 200 - 12 * 0.5), width=0.6)
    docs = [_marked_line("cour", [0.0]), _marked_line("helv", [0.75]), _marked_line("cour", ["x"]),
            _marked_line("tibo", [0.4]), _reopen(stretched)]
    prev, out = pymupdf.TOOLS.unset_quad_corrections(), {}
    try:
        for off in (False, True):
            pymupdf.TOOLS.unset_quad_corrections(off)
            out[off] = [[(w["text"], w["char_span"]) for w in st.strikethroughs_in_pdf(d)]
                        for d in docs]
            assert bool(pymupdf.TOOLS.unset_quad_corrections()) == off
    finally:
        pymupdf.TOOLS.unset_quad_corrections(prev)
    assert out[True] == out[False] == [[], [], [("bravo", (0, 5))], [("bravo", (0, 5))],
                                       [("BRAVO", (0, 5))]]


def test_em_of_a_font_shorter_than_one_em():
    # ZapfDingbats' ascender minus descender is 0.963 em; PyMuPDF draws its boxes 1 em tall
    from pdf_strikethrough import native

    def struck_at(h):
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 200), "alpha bravo charlie", fontname="zadb", fontsize=FS)
        w = page.get_text("words")[1]
        page.draw_line((w[0], 200 - FS * h), (w[2], 200 - FS * h), width=0.6)
        return [r["text"] for r in st.strikethroughs_in_pdf(_reopen(doc))], w[4]
    top = native.STRIKE_BAND_HI_LEVEL
    (inside, word), (above, _w) = struck_at(top - 0.03), struck_at(top + 0.012)
    assert inside == [word] and above == []


def test_combining_mark_does_not_set_the_em():
    # a math font's U+0338 slash, taller than the digits it crosses, is part of the word '66\u0338'
    from pdf_strikethrough.native import _word_baseline

    def char(c, x, size):
        return ({"c": c, "origin": (x, 200.0), "bbox": (x, 200 - size, x + 6, 200 + size / 4)},
                1.0, -0.25)
    base = _word_baseline([char("6", 72, 12), char("6", 78, 12), char("\u0338", 84, 20)],
                          80.0, 1.0, 0.0)
    assert base[1] == 200.0 and base[4] == pytest.approx(12)


def test_word_on_two_baselines_uses_the_box_band():
    # an equation's '=|', denominator and '|=' that MuPDF joins into one word, struck line by line:
    # one strike through the main line, one through the denominator
    doc = pymupdf.open()
    page = doc.new_page()
    tw, xs = pymupdf.TextWriter(page.rect), [72]
    for text, y in (("=|", 200), ("14.67", 208), ("|=", 200)):
        xs.append(tw.append((xs[-1], y), text, fontsize=FS)[1].x)
    tw.write_text(page)
    page.draw_line((xs[0], 200 - 3.1), (xs[3], 200 - 3.1), width=0.6)
    page.draw_line((xs[1], 208 - 3.1), (xs[2], 208 - 3.1), width=0.6)
    (rec,) = st.strikethroughs_in_pdf(_reopen(doc))
    assert (rec["text"], rec["chars"], rec["partial"]) == ("=|14.67|=", "=|14.67|=", False)


def test_untrusted_origins_fall_back_to_the_box_band():
    from pdf_strikethrough.native import _through_text
    word = (0.0, 100.0, 50.0, 112.0, "word")          # a 12 pt box, baseline at its bottom
    low, high = 110.5, 103.0                           # 0.125 and 0.75 em over the baseline
    through = _through_text(word, (25.0, 112.0, 0.0, 1.0, 12.0))
    assert through(0, 50, low) and not through(0, 50, high) and not through(0, 50, 112.0)
    # no origins, a baseline off the box, an em the box does not fit: the box-share band
    for base in (None, (25.0, 130.0, 0.0, 1.0, 12.0), (25.0, 112.0, 0.0, 1.0, 30.0)):
        through = _through_text(word, base)
        assert not through(0, 50, low) and through(0, 50, high)


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
