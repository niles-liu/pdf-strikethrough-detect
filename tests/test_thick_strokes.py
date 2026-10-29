"""Thick strikes on the scanned path: a 2-3 pt line through text is kept, bold and large text
included, while the bars of a bold heading, about as thick as its stems, are not."""
import numpy as np
import pymupdf
import pytest
from scipy import ndimage

import pdf_strikethrough as st
from pdf_strikethrough import lines, scanned
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import ScanConfig

CF = ScanConfig.confidence_free()


def _page(text, font, size, strike=None, width=2.0, dpi=200, blur=0.0):
    """One line of text rendered at `dpi`: (gray, words). `strike` = (first, last) word index,
    struck by a `width` pt line through the middle of the x-height. `blur` is a Gaussian sigma in
    px, a mild stand-in for a scanner's softening."""
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=120)
    page.insert_text((24, 64), text, fontname=font, fontsize=size)
    ws = page.get_text("words")
    if strike:
        y = 64 - size * pymupdf.Font(font).glyph_bbox(ord("x")).y1 / 2
        page.draw_line((ws[strike[0]][0], y), (ws[strike[1]][2], y), width=width)
    W, H = page.rect.width, page.rect.height
    words = [Word(w[4], (w[0] / W, w[1] / H, w[2] / W, w[3] / H), None) for w in ws]
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    if blur:
        gray = np.clip(ndimage.gaussian_filter(gray.astype(float), blur), 0, 255).astype(np.uint8)
    return gray, words


def _thick(found, dpi):
    return [ln for ln in found if ln["run_px"] > lines.THIN_STROKE_RUN_PX * dpi / lines.RENDER_DPI]


def _struck(gray, words, dpi=200):
    recs = st.detect_scanned_image(gray, words, config=CF, dpi=dpi)
    return sorted(r["text"] for r in recs if r["final"]), recs


@pytest.mark.parametrize("dpi", [150, 200, 300])
@pytest.mark.parametrize("width", [2.0, 3.0])
def test_thick_strike_through_regular_text_is_kept(width, dpi):
    """A 2 or 3 pt strike is 5.6-8.3 px at 200 dpi, past the old 4 px cap; it is clearly thicker
    than the stems of 11 pt Times, so it is kept and the word is marked."""
    gray, words = _page("hold the deposit in a segregated account", "tiro", 11, (2, 2), width, dpi)
    assert _thick(st.strike_lines(gray, dpi=dpi), dpi)
    assert _struck(gray, words, dpi)[0] == ["deposit"]


def test_thick_strike_was_dropped_by_the_old_cap(monkeypatch):
    """The regression the stem test replaces: at 0.12.1's 4 px cap the same strike is lost."""
    gray, words = _page("hold the deposit in a segregated account", "tiro", 11, (2, 2), 2.0)
    monkeypatch.setattr(lines, "MAX_STROKE_RUN_PX", 4)
    assert _struck(gray, words)[0] == []


def test_3pt_strike_at_150_dpi_uses_the_slack():
    """At 150 dpi the scaled cap is 6.75 px and a 3 pt strike measures 7: the 1 px of slack on
    the cap keeps it."""
    gray, words = _page("a summer was over now, we saw a nurse", "helv", 11, (1, 2), 3.0, 150)
    assert _struck(gray, words, 150)[0] == ["summer", "was"]


def test_blurred_3pt_strike_is_kept():
    """A softened 3 pt strike, which 0.12.1 lost. Blur can still merge a 3 pt stroke with the
    glyphs it crosses into a piece thicker than the stitching takes (11-13 pt Times), and then it
    is lost as before."""
    gray, words = _page("a summer was over now, we saw a nurse", "tiro", 9, (1, 2), 3.0, blur=0.6)
    assert _struck(gray, words)[0] == ["summer", "was"]


@pytest.mark.parametrize("text,strike,font,size,width,found", [
    ("he was over now as we came", (1, 4), "cour", 11, 2.0, ["as", "now", "over", "was"]),
    ("so we can use one", (2, 4), "helv", 9, 2.5, ["can", "one", "use"]),
    ("one more case", (0, 0), "tiro", 11, 2.0, ["one"])])
def test_thick_strike_over_words_without_ascenders(text, strike, font, size, width, found,
                                                   monkeypatch):
    """A 2 or 2.5 pt strike leaves two to four rows of the x-height above and below it. Over
    words without ascenders or descenders, read from 2 px past the strike, that is no ink above:
    a long line read as a table rule, a short one as no strike. Read from its edges
    (INK_GAP_THICK_PX = 0) the words are found; at 2, the release's reading, none is."""
    gray, words = _page(text, font, size, strike, width)
    assert _struck(gray, words)[0] == found
    monkeypatch.setattr(scanned, "INK_GAP_THICK_PX", 2)
    assert _struck(gray, words)[0] == []


def test_thick_line_with_no_text_beside_it_is_kept():
    """With no glyph ink beside it there are no stems to compare with: the line is kept."""
    gray = np.full((200, 600), 255, np.uint8)
    gray[96:102, 100:500] = 0                                   # 6 px, 2 in long
    assert any(ln["len_in"] >= 1.9 and ln["run_px"] == 6 for ln in st.strike_lines(gray, dpi=200))


@pytest.mark.parametrize("font,size", [("hebo", 18), ("hebo", 24), ("tibo", 12)])
def test_bold_heading_gives_no_strike_candidates(font, size, monkeypatch):
    """A bold glyph's bars, and chains of them stitched along a heading, are about as thick as its
    stems: no thick line comes out and no word reaches the CNN. Without the stem test (ratio 0)
    the same heading does produce thick lines."""
    gray, words = _page("TERMS OF THE AGREEMENT Effective", font, size)
    assert _thick(st.strike_lines(gray, dpi=200), 200) == []
    assert _struck(gray, words)[1] == []
    monkeypatch.setattr(lines, "STEM_RATIO", 0)
    assert _thick(st.strike_lines(gray, dpi=200), 200)


@pytest.mark.parametrize("font", ["hebo", "tibo"])
def test_thick_strike_through_small_bold_text_is_kept(font):
    """At 10 pt a bold font's stems are about 4 px, so a 2 pt strike (5.6 px) is still clearly
    thicker than them."""
    gray, words = _page("TERMS OF THE AGREEMENT", font, 10, (1, 2), 2.0)
    assert _struck(gray, words)[0] == ["OF", "THE"]


def _covers(found, strike_x, y):
    """A returned line over most of the drawn strike's x-range, at its height."""
    x0, x1 = strike_x
    return any(ln["bbox_px"][1] - 4 <= y <= ln["bbox_px"][3] + 4
               and min(ln["bbox_px"][2], x1) - max(ln["bbox_px"][0], x0) >= 0.8 * (x1 - x0)
               for ln in found)


def _strike_px(words, strike, gray, font, size):
    H, W = gray.shape
    y = (64 - size * pymupdf.Font(font).glyph_bbox(ord("x")).y1 / 2) * H / 120
    return (words[strike[0]].bbox[0] * W, words[strike[1]].bbox[2] * W), y


@pytest.mark.parametrize("font,size,text", [
    ("hebo", 18, "Effective Date and Terms"), ("hebo", 24, "Terms of Sale"),
    ("tibo", 24, "TERMS OF SALE"), ("cobo", 24, "TERMS OF SALE"), ("helv", 24, "Terms of Sale"),
    ("tiro", 18, "Effective Date and Terms")])
def test_thick_strike_through_large_or_bold_text_is_found(font, size, text):
    """Through bold text above about 10 pt, or regular text above about 18 pt, a 2 pt strike is as
    thick as the stems it crosses. It is kept because it runs on unbroken for at least two and a
    half glyph heights, where a bold glyph's bar ends with its glyph; and in larger bold text it
    opens into one piece with the glyph bars beside it, too thick as a whole, so its own band is
    taken."""
    gray, words = _page(text, font, size, (1, 2), 2.0)
    assert _covers(st.strike_lines(gray, dpi=200), *_strike_px(words, (1, 2), gray, font, size))


def test_thick_strike_through_large_bold_text_is_marked():
    """End to end: both struck words of a 24 pt bold heading come out struck, which 0.12.1 and
    the stem test alone lost."""
    gray, words = _page("TERMS OF SALE", "tibo", 24, (1, 2), 2.0)
    assert _struck(gray, words)[0] == ["OF", "SALE"]


@pytest.mark.parametrize("off", [{"SOLID_GLYPHS": 0}, {"PEEL_COVER": 0}])
def test_large_bold_strike_needs_both_the_band_and_the_solid_run(off, monkeypatch):
    """With either change off, the 2 pt strike through 24 pt bold is dropped, as before."""
    gray, words = _page("Terms of Sale", "hebo", 24, (1, 2), 2.0)
    for k, v in off.items():
        monkeypatch.setattr(lines, k, v)
    assert not _covers(st.strike_lines(gray, dpi=200),
                       *_strike_px(words, (1, 2), gray, "hebo", 24))


def test_solid_run_alone_keeps_a_strike_through_18pt_text(monkeypatch):
    """At 18 pt regular Times the strike is as thick as the stems but no piece is too thick to
    stitch: the unbroken run alone keeps it."""
    monkeypatch.setattr(lines, "PEEL_COVER", 0)
    gray, words = _page("Effective Date and Terms", "tiro", 18, (1, 2), 2.0)
    assert _covers(st.strike_lines(gray, dpi=200), *_strike_px(words, (1, 2), gray, "tiro", 18))
    monkeypatch.setattr(lines, "SOLID_GLYPHS", 0)
    assert not _covers(st.strike_lines(gray, dpi=200),
                       *_strike_px(words, (1, 2), gray, "tiro", 18))


def test_peel_takes_the_band_a_thick_fragment_fills():
    """A 6 px bar 300 px long with glyph-like blocks stuck above and below it: the piece as a whole
    is 26 px thick; its fullest 10 px band runs end to end and is kept as the stroke."""
    ink = np.zeros((120, 500), bool)
    ink[57:63, 100:400] = True
    for x in range(110, 390, 40):
        ink[47:57, x:x + 14] = True                        # blocks 10 px tall, 14 px wide
        ink[63:73, x + 20:x + 34] = True
    ys, xs = np.nonzero(ink)
    along, across = xs - xs.mean(), ys - ys.mean()
    a0, a1, mid = lines._peel(along.astype(float), across.astype(float), 10)
    assert a1 - a0 >= 295 and abs(mid + ys.mean() - 60) <= 2
    ink[57:63, 200:280] = False                            # an 80 px break: no band runs end to end
    ys, xs = np.nonzero(ink)
    assert lines._peel((xs - xs.mean()).astype(float), (ys - ys.mean()).astype(float), 10) is None


@pytest.mark.parametrize("font,size,text,strike,width,adds", [
    ("tiro", 11, "hold the deposit in a segregated account", (1, 3), 0.8, False),
    ("helv", 9, "hold the deposit in a segregated account", (1, 3), 3.0, False),
    ("cour", 12, "hold the deposit in a segregated account", (1, 3), 1.0, False),
    ("tibo", 10, "TERMS OF THE AGREEMENT", (1, 2), 2.0, False),
    ("hebo", 24, "Terms of Sale", (1, 2), 2.0, True),
    ("tiro", 18, "Effective Date and Terms", (1, 2), 2.0, True),
    ("cobo", 24, "TERMS OF SALE", (1, 2), 2.0, True)])
def test_second_pass_only_adds_lines(font, size, text, strike, width, adds, monkeypatch):
    """Every line found without the second pass (SOLID_GLYPHS = 0, the previous behavior) comes
    out unchanged and in the same order, ahead of what the second pass adds: here the 2 pt strike
    through bold or large text, and nothing where the first pass already has the strike."""
    gray, _ = _page(text, font, size, strike, width)
    both = st.strike_lines(gray, dpi=200)
    monkeypatch.setattr(lines, "SOLID_GLYPHS", 0)
    first = st.strike_lines(gray, dpi=200)
    assert both[:len(first)] == first
    assert (len(both) > len(first)) == adds


def _added(gray, dpi, monkeypatch):
    both = st.strike_lines(gray, dpi=dpi)
    with monkeypatch.context() as m:
        m.setattr(lines, "SOLID_GLYPHS", 0)
        first = st.strike_lines(gray, dpi=dpi)
    return both[len(first):]


@pytest.mark.parametrize("text,size,dpi,blur", [
    ("parties section agreement lessee", 20, 200, 0.0), ("TERMS OF THE ZONE FEE", 20, 200, 0.0),
    ("the lessee agrees to these sales terms", 14, 150, 0.6)])
def test_courier_bold_bars_add_no_line(text, size, dpi, blur, monkeypatch):
    """Strike-free bold Courier: the bars of e, a and s join into thick runs about a glyph long.
    Where a line crosses the curved side of such a glyph, the ink it crosses is much shorter than
    the glyph, so the glyph height is read from the taller stems (the upper quartile), and a run
    must reach two and a half of them. Read at the median, or held to two, these pages add a
    line."""
    gray, _ = _page(text, "cobo", size, dpi=dpi, blur=blur)
    assert _added(gray, dpi, monkeypatch) == []


def test_glyph_height_reach_scales_with_dpi(monkeypatch):
    """The stems that give the glyph height are followed 20 px either side at 200 dpi, scaled with
    dpi. At 300 dpi a bar 8 px thick and 132 px long through stems 60 px tall runs on for 2.2
    glyph heights, short of the 2.5 needed. Followed only 20 px either side, the stems would
    measure 41 px and the bar would pass."""
    gray = np.full((300, 400), 255, np.uint8)
    gray[146:154, 130:262] = 0                             # the bar
    for x in range(134, 258, 20):
        gray[120:180, x:x + 10] = 0                        # stems crossing it
    assert _added(gray, 300, monkeypatch) == []
    gray[146:154, 262:330] = 0                             # 200 px: 3.3 glyph heights
    assert len(_added(gray, 300, monkeypatch)) == 1


def test_rule_along_the_tops_of_bold_letters_adds_no_line(monkeypatch):
    """A 1 pt rule laid along the tops of 18 pt bold Courier capitals, softened as by a scanner,
    merges with their serifs into one piece too thick to stitch. Its band is thin, and the second
    pass keeps only thick lines through such bands, so it adds nothing."""
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=120)
    page.insert_text((24, 64), "TEXT OF THE STATUTE", fontname="cobo", fontsize=18)
    font = pymupdf.Font("cobo")
    top = 64 - 18 * font.glyph_bbox(ord("T")).y1
    right = 28 + font.text_length("TEXT OF THE STATUTE", 18)
    page.draw_rect(pymupdf.Rect(20, top - 1.0, right, top), color=None, fill=0)
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).astype(float)
    gray = np.clip(ndimage.gaussian_filter(gray, 0.6), 0, 255).astype(np.uint8)
    assert _added(gray, 200, monkeypatch) == []


def test_rising_line_past_the_strike_is_not_added(monkeypatch):
    """In 14 pt bold Helvetica the 2.5 pt strike stitches on to glyph pieces on other rows, into
    a line that rises across the live words before it. It runs unbroken over too little of its
    length (SOLID_SHARE), so it is not added. With SOLID_SHARE = 0 it is, from "the lessee" on."""
    gray, words = _page("the lessee shall hold the deposit in trust", "hebo", 14, (2, 4), 2.5)
    x0 = words[2].bbox[0] * gray.shape[1]
    assert all(ln["bbox_px"][0] >= x0 - 10 for ln in _added(gray, 200, monkeypatch))
    monkeypatch.setattr(lines, "SOLID_SHARE", 0)
    assert any(ln["bbox_px"][0] < x0 - 10 for ln in _added(gray, 200, monkeypatch))


@pytest.mark.parametrize("font,text,dpi", [("hebo", "Terms of Sale", 200),
                                           ("cobo", "TERMS OF SALE", 300)])
def test_one_strike_gives_one_second_pass_line(font, text, dpi, monkeypatch):
    """The second pass finds the strike through 24 pt bold text in two stitched groups. A line
    along half of which a kept line already runs is dropped (EXTRA_ALONG_MAX), so one line comes
    out; without that test, two do."""
    gray, _ = _page(text, font, 24, (1, 2), 2.0, dpi)
    assert len(_added(gray, dpi, monkeypatch)) == 1
    monkeypatch.setattr(lines, "EXTRA_ALONG_MAX", 99)
    assert len(_added(gray, dpi, monkeypatch)) == 2


@pytest.mark.parametrize("font,size,dpi", [("hebo", 24, 200), ("tiro", 11, 200), ("cobo", 24, 300)])
def test_shared_spine_read_changes_nothing(font, size, dpi):
    """The thickness, stem and solid-run tests read the spine once and share it: each gives the
    same answer as reading it on its own, on every line of a struck page."""
    gray, _ = _page("hold the deposit in a segregated account", font, size, (1, 3), 2.0, dpi)
    ink = lines.ink_mask(gray)
    found = st.strike_lines(gray, dpi=dpi, ink=ink)
    assert found
    for ln in found:
        (sx, sy), (ex, ey) = ln["ends_px"]
        vec = np.array([ex - sx, ey - sy])
        length = float(np.hypot(*vec))
        center, u = np.array([(sx + ex) / 2, (sy + ey) / 2]), vec / length
        prof = lines._spine_profile(ink, center, u, length)
        assert (lines._spine_run_thickness(ink, center, u, length, prof=prof)
                == lines._spine_run_thickness(ink, center, u, length))
        assert (lines._stem_width(ink, center, u, length, 1, 8, prof=prof)
                == lines._stem_width(ink, center, u, length, 1, 8))
        assert (lines._solid_run(ink, center, u, length, ln["run_px"], prof=prof)
                == lines._solid_run(ink, center, u, length, ln["run_px"]))
