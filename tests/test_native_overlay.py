"""Text struck with U+0336 COMBINING LONG STROKE OVERLAY characters (``o̶f̶``), as online
strikethrough generators write it: the vector detector strikes each character an overlay follows,
and its record leaves out the overlays on the characters it reports. Overlay handling never hides
a drawn strike and never leaves struck text reading as live. No base-14 font has the glyph, so the
fixtures are set in PyMuPDF's built-in Noto Serif, which does; one test needs no font at all."""
import os

import pymupdf
import pytest

import pdf_strikethrough as st
from pdf_strikethrough import native
from _synth import _reopen

FS = 12
S = "\N{COMBINING LONG STROKE OVERLAY}"        # U+0336
ACUTE = "\N{COMBINING ACUTE ACCENT}"


def _ov(text):
    """`text` with an overlay after every character, as a generator writes it."""
    return "".join(c + S for c in text)


@pytest.fixture(scope="module")
def noto():
    font = pymupdf.Font(script=0)
    if not font.has_glyph(ord(S)):
        msg = "this PyMuPDF build's Noto Serif has no U+0336"
        if os.environ.get("CI"):
            pytest.fail(msg + "; the overlay tests would all be skipped")
        pytest.skip(msg)
    return font


def _page(noto, *parts):
    """One line of `parts` set in Noto. A part given as ``(text, "drawn")`` gets a red line
    through its middle, one given as ``(text, "annot")`` a /StrikeOut annotation. Returns the
    reopened document."""
    doc = pymupdf.open()
    page = doc.new_page()
    tw = pymupdf.TextWriter(page.rect)
    pos, marks = pymupdf.Point(72, 200), []
    for part in parts:
        text, how = part if isinstance(part, tuple) else (part, None)
        x0 = pos.x
        _, pos = tw.append(pos, text, font=noto, fontsize=FS)
        if how:
            marks.append((how, x0, pos.x))
    tw.write_text(page)
    for how, x0, x1 in marks:
        if how == "drawn":
            page.draw_line((x0, 200 - 0.3 * FS), (x1, 200 - 0.3 * FS), width=0.8, color=(1, 0, 0))
        else:
            page.add_strikeout_annot(pymupdf.Rect(x0, 200 - 0.9 * FS, x1, 201))
    return _reopen(doc)


def _recs(res):
    return [(r["text"], r["chars"], r["char_span"], r["partial"]) for r in res["words"]]


def _mupdf_flagged(doc):
    """The text MuPDF flags as struck out (FZ_STEXT_STRIKEOUT) on the first page: what pymupdf4llm
    renders as ~~, so what clean_markdown() drops."""
    flags = pymupdf.TEXTFLAGS_DICT | pymupdf.TEXT_COLLECT_STYLES | pymupdf.TEXT_COLLECT_VECTORS
    return "".join(s["text"] for b in doc[0].get_text("dict", flags=flags)["blocks"]
                   for line in b.get("lines", ()) for s in line["spans"]
                   if s.get("char_flags", 0) & pymupdf.mupdf.FZ_STEXT_STRIKEOUT)


def test_overlaid_word_is_struck_and_its_overlays_leave_the_text(noto):
    res = st.detect_pdf(_page(noto, f"keep {_ov('of')} this"))
    assert _recs(res) == [("of", "of", (0, 2), False)]
    (rec,) = res["words"]
    assert rec["tier"] == "vector" and rec["coverage"] == 1.0
    assert "stroke_color" not in rec and "stroke_width" not in rec     # no stroke to report
    assert res["clean_text"] == "keep this"
    assert res["markdown"] == "keep ~~of~~ this"
    assert [p["text"] for p in res["passages"]] == ["of"]


def test_partly_overlaid_word_is_a_partial_strike(noto):
    res = st.detect_pdf(_page(noto, f"the {_ov('semi-')}monthly rate"))
    assert _recs(res) == [("semi-monthly", "semi-", (0, 5), True)]
    assert res["clean_text"] == "the monthly rate"
    assert res["markdown"] == "the ~~semi-~~monthly rate"


def test_overlay_on_a_space_strikes_nothing(noto):
    # a generator overlays the space between struck words too; MuPDF hands that overlay to the
    # next word, where it has no character to strike
    res = st.detect_pdf(_page(noto, f"keep {_ov('of the')} rest"))
    assert _recs(res) == [("of", "of", (0, 2), False), ("the", "the", (0, 3), False)]
    assert res["clean_text"] == "keep rest"
    assert [p["text"] for p in res["passages"]] == ["of the"]
    res = st.detect_pdf(_page(noto, f"x {S} y"))      # an overlay alone (MuPDF may drop it)
    assert res["words"] == [] and res["clean_text"] in ("x y", f"x {S} y")


@pytest.mark.parametrize("word", [f"e{ACUTE}{S}lan", f"e{S}{ACUTE}lan"])
def test_overlay_strikes_a_base_character_with_its_accent(noto, word):
    # the accent before the overlay, or after it as normalization (NFD) orders them
    res = st.detect_pdf(_page(noto, f"un {word}"))
    assert _recs(res) == [(f"e{ACUTE}lan", f"e{ACUTE}", (0, 2), True)]
    assert res["clean_text"] == "un lan"
    assert native.strip_struck_markdown(f"un {word}") == "un lan"


@pytest.mark.parametrize("parts, struck, clean, min_cov", [
    ((f"keep {_ov('al')}", ("pha", "drawn"), " rest"), "alpha", "keep rest", 0.95),
    (("keep ", (_ov("alpha"), "drawn"), " rest"), "alpha", "keep rest", 0.95),
    ((f"keep {_ov('al')}pha rest",), "al", "keep pha rest", 0.3),
])
def test_overlay_and_drawn_strike_on_one_word_count_once(noto, parts, struck, clean, min_cov):
    doc = _page(noto, *parts)
    for method in ("vector", "both"):
        res = st.detect_pdf(doc, method=method)
        (rec,) = res["words"]
        assert (rec["text"], rec["chars"], rec["tier"]) == ("alpha", struck, "vector")
        assert ("stroke_color" in rec) == (struck == "alpha")      # a stroke took part
        assert min_cov <= rec["coverage"] <= 1.0                    # a share of the word's width
        assert res["clean_text"] == clean


def test_a_second_overlaid_run_stays_marked(noto):
    # one record holds one span: the longer run is reported, the other keeps its overlay in the
    # record and the markdown, and clean_text drops it
    res = st.detect_pdf(_page(noto, f"keep {_ov('a')}b{_ov('cd')} x"))
    assert _recs(res) == [(f"a{S}bcd", "cd", (3, 5), True)]
    assert res["markdown"] == f"keep a{S}b~~cd~~ x"
    assert res["clean_text"] == "keep b x"


def test_overlays_never_displace_a_drawn_strike(noto):
    res = st.detect_pdf(_page(noto, f"keep {_ov('abcd')}e", ("fg", "drawn"), " rest"))
    (rec,) = res["words"]
    assert (rec["text"], rec["chars"]) == (f"{_ov('abcd')}efg", "fg")
    assert rec["stroke_color"] == (1.0, 0.0, 0.0)
    assert res["clean_text"] == "keep e rest"


@pytest.mark.parametrize("parts, chars", [
    ((f"keep {_ov('a')}b{_ov('cd')}", ("e", "drawn"), " x"), "cde"),   # merged with 'cd'
    (("keep ", ("a", "drawn"), f"b{_ov('cdefgh')} x"), "a"),             # beats six overlays
])
def test_clean_text_agrees_with_clean_markdown(noto, parts, chars):
    # overlaid characters outside the record's span leave clean_text, as clean_markdown() drops
    # them; 'flag' doesn't read overlays, so its clean_text keeps them as it always has
    doc = _page(noto, *parts)
    for method in ("vector", "both"):
        res = st.detect_pdf(doc, method=method)
        (rec,) = res["words"]
        assert rec["chars"] == chars and S in rec["text"]
        assert res["clean_text"] == "keep b x"
        assert native.strip_struck_markdown(res["markdown"]) == res["clean_text"]
    assert S in st.detect_pdf(doc, method="flag")["clean_text"]
    try:
        md = st.clean_markdown(doc)
    except ImportError:                         # no [markdown] extra
        return
    # clean_markdown() drops what MuPDF flags, and the flag can take in the live character a drawn
    # strike ends against, as 'flag' can: MuPDF 1.28.5 (PyMuPDF 2.0) flags the 'b' after the drawn
    # 'a', 1.28.2 does not. The overlaid characters go either way; 'b' only where MuPDF flags it.
    live = ["keep", "x"] if "b" in _mupdf_flagged(doc) else ["keep", "b", "x"]
    assert md.split() == live


def test_flag_and_annot_alone_leave_overlaid_text_as_it_is(noto):
    doc = _page(noto, f"keep {_ov('of')} this")
    for method in ("flag", "annot"):
        res = st.detect_pdf(doc, method=method)
        assert res["words"] == [] and res["clean_text"] == f"keep {_ov('of')} this"
    doc = _page(noto, "keep ", (_ov("of"), "annot"), " this")
    res = st.detect_pdf(doc, method="annot")
    assert _recs(res) == [(_ov("of"), _ov("of"), (0, 4), False)]
    assert res["clean_text"] == "keep this"
    # 'both': the vector record (it reads the overlays) covers the annotation's word
    res = st.detect_pdf(doc, method="both")
    assert [(r["tier"], r["text"]) for r in res["words"]] == [("vector", "of")]


def test_annotation_and_overlays_on_one_word_join(noto):
    # the annotation's appearance stream is a drawn line over 'ab', the overlays strike 'cd'
    doc = _page(noto, "keep ", ("ab", "annot"), f"{_ov('cd')} x")
    for method in ("vector", "both"):
        res = st.detect_pdf(doc, method=method)
        assert [(r["tier"], r["chars"]) for r in res["words"]] == [("vector", "abcd")]
        assert res["clean_text"] == "keep x"


@pytest.mark.parametrize("flag_span, kept", [((0, 2), ("flag", "ab")), ((2, 6), ("vector", "cd"))])
def test_both_keeps_a_flag_span_the_overlaid_run_misses(noto, monkeypatch, flag_span, kept):
    # one record per word: a flag span outside an overlays-only vector record replaces it (its
    # text keeps the overlays, so 'cd' still reads as struck); one inside it changes nothing
    doc = _page(noto, f"keep ab{_ov('cd')} x")
    (vec,) = st.detect_pdf(doc, method="vector")["words"]
    raw = f"ab{_ov('cd')}"
    flag = {k: vec[k] for k in ("page", "bbox_frac")}
    flag.update(text=raw, chars=raw[slice(*flag_span)], char_span=flag_span, partial=True,
                coverage=0.5, tier="flag", verdict="struck", final=True)
    monkeypatch.setattr(native, "native_flag_strikes", lambda *a, **k: [dict(flag)])
    res = st.detect_pdf(doc, method="both")
    assert [(r["tier"], r["chars"].replace(S, "")) for r in res["words"]] == [kept]
    assert res["clean_text"] == ("keep x" if kept[0] == "flag" else "keep ab x")


def test_overlay_set_in_a_second_font(noto):
    """Base letters in Helvetica, each overlay in Noto centred on its letter with no advance: the
    layout a generator's text takes when the body font lacks U+0336. Older MuPDF splits such a word
    at each overlay, so only the struck text and the clean text are pinned."""
    doc = pymupdf.open()
    page = doc.new_page()
    tw = pymupdf.TextWriter(page.rect)
    helv = pymupdf.Font("helv")
    pos = pymupdf.Point(72, 200)
    for text, over in (("keep ", False), ("of", True), (" this", False)):
        for ch in text:
            _, pos = tw.append(pos, ch, font=helv, fontsize=FS)
            if over:
                tw.append((pos.x - helv.text_length(ch, FS) / 2, pos.y), S, font=noto, fontsize=FS)
    tw.write_text(page)
    res = st.detect_pdf(_reopen(doc))
    assert "".join(r["chars"] for r in res["words"]) == "of"
    assert res["clean_text"] == "keep this"


def test_other_overlays_are_not_read(noto):
    # U+0335/U+0337 are short overlays, U+0338 makes a decomposed ≠: none is read as a strike
    for mark in ("\N{COMBINING SHORT STROKE OVERLAY}", "\N{COMBINING SHORT SOLIDUS OVERLAY}",
                 "\N{COMBINING LONG SOLIDUS OVERLAY}"):
        text = f"keep o{mark}f{mark} this"
        res = st.detect_pdf(_page(noto, text))
        assert res["words"] == [] and res["clean_text"] == text


def test_supplied_words_need_no_glyph_or_size_floor():
    # words= with no matching text layer: the overlays alone decide, and a word under the 3 pt
    # width floor (which keeps gridlines off tiny glyphs) is still struck by them
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 200), "anything", fontsize=FS)
    words = [(72, 190, 90, 203, f"keep{S}s", 0, 0, 0), (100, 190, 102, 203, f"i{S}", 0, 0, 1),
             (110, 190, 130, 203, f"{S}{S}", 0, 0, 2), (140, 190, 160, 203, "live", 0, 0, 3)]
    recs = native.native_page_strikes(page, 0, words=words)
    assert [(r["text"], r["chars"], r["partial"]) for r in recs] == [("keeps", "p", True),
                                                                     ("i", "i", False)]


@pytest.mark.parametrize("md, clean", [
    (f"keep {_ov('of')} ~~old~~ this {S}x", "keep this x"),
    (f"z~~abc~~{S}d{S} x", "z x"),               # an overlay after a struck span strikes it
    (f"keep **{S}{_ov('of')}** x", "keep **** x"),  # one after emphasis markup strikes nothing
    (f"keep {_ov('old_rate')} and {_ov('a*b')} x", "keep and x"),  # a '_' or '*' in a word
    (f"keep ~~old~~{_ov('_x')} y", "keep y"),
])
def test_strip_struck_markdown_drops_overlaid_characters(md, clean):
    assert native.strip_struck_markdown(md) == clean


def test_strip_struck_reads_every_star_and_underscore_as_text():
    # detect_pdf's markdown writes no emphasis markup, so a struck '*' opening a word is struck
    from pdf_strikethrough import markdown
    md = f"keep {_ov('*a1')} {_ov('_note')} x"
    assert markdown.strip_struck(md) == "keep x"
    assert native.strip_struck_markdown(md) == "keep * _ x"      # pymupdf4llm's: read as markup
