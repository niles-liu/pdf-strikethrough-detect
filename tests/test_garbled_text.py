"""Garbled text layers: a font whose glyph-to-Unicode map is wrong (Word's Cambria Math export
reads '·(0.955) −' as '⋅ሺ0.955ሻെ') is flagged and reported, and real text in those scripts, digits
and Latin abbreviations included, is not. The text-level and synthetic-PDF tests always run; the
genuine-script PDF tests need a system font for the script and are skipped without one."""
import functools
import json
import os
import warnings
from pathlib import Path

import pymupdf
import pytest

import pdf_strikethrough as st
from pdf_strikethrough import __main__ as cli
from pdf_strikethrough.native import _garbled_text

# ------------------------------------------------------------------------- the text test

GARBLED = [
    "⋅ሺ0.955ሻെ",                   # Ethiopic brackets beside a Malayalam vowel sign
    "ሾemissionሿ\u0bdc",             # an unassigned code point in the Tamil block
    "\uffff", "\ufdd0",               # noncharacters
    "𝐹𝐿ୣୟ୰୪୷ୡ୰ୣୢ",                   # mathematical letters beside Oriya
    "20െ",                            # a Malayalam vowel sign after a digit, none after it
    "ሻെx",                            # ... after an Ethiopic letter
    "ሺ\ue000", "𝑥\ue000",             # a private-use character beside Ethiopic or a math letter
]

REAL = {
    "amharic": "በ2015 ዓ.ም. የUNDP ሪፖርት (ገጽ 3) መሠረት 45% የሚሆኑ ሰዎች በገጠር ይኖራሉ። ዋና ከተማዋ "
               "አዲስ አበባ ናት፤ ከ10% በላይ፣ የWHO መረጃ (2020)፣ ዓመት ፲፱፻፹፯ ዓ.ም.",
    "devanagari": "भारत में 2015 की UNDP रिपोर्ट (पृष्ठ 3) के अनुसार 45% लोग गाँवों में रहते हैं। "
                  "ज़िंदगी, क्षेत्र, श्री, (कि), ई-मेल, WHO-के, १९४७। अ॒ग्निमी॑ळे",
    "tamil": "2015-ஆம் ஆண்டு UNDP அறிக்கையின்படி (பக்கம் 3) 45% மக்கள் கிராமங்களில் வாழ்கின்றனர். "
             "(கொ) கெ, கை, கௌ.",
    "bengali": "২০১৫ সালের UNDP-এর প্রতিবেদন (পৃষ্ঠা ৩) অনুযায়ী ৪৫% মানুষ গ্রামে বাস করে। "
               "ঢাকা বাংলাদেশের রাজধানী।",
    "malayalam": "2015-ലെ UNDP റിപ്പോർട്ട് (പേജ് 3) പ്രകാരം 45% ആളുകൾ ഗ്രാമങ്ങളിൽ താമസിക്കുന്നു.",
    "sinhala": "2015 UNDP වාර්තාවට (පිටුව 3) අනුව 45% ජනතාව. ශ්‍රී ලංකාව.",
    "thai": "ในปี 2558 รายงานของ UNDP (หน้า 3) ระบุว่า 45% ของประชากรอาศัยอยู่ในชนบท",
    "arabic": "وفقا لتقرير UNDP لعام 2015م (ص 3) فإن 45% من السكان يعيشون في الريف. ٢٠١٥، ۱۳۹۴.",
    "hebrew": "לפי דו״ח UNDP לשנת 2015 (עמ׳ 3), 45% מהאוכלוסייה גרים בכפרים. ב-2015 בְּרֵאשִׁית.",
}


@pytest.mark.parametrize("text", GARBLED)
def test_garbled_text_is_flagged(text):
    assert _garbled_text(text)


@pytest.mark.parametrize("script", sorted(REAL))
def test_real_text_is_not_flagged(script):
    for word in REAL[script].split():
        assert not _garbled_text(word), word
        assert not _garbled_text(word, ["SomeFont"] * len(word)), word


def test_visual_order_vowel_sign_and_lone_symbols_are_not_flagged():
    # extraction in visual order puts a vowel sign before its consonant, also right after a digit
    # or a Latin letter ('2015ലെ' reads '2015െല'); a lone vowel sign is what a split cluster leaves
    for word in ("(िक", "िक", "െ", "◌ൌ", "2015െല", "UNDPെയ", "UNDPைய", "3িট", "2015िक",
                 "2015ലെ", "UNDPയെ", "UNDPயை", "3টি", "2015कि"):
        assert not _garbled_text(word), word


def test_private_use_beside_latin_or_cjk_is_a_symbol_not_garbled():
    # a Symbol-font degree sign, an icon font, a ligature, a bullet, a CJK line MuPDF reads as one
    # word: none is a broken map's output
    for word in ("25\uf0b0C", "\uf095Call", "of\ue001ce", "\uf0b7", "申請者は東京都千代田区\ue000町"):
        assert not _garbled_text(word), word


def test_characters_newer_than_pythons_unicode_tables_are_not_flagged():
    # U+20C1 SAUDI RIYAL SIGN (Unicode 17) is unassigned in every supported Python's tables; the
    # others are on 3.10: the SOM sign, Telugu nukta, an Arabic ligature, CJK, Arabic pepet; the
    # last four are Unicode 17 and 18 letters inside the checked script blocks
    for text in ("Total \u20c1 1,000", "\u20c11,000", "\u20c0", "\u0c15\u0c3c", "\ufd40",
                 "\u9ffd", "\u0897", "\u088f", "\u0c5c", "\u0cdc", "\ufdc8"):
        for word in text.split():
            assert not _garbled_text(word), word


def test_script_characters_from_a_math_font_are_flagged():
    for font in ("Cambria", "CambriaMath", "Cambria Math", "PMLBNH+Cambria", "STIXTwoMath-Regular",
                 "XITSMath"):
        assert _garbled_text("ሺ20", [font] * 3), font
        assert _garbled_text("൨", [font]), font
    for font in ("Ebrima", "Nyala", "Mathura", None):     # a script font; "Math" not a word
        assert not _garbled_text("ሺ20", [font] * 3), font
    assert not _garbled_text("x2", ["Cambria Math"] * 2)    # Latin from a math font is fine


# ---------------------------------------------------------------- a broken map in a real PDF

# Word maps Cambria Math's stretched brackets and minus sign to their glyph numbers; this gives a
# base-14 font the same kind of /ToUnicode: '(' ')' '-' read as Ethiopic and Malayalam, '=' as an
# unassigned code point. No font file is needed, so this always runs.
_BROKEN_CMAP = b"""/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<00> <FF>
endcodespacerange
4 beginbfchar
<28> <123A>
<29> <123B>
<2D> <0D46>
<3D> <0D45>
endbfchar
4 beginbfrange
<20> <27> <0020>
<2A> <2C> <002A>
<2E> <3C> <002E>
<3E> <7E> <003E>
endbfrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end"""


def _break_font(doc, page, fontname):
    for xref, _ext, _type, _base, name, *_ in page.get_fonts():
        if name == fontname:
            cmap = doc.get_new_xref()
            doc.update_object(cmap, "<<>>")
            doc.update_stream(cmap, _BROKEN_CMAP)
            doc.xref_set_key(xref, "ToUnicode", f"{cmap} 0 R")
            return
    raise AssertionError(f"no font {fontname} on the page")


def _strike(page, word):
    r = {w[4]: pymupdf.Rect(w[:4]) for w in page.get_text("words")}[word]
    y = (r.y0 + r.y1) / 2
    page.draw_line((r.x0 - 1, y), (r.x1 + 1, y), width=0.8)


def _broken_pdf():
    """Prose in Helvetica with 'old' struck, and an equation line in Times-Roman whose map is
    broken, with its last word struck: 'E = k(0.955)-x' reads 'E ൅ kሺ0.955ሻെx'."""
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=300)
    page.insert_text((40, 60), "The old rate is computed as follows:", fontname="helv",
                     fontsize=11)
    page.insert_text((40, 90), "E = k(0.955)-x", fontname="tiro", fontsize=11)
    _break_font(doc, page, "tiro")
    doc = pymupdf.open("pdf", doc.tobytes())          # reopened, so the text reads the new map
    _strike(doc[0], "old")
    _strike(doc[0], "kሺ0.955ሻെx")
    return doc.tobytes()


def test_broken_map_is_flagged_and_warned():
    with pytest.warns(UserWarning, match="broken character map"):
        res = st.detect_pdf(_broken_pdf())
    assert [g["text"] for g in res["garbled_words"]] == ["\u0d45", "kሺ0.955ሻെx"]
    assert all(g["page"] == 0 and len(g["bbox_frac"]) == 4 for g in res["garbled_words"])
    msgs = [w for w in res["warnings"] if "broken character map" in w]
    assert msgs == [res["warnings"][-1]] and msgs[0].startswith("page 0: the text of 2 words")
    struck = {r["text"]: r for r in res["words"]}
    assert struck["kሺ0.955ሻെx"]["garbled"] is True
    assert "garbled" not in struck["old"]
    # kept by default: clean_text still says what the text layer says, and the strike holds
    assert "E \u0d45" in res["clean_text"] and "kሺ" not in res["clean_text"]
    assert "~~kሺ0.955ሻെx~~" in res["markdown"]


def test_on_garbled_drop_leaves_the_words_out_of_the_text():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.detect_pdf(_broken_pdf(), on_garbled="drop")
    assert "\u0d45" not in res["clean_text"] and "kሺ" not in res["markdown"]
    assert res["clean_text"].splitlines()[-1] == "E"
    assert [p["text"] for p in res["passages"]] == ["old"]
    assert len(res["garbled_words"]) == 2                 # still listed, and still struck
    assert any(r.get("garbled") for r in res["words"])


def _two_broken_pages():
    doc = pymupdf.open()
    for _ in range(2):
        doc.insert_pdf(pymupdf.open("pdf", _broken_pdf()))
    return doc.tobytes()


def test_one_warning_per_document_and_a_note_per_page():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = st.detect_pdf(_two_broken_pages(), on_garbled="drop")
    garbled = [w for w in caught if issubclass(w.category, st.GarbledTextWarning)]
    assert len(garbled) == 1 and len(caught) == 1
    msg = str(garbled[0].message)
    assert "4 words on pages 0, 1" in msg and "leave them out" in msg
    assert [w[:7] for w in res["warnings"]] == ["page 0:", "page 1:"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        st.detect_pdf(_two_broken_pages(), include_markdown=False)
    assert "markdown" not in str(caught[0].message)       # no text was assembled


def test_low_level_detectors_flag_the_record():
    with pymupdf.open("pdf", _broken_pdf()) as doc:
        page = doc[0]
        for detect in (st.native_page_strikes, st.native_flag_strikes):
            recs = {r["text"]: r for r in detect(page, 0)}
            assert recs["kሺ0.955ሻെx"].get("garbled") is True, detect.__name__


def test_cli_json_lists_garbled_words_without_doubling_the_warning(tmp_path, capsys):
    src, out = tmp_path / "broken.pdf", tmp_path / "out.json"
    src.write_bytes(_two_broken_pages())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert cli.main(["detect", str(src), "--json", str(out)]) == 0
    assert not [w for w in caught if issubclass(w.category, st.GarbledTextWarning)]
    assert capsys.readouterr().err.count("broken character map") == 2   # one line a page
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [g["text"] for g in payload["garbled_words"]] == ["\u0d45", "kሺ0.955ሻെx"] * 2
    assert all(w["garbled"] for w in payload["words"] if w["text"].startswith("k"))


def test_on_garbled_is_validated():
    with pytest.raises(ValueError, match="on_garbled"):
        st.detect_pdf(_broken_pdf(), on_garbled="Drop")


def test_strikethroughs_in_pdf_flags_the_record():
    recs = {r["text"]: r for r in st.strikethroughs_in_pdf(_broken_pdf(), method="both")}
    assert recs["kሺ0.955ሻെx"].get("garbled") is True and "garbled" not in recs["old"]


def test_clean_pdf_has_no_garbled_words():
    doc = pymupdf.open()
    doc.new_page().insert_text((40, 60), "E = k(0.955) - x, all in one font", fontname="tiro")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = st.detect_pdf(doc.tobytes())
    assert res["garbled_words"] == [] and res["warnings"] == []


# ------------------------------------------------- genuine scripts in their own fonts (if found)

_FONT_DIRS = ([Path(os.environ["WINDIR"]) / "Fonts"] if os.environ.get("WINDIR") else []) + [
    Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path("/Library/Fonts"),
    Path("/System/Library/Fonts"), Path.home() / "Library" / "Fonts"]
_FONTS = {   # script -> font files that draw it, the Windows, Noto and macOS ones
    "amharic": ["ebrima.ttf", "nyala.ttf", "NotoSansEthiopic-Regular.ttf",
                "NotoSerifEthiopic-Regular.ttf", "Kefa.ttc"],
    "devanagari": ["Nirmala.ttc", "Nirmala.ttf", "mangal.ttf", "NotoSansDevanagari-Regular.ttf",
                   "Lohit-Devanagari.ttf", "Kohinoor.ttc"],
    "tamil": ["Nirmala.ttc", "Nirmala.ttf", "latha.ttf", "NotoSansTamil-Regular.ttf",
              "Lohit-Tamil.ttf"],
    "bengali": ["Nirmala.ttc", "Nirmala.ttf", "vrinda.ttf", "NotoSansBengali-Regular.ttf",
                "Lohit-Bengali.ttf"],
    "malayalam": ["Nirmala.ttc", "Nirmala.ttf", "kartika.ttf", "NotoSansMalayalam-Regular.ttf"],
    "arabic": ["arial.ttf", "NotoSansArabic-Regular.ttf", "NotoNaskhArabic-Regular.ttf"],
    "hebrew": ["arial.ttf", "NotoSansHebrew-Regular.ttf"],
}


@functools.cache
def _font_index():
    found = {}
    for d in _FONT_DIRS:
        if d.is_dir():
            for f in d.rglob("*"):
                found.setdefault(f.name.lower(), f)
    return found


def _font(script):
    """The first font file found that has a glyph for every character of the script's text."""
    index = _font_index()
    for name in _FONTS[script]:
        if name.lower() in index:
            font = pymupdf.Font(fontfile=str(index[name.lower()]))
            if all(font.has_glyph(ord(c)) for c in REAL[script] if not c.isspace()):
                return index[name.lower()]
    return None


def _genuine_pdf(script, font, broken=False):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    rc = page.insert_textbox(pymupdf.Rect(40, 120, 555, 800), REAL[script], fontsize=11,
                             fontname="G", fontfile=str(font))
    assert rc >= 0, "the text did not fit"
    if broken:
        page.insert_text((40, 80), "E = k(0.955)-x", fontname="tiro", fontsize=11)
        _break_font(doc, page, "tiro")
    return doc.tobytes()


@pytest.mark.parametrize("script", sorted(_FONTS))
def test_genuine_script_text_is_not_flagged(script):
    font = _font(script)
    if font is None:
        pytest.skip(f"no {script} font on this machine")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.detect_pdf(_genuine_pdf(script, font))
    extracted = res["clean_text"].split()
    assert len(extracted) >= 0.8 * len(REAL[script].split())   # the text layer is the text
    assert res["garbled_words"] == [], res["garbled_words"]
    assert not any("broken character map" in w for w in res["warnings"])


def test_genuine_amharic_beside_a_broken_map_keeps_its_words():
    font = _font("amharic")
    if font is None:
        pytest.skip("no Ethiopic font on this machine")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.detect_pdf(_genuine_pdf("amharic", font, broken=True))
    assert [g["text"] for g in res["garbled_words"]] == ["\u0d45", "kሺ0.955ሻെx"]
