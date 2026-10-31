"""With Azure DI words, the OCR-confidence rescue passes a word under a short line that lacks glyph
ink on both sides. It used to pass it before any retry ran, so a double strike or a thick bar there
got no strike geometry and the DI-confidence veto dropped it when it was read above
max_clean_conf (0.955): from 0.975 the rescue no longer applies, a retry passes the word with
geometry, and it was kept. The retries now run after a rescue too, and the rescued state stands
when none gives strike geometry (in the band with ink on both sides, or a pair)."""
import numpy as np
import pytest

from pdf_strikethrough import cnn, detect, scanned
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import ScanConfig, classify_lines

DI, CF = ScanConfig.azure_di(), ScanConfig.confidence_free()
H, W, X0, ADV = 300, 900, 200, 18


@pytest.fixture(autouse=True)
def _strikenet_says_struck(monkeypatch):
    monkeypatch.setattr(cnn, "score_crops", lambda crops: np.full(len(crops), 0.99))


def _x1(n):
    return X0 + (n - 1) * ADV + 14


def _letters(gray, n, top, bot):
    for i in range(n):
        for s in (0, 5, 11):
            gray[top:bot, X0 + i * ADV + s:X0 + i * ADV + s + 3] = 20


def _double_strike(n=4):
    """Letters on rows 140-157, 2 px strokes on rows 142 and 152, the line on the upper one:
    0.36 in long, under INK_SHORT_LEN_IN."""
    g = np.full((H, W), 255, np.uint8)
    _letters(g, n, 140, 158)
    xa, xb = X0 - 2, _x1(n) + 2
    for y in (142, 152):
        g[y:y + 2, xa:xb] = 20
    ln = {"bbox_px": (xa, 142, xb, 143), "ends_px": ((float(xa), 142.5), (float(xb), 142.5)),
          "len_in": (xb - xa) / 200, "angle_deg": 0.0, "fill": 0.95, "run_px": 2.0,
          "straightness": 1.0}
    return g, ln, (X0 / W, 119 / H, _x1(n) / W, 179 / H), "abcd"


def _thick_bar(n=3):
    """Letters on rows 140-152, a 6 px bar on rows 143-148 under a 0.27 in line."""
    g = np.full((H, W), 255, np.uint8)
    _letters(g, 12, 140, 153)
    xa, xb = X0 - 2, X0 + (n - 1) * ADV + 16
    g[143:149, X0 - 2:_x1(12) + 2] = 20
    ln = {"bbox_px": (xa, 142, xb, 148), "ends_px": ((float(xa), 145.6), (float(xb), 145.6)),
          "len_in": (xb - xa) / 200, "angle_deg": 0.0, "fill": 0.95, "run_px": 6.0,
          "straightness": 1.0}
    return g, ln, (X0 / W, 116 / H, (xb - 2) / W, 175 / H), "abc"


def _stroke_row_out_of_band():
    """A short word over a double underline (rows 102-103 and 106-107), the line 4.5 px above the
    upper one and in the band. The stroke's row is out of the band, where the rescue does not
    apply, so a retry there must not cost the word its rescue."""
    g = np.full((220, 760), 255, np.uint8)
    for i in range(5):
        for s in (0, 6, 12):
            g[95:105, 40 + i * 18 + s:40 + i * 18 + s + 4] = 20
    g[105:112, 76:80] = 20                                     # a descender
    for y in (102, 106):
        g[y:y + 2, 38:131] = 20
    ln = {"bbox_px": (38, 96, 131, 98), "ends_px": ((38.0, 97.5), (131.0, 97.5)),
          "len_in": 0.465, "angle_deg": 0.0, "fill": 0.9, "run_px": 2.0, "straightness": 0.6}
    return g, ln, (40 / 760, 79 / 220, 128 / 760, 109 / 220), "abcde"


def _records(page, conf, config=DI, retries=True, monkeypatch=None):
    gray, line, box, txt = page
    monkeypatch.setattr(scanned, "RESCUE_RETRIES", retries)
    _, recs = classify_lines([line], [Word(txt, box, conf)], gray, config=config, dpi=200)
    return detect.apply_cnn_verdict(recs, gray, config=config)


@pytest.mark.parametrize("make", [_double_strike, _thick_bar])
@pytest.mark.parametrize("conf", [0.956, 0.96, 0.97, 0.974])
def test_a_rescued_strike_gets_geometry_and_survives_the_veto(monkeypatch, make, conf):
    page = make()
    before, = _records(page, conf, retries=False, monkeypatch=monkeypatch)
    assert not before["geom_corroborated"] and before["conf_veto"] and not before["final"]
    after, = _records(page, conf, monkeypatch=monkeypatch)
    assert after["geom_corroborated"] and after["final"] and not after.get("conf_veto")
    assert after["char_span"] == before["char_span"]


@pytest.mark.parametrize("make", [_double_strike, _thick_bar])
def test_the_rescue_no_longer_decides_the_record(monkeypatch, make):
    """Read at 0.975 the rescue never applied, and the retries passed the word; a word read at
    0.974 now gets the same record."""
    page = make()
    above, = _records(page, 0.975, retries=False, monkeypatch=monkeypatch)
    rescued, = _records(page, 0.974, monkeypatch=monkeypatch)
    for key in ("off", "char_span", "geom_corroborated", "tier", "final"):
        assert rescued[key] == above[key], key


@pytest.mark.parametrize("make", [_double_strike, _thick_bar, _stroke_row_out_of_band])
@pytest.mark.parametrize("conf", [0.5, 0.9, 0.975])
def test_confidence_free_and_unrescued_words_are_unchanged(monkeypatch, make, conf):
    page = make()
    for config, c in ((CF, None), (DI, conf)):
        if config is DI and c <= DI.inkfail_max_conf and make is not _stroke_row_out_of_band:
            continue                                       # rescued: the tests above
        old = _records(page, c, config, retries=False, monkeypatch=monkeypatch)
        new = _records(page, c, config, monkeypatch=monkeypatch)
        assert new == old


def _isolated_partial():
    """Eight letters, a 2 px stroke over letters 3-4 (0.15 in), the line 4 px under it at the
    letters' foot, in the middle of the box: no glyph ink below, so the rescue passes the word,
    an isolated partial strike in the middle. The stroke's row is in the band but 0.13 of the box
    up, past PARTIAL_ISO_MAX_OFF, where an isolated partial strike is dropped."""
    g = np.full((260, 800), 255, np.uint8)
    for i in range(8):
        for s in (0, 5, 10):
            g[100:124, 100 + i * 16 + s:100 + i * 16 + s + 3] = 20
    g[117:119, 148:179] = 20
    line = {"bbox_px": (148, 122, 178, 124), "ends_px": ((148.0, 123.0), (178.0, 123.0)),
            "len_in": 30 / 200, "angle_deg": 0.0, "fill": 0.9, "run_px": 2.0,
            "straightness": 1.0}
    return g, line, (100 / 800, 108 / 260, (100 + 7 * 16 + 13) / 800, 138 / 260), "abcdefgh"


@pytest.mark.parametrize("conf", [0.5, 0.9, 0.955])
def test_a_retry_never_moves_a_partial_word_where_it_is_dropped(monkeypatch, conf):
    page = _isolated_partial()
    old = _records(page, conf, retries=False, monkeypatch=monkeypatch)
    assert len(old) == 1 and old[0]["char_span"] == (3, 5) and old[0]["final"]
    assert _records(page, conf, monkeypatch=monkeypatch) == old


def test_an_isolated_partial_strike_restored_at_the_line_is_judged_there(monkeypatch):
    """The line over a third of the word, its stroke's row out of the band: restored at the line,
    the word is a partial strike in the band but off its middle, which an isolated line drops."""
    gray, line, box, txt = _stroke_row_out_of_band()
    line = {**line, "bbox_px": (38, 96, 70, 98), "ends_px": ((38.0, 97.5), (70.0, 97.5)),
            "len_in": 0.16}
    for retries in (False, True):
        assert _records((gray, line, box, txt), 0.9, retries=retries, monkeypatch=monkeypatch) == []


# Pages from a seeded fuzz of 16,000 cases (letters of three stems, strokes and finder lines
# drawn from a few numbers), where a wrong version of this change differs from it.
def _fuzz_page(top, bot, stem, adv, words, marks, strokes):
    g = np.full((220, 760), 255, np.uint8)
    for x0, n in words:
        for i in range(n):
            for s in (0, adv // 3, (2 * adv) // 3):
                g[top:bot, x0 + i * adv + s:x0 + i * adv + s + stem] = 20
    for x, up in marks:                                        # an ascender or a descender
        y0, y1 = (top - 7, top) if up else (bot, bot + 7)
        g[y0:y1, x:x + stem] = 20
    for y, t, x0, x1, gap in strokes:
        g[y:y + t, x0:x1] = 20
        if gap:
            g[y:y + t, gap[0]:gap[1]] = 255
    return g


def _line(x0, y0, x1, y1, ly, run, fill=None, wobble=None):
    ln = {"bbox_px": (x0, y0, x1, y1), "ends_px": ((float(x0), ly), (float(x1), ly)),
          "len_in": (x1 - x0) / 200, "angle_deg": 0.0, "run_px": run, "straightness": wobble}
    return ln if fill is None else {**ln, "fill": fill}


def _fuzz_records(case, retries, monkeypatch):
    page, lines, words, config, dpi = case
    monkeypatch.setattr(scanned, "RESCUE_RETRIES", retries)
    gray = _fuzz_page(**page)
    ws = [Word(t, (0.9, 0.05, 0.95, 0.1) if b == "far" else
               (b[0] / 760, b[1] / 220, b[2] / 760, b[3] / 220), c) for t, b, c in words]
    _, recs = classify_lines(lines, ws, gray, config=config, dpi=dpi)
    return detect.apply_cnn_verdict(recs, gray, config=config)


FUZZ = {
    # a double strike: the rescued word's stroke-row retry finds glyph ink on both sides
    "snap gives geometry": (
        dict(top=95, bot=109, stem=3, adv=15, words=[(40, 3), (92, 5)], marks=[(40, True)],
             strokes=[(100, 3, 5, 755, None), (109, 3, 5, 755, None)]),
        [_line(47, 99, 79, 100, 100.0, 1.5, 1.0, 2.0), _line(37, 104, 83, 107, 106.0, 3.0, 1.0, 1.3)],
        [("abc", (40, 83.0, 83, 135.0), 0.85), ("abcde", (92, 83.0, 165, 135.0), 0.974)], DI, 200),
    # a word under a short line that passes with ink on both sides is not retried
    "two-sided pass": (
        dict(top=98, bot=121, stem=3, adv=15, words=[(40, 4), (109, 8)], marks=[(70, True)],
             strokes=[(102, 3, 108, 229, (197, 200)), (111, 2, 108, 229, None)]),
        [_line(131, 101, 206, 104, 103.0, 3.0, 0.8, 0.6),
         _line(108, 110, 229, 112, 111.5, 2.0, 0.8, 1.3)],
        [("abcd", (40, 91.5, 98, 121.5), 0.98), ("abcdefgh", (109, 91.5, 227, 121.5), 0.955),
         ("zz", "far", 0.6)], DI, 200),
    # confidence-free, a long in-band line passing without ink on both sides: no rescue to retry
    "long line": (
        dict(top=98, bot=119, stem=4, adv=19, words=[(40, 3), (108, 3), (176, 3)],
             marks=[(146, False)], strokes=[(100, 4, 37, 232, (163, 165)), (113, 2, 37, 232, None)]),
        [_line(37, 100, 232, 102, 101.5, 1.5, None, 1.25),
         _line(37, 112, 232, 114, 113.5, 2.0, 0.8, 0.6)],
        [("abc", (40, 79.0, 94, 126.0), None), ("abc", (108, 79.0, 162, 126.0), None),
         ("abc", (176, 79.0, 230, 126.0), None)], CF, 300),
    # a double underline: the stroke-row retry passes on the rescue alone, so it is not taken
    "snap on the rescue alone": (
        dict(top=103, bot=119, stem=2, adv=16, words=[(40, 4)], marks=[(56, True)],
             strokes=[(117, 2, 5, 755, None), (121, 2, 39, 100, None)]),
        [_line(39, 121, 100, 123, 122.5, 2.0, None, 2.0)],
        [("abcd", (40, 90.5, 100, 143.5), 0.97)], DI, 200),
    # at 100 dpi, the thick retry after the stroke-row move passes on the rescue alone
    "thick on the rescue alone": (
        dict(top=94, bot=108, stem=2, adv=15, words=[(40, 6)], marks=[(40, True)],
             strokes=[(106, 2, 5, 755, None), (109, 2, 37, 128, None)]),
        [_line(37, 100, 128, 104, 102.5, 4.5, 1.0, 2.0)],
        [("abcdef", (40, 91.5, 127, 114.5), 0.955), ("zz", "far", 0.6)], DI, 100),
    # restored at the line, the partial word's span is read at the line, not the stroke's row
    "span at the line": (
        dict(top=92, bot=102, stem=4, adv=15, words=[(40, 6), (139, 7), (261, 8)],
             marks=[(70, False), (115, False), (139, False), (276, False)],
             strokes=[(92, 3, 259, 381, (350, 354)), (100, 4, 5, 755, None)]),
        [_line(276, 89, 370, 97, 93.0, 8.0, 0.86, 1.3), _line(282, 99, 356, 103, 101.5, 4.0, 0.9, 0.6)],
        [("abcdef", (40, 69.0, 129, 109.0), 0.955), ("abcdefg", (139, 69.0, 243, 109.0), 0.5),
         ("abcdefgh", (261, 69.0, 380, 109.0), 0.95)], DI, 200),
}


def test_a_stroke_row_retry_with_two_sided_ink_gives_the_rescued_word_geometry(monkeypatch):
    old, = _fuzz_records(FUZZ["snap gives geometry"], False, monkeypatch)
    new, = _fuzz_records(FUZZ["snap gives geometry"], True, monkeypatch)
    assert not old["geom_corroborated"] and new["geom_corroborated"] and new["final"]
    assert new["off"] == pytest.approx(-0.14, abs=0.005) and old["off"] == pytest.approx(-0.12)


@pytest.mark.parametrize("name", [k for k in FUZZ if k != "snap gives geometry"])
def test_what_the_change_leaves_alone(monkeypatch, name):
    old = _fuzz_records(FUZZ[name], False, monkeypatch)
    assert old                                                 # a record to compare
    assert _fuzz_records(FUZZ[name], True, monkeypatch) == old


@pytest.mark.parametrize("conf", [0.9, 0.96])
def test_a_retry_that_leaves_the_band_keeps_the_rescue(monkeypatch, conf):
    """The stroke's row is out of the band: the retry there finds no ink on both sides, and the
    word keeps the record the rescue gave it at the line (an earlier version of this change took
    the retry's state and lost the word)."""
    page = _stroke_row_out_of_band()
    old = _records(page, conf, retries=False, monkeypatch=monkeypatch)
    new = _records(page, conf, monkeypatch=monkeypatch)
    assert len(new) == 1 and new == old
    assert new[0]["off"] == pytest.approx(0.12, abs=0.005)   # at the line, not the stroke's row
    assert not new[0]["geom_corroborated"]


def _top_row_line(n, line_x, stroke_x, top, stroke_y, line_y):
    """`n` letters on rows top-119, a 2 px stroke on rows stroke_y and stroke_y + 1 over stroke_x,
    a 2 px finder line at line_y over line_x: at the letters' top row there is no glyph ink above
    it, so the rescue passes the word; the stroke's row has glyph ink on both sides."""
    g = np.full((H, W), 255, np.uint8)
    _letters(g, n, top, 120)
    g[stroke_y:stroke_y + 2, stroke_x[0]:stroke_x[1]] = 20
    return g, _line(line_x[0], int(line_y) - 1, line_x[1], int(line_y) + 1, line_y, 2.0, 0.9, 1.0)


@pytest.mark.parametrize("box", [(72, 128), (75.76, 124.24)])
@pytest.mark.parametrize("conf", [0.5, 0.9, 0.955])
def test_a_rescued_partial_word_takes_a_stroke_row_retry_near_the_middle(monkeypatch, conf, box):
    """A line over letters 4-7 of twelve at their top row, the stroke 5 px below it. The box puts
    the retry 0.07 off the middle, or 0.0825, which a record stores as 0.08: within
    PARTIAL_ISO_MAX_OFF either way, so the partial word takes it and gets geometry, as the release
    gives it at a confidence the rescue does not reach."""
    xs = (X0 + 3 * ADV, X0 + 7 * ADV - 4)
    g, ln = _top_row_line(12, xs, xs, 99, 105, 100.0)
    page = (g, ln, (X0 / W, box[0] / H, _x1(12) / W, box[1] / H), "abcdefghijkl")
    old, = _records(page, conf, retries=False, monkeypatch=monkeypatch)
    new, = _records(page, conf, monkeypatch=monkeypatch)
    above, = _records(page, 0.975, retries=False, monkeypatch=monkeypatch)
    assert not old["geom_corroborated"] and old["off"] == pytest.approx(0.0, abs=0.005)
    assert new["geom_corroborated"]
    assert new["off"] == pytest.approx(4 / (box[1] - box[0]), abs=0.005)
    for key in ("off", "char_span", "geom_corroborated"):
        assert new[key] == above[key], key


@pytest.mark.parametrize("conf", [0.5, 0.9, 0.955])
def test_a_partial_word_the_filter_keeps_after_rounding_keeps_its_record(monkeypatch, conf):
    """The line 0.0825 off the middle of rows 65.05-125.05, which a record stores as 0.08, within
    PARTIAL_ISO_MAX_OFF; the stroke-row retry would move the word to 0.15, where an isolated
    partial strike is dropped. Compared as stored, the retry is not taken."""
    xs = (X0 + 3 * ADV, X0 + 7 * ADV - 4)
    g, ln = _top_row_line(12, xs, xs, 99, 105, 100.0)
    page = (g, ln, (X0 / W, 65.05 / H, _x1(12) / W, 125.05 / H), "abcdefghijkl")
    old = _records(page, conf, retries=False, monkeypatch=monkeypatch)
    assert len(old) == 1 and old[0]["off"] == pytest.approx(0.08) and old[0]["final"]
    assert _records(page, conf, monkeypatch=monkeypatch) == old


@pytest.mark.parametrize("conf", [0.5, 0.9])
def test_a_stroke_row_retry_out_of_the_band_is_not_taken(monkeypatch, conf):
    """A whole word, the line at row 103 (0.075 off the middle of rows 80-120), the stroke 5 px
    below it: the retry finds glyph ink on both sides there, but 0.175 off the middle, out of the
    band, so it gives no strike geometry and the rescue stands at the line."""
    xs = (X0 - 2, _x1(4) + 2)
    g, ln = _top_row_line(4, xs, xs, 101, 108, 103.0)
    page = (g, ln, (X0 / W, 80 / H, _x1(4) / W, 120 / H), "abcd")
    old = _records(page, conf, retries=False, monkeypatch=monkeypatch)
    new = _records(page, conf, monkeypatch=monkeypatch)
    assert len(new) == 1 and new == old and not new[0]["geom_corroborated"]


def test_a_rescue_first_passing_at_the_stroke_row_still_gets_the_retries(monkeypatch):
    """The line just above the band: the ink tests drop the word there, the stroke-row retry moves
    it onto the upper stroke, in the band, where the rescue alone passes it. The pair test after
    that gives it geometry, as it does at 0.975, where the rescue does not apply."""
    gray, line, box, txt = _double_strike()
    line = {**line, "bbox_px": (line["bbox_px"][0], 137, line["bbox_px"][2], 139),
            "ends_px": ((line["ends_px"][0][0], 138.5), (line["ends_px"][1][0], 138.5))}
    page = (gray, line, box, txt)
    above, = _records(page, 0.975, monkeypatch=monkeypatch)
    assert above["geom_corroborated"] and above["final"]
    for conf in (0.96, 0.974):
        rec, = _records(page, conf, monkeypatch=monkeypatch)
        for key in ("off", "char_span", "geom_corroborated", "tier", "final"):
            assert rec[key] == above[key], (conf, key)


def _partial_beside_strong():
    """Word A ("ab") wholly under a 0.35 in line, word B the first three of its twelve letters; the
    stroke 5 px under the finder line."""
    g = np.full((300, 900), 255, np.uint8)
    for x0, n in ((200, 2), (232, 12)):
        for i in range(n):
            for s in (0, 4, 8):
                g[99:120, x0 + i * 12 + s:x0 + i * 12 + s + 2] = 20
    g[105:107, 198:268] = 20
    line = {"bbox_px": (198, 99, 268, 101), "ends_px": ((198.0, 100.0), (268.0, 100.0)),
            "len_in": 0.35, "angle_deg": 0.0, "fill": 0.9, "run_px": 2.0, "straightness": 1.0}
    a = (200 / 900, 65.05 / 300, 222 / 900, 125.05 / 300)
    b = (232 / 900, 65.05 / 300, 374 / 900, 125.05 / 300)
    return g, line, a, b


@pytest.mark.parametrize("conf", [0.96, 0.974])
def test_a_rescued_partial_word_beside_a_strong_hit_takes_the_stroke_row(monkeypatch, conf):
    """Beside a strong hit an isolated partial strike is never dropped, so B's stroke-row retry is
    taken (it gives geometry), as for B read at 0.975."""
    gray, line, a, b = _partial_beside_strong()

    def recs(c):
        _, out = classify_lines([line], [Word("ab", a, 0.5), Word("abcdefghijkl", b, c)], gray,
                                config=DI, dpi=200)
        return {r["text"]: r for r in detect.apply_cnn_verdict(out, gray, config=DI)}
    above, rescued = recs(0.975)["abcdefghijkl"], recs(conf)["abcdefghijkl"]
    assert above["geom_corroborated"] and above["final"]
    for key in ("off", "char_span", "geom_corroborated", "final"):
        assert rescued[key] == above[key], key
