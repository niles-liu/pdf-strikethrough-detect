"""issue #7 / v0.10.0 — the printed-rule veto for degraded ruled-form scans.

A faint table/form rule crossing text on a degraded scan mimics a pen strike (in-band, two-sided
ink, shattered fill) and the CNN over-fires on it. ``ScanConfig.ruled_forms()`` drops a detected
line that is SOLID (fill high) and/or DEAD-STRAIGHT (low perpendicular wobble) as a drawn rule.

It is OFF by default because on a CLEAN scan a real strike is also solid and straight — these tests
pin both that default-off keeps such a strike and that the opt-in drops the rule while sparing a
genuinely degraded (shattered + wobbly) strike.
"""
import numpy as np

from pdf_strikethrough.lines import (STRAIGHTNESS_MIN_SAMPLES, _spine_straightness, strike_lines)
from pdf_strikethrough.ocr import Word
from pdf_strikethrough.scanned import (PRINTED_RULE_FILL_MAX, PRINTED_RULE_STRAIGHT_MAX,
                                        ScanConfig, _is_printed_rule, classify_lines)


def _line(fill, straightness, x0=100.0, x1=700.0, y=150.0):
    return {"bbox_px": (int(x0), int(y) - 1, int(x1), int(y) + 1),
            "ends_px": ((x0, y), (x1, y)), "len_in": 3.0, "angle_deg": 0.5,
            "fill": fill, "run_px": 3.0, "straightness": straightness}


def _words(n=5, y=150.0, pix_w=800, pix_h=400):
    out = []
    for i in range(n):
        cx = 160 + i * 110
        out.append(Word(f"w{i}", ((cx - 45) / pix_w, (y - 14) / pix_h,
                                   (cx + 45) / pix_w, (y + 14) / pix_h), 0.6))
    return out


def test_ruled_forms_config_flag():
    assert ScanConfig().veto_printed_rules is False
    assert ScanConfig.ruled_forms().veto_printed_rules is True
    assert ScanConfig.azure_di().veto_printed_rules is False


def test_is_printed_rule_predicate():
    assert _is_printed_rule(_line(fill=0.95, straightness=3.0))          # solid -> rule
    assert _is_printed_rule(_line(fill=0.70, straightness=0.5))          # dead-straight -> rule
    assert not _is_printed_rule(_line(fill=0.70, straightness=3.0))      # shattered + wobbly -> strike
    # thresholds are the documented operating point
    assert PRINTED_RULE_FILL_MAX == 0.88 and PRINTED_RULE_STRAIGHT_MAX == 1.80


def test_veto_off_by_default_keeps_solid_straight_strike():
    """A clean strike is solid and straight; the DEFAULT config must still flag it (no regression)."""
    gray = np.full((400, 800), 255, np.uint8)
    line, words = _line(fill=0.95, straightness=0.4), _words()
    _tagged, struck = classify_lines([line], words, gray)                # default: veto off
    assert {h["text"] for h in struck} == {f"w{i}" for i in range(5)}


def test_ruled_forms_drops_the_rule():
    """The SAME solid/straight line is dropped under ruled_forms() — it reads as a drawn rule."""
    gray = np.full((400, 800), 255, np.uint8)
    line, words = _line(fill=0.95, straightness=0.4), _words()
    _tagged, struck = classify_lines([line], words, gray, config=ScanConfig.ruled_forms())
    assert struck == []


def test_ruled_forms_spares_degraded_strike():
    """A genuinely degraded strike (shattered fill + wobble) survives the veto even when on."""
    gray = np.full((400, 800), 255, np.uint8)
    line, words = _line(fill=0.72, straightness=2.6), _words()
    _tagged, struck = classify_lines([line], words, gray, config=ScanConfig.ruled_forms())
    assert {h["text"] for h in struck} == {f"w{i}" for i in range(5)}


def test_strike_lines_reports_straightness():
    """strike_lines attaches a numeric, resolution-normalized straightness to every line."""
    gray = np.full((120, 600), 255, np.uint8)
    gray[60, 40:560] = 0                                  # one dead-straight rule
    lines = strike_lines(gray, dpi=200)
    assert lines and all(isinstance(ln["straightness"], float) for ln in lines)
    assert min(ln["straightness"] for ln in lines) < PRINTED_RULE_STRAIGHT_MAX


# --- fail-safe behaviour (0.10.0). The veto DROPS detections, so every "can't tell" path must
# resolve to "not a rule". These pin the directions, because getting one backwards loses real
# strikes silently -- there is no error, just missing output.

def test_missing_metrics_are_never_a_rule():
    """A line dict without `fill` / `straightness` must NOT be vetoed. Before 0.10.0 an absent
    `fill` defaulted to 1.0 (> the max), so a caller-built line dict lost EVERY detection."""
    assert not _is_printed_rule({})
    assert not _is_printed_rule({"len_in": 3.0})                       # no fill, no straightness
    assert not _is_printed_rule({"straightness": 3.0})                 # wobbly, fill unknown
    assert not _is_printed_rule({"fill": 0.70})                        # shattered, wobble unknown


def test_none_metrics_are_never_a_rule():
    """None means "not measurable", not "straight" — strike_lines emits None for an unmeasurable
    wobble, and that must not read as a drawn rule."""
    assert not _is_printed_rule({"fill": 0.70, "straightness": None})
    assert not _is_printed_rule({"fill": None, "straightness": 3.0})
    # ...but a genuinely solid line is still a rule even with the wobble unknown
    assert _is_printed_rule({"fill": 0.95, "straightness": None})


def test_veto_keeps_struck_words_when_line_dict_lacks_fill():
    """End-to-end form of the above: the opt-in must not silently empty the output for a caller
    that passes hand-built line dicts."""
    gray = np.full((400, 800), 255, np.uint8)
    line = {"bbox_px": (100, 149, 700, 151), "ends_px": ((100.0, 150.0), (700.0, 150.0)),
            "len_in": 3.0, "angle_deg": 0.5, "run_px": 3.0}          # no fill, no straightness
    _tagged, struck = classify_lines([line], _words(), gray, config=ScanConfig.ruled_forms())
    assert {h["text"] for h in struck} == {f"w{i}" for i in range(5)}


def test_ruled_forms_accepts_an_explicit_flag():
    """ruled_forms(veto_printed_rules=...) must not raise (it used to collide with the **kw
    passthrough as 'multiple values for keyword argument')."""
    assert ScanConfig.ruled_forms(veto_printed_rules=False).veto_printed_rules is False
    assert ScanConfig.ruled_forms(cnn_p_hi=0.97).cnn_p_hi == 0.97
    assert ScanConfig.ruled_forms(cnn_p_hi=0.97).veto_printed_rules is True


def test_vetoed_lines_keep_tagged_index_aligned_with_lines():
    """score_struck() resolves a hit's line as ``tagged[line_idx]``, so `tagged` must stay
    index-aligned with the input `lines` — a vetoed line still has to occupy its slot. If the veto
    ever skipped the append, every later word on the page would silently score off the wrong line.
    """
    gray = np.full((400, 800), 255, np.uint8)
    rule = _line(fill=0.95, straightness=0.4, y=80.0)                    # vetoed
    strike = _line(fill=0.72, straightness=2.6, y=150.0)                 # kept
    lines = [rule, strike]

    tagged, struck = classify_lines(lines, _words(), gray, config=ScanConfig.ruled_forms())
    assert len(tagged) == len(lines)
    assert tagged[0]["label"] == "rule" and tagged[0]["struck"] == []
    assert [h["text"] for h in tagged[1]["struck"]]
    # the surviving word's line_idx must point at the strike (index 1), not the vetoed rule
    assert all(rec["line_idx"] == [1] for rec in struck)


def test_straightness_unmeasurable_is_inf_not_zero():
    """The helper's fail-safe sentinel. inf (-> None in the public field) keeps the line; 0.0 would
    read as dead-straight and drop it."""
    ink = np.zeros((60, 200), bool)
    center, u = np.array([100.0, 30.0]), np.array([1.0, 0.0])
    assert _spine_straightness(ink, center, u, 100.0, float("inf")) == float("inf")  # bad run_px
    assert _spine_straightness(ink, center, u, 100.0, 3.0) == float("inf")           # no ink at all
    ink[30, 100:100 + STRAIGHTNESS_MIN_SAMPLES - 1] = True                           # too few
    assert _spine_straightness(ink, center, u, 100.0, 3.0) == float("inf")
    ink[30, 100:140] = True                                                          # measurable
    assert np.isfinite(_spine_straightness(ink, center, u, 100.0, 3.0))
