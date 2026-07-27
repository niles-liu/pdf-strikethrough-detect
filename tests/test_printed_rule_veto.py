"""issue #7 / v0.9.2 — the printed-rule veto for degraded ruled-form scans.

A faint table/form rule crossing text on a degraded scan mimics a pen strike (in-band, two-sided
ink, shattered fill) and the CNN over-fires on it. ``ScanConfig.ruled_forms()`` drops a detected
line that is SOLID (fill high) and/or DEAD-STRAIGHT (low perpendicular wobble) as a drawn rule.

It is OFF by default because on a CLEAN scan a real strike is also solid and straight — these tests
pin both that default-off keeps such a strike and that the opt-in drops the rule while sparing a
genuinely degraded (shattered + wobbly) strike.
"""
import numpy as np

from pdf_strikethrough.lines import strike_lines
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
