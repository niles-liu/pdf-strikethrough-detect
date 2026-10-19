"""The shared guards and files: no test page before prereg-2, and JSONL that survives a killed
write (an unfinished last line is skipped, then cut before the next record goes on)."""

import json

import pytest

from bench import common


def test_test_pages_stay_locked_until_the_prereg_2_tag(monkeypatch):
    pages = [{"split": "dev"}, {"split": "test"}]
    common.guard_test_pages(pages[:1])  # dev pages are never locked
    for tags in ([], ["prereg-1"]):
        monkeypatch.setattr(common, "tags_in_force", lambda tags=tags: tags)
        with pytest.raises(PermissionError, match="prereg-2"):
            common.guard_test_pages(pages)
    monkeypatch.setattr(common, "tags_in_force", lambda: ["prereg-1", "prereg-2"])
    common.guard_test_pages(pages)


def test_after_prereg_2_a_change_to_the_rules_is_listed_the_drops_and_prices_apart(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(common, "tags_in_force", lambda: ["prereg-1"])
    assert common.changed_since_prereg_2() is None  # before the tag: nothing to compare with
    monkeypatch.setattr(common, "tags_in_force", lambda: ["prereg-1", "prereg-2"])
    diff = "bench/report.py\nconfig/prices.toml\nconfig/budget.toml\n"
    monkeypatch.setattr(common, "git", lambda *args: diff)
    frozen = {"cap_b3_usd": 300, "reserve_usd": 50, "drop_order": ["fable"], "dropped": []}
    monkeypatch.setattr(common, "PREREG2", tmp_path / "prereg2.json")
    common.PREREG2.write_text(json.dumps({"budget": frozen}), encoding="utf-8")
    budget = dict(frozen, dropped=["fable"])  # a drop the cap called for: no change
    monkeypatch.setattr(common, "config", {"budget": budget}.__getitem__)
    assert common.changed_since_prereg_2() == ["bench/report.py"]
    budget["cap_b3_usd"] = 400  # a raised cap is a change to a rule the tag fixed
    assert common.changed_since_prereg_2() == ["bench/report.py", common.BUDGET_CHANGED]


def test_an_unfinished_last_line_is_skipped_then_cut(tmp_path, capsys):
    path = tmp_path / "x.jsonl"
    path.write_bytes(b'{"n": 1}\n{"n": 2, "te')  # a write killed mid-line
    assert common.read_jsonl(path) == [{"n": 1}]
    assert "unfinished last line" in capsys.readouterr().err
    common.append_jsonl(path, {"n": 3})
    assert path.read_bytes().decode().splitlines() == [json.dumps({"n": 1}), json.dumps({"n": 3})]


def test_a_damaged_line_before_the_last_is_an_error(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_bytes(b'{"n": 1}\n{"n": \n{"n": 3}\n')
    with pytest.raises(json.JSONDecodeError):
        common.read_jsonl(path)
