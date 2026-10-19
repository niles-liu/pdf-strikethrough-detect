"""The audit: G2's span check, the post-test relabel, the item pages, and the labels coming back
(κ, missing and blank answers, disagreements, corrections)."""

import csv
import json

import pytest

from bench import audit, g1
from bench.score import Token


def w(text, y, x=0.1, struck=False, masked=False):
    return {"text": text, "bbox": [x, y - 0.005, x + 0.05, y + 0.005], "line": [0, round(y * 100)],
            "struck": [[0, len(text)]] if struck else [], "masked": masked, "furniture": False}  # fmt: skip


def test_g2_spans_follow_printed_lines_and_put_boundaries_first(monkeypatch):
    words = [
        w("a", 0.1), w("b", 0.1, 0.2, True), w("c", 0.1, 0.3, True),  # a boundary at its start
        w("d", 0.2, struck=True), w("‘‘;", 0.2, 0.2), w("e", 0.2, 0.3, True),  # ‘‘; passed over
        w("f", 0.3, struck=True),
        w("g", 0.4, struck=True), w("h", 0.4, 0.2, masked=True),
    ]  # fmt: skip
    words[5]["line"] = [9, 9]  # PyMuPDF split the printed line: still one span
    asked = []
    monkeypatch.setattr(audit, "load_pages",
                        lambda *s, **k: asked.append(k) or [{"page_id": "P", "words": words}])  # fmt: skip
    spans = audit.g2_items(n=4)
    assert asked == [{"split": "test"}]  # G2-dev's bills are never checked as G2's
    assert spans[0]["words"] == [1, 2] and spans[0]["neighbours"] == [0]
    assert sorted(s["words"] for s in spans[1:]) == [[3, 5], [6], [7]]
    assert all(not s["neighbours"] for s in spans[1:])
    with pytest.raises(ValueError, match="fewer than"):
        audit.g2_items(n=5)


def test_the_notation_sweep_lists_leaks_at_r0_and_r1_and_passes_over_failed_answers(
    monkeypatch, tmp_path
):
    page = {"page_id": "P", "words": [w("old", 0.1, struck=True), w("new", 0.1, 0.2)]}
    monkeypatch.setattr(audit, "load_pages", lambda *s, **k: [page])
    monkeypatch.setattr(audit, "RUNS", tmp_path)
    (tmp_path / "ladder").mkdir()
    leak = [Token("[struck:", [False] * 8), Token("old]", [False] * 4), Token("new", [False] * 3)]
    outs = [
        {"arm": "m:R0:C2", "rung": "R0", "page_id": "P", "tokens": leak},
        {"arm": "m:R1:C2", "rung": "R1", "page_id": "P", "tokens": []},  # nothing usable
        {"arm": "m:R2:C2", "rung": "R2", "page_id": "P", "tokens": leak},
    ]  # a marking rung
    monkeypatch.setattr(audit.harness, "outputs", lambda stage: outs)
    report = audit.notations(["ladder"])
    with (tmp_path / "ladder" / "notations.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [(r["arm"], r["word"]) for r in rows] == [("m:R0:C2", "0")]  # the failure is not a leak
    assert "'[struck:' 1" in report


def test_a_relabel_needs_three_flagships_and_ignores_failed_pages(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "RUNS", tmp_path)  # no test audit labelled yet
    page = {"page_id": "G1.d.p0", "words": [w("abcdef", 0.1), w("ok", 0.1, 0.3)]}
    page["words"][0]["struck"] = [[2, 4]]
    models = {f"m{k}": {"provider": f"p{k}", "role": "flagship"} for k in range(4)}
    monkeypatch.setattr(audit, "config", {"models": models}.__getitem__)
    # each flagship's own rung (the flagship check's), not its provider's R-best
    monkeypatch.setattr(
        audit.plans, "choice", lambda prov, key: "R3" if key == "flagship" else "C2"
    )
    monkeypatch.setattr(audit, "load_pages", lambda *s, **k: [page])

    def out(k: int, tokens: list[Token]) -> dict:
        return {"arm": f"m{k}:R3:C2", "page_id": page["page_id"], "tokens": tokens}

    whole = [Token("abcdef", [True] * 6), Token("ok", [False] * 2)]  # two sub-tokens over-deleted
    outputs = [out(0, whole), out(1, whole)]
    monkeypatch.setattr(audit.harness, "outputs", lambda stage: outputs)
    assert audit.relabel_items() == []  # two arms, though four disputed sub-tokens
    outputs.append(out(2, whole))
    assert audit.relabel_items() == [{"page_id": "G1.d.p0", "words": [0], "kind": "g1-relabel"}]
    outputs[:] = [out(k, []) for k in range(4)]
    assert audit.relabel_items() == []  # empty answers fail; they dispute nothing


def test_the_test_audits_key_errors_on_g1_join_the_relabel(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "RUNS", tmp_path)
    page = {"page_id": "G1.d.p0", "words": [w("abcdef", 0.1), w("ok", 0.1, 0.3)]}
    monkeypatch.setattr(audit, "config", {"models": {}}.__getitem__)
    monkeypatch.setattr(audit, "load_pages", lambda *s, **k: [page])
    monkeypatch.setattr(audit.harness, "outputs", lambda stage: [])
    found = tmp_path / "test" / "audit"
    found.mkdir(parents=True)
    keys = [{"item_id": "i001", "page_id": "G1.d.p0", "words": [1]},
            {"item_id": "i002", "page_id": "G2.b.p0", "words": [3]}]  # fmt: skip
    (found / "key.jsonl").write_text("".join(json.dumps(k) + "\n" for k in keys), encoding="utf-8")
    (found / "ingest.json").write_text(json.dumps({"ground_truth_no": ["i001", "i002"]}))
    assert audit.relabel_items() == [{"page_id": "G1.d.p0", "words": [1], "kind": "g1-relabel"}]


@pytest.mark.parametrize("text, chars, spans", [
    ("CO2CO2", "CO2~CO2~", [[3, 6]]),  # the second of two equal parts
    ("Eaffects", "~E~affect~s~", [[0, 1], [7, 8]]),  # struck in two places
    ("Eaffects", "E s", [[0, 1], [7, 8]]),  # the same as plain parts
    ("Llist", "L", [[0, 1]]),
    ("abc", "", [[0, 3]]),  # nothing typed: the whole word
    ("abc", "x", None),
    ("abc", "a~b~", None),  # the tildes' word must be the word
])  # fmt: skip
def test_struck_characters_come_as_the_word_with_tildes_or_as_its_parts(text, chars, spans):
    assert g1.struck_spans(text, chars) == spans


def test_a_page_printed_sideways_turns_upright():
    def tall(text, y, x=0.5):
        return {"text": text, "bbox": [x, y, x + 0.02, y + 0.1], "line": [0, 0], "furniture": False}

    down = {"words": [tall("first", 0.1), tall("second", 0.3), tall("third", 0.5)]}
    assert audit._turn(down) == 90  # its lines run down the page
    up = {"words": [tall("first", 0.5), tall("second", 0.3), tall("third", 0.1)]}
    assert audit._turn(up) == 270
    assert audit._turn({"words": [w("upright", 0.1), w("words", 0.1, 0.3)]}) == 0


def test_g5_is_outside_the_audit(monkeypatch):
    page = {"page_id": "G0.d.p0", "words": [w("gone", 0.1, struck=True), w("kept", 0.1, 0.3)]}
    monkeypatch.setattr(audit, "load_pages", lambda *s, **k: [page])  # G0–G2 only
    leak = [Token("gone", [False] * 4), Token("kept", [False] * 4)]
    outputs = [{"arm": "P-scan", "page_id": pid, "tokens": leak}
               for pid in (page["page_id"], "G5.0001-001.p0")]  # fmt: skip
    monkeypatch.setattr(audit.harness, "outputs", lambda stage: outputs)
    pool = audit.disagreements("test")
    assert [x["page_id"] for x in pool[("P-scan", "leak")]] == ["G0.d.p0"]


def test_question_three_names_the_part_of_the_word_it_is_about(monkeypatch):
    monkeypatch.setattr(audit, "_crop", lambda pid, boxes, colors, **k: "x")
    page = {"words": [w("semi-monthly", 0.1)]}
    item = {"item_id": "i001", "page_id": "P", "words": [0], "span": [0, 5], "kind": "leak",
            "outcome": "live", "output": "semi-monthly"}  # fmt: skip
    live = f"<b>{audit.OUTCOME['live']}</b>"
    part = f"the output has “semi-” (part of the boxed word) {live}"
    assert part in audit._item_html(item, {"P": page})
    item["span"] = [0, 12]  # the whole word
    assert f"the output has the boxed word {live}" in audit._item_html(item, {"P": page})


def test_a_span_is_boxed_blue_and_its_neighbours_orange(monkeypatch):
    seen = []
    monkeypatch.setattr(audit, "_crop", lambda pid, boxes, colors, **k: seen.append(colors) or "x")
    page = {"words": [w("a", 0.1), w("b", 0.1, 0.2, True), w("c", 0.1, 0.3)]}
    item = {"item_id": "i001", "page_id": "P", "words": [1], "neighbours": [0, 2],
            "kind": "g2-span"}  # fmt: skip
    audit._item_html(item, {"P": page}, q1_only=True)
    assert seen == [[audit.BLUE, audit.ORANGE, audit.ORANGE]]


FIELDS = ("annotator", "item_id", "q1", "chars", "q2", "q3")


def write_csv(path, who, labels, fields=FIELDS):
    """labels: item -> (q1, chars, q2, q3), as far as `fields` reach."""
    with path.open("w", encoding="utf-8", newline="") as f:
        out = csv.writer(f)
        out.writerow(fields)
        for item, row in labels.items():
            out.writerow([who, item, *row][: len(fields)])


@pytest.fixture
def folder(tmp_path, monkeypatch):
    key = [
        {"item_id": "i001", "kind": "leak", "page_id": "P", "words": [0], "annotators": "AB"},
        {"item_id": "i002", "kind": "hallucination", "page_id": "P", "words": [],
         "annotators": "AB"},
        {"item_id": "i003", "kind": "g1-struck", "page_id": "P", "words": [1], "annotators": "A"},
        {"item_id": "i004", "kind": "g1-relabel", "page_id": "P", "words": [2], "annotators": "AB"},
        {"item_id": "i005", "kind": "over-deletion", "page_id": "P", "words": [3],
         "annotators": "AB"},
        {"item_id": "i006", "kind": "g1-live", "page_id": "P", "words": [4], "annotators": "B"},
        {"item_id": "i007", "kind": "g1-relabel", "page_id": "P", "words": [5], "annotators": "AB"},
        {"item_id": "i008", "kind": "leak", "page_id": "P", "words": [6], "annotators": "AB"},
    ]  # fmt: skip
    (tmp_path / "key.jsonl").write_text("".join(json.dumps(k) + "\n" for k in key), "utf-8")
    words = [w(t, 0.1) for t in ("a", "b", "semi-", "d", "e", "abcdef", "g")]
    monkeypatch.setattr(audit, "load_pages", lambda *s, **k: [{"page_id": "P", "words": words}])
    monkeypatch.setattr(g1, "CORRECTIONS", tmp_path / "G1-corrections.jsonl")
    a = {"i001": ("struck", "", "yes", "yes"), "i002": ("", "", "", "no"), "i003": ("", "", "", ""),
         "i004": ("struck", "semi-", "", ""), "i005": ("struck", "", "", ""),
         "i007": ("struck", "cd", "", ""), "i008": ("struck", "", "", "")}  # fmt: skip
    b = {"i001": ("struck", "", "no", "yes"), "i002": ("", "", "", "yes"),
         "i004": ("struck", "", "", ""), "i005": ("not struck", "", "", ""),
         "i006": ("cannot tell", "", "", ""), "i007": ("struck", "cd", "", "")}  # fmt: skip
    write_csv(tmp_path / "A.csv", "A", a)
    write_csv(tmp_path / "B.csv", "B", b)
    return tmp_path


def test_ingest_counts_kappa_on_question_one_and_lists_what_needs_a_person(folder):
    out = audit.ingest(folder, [folder / "A.csv", folder / "B.csv"])
    assert out["shared"] == 4  # i001, i004, i005, i007: not the hallucination item left blank
    assert out["missing"] == ["i008"]  # assigned to both, returned by A only
    assert out["q1_blank"] == ["i003"]
    assert out["q1_disagree"] == ["i004", "i005"]  # i004: the struck characters differ
    assert out["scorer_no"] == ["i002"] and out["ground_truth_no"] == ["i001"]
    assert out["g1-live_cannot_tell"] == 1 and "g1-live_error_rate" not in out
    assert out["corrections"].startswith("not written") and not g1.CORRECTIONS.exists()


def test_corrections_are_written_once_every_relabel_item_is_decided(folder):
    files = [folder / "A.csv", folder / "B.csv"]
    write_csv(folder / "adj.csv", "", {"i004": ("struck",)}, fields=FIELDS[:3])
    with pytest.raises(ValueError, match="chars"):  # an adjudication must give the characters
        audit.ingest(folder, files, folder / "adj.csv")
    write_csv(folder / "adj.csv", "", {"i004": ("struck", "semi")}, fields=FIELDS[:4])
    out = audit.ingest(folder, files, folder / "adj.csv")
    assert "i004" not in out["q1_disagree"]
    fixes = [json.loads(x) for x in g1.CORRECTIONS.read_text(encoding="utf-8").splitlines()]
    assert fixes == [
        {"page_id": "P", "word": 2, "text": "semi-", "label": "struck", "chars": "semi"},
        {"page_id": "P", "word": 5, "text": "abcdef", "label": "struck", "chars": "cd"},
    ]  # fmt: skip


def test_struck_characters_not_in_the_word_write_no_corrections(folder):
    write_csv(folder / "adj.csv", "", {"i004": ("struck", "Semi")}, fields=FIELDS[:4])
    out = audit.ingest(folder, [folder / "A.csv", folder / "B.csv"], folder / "adj.csv")
    assert out["corrections"].startswith("not written: struck characters")
    assert not g1.CORRECTIONS.exists()


def test_an_adjudication_decides_only_what_the_annotators_left_undecided(folder):
    files = [folder / "A.csv", folder / "B.csv"]
    write_csv(folder / "adj.csv", "", {"i007": ("not struck", "")}, fields=FIELDS[:4])
    with pytest.raises(ValueError, match=r"already decided: \['i007'\]"):  # both said "cd"
        audit.ingest(folder, files, folder / "adj.csv")
    assert not g1.CORRECTIONS.exists()
    write_csv(folder / "adj.csv", "", {"i002": ("struck", "")}, fields=FIELDS[:4])
    with pytest.raises(ValueError, match="no question 1"):  # a hallucination asks only q3
        audit.ingest(folder, files, folder / "adj.csv")
    write_csv(folder / "adj.csv", "", {"i003": ("struck", ""), "i005": ("struck", "")},
              fields=FIELDS[:4])  # fmt: skip  # blank, and a disagreement: the author's to decide
    out = audit.ingest(folder, files, folder / "adj.csv")
    assert "i005" not in out["q1_disagree"] and "i003" not in out["unresolved"]


def test_the_pilot_audit_is_one_page_for_the_author_with_no_practice(monkeypatch, tmp_path):
    words = [w(t, 0.1 * (i + 1)) for i, t in enumerate("abcdefgh")]
    monkeypatch.setattr(audit, "load_pages", lambda *s, **k: [{"page_id": "P", "words": words}])
    monkeypatch.setattr(audit.render, "require_current", lambda pages: None)
    monkeypatch.setattr(audit, "_crop", lambda *a, **k: "x")
    monkeypatch.setattr(audit, "_practice", lambda: pytest.fail("no practice for the author"))
    items = [{"page_id": "P", "words": [i], "kind": "leak", "outcome": "live", "span": [0, 1]}
             for i in range(8)]  # fmt: skip
    audit.write_pages(tmp_path, "Audit: pilot", items, who="A")
    assert sorted(f.name for f in tmp_path.iterdir()) == ["A.html", "key.jsonl"]
    key = [json.loads(x) for x in (tmp_path / "key.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(key) == 8 and {k["annotators"] for k in key} == {"A"}


def test_the_g1_sample_waits_for_prereg_2_to_fix_it(monkeypatch, tmp_path):
    sample = tmp_path / "G1-error-sample.jsonl"
    sample.write_text(json.dumps({"page_id": "P", "word": 0, "kind": "struck"}) + "\n", "utf-8")
    monkeypatch.setattr(audit, "GT", tmp_path)
    monkeypatch.setattr(audit, "PREREG2", tmp_path / "prereg2.json")
    with pytest.raises(ValueError, match="not fixed it yet"):
        audit.g1_items()
    digest = audit.sha256_file(sample)
    (tmp_path / "prereg2.json").write_text(json.dumps({"g1_error_sample": digest}), "utf-8")
    assert audit.g1_items() == [{"page_id": "P", "words": [0], "kind": "g1-struck"}]
