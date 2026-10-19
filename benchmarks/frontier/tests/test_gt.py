"""G1's furniture rule and page kinds, and G2's alignment: struck spans, a word hyphenated across
a page break, and a struck run the substitute repeats."""

import datetime
import json

import pymupdf
import pytest

from bench import g1, g2


def word(text, x, y, line, struck=None):
    return {"text": text, "bbox": [x, y - 0.005, x + 0.01 * len(text), y + 0.005],
            "struck": struck or [], "masked": False, "furniture": False, "line": line}  # fmt: skip


def doc_pages(n=5):
    """n pages, each with a running head (its page number in it), a first body line of its own,
    a centred page number, and a lone figure in a table column that moves from page to page."""
    pages, first = {}, ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
    for p in range(n):
        pages[p] = [
            word("Report", 0.1, 0.03, [0, 0]), word(f"{p + 1}", 0.2, 0.03, [0, 0]),
            word(first[p], 0.1, 0.05, [1, 0], struck=[[0, 4]]),
            word("12.5", 0.2 + 0.1 * p, 0.9, [2, 0]),
            word(f"{p + 1}", 0.5, 0.97, [3, 0]),
        ]  # fmt: skip
    return pages


def test_repeated_edge_lines_are_furniture_and_unique_ones_are_not():
    pages = doc_pages()
    g1.mark_furniture(pages)
    flags = [(w["text"], w["furniture"]) for w in pages[3]]
    assert flags == [("Report", True), ("4", True), ("Delta", False), ("12.5", False), ("4", True)]


def test_a_line_without_letters_must_also_recur_in_place():
    pages = doc_pages()
    for words in pages.values():
        words[3]["bbox"][0] = 0.2  # the figure now sits in the same place on every page
        words[3]["bbox"][2] = 0.24
    g1.mark_furniture(pages)
    assert all(words[3]["furniture"] for words in pages.values())


def test_a_page_struck_only_in_its_furniture_is_strike_free():
    words = [word("Draft", 0.1, 0.03, [0, 0], struck=[[0, 5]]), word("body", 0.1, 0.5, [1, 0])]
    words[0]["furniture"] = True
    assert g1._kind(words) == "strike-free"
    words[1]["struck"] = [[0, 4]]
    assert g1._kind(words) == "struck"


def test_a_strike_short_of_a_words_edge_punctuation_is_not_partial():
    def page(text, span):
        return {"words": [word(text, 0.1, 0.5, [0, 0], struck=[span])]}

    assert not g1._partial(page("Board.", [0, 5]))  # the scorer drops the period
    assert g1._partial(page("semi-monthly", [0, 5])) and g1._partial(page("Board", [0, 3]))


def test_g2_takes_one_partly_struck_page_per_bill_and_a_strike_free_one_on_request():
    def page(n, struck, live, coverage=1.0):
        words = [word("gone", 0.1, 0.5, [0, 0], struck=[[0, 4]]) for _ in range(struck)]
        words += [word("kept", 0.1, 0.6, [1, 0]) for _ in range(live)]
        return {"page": n, "words": words, "coverage": coverage}

    pages = [page(0, 0, 9), page(1, 10, 0), page(2, 10, 0), page(3, 5, 5), page(4, 0, 9),
             page(5, 10, 0), page(6, 9, 1), page(7, 5, 5, coverage=0.9)]  # fmt: skip
    # 3 is the first holding at least 10% live words; 7 is under 98% coverage
    assert [p["page"] for p in g2.pick(pages)] == [3]
    assert [p["page"] for p in g2.pick(pages, free=True)] == [0, 3]  # a strict or dev bill
    wholly = [page(0, 0, 9), page(1, 10, 0), page(2, 10, 0)]  # none 10% live: the first struck
    assert [p["page"] for p in g2.pick(wholly, free=True)] == [0, 1]
    assert g2.pick([page(0, 0, 9)], free=True) == []  # no struck page, no pages


def test_g2_test_takes_45_bills_the_inline_bill_last_then_three_for_dev(monkeypatch):
    # b00–b30 introduced after the cutoff (strict, newest first), then b59 down to b31
    bills = [{"package": f"b{i:02d}", "bill": f"b{i:02d}", "issued": "2026-08-01",
              "introduced": f"2026-07-{i + 1:02d}" if i < 31 else None, "struck_phrases": 0}
             for i in range(60)]  # fmt: skip
    bills[32]["struck_phrases"] = 9  # the most inline strikes, second to last in the order
    monkeypatch.setattr(g2, "read_jsonl", lambda path: bills)
    monkeypatch.setattr(g2, "cutoff", lambda: datetime.date(2026, 6, 30))
    tried = []

    def one(r, split):
        tried.append((r["package"], split))
        usable = r["package"] != "b28"  # one bill gives no usable page: skipped, not counted
        return ([{"cluster": r["bill"], "split": split}] if usable else []), []

    monkeypatch.setattr(g2, "_bill", one)
    pages, _ = g2.build()
    test = [p["cluster"] for p in pages if p["split"] == "test"]
    dev = [p["cluster"] for p in pages if p["split"] == "dev"]
    # 30 usable strict bills and b59–b46 make 44; the inline bill takes the 45th place
    assert len(test) == g2.N_BILLS and test[-1] == "b32" and "b28" not in test
    assert dev == ["b45", "b44", "b43"]  # the bills after G2-test's, in order
    assert [b for b, s in tried if s == "dev"] == dev


def test_a_wholly_struck_word_is_struck_to_its_edges():
    assert g2._spans("(a)", {1: True}) == [[0, 3]]
    assert g2._spans("annex.", dict.fromkeys(range(5), True)) == [[0, 6]]
    assert g2._spans("sections", {**dict.fromkeys(range(7), False), 7: True}) == [[7, 8]]


def bill(tmp_path, body_xml, pages):
    """A bill's XML and a PDF printing `pages`, each a list of lines in the body area."""
    xml = (f'<bill><legis-body display-enacting-clause="no-display-enacting-clause">'
           f"<section><text>{body_xml}</text></section></legis-body></bill>").encode()  # fmt: skip
    pdf = tmp_path / "bill.pdf"
    with pymupdf.open() as doc:
        for lines in pages:
            page = doc.new_page(width=612, height=792)
            for k, line in enumerate(lines):
                page.insert_text((200, 200 + 20 * k), line, fontsize=11)
        doc.save(pdf)
    return xml, pdf


def test_a_word_hyphenated_across_a_page_break_is_aligned_on_both_pages(tmp_path):
    pages = [["The princi-"], ["pal amount is due."]]
    xml, pdf = bill(tmp_path, "The principal amount is due.", pages)
    first, second = g2.align_bill(xml, pdf)
    assert [w["masked"] for w in first["words"] + second["words"]] == [False] * 6
    assert second["coverage"] == 1.0


def test_a_struck_run_the_substitute_repeats_keeps_its_strike(tmp_path):
    # The run outlasts the lookahead, so the first page could take either copy at the same cost.
    run = [f"term{k}" for k in range(60)]
    text = f"<deleted-phrase>{' '.join(run)}</deleted-phrase> {' '.join(run)} end"

    def lines(ws):
        return [" ".join(ws[k : k + 5]) for k in range(0, len(ws), 5)]

    xml, pdf = bill(tmp_path, text, [lines(run[:20]), lines(run[20:] + run + ["end"])])
    first, second = g2.align_bill(xml, pdf)
    assert first["coverage"] == 1.0 and all(w["struck"] for w in first["words"])
    assert [bool(w["struck"]) for w in second["words"]] == [True] * 40 + [False] * 61


def test_a_running_foot_whose_date_changes_is_furniture_too():
    pages = doc_pages()
    for p, words in pages.items():
        date = "September 2015" if p == 2 else "December 2024"
        words.append(word(f"Manual — {date}", 0.1, 0.95, [4, 0], struck=[[0, 4]]))
    g1.mark_furniture(pages)
    assert all(words[-1]["furniture"] for words in pages.values())
    assert not pages[2][2]["furniture"]  # the page's own first body line stays


def test_a_page_number_where_the_layout_moves_it_is_furniture():
    pages = doc_pages()
    pages[4][4]["bbox"] = [0.03, 0.495, 0.05, 0.505]  # a rotated page: its number at the left
    pages[3].append(word("7", 0.03, 0.5, [5, 0]))  # a lone figure that isn't the page's number
    g1.mark_furniture(pages)
    assert pages[4][4]["furniture"] and not pages[3][-1]["furniture"]


def test_corrections_relabel_words_and_refuse_a_different_word(tmp_path, monkeypatch):
    path = tmp_path / "G1-corrections.jsonl"
    monkeypatch.setattr(g1, "CORRECTIONS", path)
    words = [word("Oother", 0.1, 0.5, [0, 0]), word("x", 0.2, 0.5, [0, 0])]
    pages = [{"page_id": "P", "words": words}]
    fixes = [{"page_id": "P", "word": 0, "text": "Oother", "label": "struck", "chars": "O"},
             {"page_id": "P", "word": 1, "text": "x", "label": "cannot tell", "chars": ""}]  # fmt: skip
    path.write_text("".join(json.dumps(f) + "\n" for f in fixes), encoding="utf-8")
    g1._correct(pages)
    assert pages[0]["words"][0]["struck"] == [[0, 1]] and pages[0]["words"][1]["masked"]
    fixes[0]["text"] = "Other"
    path.write_text("".join(json.dumps(f) + "\n" for f in fixes), encoding="utf-8")
    with pytest.raises(ValueError, match="not 'Other'"):
        g1._correct(pages)
    fixes[0] |= {"text": "Oother", "chars": "Q"}  # not in the word: never the whole word struck
    path.write_text("".join(json.dumps(f) + "\n" for f in fixes), encoding="utf-8")
    with pytest.raises(ValueError, match="can't place 'Q' in word 0"):
        g1._correct(pages)


def test_gpos_drawn_seal_on_a_first_page_is_furniture_with_its_words():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 200), "SEC. 2. Short title.", fontsize=12)
    for i in range(12):  # the seal's paths, in the top band's left part
        page.draw_line((55 + 4 * i, 10), (58 + 4 * i, 30))
    words = g2.page_words(page)
    seal = [w["text"] for w in words if w["furniture"] and w["line"] == g2.SEAL_LINE]
    assert seal == list(g2.SEAL) and not any(
        w["furniture"] for w in words if w["line"] != g2.SEAL_LINE
    )
    plain = doc.new_page(width=612, height=792)
    plain.insert_text((72, 200), "SEC. 3. Findings.", fontsize=12)
    assert not any(w["furniture"] for w in g2.page_words(plain))
