"""G5's ground truth from BRESSAY's transcriptions, and its C2."""

import io

import pytest
from PIL import Image

from bench import arms, g5, render
from bench.score import UNREAD


def words(text: str) -> list[tuple[str, list, bool]]:
    return [(w["text"], w["struck"], w["masked"]) for w in g5.parse(text)]


def test_struck_text_whole_words_and_part_of_one():
    assert words("o --custo dos-- preço pala--vra--") == [
        ("o", [], False), ("custo", [[0, 5]], False), ("dos", [[0, 3]], False),
        ("preço", [], False), ("palavra", [[4, 7]], False),
    ]  # fmt: skip


def test_additions_are_live_where_they_stand_and_struck_where_marked_inside():
    assert words("##--um--## --o-- um ##de## $$nota$$") == [
        ("um", [[0, 2]], False), ("o", [[0, 1]], False), ("um", [], False),
        ("de", [], False), ("nota", [], False),
    ]  # fmt: skip


def test_unreadable_and_illegible_words_are_masked_as_unread():
    got = words("a --xxx-- b @@???@@ c --(xxx)-- ##xxx## --xxx proble--")
    assert got == [
        ("a", [], False), (UNREAD, [[0, 1]], True), ("b", [], False), (UNREAD, [], True),
        ("c", [], False), (UNREAD, [[0, 1]], True), (UNREAD, [], True),
        (UNREAD, [[0, 1]], True), ("proble", [[0, 6]], False),
    ]  # fmt: skip
    glued = words("cinema--xxx--, d--xxx-- fé@@???@@")  # all unreadable, as in 1484-029
    assert glued == [(UNREAD, [[0, 1]], True), (UNREAD, [[0, 1]], True), (UNREAD, [], True)]


def test_g5_holds_every_struck_page_and_its_core_takes_them_until_both_floors():
    free = [f"f{i}" for i in range(20)]
    for per_page, want in ((10, 40), (4, 50), (2, 60)):  # 60 pages hold only 120: all of them
        struck = {f"s{i:02d}": per_page for i in range(60)}
        drawn, core = g5.draw(struck, free)
        assert sum(p.startswith("s") for p in drawn) == 60 and drawn == sorted(drawn)
        assert sum(p.startswith("s") for p in core) == want
        clean = [p for p in drawn if p.startswith("f")]
        assert len(clean) == g5.N_FREE and set(clean) <= core  # the strike-free pages are core
    _, a = g5.draw(dict.fromkeys(map(str, range(60)), 4), free)
    _, b = g5.draw(dict.fromkeys(map(str, range(60)), 4), free)
    assert a == b  # seeded


def test_a_strike_split_by_a_line_break_and_the_line_index():
    parsed = g5.parse("a --popula--\n##de## --ção--\n\nfim")
    assert [(w["text"], bool(w["struck"]), w["line"]) for w in parsed] == [
        ("a", False, [0, 0]), ("popula", True, [0, 0]), ("de", False, [0, 1]),
        ("ção", True, [0, 1]), ("fim", False, [1, 0]),
    ]  # fmt: skip


@pytest.mark.parametrize("text", ["um --dois", "##três", "quatro $$", "cinco @ seis"])
def test_an_unpaired_or_leftover_marker_does_not_parse(text):
    with pytest.raises(g5.ParseError):
        g5.parse(text)


def test_page_kinds():
    assert g5.kind(g5.parse("um --dois-- três")) == "struck"
    assert g5.kind(g5.parse("um dois ##três##")) == "free"
    assert g5.kind(g5.parse("um --xxx-- dois")) is None  # its only strike is unreadable


def test_c2_widens_a_narrow_scan_and_leaves_a_wide_one(tmp_path):
    for size, want in (((500, 560), (1700, 1904)), ((2000, 2200), (2000, 2200))):
        path = tmp_path / f"{size[0]}.png"
        Image.new("L", size, 255).save(path)
        with Image.open(io.BytesIO(render.scan_c2(path))) as c2:
            assert (c2.size, c2.mode) == (want, "RGB")


def test_only_the_scan_arms_run_on_a_g5_page():
    assert [a for a in arms.ARMS if arms.applies(a, "G5.0001-001.p0")] == list(arms.SCAN)
    assert [a for a in arms.ARMS if arms.applies(a, "G0.dev-000.p0")] == list(arms.ARMS)
