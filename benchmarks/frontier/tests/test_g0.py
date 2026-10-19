"""G0's generator: byte determinism, ground truth against the page, strata and the record schema."""

import hashlib

import pymupdf
import pytest

from bench import g0

SEED = 20261008  # G0-dev's public seed
WORD_KEYS = ["text", "bbox", "struck", "masked", "furniture", "line", "strata"]
PAGE_KEYS = ["page_id", "set", "split", "doc", "page", "cluster", "pdf", "pdf_sha256", "words"]


@pytest.fixture(scope="module")
def pages():
    """G0-dev's first five pages, built in memory: page 0 is strike-free (N1), 1-4 are struck."""
    return [g0.make_page(SEED, i) for i in range(5)]


def _words(pages, stratum):
    return [w for _, words, _, _ in pages for w in words if stratum in w["strata"]]


def test_same_seed_and_index_give_identical_bytes(pages):
    assert g0.make_page(SEED, 2)[0] == pages[2][0]
    assert g0.make_page(SEED + 1, 2)[0] != pages[2][0]


def test_ground_truth_is_the_page_words(pages):
    for data, words, _, _ in pages:
        found = pymupdf.open(stream=data, filetype="pdf")[0].get_text("words")
        assert [w["text"] for w in words] == [f[4] for f in found]
        assert [w["line"] for w in words] == [[f[5], f[6]] for f in found]
        for w, f in zip(words, found):
            box = [f[0] / g0.W, f[1] / g0.H, f[2] / g0.W, f[3] / g0.H]
            assert w["bbox"] == pytest.approx(box, abs=1e-5)


def test_partial_strikes_keep_two_live_characters(pages):
    partial = _words(pages, "S5")
    assert partial
    for w in partial:
        [(c0, c1)] = w["struck"]
        assert (c0 == 0) != (c1 == len(w["text"]))  # a prefix or a suffix, never the whole
        assert c1 - c0 >= 2 and len(w["text"]) - (c1 - c0) >= 2


def test_combining_strokes_are_full_strikes(pages):
    e6 = _words(pages, "E6")
    assert e6
    for w in e6:
        assert w["text"][1::2] == g0.STROKE * (len(w["text"]) // 2)
        assert w["struck"] == [[0, len(w["text"])]]


def test_furniture_is_flagged_and_live(pages):
    for _, words, _, _ in pages:
        furniture = [w for w in words if w["furniture"]]
        assert len(furniture) >= 3  # the running head's two parts and the page number
        assert not any(w["struck"] or w["strata"] for w in furniture)
        tops = [w["bbox"][1] for w in words if not w["furniture"]]
        assert all(w["bbox"][1] < min(tops) or w["bbox"][1] > max(tops) for w in furniture)


def test_strike_free_page_has_no_strikes(pages):
    _, words, extras, qa = pages[0]
    assert extras["n1"] and not qa
    assert not any(w["struck"] for w in words)


def test_page_record_matches_the_schema(pages, tmp_path):
    data, words, extras, _ = pages[1]
    rec = g0.page_record("dev", 1, data, words, extras, tmp_path / "dev-001.pdf")
    assert list(rec) == [*PAGE_KEYS, "font", "fontsize", "n1"]
    assert rec["page_id"] == rec["cluster"] == "G0.dev-001.p0"
    assert (rec["set"], rec["split"], rec["doc"], rec["page"]) == ("G0", "dev", "dev-001", 0)
    assert rec["pdf_sha256"] == hashlib.sha256(data).hexdigest()
    assert rec["font"] == "Times-Roman" and 9 <= rec["fontsize"] <= 13  # page 1 of 3 fonts
    strata = set(g0.LABELS) - {"N1", "live"}
    for w in rec["words"]:
        assert list(w) == WORD_KEYS
        assert len(w["bbox"]) == 4 and all(0 <= v <= 1 for v in w["bbox"])
        assert w["masked"] is False and isinstance(w["furniture"], bool) and len(w["line"]) == 2
        assert all(0 <= c0 < c1 <= len(w["text"]) for c0, c1 in w["struck"])
        assert set(w["strata"]) <= strata
        assert bool(w["struck"]) == any(s.startswith("E") for s in w["strata"])


def test_clause_attributes_are_unique():
    attrs = [attr for _, text in g0.CLAUSES for _, attr in g0.SLOT.findall(text) if attr]
    assert attrs and len(attrs) == len(set(attrs))


def test_no_qa_item_is_answerable_elsewhere_on_its_page(pages):
    recs = [{"page_id": f"p{i}", "words": words} for i, (_, words, _, _) in enumerate(pages)]
    qa = [
        {"qa_id": f"p{i}.q{j}", "page_id": f"p{i}", "question": q, "live": live, "struck": old}
        for i, (*_, asked) in enumerate(pages)
        for j, (q, live, old) in enumerate(asked)
    ]
    assert qa and any("N3" in w["strata"] for r in recs for w in r["words"])
    assert g0._conflicts(recs, qa) == []
    # a table cell naming the question's concept, or holding its struck value, is caught
    item = qa[0]
    attr = item["question"].partition("what is the ")[2].rstrip("?")
    for cell in (attr.title(), f"Event {item['struck']}"):
        row = [{"text": t, "strata": ["N3"], "line": [99, 0]} for t in cell.split()]
        assert g0._conflicts([{"page_id": item["page_id"], "words": row}], [item])
    assert g0._conflicts(recs, [item, item]) == [f"{item['qa_id']}: asked twice"]


def test_heldback_seed_is_private_and_kept(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(g0, "HELDBACK_SEED", tmp_path / "private" / "seed.txt")
    monkeypatch.delenv("G0_HELDBACK_SEED", raising=False)
    with pytest.raises(SystemExit):
        g0._heldback_seed()
    g0.cmd_heldback(None)
    value = (tmp_path / "private" / "seed.txt").read_text().strip()
    assert capsys.readouterr().out.strip() == hashlib.sha256(value.encode()).hexdigest()
    assert int(value) < 2**128 and g0._heldback_seed() == int(value)
    with pytest.raises(SystemExit):
        g0.cmd_heldback(None)
    assert (tmp_path / "private" / "seed.txt").read_text().strip() == value
    monkeypatch.setenv("G0_HELDBACK_SEED", "12345")
    assert g0._heldback_seed() == 12345
