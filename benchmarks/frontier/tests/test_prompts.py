"""The prompts (PREREG.md, Prompts): the order of a request's parts, the JSON rungs' verdicts,
stage B4's M+P context from P-scan's reading, and R6's example page."""

import pytest

from bench import prompts
from bench.providers import Part

IMAGE = Part("image", path="data/inputs/p/c2.png", sha256="x")


@pytest.mark.parametrize("provider", ["anthropic", "google", "openai", "xai"])
def test_the_page_comes_before_the_instruction_for_every_provider(monkeypatch, provider):
    monkeypatch.setattr(prompts, "page_parts", lambda page_id, input_: [IMAGE])
    system, parts = prompts.assemble({"provider": provider}, "R2", "C2", "p")
    assert parts[0] == IMAGE and parts[-1].kind == "text" and "~~" in parts[-1].text
    assert system == prompts.text("R1")[0]  # the system line from R1 on


def test_the_json_rungs_offer_struck_and_clean_alone():
    for rung in ("R4", "R5", "R6"):
        for para in range(prompts.PARAPHRASES):
            instruction = prompts.text(rung, para)[1]
            assert "`struck`" in instruction and "`clean`" in instruction
            assert "unsure" not in instruction


def test_m_plus_p_hands_the_model_p_scans_reading_of_the_page(monkeypatch):
    words = [["The", []], ["semi-monthly", [[0, 5]]], ["fee", [[0, 3]]], ["applies", []]]
    monkeypatch.setattr(prompts, "_p_scan", lambda: {("P-scan", "p"): {"words": words}})
    monkeypatch.setattr(prompts, "page_parts", lambda page_id, input_: [IMAGE])
    system, parts = prompts.m_plus_p({"provider": "google"}, "p")
    image, context, instruction = parts
    assert image == IMAGE and instruction.text == prompts.text("R2")[1]
    assert "semi-monthly [semi-]\nfee [fee]" in context.text
    assert context.text.endswith("The monthly applies")  # the live characters, in its order
    monkeypatch.setattr(prompts, "_p_scan", lambda: {("P-scan", "p"): {"error": "refused"}})
    _, (image, context, instruction) = prompts.m_plus_p({"provider": "google"}, "p")
    assert image == IMAGE and instruction.text == prompts.text("R2")[1]
    assert context.text == "A strikethrough detector was run on this page but produced no reading."
    monkeypatch.setattr(prompts, "_p_scan", lambda: {})
    with pytest.raises(FileNotFoundError, match="no P-scan output"):
        prompts.m_plus_p({"provider": "google"}, "p")


def test_r6s_examples_come_from_their_page_which_no_dev_list_draws(monkeypatch):
    monkeypatch.setattr(prompts, "dev_pages", lambda: {"pilot": [prompts.FEWSHOT_PAGE]})
    with pytest.raises(ValueError, match="R6's example page, is in a dev list"):
        prompts._example(frozenset({"N3"}))
    words = [{"text": t, "strata": ["N3"], "furniture": False, "line": [0, 0]} for t in "abc"]
    pages = [{"page_id": "G0.dev-000.p0", "words": words},  # outside every list, but not R6's
             {"page_id": prompts.FEWSHOT_PAGE, "words": words}]  # fmt: skip
    monkeypatch.setattr(prompts, "dev_pages", lambda: {"pilot": []})
    monkeypatch.setattr(prompts, "load_pages", lambda *s, split=None: pages)
    page, run = prompts._example(frozenset({"N3"}))
    assert page["page_id"] == prompts.FEWSHOT_PAGE and run == [0, 1, 2]
