"""Golden tests for the scorer: hand-made outputs with known scores (PREREG.md, stage B1 step 7)."""

import json

import pytest

from bench.score import (
    UNREAD, ParseError, align_words, normalize, parse, score_page, tokens_from_words, uncertain,
)  # fmt: skip


def page(*words) -> dict:
    """Words as `text`, `(text, struck spans)` or a dict of overrides."""
    out = []
    for w in words:
        if isinstance(w, str):
            w = {"text": w}
        elif isinstance(w, tuple):
            w = {"text": w[0], "struck": w[1]}
        out.append({"struck": [], "masked": False, "furniture": False, **w})
    return {"words": out}


def outcomes(pg: dict, text: str | None, json_answer: bool = False) -> list[tuple[str, str]]:
    tokens = None if text is None else parse(text, json_answer)
    rows = score_page(pg, tokens, details=True)["details"]
    return [(pg["words"][r["word"]]["text"][slice(*r["span"])], r["outcome"]) for r in rows]


FULL = [[0, 5]]
STRUCK = page("The", ("fifth", FULL), "clause")


def test_struck_word_live_is_a_leak():
    assert outcomes(STRUCK, "The fifth clause") == [
        ("The", "live"), ("fifth", "live"), ("clause", "live"),
    ]  # fmt: skip
    counts = score_page(STRUCK, parse("The fifth clause"))
    assert counts["struck"] == 1 and counts["struck_live"] == 1


def test_struck_word_marked():
    assert outcomes(STRUCK, "The ~~fifth~~ clause")[1] == ("fifth", "marked")


@pytest.mark.parametrize("marker", ["<del>fifth</del>", "<s>fifth</s>", "<strike>fifth</strike>",
                                    "[deleted: fifth]", "<DEL class='x'>fifth</DEL>",
                                    "[red strikethrough: fifth]",
                                    "[Blue Strikethrough: fifth]"])  # fmt: skip
def test_every_marker_on_the_list(marker):
    assert outcomes(STRUCK, f"The {marker} clause")[1] == ("fifth", "marked")


@pytest.mark.parametrize("notation", ["[DEL: fifth]", "[INS-RED: fifth]", "[added: fifth]"])
def test_notations_the_audit_left_out(notation):
    assert outcomes(STRUCK, f"The {notation} clause")[1] == ("fifth", "live")


def test_struck_word_absent():
    counts = score_page(STRUCK, parse("The clause"))
    assert counts["struck_absent"] == 1 and counts["live_live"] == 2 and counts["inserted"] == 0


def test_partial_strike_semi_monthly():
    pg = page(("semi-monthly", [[0, 5]]), "payments")
    assert outcomes(pg, "~~semi-~~monthly payments") == [
        ("semi-", "marked"), ("monthly", "live"), ("payments", "live"),
    ]  # fmt: skip
    assert outcomes(pg, "monthly payments")[0] == ("semi-", "absent")
    assert outcomes(pg, "~~semi-monthly~~ payments")[1] == ("monthly", "marked")


def test_struck_word_beside_its_correction():
    pg = page(("DecemberMay", [[0, 8]]))
    assert outcomes(pg, "~~December~~May") == [("December", "marked"), ("May", "live")]
    pg = page(("December", [[0, 8]]), "May")
    assert outcomes(pg, "~~December~~ May") == [("December", "marked"), ("May", "live")]
    assert outcomes(pg, "May") == [("December", "absent"), ("May", "live")]


def answer(*items) -> str:
    return "Here it is:\n```json\n" + json.dumps({"words": list(items)}) + "\n```"


def test_json_partial_chars():
    pg = page(("semi-monthly", [[0, 5]]))
    text = answer(
        {"text": "semi-monthly", "chars": "semi-", "verdict": "struck", "confidence": 0.9}
    )
    assert outcomes(pg, text, json_answer=True) == [("semi-", "marked"), ("monthly", "live")]


def test_json_chars_occurring_twice_marks_the_first():
    pg = page(("lalaland", [[0, 2]]))
    text = answer({"text": "lalaland", "chars": "la", "verdict": "struck"})
    assert outcomes(pg, text, json_answer=True) == [("la", "marked"), ("laland", "live")]


def test_json_empty_or_missing_chars_marks_the_whole_word_and_unsure_is_live():
    pg = page(("fifth", FULL), "clause")
    text = answer({"text": "fifth", "chars": "", "verdict": "struck"},
                  {"text": "clause", "chars": "cl", "verdict": "unsure"})  # fmt: skip
    assert outcomes(pg, text, json_answer=True) == [("fifth", "marked"), ("clause", "live")]
    text = answer({"text": "fifth", "chars": "zz", "verdict": "struck"},
                  {"text": "clause", "verdict": "clean"})  # fmt: skip
    assert outcomes(pg, text, json_answer=True)[0] == ("fifth", "marked")


def test_json_uses_the_last_object():
    pg = page(("fifth", FULL))
    text = answer({"text": "fifth", "verdict": "clean"}) + answer(
        {"text": "fifth", "verdict": "struck"}
    )
    assert outcomes(pg, text, json_answer=True) == [("fifth", "marked")]


@pytest.mark.parametrize("text", ["no json here", '{"items": []}', '{"words": [3]}'])
def test_json_parse_errors(text):
    with pytest.raises(ParseError):
        parse(text, json_answer=True)


def test_combining_strokes_mark_their_characters():
    assert outcomes(STRUCK, "The f̶i̶f̶t̶h̶ clause")[1] == (
        "fifth",
        "marked",
    )
    pg = page(("f̶i̶f̶t̶h̶", [[0, 10]]), "clause")
    assert outcomes(pg, "~~fifth~~ clause")[0][1] == "marked"
    assert outcomes(pg, "fifth clause")[0][1] == "live"


def test_misread_struck_word_is_present():
    pg = page(("December", [[0, 8]]), "1")
    assert outcomes(pg, "~~Decernber~~ 1")[0] == ("December", "marked")
    assert outcomes(pg, "Decernber 1")[0] == ("December", "live")


def test_distant_pair_is_absent_plus_an_insertion():
    pg = page(("December", [[0, 8]]))
    counts = score_page(pg, parse("zzzzzzzz"))
    assert counts["struck_absent"] == 1 and counts["inserted"] == 1


def test_hyphen_split_across_a_line_end():
    pg = page(("semi-", [[0, 5]]), "monthly", "payments")
    assert outcomes(pg, "~~semi-~~monthly payments") == [
        ("semi-", "marked"), ("monthly", "live"), ("payments", "live"),
    ]  # fmt: skip
    assert align_words(["semimonthly", "payments"], ["semi", "monthly", "payments"])[0] == (
        (0,),
        (0, 1),
    )
    assert score_page(pg, parse("~~semi-~~monthly payments"))["wer_num"] == 0


def test_markdown_table_and_heading():
    pg = page("Fees", "Item", ("Rate", [[0, 4]]), "Setup", ("5%", [[0, 2]]), "3%")
    text = "# Fees\n\n| Item | ~~Rate~~ |\n|---|:---:|\n| **Setup** | ~~5%~~ 3% |\n"
    got = outcomes(pg, text)
    assert got == [
        ("Fees", "live"), ("Item", "live"), ("Rate", "marked"),
        ("Setup", "live"), ("5%", "marked"), ("3%", "live"),
    ]  # fmt: skip
    assert score_page(pg, parse(text))["inserted"] == 0


def test_numbered_label_is_page_text_and_bullets_are_markup():
    pg = page("1.", "Term", "Scope")
    assert score_page(pg, parse("1. Term\n- Scope"))["inserted"] == 0
    assert outcomes(pg, "1. Term\n- Scope")[0] == ("1.", "live")


def test_sub_token_that_normalizes_to_nothing_is_dropped():
    pg = page(("—", [[0, 1]]), ("rate;", [[4, 5]]), "due")
    counts = score_page(pg, parse("rate due"))
    assert counts["dropped"] == 2  # the struck dash and the struck semicolon
    assert counts["struck"] == 0 and counts["live"] == 2


def test_furniture_and_masked_words_are_not_scored_and_absorb_output():
    pg = page({"text": "Page", "furniture": True}, {"text": "7", "furniture": True},
              ("fifth", FULL), {"text": "draft", "masked": True}, "clause")  # fmt: skip
    counts = score_page(pg, parse("Page 7 ~~fifth~~ draft clause"))
    assert counts["struck"] == 1 and counts["struck_marked"] == 1
    assert counts["live"] == 1 and counts["inserted"] == 0
    assert counts["wer_num"] == 0 and counts["cer_num"] == 0


def test_furniture_typed_out_of_order_is_ignored_too():
    # margin line numbers last in the ground truth's reading order, typed beside their lines
    pg = page("Be", "it", ("enacted", [[0, 7]]), "by",
              {"text": "1", "furniture": True}, {"text": "2", "furniture": True})  # fmt: skip
    counts = score_page(pg, parse("1 Be it\n2 ~~enacted~~ by"))
    assert counts["inserted"] == 0 and counts["struck_marked"] == 1
    assert counts["live_live"] == 3 and counts["cer_num"] == 0


def test_a_furniture_word_takes_one_out_of_order_token():
    pg = page("clause", {"text": "7", "furniture": True})
    assert score_page(pg, parse("7 7 clause"))["inserted"] == 1


def test_a_furniture_word_paired_with_another_number_still_takes_its_own():
    # the alignment pairs the output's 10 with the key's 1 (distance 0.5), leaving the output's 1
    pg = page("alpha", {"text": "10", "furniture": True}, "beta", {"text": "1", "furniture": True})
    assert score_page(pg, parse("1 alpha beta 10"))["inserted"] == 0


def test_latex_reads_as_the_printed_symbol_as_markup_or_as_a_name():
    pg = page("Δp", "≤", "5", "·", "10", "kg", "max")
    counts = score_page(pg, parse(r"$\Delta p \leq 5 \cdot 10$ \mathrm{kg} \max"))
    assert counts["inserted"] == 0 and counts["live_live"] == counts["live"] == 5
    text = answer({"text": r"\Delta p", "verdict": "struck"})
    assert outcomes(page(("Δp", [[0, 2]])), text, json_answer=True) == [("Δp", "marked")]


def test_a_table_read_row_by_row_against_a_key_that_lists_a_column_first():
    pg = page("698", ("699", [[0, 3]]), "74.8", "72.74")
    counts = score_page(pg, parse("698 74.8 ~~699~~ 72.74"))
    assert counts["inserted"] == 0 and counts["live_absent"] == 0
    assert counts["struck_marked"] == 1 and counts["live_live"] == 3


def test_latex_environments_and_braces_are_markup():
    pg = page("4.0", "0.399", "10", "−3")
    counts = score_page(pg, parse(r"\begin{aligned} \frac{4.0}{0.399} 10^{-3} \end{aligned}"))
    assert counts["inserted"] == 0 and counts["live_live"] == 4
    text = answer({"text": r"10^{-3}", "verdict": "clean"})
    assert score_page(page("10", "−3"), parse(text, json_answer=True))["live_live"] == 2


def test_other_latex_commands_and_paths_stay_text():
    pg = page(r"E:\BILLS\S5249.RS", r"\newline")
    assert score_page(pg, parse(r"E:\BILLS\S5249.RS \newline"))["live_live"] == 2


UNREAD_PAGE = page("the", {"text": UNREAD, "masked": True}, ("fifth", FULL), "clause")


def test_an_unreadable_word_absorbs_one_token_and_is_not_scored():
    counts = score_page(UNREAD_PAGE, parse("the guess ~~fifth~~ clause"))
    assert counts["inserted"] == 0 and counts["struck_marked"] == 1 and counts["live_live"] == 2
    assert counts["wer_num"] == 0 and counts["cer_num"] == 0
    assert score_page(UNREAD_PAGE, parse("the two guesses ~~fifth~~ clause"))["inserted"] == 1


def test_an_unreadable_word_never_takes_a_readable_words_token():
    counts = score_page(UNREAD_PAGE, parse("the ~~fifth~~ clause"))
    assert counts["inserted"] == 0 and counts["struck_marked"] == 1 and counts["live_live"] == 2
    misread = page("the", ("rate", [[0, 4]]), {"text": UNREAD, "masked": True})
    assert outcomes(misread, "the ~~raxy~~") == [("the", "live"), ("rate", "marked")]  # 0.5 apart


def test_adjacent_unreadable_words_each_take_one_token_in_either_direction():
    unread = {"text": UNREAD, "masked": True}
    assert score_page(page(unread, unread, "clause"), parse("one two clause"))["inserted"] == 0
    for early in (False, True):
        steps = align_words(["one", "two"], [UNREAD, UNREAD], wild=[0, 1], early=early)
        assert steps == [((0,), (0,)), ((1,), (1,))]


def test_only_a_masked_unreadable_word_is_a_wildcard():
    counts = score_page(page("the", UNREAD, "clause"), parse("the guess clause"))  # a PDF's U+FFFD
    assert counts["inserted"] == 1 and counts["live_absent"] == 1


def test_hallucinated_tokens_counted_with_digits_separately():
    counts = score_page(STRUCK, parse("The ~~fifth~~ clause expires 2031 soon"))
    assert counts["inserted"] == 3 and counts["inserted_digit"] == 1


def test_failure_scores_worst_case():
    counts = score_page(STRUCK, None)
    assert counts["failed"]
    assert counts["struck_live"] == counts["struck"] == 1
    assert counts["live_absent"] == counts["live"] == 2


def test_live_text_error_rates():
    counts = score_page(STRUCK, parse("The ~~fifth~~ clauses"))
    assert counts["wer_num"] == 1 and counts["wer_den"] == 2
    assert counts["cer_num"] == 1 and counts["cer_den"] == len("the clause")


def test_normalize_keeps_an_offset_map():
    assert normalize("“Fiﬁ—”") == ("fifi", [1, 2, 3, 3])
    assert normalize("s̶t̶") == ("st", [0, 2])


def test_package_tokens_from_word_records():
    tokens = tokens_from_words([("The", []), ("semi-monthly", [[0, 5]])])
    pg = page("The", ("semi-monthly", [[0, 5]]))
    assert outcomes(pg, None)[0] == ("The", "absent")  # a failure, for contrast
    rows = score_page(pg, tokens, details=True)["details"]
    assert [r["outcome"] for r in rows] == ["live", "marked", "live"]


def test_strata_counts():
    pg = page({"text": "fifth", "struck": FULL, "strata": ["E1", "S2"]}, "clause")
    strata = score_page(pg, parse("fifth clause"))["strata"]
    assert strata["E1"] == {"struck": 1, "struck_live": 1} == strata["S2"]


# ------------------------------------------------- the bug-detection pass's cases, 2026-10-14


def test_the_package_arms_marks_come_from_their_spans_only():
    e6 = "f̶i̶f̶t̶h̶"
    pg = page((e6, [[0, len(e6)]]), "clause")
    tokens = tokens_from_words([(e6, []), ("clause", [])])
    assert [r["outcome"] for r in score_page(pg, tokens, details=True)["details"]][0] == "live"
    assert score_page(pg, parse(f"{e6} clause", marks=False))["struck_live"] == 1
    assert score_page(pg, parse(f"{e6} clause"))["struck_marked"] == 1  # a model copying them


@pytest.mark.parametrize("text", ["", "   \n", "**", "| --- |"])
def test_an_empty_output_is_a_failure(text):
    counts = score_page(STRUCK, parse(text))
    assert counts["failed"] and counts["struck_live"] == 1 and counts["live_absent"] == 2


def test_an_empty_json_words_list_fails_to_parse():
    with pytest.raises(ParseError):
        parse('{"words": []}', json_answer=True)


def test_the_backtrace_picks_the_cheaper_pair_after_a_long_prefix():
    # PREREG: global alignment. A relative tolerance once let a costlier pair through here.
    out = ["99999"] * 1000 + ["abcdefghijklmnopqrst"]  # junk sharing no character with gt
    gt = ["abcdefghijklmnopqXYZ", "abcdefghijklmnoQXYZW"]
    assert ((1000,), (0,)) in align_words(out, gt)


def test_a_join_keeps_edge_punctuation_inside():
    pg = page(("5%", [[0, 2]]), "3%")
    assert outcomes(pg, "~~5%~~3%") == [("5%", "marked"), ("3%", "live")]
    pg = page(("$500", [[0, 4]]), "$450")
    assert outcomes(pg, "~~$500~~$450") == [("$500", "marked"), ("$450", "live")]


def test_two_output_tokens_split_from_one_word():
    pg = page(("semi-monthly", [[0, 5]]), "payments")
    assert outcomes(pg, "~~semi-~~ monthly payments") == [
        ("semi-", "marked"), ("monthly", "live"), ("payments", "live"),
    ]  # fmt: skip
    counts = score_page(pg, parse("~~semi-~~ monthly payments"))
    assert counts["inserted"] == 0 and counts["wer_num"] == 0


@pytest.mark.parametrize("text, want", [
    ("The ~fifth~ clause", "marked"),  # GFM's single-tilde strikethrough
    ("The ~fifth clause~", "marked"),  # spaces inside are fine
    ("The ~ fifth ~ clause", "live"),  # not flanking: text
    ("The \\~fifth~ clause", "live"),  # escaped
    ("The ~fifth\nclause~", "live"),  # never across a line
    ("The ~~fifth~ clause", "live"),  # part of a `~~`, unpaired
    ("~The~ ~~fifth~~ clause", "marked"),  # beside a double pair, each read once
])  # fmt: skip
def test_single_tildes_mark_as_gfm_reads_them(text, want):
    assert outcomes(STRUCK, text)[1] == ("fifth", want)


def test_single_tildes_mark_only_what_they_wrap():
    pg = page(("5%", [[0, 2]]), "3%")
    assert outcomes(pg, "~5%~ 3%") == [("5%", "marked"), ("3%", "live")]
    assert outcomes(pg, "~5%~3%") == [("5%", "marked"), ("3%", "live")]  # one token, split


def test_a_tilde_code_fence_is_not_a_strike():
    assert outcomes(STRUCK, "~~~markdown\nThe fifth clause\n~~~") == [
        ("The", "live"), ("fifth", "live"), ("clause", "live"),
    ]  # fmt: skip
    for fence in ("~~~~", "~~~~~", "````"):  # a longer fence, as CommonMark allows
        assert outcomes(STRUCK, f"{fence}\nThe fifth clause\n{fence}") == [
            ("The", "live"), ("fifth", "live"), ("clause", "live"),
        ], fence  # fmt: skip


def test_a_decomposed_accent_composes():
    assert score_page(page("né", "clause"), parse("né clause"))["live_live"] == 2


def test_html_entities_decode_and_angle_brackets_stay_text():
    counts = score_page(page("R&D", "and", "<Party", "A>"), parse("R&amp;D and <Party A>"))
    assert counts["live_live"] == 4 and counts["inserted"] == 0


def test_escaped_emphasis_characters_stay():
    assert score_page(page("snake_case"), parse(r"snake\_case"))["live_live"] == 1


def test_a_json_verdict_is_read_case_insensitively():
    text = answer({"text": "fifth", "verdict": " Struck"})
    assert outcomes(page(("fifth", FULL)), text, json_answer=True) == [("fifth", "marked")]


# ------------------------------------------------- the second bug-detection pass's cases


def test_cer_and_wer_follow_the_alignments_joins_and_splits():
    joined = page("semi-", "monthly", "rate")
    for text in ("semimonthly rate", "semi-monthly rate"):
        c = score_page(joined, parse(text))
        assert (c["wer_num"], c["cer_num"]) == (0, 0), text
    split = page("semi-monthly", "rate")
    c = score_page(split, parse("semi- monthly rate"))
    assert (c["wer_num"], c["cer_num"]) == (0, 0)


def test_a_clean_json_word_stays_live_whatever_its_text_carries():
    e6 = "f̶i̶f̶t̶h̶"
    text = answer({"text": e6, "verdict": "clean"})
    assert outcomes(page((e6, [[0, len(e6)]])), text, json_answer=True)[0][1] == "live"


def test_only_a_lone_fence_line_is_a_fence():
    pg = page("x", ("y", [[0, 1]]))
    assert outcomes(pg, "~~~text\nx ~~y~~\n~~~") == [("x", "live"), ("y", "marked")]
    assert outcomes(page("x", "~~~", "y"), "x ~~~ y")[0] == ("x", "live")  # not a fence line


def test_nested_strike_tags_mark_everything_inside_the_outer_pair():
    pg = page(("a", [[0, 1]]), ("b", [[0, 1]]), ("c", [[0, 1]]))
    for text in ("<del>a <del>b</del> c</del>", "<del>a <s>b</s> c</del>"):
        assert outcomes(pg, text) == [("a", "marked"), ("b", "marked"), ("c", "marked")]
        assert score_page(pg, parse(text))["inserted"] == 0
    assert outcomes(page("x", "y"), "x <del>y") == [("x", "live"), ("y", "live")]  # unpaired


def test_a_json_item_without_text_is_left_out_and_its_probability_kept():
    pg = page("the", ("fifth", FULL), "clause")
    text = answer({"text": "the", "verdict": "clean"}, {"text": None, "verdict": "unsure"},
                  {"text": "fifth", "verdict": "struck", "confidence": 0.9},
                  {"text": "clause", "verdict": "clean", "confidence": "high"})  # fmt: skip
    assert outcomes(pg, text, json_answer=True) == [
        ("the", "live"), ("fifth", "marked"), ("clause", "live"),
    ]  # fmt: skip
    assert [t.p_struck for t in parse(text, json_answer=True)] == [None, 0.9, None]


# ------------------------------------------------- the calibration and the cascade, 2026-10-15


@pytest.mark.parametrize("verdict, confidence, p", [
    ("struck", 0.9, 0.9), ("clean", 0.8, 0.2), ("Clean", "0.75", 0.25), ("unsure", 0.7, 0.5),
    ("unsure", None, 0.5), ("deleted", 0.9, None),
    ("struck", 1.5, None), ("struck", True, None), ("struck", "nan", None), ("clean", None, None),
])  # fmt: skip
def test_a_json_words_probability_of_being_struck(verdict, confidence, p):
    (token,) = parse(answer({"text": "x", "verdict": verdict, "confidence": confidence}), True)
    assert token.p_struck == pytest.approx(p) if p is not None else token.p_struck is None


def test_a_word_takes_its_tokens_highest_probability_and_whether_it_holds_a_strike():
    pg = page("the", ("semi-monthly", [[0, 5]]), "clause", {"text": "Page", "furniture": True})
    words = [("the", []), ("semi-", [[0, 5]]), ("monthly", []), ("clause", []), ("Page", [])]
    tokens = tokens_from_words(words, [0.01, 0.9, 0.5, None, 0.2])
    # every scored word is listed, None where no token gives a probability; furniture apart
    assert score_page(pg, tokens)["probs"] == [[0, 0, 0.01], [1, 1, 0.9], [2, 0, None]]
    assert "probs" not in score_page(pg, parse("the semi-monthly clause"))  # no probabilities
    failed = score_page(pg, None, probabilities=True)  # an arm that gives them, failed here
    assert failed["probs"] == [[0, 0, None], [1, 1, None], [2, 0, None]]
    with pytest.raises(ValueError, match="2 probabilities for 5 words"):
        tokens_from_words(words, [0.1, 0.2])


def test_the_words_p_scan_leaves_uncertain_are_counted_apart():
    pg = page("the", ("fifth", FULL), "clause")
    pscan = tokens_from_words([("the", []), ("fifth", []), ("clause", [])], [0.01, 0.6, 0.9])
    assert uncertain(pg, pscan, 0.15, 0.85) == {1}
    model = score_page(pg, parse("the ~~fifth~~ clause"), subsets={"uncertain": {1}})
    assert model["subsets"] == {"uncertain": {"struck": 1, "struck_marked": 1}}
    failed = score_page(pg, None, subsets={"uncertain": {1, 2}})  # worst case there too
    assert failed["subsets"]["uncertain"] == {"struck": 1, "struck_live": 1, "live": 1,
                                              "live_absent": 1}  # fmt: skip


def test_early_ties_take_the_earliest_copy_and_keep_joins_and_splits():
    # G2 aligns with early ties: a bill can repeat its struck text in the substitute after it.
    out, gt = ["a", "b", "x", "a", "b"], ["a", "b"]
    assert align_words(out, gt) == [((0,), ()), ((1,), ()), ((2,), ()), ((3,), (0,)), ((4,), (1,))]
    assert align_words(out, gt, early=True) == [
        ((0,), (0,)), ((1,), (1,)), ((2,), ()), ((3,), ()), ((4,), ()),
    ]  # fmt: skip
    assert align_words(["semimonthly", "pay"], ["semi", "monthly", "pay"], early=True) == [
        ((0,), (0, 1)), ((1,), (2,)),
    ]  # fmt: skip
    assert align_words(["semi", "monthly", "pay"], ["semimonthly", "pay"], early=True) == [
        ((0, 1), (0,)), ((2,), (1,)),
    ]  # fmt: skip


def test_a_soft_hyphen_aligns_like_the_printed_hyphen():
    # A tagged PDF can carry U+00AD where the page prints "-" (G1's EPA documents).
    pg = page("regu­", "lation", "applies")
    assert outcomes(pg, "regulation applies") == [
        ("regu­", "live"), ("lation", "live"), ("applies", "live"),
    ]  # fmt: skip
    assert outcomes(pg, "regu- lation applies") == outcomes(pg, "regulation applies")


def test_the_uncertain_band_excludes_its_edges_and_takes_any_aligned_token():
    pg = page("a", "b", "c", "pre-test")
    pscan = tokens_from_words([("a", []), ("b", []), ("c", []), ("pre", []), ("test", [[0, 4]])],
                              [0.15, 0.85, 0.1501, 0.036, 0.965])  # fmt: skip
    # at 0.15 and 0.85 the package has decided; "pre-test" averages 0.5 but both halves decided
    assert uncertain(pg, pscan, 0.15, 0.85) == {2}
    assert uncertain(pg, None, 0.15, 0.85) == set()  # a failed page leaves nothing uncertain


def test_numbers_too_large_to_read_fail_quietly():
    (token,) = parse(answer({"text": "x", "verdict": "struck", "confidence": 10**400}), True)
    assert token.p_struck is None
    huge = '{"words": [{"text": "x", "verdict": "clean", "confidence": 1' + "0" * 5000 + "}]}"
    with pytest.raises(ParseError):  # a parse failure, scored as one, not a crash
        parse(huge, json_answer=True)


def test_a_struck_items_probability_reaches_its_struck_tokens_alone():
    text = answer({"text": "the old", "chars": "old", "verdict": "struck", "confidence": 0.95},
                  {"text": "new rule", "verdict": "clean", "confidence": 0.8})  # fmt: skip
    got = [(t.text, t.p_struck) for t in parse(text, json_answer=True)]
    assert got == [("the", None), ("old", 0.95), ("new", pytest.approx(0.2)),
                   ("rule", pytest.approx(0.2))]  # fmt: skip


def test_a_word_hyphenated_across_a_margin_line_number_joins():
    # the bill prints "Avia-", then line 4's number, then "tion": the output types "Aviation"
    pg = page(("Avia-", [[0, 5]]), {"text": "4", "furniture": True}, ("tion", [[0, 4]]), "Act")
    counts = score_page(pg, parse("4 ~~Aviation~~ Act"))
    assert counts["struck"] == 2 and counts["struck_marked"] == 2
    assert counts["inserted"] == 0 and counts["live_live"] == 1


def test_a_line_the_text_layer_garbles_is_not_scored():
    line = {"line": [0, 0]}
    pg = page({"text": "x", **line}, {"text": "ሺ0.955ሻ", **line}, {"text": "1.5", **line},
              {"text": "clause", "line": [1, 0]})  # fmt: skip
    counts = score_page(pg, parse("x cos (0.955) 1.5 clause"))
    assert counts["live"] == 1 and counts["live_live"] == 1  # only the readable line's word
    assert counts["inserted"] == 0 and counts["cer_num"] == 0  # "cos" sits inside the garbled line


def test_a_garbled_line_at_the_page_edge_absorbs_its_reading_but_not_a_readable_lines():
    l0, l1, l2 = {"line": [0, 0]}, {"line": [1, 0]}, {"line": [2, 0]}
    first = page({"text": "ሺ1ሻ", **l0}, {"text": "the", **l1})
    assert score_page(first, parse("(1) the"))["inserted"] == 0  # no aligned token before it
    between = page({"text": "ሺ1ሻ", **l0}, {"text": "x", **l0}, {"text": "the", **l1},
                   {"text": "rule", **l1}, {"text": "y", **l2}, {"text": "ሺ2ሻ", **l2})  # fmt: skip
    counts = score_page(between, parse("(1) x foo bar baz y (2)"))
    assert counts["live_absent"] == 2 and counts["inserted"] == 3  # a readable line stays a miss
    across = page({"text": "x", **l0}, {"text": "ሺ1ሻ", **l0}, {"text": "alpha", **l1},
                  {"text": "y", **l2}, {"text": "ሺ2ሻ", **l2})  # fmt: skip
    assert score_page(across, parse("y qqqq x"))["inserted"] == 1  # neighbours out of order


def test_the_out_of_order_passes_give_a_scored_word_its_own_token_first():
    furniture = {"text": "13", "furniture": True}  # a margin number near the struck "12"
    pg = page(("12", [[0, 2]]), "alpha", "beta", "gamma", furniture)
    assert outcomes(pg, "alpha beta gamma ~~12~~")[0] == ("12", "marked")
    masked = page({"text": "fee", "masked": True}, "a", "b", "c", ("fee", [[0, 3]]))
    assert outcomes(masked, "a b ~~fee~~ c")[-1] == ("fee", "marked")


def test_garbled_reads_only_wrong_glyphs():
    from bench.score import garbled

    assert garbled("⋅ሺ0.955ሻെ") and garbled("10ିଷ") and garbled("\u0d45")
    assert not any(garbled(t) for t in ("µg", "nº", "𝑣ref1000", "Δp", "–", UNREAD))
