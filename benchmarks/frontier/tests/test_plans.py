"""What the stages send (PREREG.md, Arms and inputs, Cells, Operations), the dev lists, the caps
and what the tags freeze."""

import json

import pytest

from bench import plans

TEST, DEV, SCAN, EXTRA = "G0.test-000.p0", "G0.dev-000.p0", "G5.0001-001.p0", "G5.0002-001.p0"
BILL = "G2.BILLS-119s1rs.p0"


@pytest.fixture
def arms(monkeypatch):
    """The arms a stage sends, for two providers' flagships, a ladder model, Grok and the ceiling
    model, over a G0 and a G2 test page, one dev page and two G5 pages, one of them the core's."""
    models = {
        "flag-a": {"role": "flagship", "provider": "a"},
        "small-a": {"role": "ladder", "provider": "a"},
        "flag-b": {"role": "flagship", "provider": "b"},
        "grok": {"role": "flagship", "provider": "xai"},
        "ceiling": {"role": "ceiling", "provider": "a"},  # runs only its own subset
    }
    choices = {  # flag-a keeps R-best; flag-b and grok switched to their runner-up
        "a": {"rbest": "R4", "image": "C2-tiled", "flagship": "R4", "runner_up": "R2"},
        "b": {"rbest": "R2", "image": "C2", "flagship": "R5", "runner_up": "R5"},
        "xai": {"rbest": "R3", "image": "C2", "flagship": "R6", "runner_up": "R6"},
    }
    budget = {"drop_order": list(plans.DROPS), "dropped": []}
    config = {"models": models, "choices": choices, "budget": budget}
    tests = [{"page_id": TEST, "set": "G0"}, {"page_id": BILL, "set": "G2"}]
    scans = [
        {"page_id": SCAN, "set": "G5", "core": True},
        {"page_id": EXTRA, "set": "G5", "core": False},
    ]
    monkeypatch.setattr(plans, "config", config.__getitem__)
    monkeypatch.setattr(plans, "test_pages", lambda guard=True: tests)
    monkeypatch.setattr(plans, "g5_pages", lambda guard=True: scans)
    monkeypatch.setattr(plans, "dev_pages",
                        lambda: {"ladder": [DEV], "repeats": [DEV], "canary": [DEV]})  # fmt: skip
    splits = {TEST: "test", BILL: "test", DEV: "dev", SCAN: "test", EXTRA: "test"}
    monkeypatch.setattr(plans, "_splits", lambda: splits)
    monkeypatch.setattr(plans, "guard_test_pages", lambda pages: None)  # tested in test_arms
    monkeypatch.setattr(plans, "LATENCY_PAGES", 3)
    return lambda stage, page=None: {
        i.cell.arm for i in plans.items(stage) if page in (None, i.page_id)
    }


def test_gpt6_goes_at_flex_but_the_smoke_test_and_latency_sample_use_the_standard_tier():
    flex = {"batch": False, "service_tier": "flex"}
    batched, plain = {"batch": True}, {"batch": False}
    modes = [plans.mode("test", m) for m in (flex, batched, plain)]
    assert modes == ["flex", "batch", "sequential"]
    assert [plans.mode("smoke-batch", m) for m in (flex, batched)] == ["flex", "batch"]
    assert {plans.mode(stage, flex) for stage in ("smoke", "latency")} == {"sequential"}


def test_the_cache_probe_sends_one_dev_page_to_gpt6_alone_after_the_stages_sent_cached(
    arms, monkeypatch
):
    cfg = {k: plans.config(k) for k in ("choices", "budget")}
    cfg["models"] = plans.config("models") | {"gpt": {"role": "ladder", "provider": "openai"}}
    monkeypatch.setattr(plans, "config", cfg.__getitem__)
    monkeypatch.setattr(plans, "dev_pages", lambda: {"pilot": [DEV, TEST]})
    assert [(i.cell.arm, i.page_id) for i in plans.items("cache-probe")] == [("gpt:R0:C2", DEV)]
    assert plans.group("cache-probe") == "b2" and "cache-probe" not in plans.CACHE_ON
    assert {"smoke", "smoke-batch", *plans.GROUPS["b2"]} - set(plans.CACHE_ON) == {"cache-probe"}


def test_every_model_runs_r0_r1_and_r_best_on_c2_and_r0_on_c1(arms):
    for page in (TEST, BILL):
        sent = arms("test", page)
        for k, r in (("flag-a", "R4"), ("small-a", "R4"), ("flag-b", "R5"), ("grok", "R6")):
            assert {f"{k}:{x}:C2" for x in ("R0", "R1", r)} | {f"{k}:R0:C1"} <= sent
        assert {a for a in sent if "C2-tiled" in a} == {"flag-a:R4:C2-tiled", "small-a:R4:C2-tiled"}
        assert not any(a.startswith("ceiling") for a in sent)


def test_c1_beyond_r0_and_c1p_are_the_flagships_and_c1p_skips_g1(arms, monkeypatch):
    sent = arms("test", TEST)
    assert {a for a in sent if ":C1" in a and ":R0:C1" not in a} == {
        "flag-a:R1:C1", "flag-a:R4:C1", "flag-b:R1:C1", "flag-b:R5:C1", "grok:R1:C1", "grok:R6:C1",
    }  # fmt: skip  # no model but a flagship past R0
    assert {a for a in sent if a.endswith(":C1p")} == {
        "flag-a:R0:C1p", "flag-b:R0:C1p", "grok:R0:C1p",
    }  # fmt: skip
    g1 = "G1.doc.p0"
    monkeypatch.setattr(plans, "test_pages", lambda guard=True: [{"page_id": g1, "set": "G1"}])
    plans._splits()[g1] = "test"
    assert not any(a.endswith(":C1p") for a in arms("test", g1))


def test_the_r1_wordings_run_on_g2_alone_for_the_flagships(arms):
    assert {a for a in arms("test", BILL) if ":p" in a} == {
        f"{k}:R1:C2:p{n}" for k in ("flag-a", "flag-b", "grok") for n in (1, 2, 3)
    }  # fmt: skip
    assert not any(":p" in a for a in arms("test", TEST))


def test_g5_runs_r0_on_its_core_and_r_best_everywhere_tiled_where_chosen(arms):
    every = {"flag-a:R4:C2", "flag-a:R4:C2-tiled", "small-a:R4:C2", "small-a:R4:C2-tiled",
             "flag-b:R5:C2", "grok:R6:C2"}  # fmt: skip
    assert arms("test", EXTRA) == every
    assert arms("test", SCAN) == every | {
        f"{k}:R0:C2" for k in ("flag-a", "small-a", "flag-b", "grok")
    }


def test_cells_dropped_for_the_cap_leave_the_plan_in_drop_orders_order(arms):
    budget = plans.config("budget")
    budget["dropped"] = ["test:C1 marks", "test:C1prime", "fable"]
    sent = arms("test")
    assert not any(":C1p" in a or ":R1:C1" in a for a in sent) and arms("fable") == set()
    assert {a for a in sent if ":C1" in a} == {f"{k}:R0:C1" for k in ("flag-a", "small-a",
                                                                        "flag-b", "grok")}  # fmt: skip
    assert not any(a.startswith("ceiling") for a in arms("canary-1"))  # nor its canary
    budget["dropped"] = ["fable"]  # skips the first item
    with pytest.raises(ValueError, match="must start drop_order"):
        arms("test")
    budget["dropped"], budget["drop_order"] = [], ["test:R1 paraphrases", "fable"]
    with pytest.raises(ValueError, match="not PREREG's"):  # a name no cell answers to: refused
        arms("test")
    budget["drop_order"] = [*plans.DROPS[::-1]]  # the right names, reordered: refused too
    with pytest.raises(ValueError, match="not PREREG's"):
        arms("test")


def test_the_flagship_runs_its_own_rung_and_the_ladder_its_providers(arms):
    assert arms("ladder-tiled") == {"small-a:R4:C2-tiled"}
    assert arms("flagship-check", DEV) == {"flag-a:R4:C2", "flag-a:R2:C2", "flag-b:R2:C2",
                                           "flag-b:R5:C2", "grok:R3:C2", "grok:R6:C2"}  # fmt: skip
    timed = {"flag-a:R4:C2", "small-a:R4:C2", "flag-b:R5:C2", "grok:R6:C2"}
    assert arms("latency", TEST) == timed
    assert arms("latency", EXTRA) == set()  # G5's core alone is in the latency sample


def test_the_smoke_test_sends_every_route_once_and_b4_reads_p_scan_on_g0_and_g2(arms, monkeypatch):
    monkeypatch.setattr(plans, "dev_pages", lambda: {"pilot": [DEV]})
    plans.config("models")["gpt"] = {"role": "ladder", "provider": "openai"}
    smoke = arms("smoke")
    assert "flag-a:M+P:C2" in smoke and not any("P→M" in a for a in smoke)  # P→M is deferred
    # an image-only PDF shows whether xAI's and GPT-6's PDF routes pass the page or its text
    assert {a for a in smoke if a.endswith(":C1-scan")} == {"grok:R0:C1-scan", "gpt:R0:C1-scan"}
    # R6's examples with a PDF, which a flagship's C1 at its rung may send in the test run
    assert {a for a in smoke if a.endswith(":R6:C1")} == {"flag-a:R6:C1", "flag-b:R6:C1",
                                                           "grok:R6:C1"}  # fmt: skip
    assert arms("b4") == {"flag-a:M+P:C2", "flag-b:M+P:C2", "grok:M+P:C2"}


def test_the_latency_sample_takes_every_test_set_in_proportion(monkeypatch):
    sizes = {"G0": 53, "G1": 73, "G2": 51, "G5": 50}
    pages = [{"page_id": f"{s}.d.p{i}", "set": s, "core": True}
             for s, n in sizes.items() for i in range(n)]  # fmt: skip
    pages += [{"page_id": f"G5.x.p{i}", "set": "G5", "core": False} for i in range(93)]
    models, choices = {"m": {"role": "flagship", "provider": "a"}}, {"a": {"flagship": "R2"}}
    monkeypatch.setattr(plans, "config", {"models": models, "choices": choices}.__getitem__)
    monkeypatch.setattr(plans, "test_pages",
                        lambda guard=True: [p for p in pages if p["set"] != "G5"])  # fmt: skip
    monkeypatch.setattr(
        plans, "g5_pages", lambda guard=True: [p for p in pages if p["set"] == "G5"]
    )
    monkeypatch.setattr(plans, "_splits", lambda: {p["page_id"]: "test" for p in pages})
    monkeypatch.setattr(plans, "guard_test_pages", lambda pages: None)
    drawn = [i.page_id for i in plans.items("latency")]
    assert [sum(p.startswith(f"{s}.d.") for p in drawn) for s in sizes] == [7, 10, 7, 6]


def test_the_ablation_runs_low_effort_against_the_repeats_three_runs(arms):
    high = {f"small-a:R4:C2:r{n}" for n in (1, 2, 3)}
    assert arms("ablation") == high | {"small-a:R4:C2:low"}
    assert high <= arms("repeats")  # the same requests: the ablation reuses their answers
    assert plans.SHARED == ("repeats", "ablation")  # the ladder's R2 is sent with its rungs


def _dev_pool() -> list[dict]:
    """G0-dev (6 strike-free of 21, R6's example page among the struck), two G1-dev documents and
    G2-dev's six pages (three struck, three strike-free)."""

    def page(set_, doc, n, struck):
        words = [{"text": "word", "struck": [[0, 4]] if struck else [], "masked": False,
                  "furniture": False}]  # fmt: skip
        return {"page_id": f"{set_}.{doc}.p{n}", "set": set_, "doc": doc, "words": words}

    pages = [page("G0", f"dev-{i:03d}", 0, i not in (0, 5, 6, 14, 16, 18)) for i in range(21)]
    pages += [page("G1", "cppa", i, i % 3 != 0) for i in range(60)]
    pages += [page("G1", "epa", i, i % 3 != 0) for i in range(400)]
    pages += [page("G2", f"BILLS-{b}", n, n == 1) for b in "xyz" for n in (0, 1)]
    return pages


def test_the_dev_lists_draw_struck_pages_by_part_and_never_r6s_example(monkeypatch):
    pages = _dev_pool()
    monkeypatch.setattr(plans, "load_pages", lambda *s, split=None: pages)
    monkeypatch.setattr(plans, "seed", lambda name: 1)
    drawn = plans.draw_dev()
    assert {k: len(v) for k, v in drawn.items()} == {"pilot": 34, "ladder": 24, "repeats": 20,
                                                     "canary": 10}  # fmt: skip
    assert all(set(v) <= set(drawn["pilot"]) for v in drawn.values())
    assert plans.FEWSHOT_PAGE not in drawn["pilot"]
    kind = {p["page_id"]: plans.page_kind(p) for p in pages}

    def count(name: str, prefix: str, k: str) -> int:
        return sum(p.startswith(prefix) and kind[p] == k for p in drawn[name])

    parts = ("G0.", "G1.cppa", "G1.epa", "G2.")
    assert [count("pilot", x, "free") for x in parts] == [2, 1, 1, 1]
    assert [count("pilot", x, "struck") for x in parts] == [13, 7, 6, 3]
    assert [count("ladder", x, "struck") for x in parts] == [8, 4, 4, 3]
    assert not any(p.startswith("G2.") for p in drawn["repeats"] + drawn["canary"])
    assert all(kind[p] == "struck" for p in drawn["canary"])
    assert drawn == plans.draw_dev()  # seeded


def test_a_page_whose_only_strike_scores_nothing_is_neither_kind():
    lone = {"words": [{"text": "—", "struck": [[0, 1]], "masked": False, "furniture": False}]}
    assert plans.page_kind(lone) is None


def test_each_group_has_its_own_cap_and_a_quoted_one_is_refused():
    budget = {"cap_b2_usd": 120, "cap_b3_usd": "TBD (B2)", "reserve_usd": 50}
    assert plans.cap("smoke", budget) is None  # before any cap
    assert plans.cap("ladder", budget) == 120
    assert plans.cap("test", budget) is None  # not set yet: nothing in stage B3 can send
    budget["cap_b3_usd"] = 300
    assert (plans.cap("canary-4", budget), plans.cap("b4", budget)) == (300, 350)  # the reserve
    for wrong in ("120", float("nan"), float("inf"), -1):  # each would turn the cap off
        budget["cap_b2_usd"] = wrong
        with pytest.raises(ValueError, match="not a number of dollars"):
            plans.cap("pilot", budget)
    assert {plans.group(s) for s in plans.GROUPS["b3"]} == {"b3"}


@pytest.fixture
def frozen(monkeypatch, tmp_path):
    """freeze() with everything it reads stubbed to pass: each test breaks one thing."""
    prereg = tmp_path / "PREREG.md"
    prereg.write_text("`TBD (B1)` marks a gap. Seed published: only the sha256 of its decimal "
                      "string is published:\n  `" + "ab" * 32 + "`.\n", encoding="utf-8")  # fmt: skip
    models = {"m": {"id": "m-1", "expect": "m-1", "cutoff": "2026-06", "cutoff_source": "s",
                    "c1_route": "text"}}  # fmt: skip
    budget = {"cap_b2_usd": 120, "cap_b3_usd": "TBD (B2)", "reserve_usd": 50,
              "drop_order": list(plans.DROPS), "dropped": []}  # fmt: skip
    monkeypatch.setattr(plans, "PREREG", prereg)
    monkeypatch.setattr(plans, "config", {"models": models, "budget": budget}.__getitem__)
    monkeypatch.setattr(plans, "dirty", lambda exclude=(): "")
    monkeypatch.setattr(plans, "harness_commit", lambda exclude=(): "abc")
    monkeypatch.setattr(plans, "_seed_digest", lambda path: "ab" * 32)
    monkeypatch.setattr(plans, "RUNS", tmp_path)
    (tmp_path / "smoke").mkdir()
    ok = {"model": "m", "status": "ok", "reported_model": "m-1"}
    (tmp_path / "smoke" / "responses.jsonl").write_text(json.dumps(ok) + "\n", encoding="utf-8")
    monkeypatch.setattr(plans, "load_pages", lambda s: [])
    monkeypatch.setattr(plans, "dev_pages", lambda: {"pilot": ["G0.dev-000.p0"]})
    monkeypatch.setattr("bench.g2.cutoff", lambda: __import__("datetime").date(2026, 6, 30))
    monkeypatch.setattr("bench.prompts.hashes", lambda: {"R0.p0": "h"})
    return {"prereg": prereg, "models": models, "budget": budget, "smoke": ok, "tmp": tmp_path}


def test_freeze_writes_what_prereg_1_tags_once_nothing_is_missing(frozen):
    out = plans.freeze()
    assert out["commit"] == "abc" and len(out["prereg_sha256"]) == 64
    assert out["models"]["m"]["c1_route"] == "text" and out["budget"]["cap_b3_usd"] == "TBD (B2)"


def test_freeze_refuses_a_dirty_tree(frozen, monkeypatch):
    monkeypatch.setattr(plans, "dirty", lambda exclude=(): " M bench/report.py")
    with pytest.raises(ValueError, match="commit first"):
        plans.freeze()


def test_freeze_refuses_a_tbd_left_in_prereg(frozen):
    text = frozen["prereg"].read_text(encoding="utf-8")
    frozen["prereg"].write_text(text + "cap: TBD (B1).\n", encoding="utf-8")
    with pytest.raises(ValueError, match="1 TBD"):  # the backticked note is not a gap
        plans.freeze()
    frozen["prereg"].write_text(text + "cap: TBD (B1), and a wrapped one: TBD\n  (B1).\n",
                                encoding="utf-8")  # fmt: skip
    with pytest.raises(ValueError, match="2 TBD"):  # a line break doesn't hide one
        plans.freeze()


def test_freeze_refuses_a_model_no_smoke_answer_confirms(frozen):
    frozen["models"]["m"]["expect"] = "m-2"
    with pytest.raises(ValueError, match="no smoke-test answer reports"):
        plans.freeze()
    # served as m-2 while m-1 was pinned, then re-pinned to it: the harness counts that answer,
    # never sending the request again, so the freeze counts it too
    served = {"model": "m", "status": "model_mismatch", "answer_status": "ok",
              "reported_model": "m-2"}  # fmt: skip
    (frozen["tmp"] / "smoke" / "responses.jsonl").write_text(json.dumps(served) + "\n", "utf-8")
    assert plans.freeze()["models"]["m"]["expect"] == "m-2"


def test_freeze_records_g2_devs_sources_and_refuses_another_drop_order(frozen, monkeypatch):
    g2 = [{"page_id": "G2.T.p0", "set": "G2", "split": "test", "doc": "T", "pdf_sha256": "a",
           "xml_sha256": "b"},
          {"page_id": "G2.D.p0", "set": "G2", "split": "dev", "doc": "D", "pdf_sha256": "c",
           "xml_sha256": "d"}]  # fmt: skip
    monkeypatch.setattr(plans, "load_pages", lambda s: g2 if s == "G2" else [])
    out = plans.freeze()
    assert (out["g2_bills"], out["g2_dev_bills"]) == ({"T": ["a", "b"]}, {"D": ["c", "d"]})
    frozen["budget"]["drop_order"] = ["test:C1 mark", "test:C1prime", "fable", "qa"]  # a typo
    with pytest.raises(ValueError, match="not PREREG's"):
        plans.freeze()


def test_freeze_refuses_another_held_back_seed(frozen, monkeypatch):
    monkeypatch.setattr(plans, "_seed_digest", lambda path: "cd" * 32)
    with pytest.raises(ValueError, match="held-back seed"):
        plans.freeze()


@pytest.mark.parametrize("value, match", [("TBD (B1)", "must be set"), ("120", "not a number")])
def test_freeze_refuses_a_cap_not_set_or_quoted(frozen, value, match):
    frozen["budget"]["cap_b2_usd"] = value
    with pytest.raises(ValueError, match=match):
        plans.freeze()


def test_freeze_names_every_value_still_tbd():
    frozen = {"models": {"m": {"expect": "TBD (B1)", "id": "m-1"}}, "budget": {"cap_usd": "TBD"},
              "dev_pages": {"pilot": ["a", "TBD"]}, "choices": {"a": "TBD (B2)"}}  # fmt: skip
    assert list(plans._tbd(frozen)) == ["models.m.expect", "budget.cap_usd", "dev_pages.pilot[1]",
                                        "choices.a"]  # fmt: skip
    assert "choices.a" not in plans._tbd(frozen, later="TBD (B2)")  # prereg-2 fills it


def test_freeze_2_hashes_the_rules_and_the_test_labels_once_nothing_is_tbd(frozen, monkeypatch):
    tmp = frozen["tmp"]
    for name in ("G1-error-sample.jsonl", "G0-qa.jsonl", "G2-coverage.jsonl"):
        (tmp / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(plans, "GT", tmp)
    monkeypatch.setattr(plans, "PREREG1", tmp / "prereg1.json")
    monkeypatch.setattr(plans, "PREREG2", tmp / "prereg2.json")
    monkeypatch.setattr(plans, "_file_hashes", lambda: {"bench/report.py": "r"})
    monkeypatch.setattr(plans, "ground_truth_hashes", lambda: {"G2.b.p0": "g"})
    choices = {"a": {"rbest": "R2", "runner_up": "none", "image": "C2", "flagship": "R2"}}
    config = {"models": frozen["models"], "budget": frozen["budget"], "choices": choices}
    monkeypatch.setattr(plans, "config", config.__getitem__)
    with pytest.raises(FileNotFoundError, match="prereg-1's lists first"):
        plans.freeze_2()
    (tmp / "prereg1.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="cap_b3_usd must be set"):
        plans.freeze_2()
    frozen["budget"]["cap_b3_usd"] = "300"
    with pytest.raises(ValueError, match="not a number of dollars"):
        plans.freeze_2()
    monkeypatch.setattr("bench.harness.spent", lambda: 0.65)
    frozen["budget"]["cap_b3_usd"] = 0.01  # a typo: less than is already spent
    with pytest.raises(ValueError, match="not above the \\$0.65 already spent"):
        plans.freeze_2()
    frozen["budget"]["cap_b3_usd"] = 300
    out = plans.freeze_2()
    assert out["files"] == {"bench/report.py": "r"} and out["ground_truth"] == {"G2.b.p0": "g"}
    assert len(out["g1_error_sample"]) == 64 and out["choices"] == choices
    assert len(out["g2_coverage"]) == 64  # what the alignment rule kept
    choices["a"]["flagship"] = "TBD (B2)"
    with pytest.raises(ValueError, match="choices.a.flagship"):
        plans.freeze_2()
    choices["a"]["flagship"] = "R2"
    frozen["prereg"].write_text("R-best: TBD (B2).", encoding="utf-8")
    with pytest.raises(ValueError, match="1 TBD"):
        plans.freeze_2()


def test_the_rules_files_cover_the_code_prompts_config_and_environment():
    files = plans._file_hashes()
    for path in (
        "bench/report.py",
        "bench/stats.py",
        "bench/harness.py",
        "bench/score.py",
        "prompts/ladder.toml",
        "config/budget.toml",
        "PREREG.md",
        "requirements.txt",
    ):
        assert path in files
    assert not any("__pycache__" in f or f.endswith("prereg2.json") for f in files)
