"""The harness against the fake provider (PREREG.md, stage B1 step 9): caching, the model-ID
check, resubmission, batches, shared answers, the dry run and the free arms' cost."""

import json
import math
import random
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from bench import harness, plans, report
from bench.providers import Part, PartialBatch, Reply, fake

PAGE = "G0.dev-000.p0"


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A one-page stage on one fake model; returns a function setting the model's options."""
    fake.reset()
    model = {"provider": "fake", "id": "fake-1", "expect": "fake-1", "role": "flagship",
             "batch": False}  # fmt: skip
    prices = {"fake": {"input": 2.0, "output": 10.0, "batch": 0.5, "flex": 0.5}}
    budget = {"cap_b2_usd": 1.0, "cap_b3_usd": 1.0, "reserve_usd": 0}
    cfg = {"models": {"fake": model}, "prices": prices, "budget": budget}
    rungs = {"cells": [plans.Cell("fake", "R0", "C2")]}
    monkeypatch.setattr(harness, "RUNS", tmp_path)
    monkeypatch.setattr(harness, "config", cfg.__getitem__)
    monkeypatch.setattr(harness, "_run_meta", lambda: {"commit": "test"})
    monkeypatch.setattr(harness, "tags_in_force", lambda: ["prereg-1", "prereg-2"])  # both tagged
    monkeypatch.setattr(harness.render, "manifest", lambda: {})
    monkeypatch.setattr(harness.render, "require_current", lambda pages: None)
    monkeypatch.setattr(harness.arms, "cached", lambda path=None: {})
    words = [{"text": "page", "struck": [], "masked": False, "furniture": False}]
    page = {"page_id": PAGE, "set": "G0", "split": "dev", "cluster": PAGE, "words": words}
    monkeypatch.setattr(harness, "load_pages", lambda *sets, split=None: [page])
    monkeypatch.setattr(harness.plans, "items",
                        lambda stage, guard=True: [plans.Item(c, PAGE) for c in rungs["cells"]])  # fmt: skip
    # the cap's guard has its own tests; elsewhere a stage sends without usage to price from
    monkeypatch.setattr(harness, "_check_cap", lambda stage, metas, seen=None: None)
    monkeypatch.setattr(harness, "_room", lambda stage, mine, ahead, seen=None, over=0.0: None)
    monkeypatch.setattr(harness, "_prompt",
                        lambda m, item: (None, (Part("text", text=f"{item.cell.rung} page"),)))  # fmt: skip

    def configure(cells=None, **options):
        model.update(options)
        if cells:
            rungs["cells"] = cells
        return model

    return configure


def responses(stage="pilot"):
    return harness.attempts(stage)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(harness, "BACKOFF_S", 0.0)


def test_a_rerun_never_calls_the_api(run):
    run()
    assert harness.submit("pilot") == 1
    assert harness.submit("pilot") == 0
    assert len(fake.CALLS) == 1
    (atts,) = responses().values()
    assert atts[0]["status"] == "ok" and atts[0]["reported_model"] == "fake-1"


def test_request_errors_are_resubmitted_three_times(run):
    run(fail_times=2)
    harness.submit("pilot")
    (atts,) = responses().values()
    assert [a["status"] for a in atts] == ["request_error", "request_error", "ok"]


def test_a_fourth_request_error_is_final(run):
    run(fail_times=4)
    harness.submit("pilot")
    (atts,) = responses().values()
    assert len(atts) == harness.MAX_ATTEMPTS and harness.is_final(atts)
    assert harness.submit("pilot") == 0 and len(fake.CALLS) == harness.MAX_ATTEMPTS
    (row,) = harness.score("pilot", partial=True)  # partial: the fixture runs no free arms
    assert row["failed"] and row["status"] == "request_error"


def test_a_rate_limit_is_not_an_attempt_and_waits_for_the_next_run(run):
    run(fail_times=99, fail_error="RateLimitError: 429 too many requests")
    harness.submit("pilot")
    (atts,) = responses().values()
    assert len(atts) == harness.MAX_TRANSIENT + 1 and {a["status"] for a in atts} == {"transient"}
    assert not harness.is_final(atts)  # still pending once the limit lifts
    run(fail_times=0)
    fake.reset()
    assert harness.submit("pilot") == 1 and responses()[atts[0]["custom_id"]][-1]["status"] == "ok"


@pytest.mark.parametrize("error", [
    "APIConnectionError: down",  # a request error
    "AuthenticationError: Error code: 401 - invalid api key",  # waiting won't fix these
    "RateLimitError: Error code: 429 - {'error': {'code': 'insufficient_quota'}}",
])  # fmt: skip
def test_errors_in_a_row_stop_only_the_failing_model(run, monkeypatch, error):
    cells = [plans.Cell("fake", f"R{i}", "C2") for i in range(4)]
    run(cells=cells, fail_times=99, fail_error=error)
    monkeypatch.setattr(harness, "TRIP", 3)
    with pytest.raises(harness.ProviderDown):
        harness.submit("pilot", workers=1)
    assert sum(len(a) for a in responses().values()) == 3
    assert all(not harness.is_final(a) for a in responses().values())


def test_a_capacity_limit_waits_and_never_stops_the_model(run, monkeypatch):
    cells = [plans.Cell("fake", f"R{i}", "C2") for i in range(4)]
    run(
        cells=cells, fail_times=99, fail_error="RateLimitError: Error code: 429 - too many requests"
    )
    monkeypatch.setattr(harness, "TRIP", 3)
    assert harness.submit("pilot", workers=1) == 4  # every request tried, none stopped
    atts = responses().values()
    assert all(len(a) == harness.MAX_TRANSIENT + 1 for a in atts)  # then left for the next run
    assert all(not harness.is_final(a) for a in atts)


def test_a_failing_model_leaves_the_others_sending(run, monkeypatch):
    cells = [plans.Cell(m, f"R{i}", "C2") for m in ("fake", "good") for i in range(3)]
    run(cells=cells, fail_times=99, fail_error="APIConnectionError: down")
    harness.config("models")["good"] = {"provider": "fake", "id": "good-1", "expect": "good-1",
                                        "role": "ladder", "batch": False}  # fmt: skip
    monkeypatch.setattr(harness, "TRIP", 2)
    with pytest.raises(harness.ProviderDown, match="fake"):
        harness.submit("pilot", workers=1)
    done = [a[-1]["status"] for a in responses().values() if a[-1]["model"] == "good"]
    assert done == ["ok"] * 3


@pytest.mark.parametrize("error", [
    "BadRequestError: Error code: 400 - {'error': {'type': 'invalid_request_error', 'message': "
    "'You have reached your specified API usage limits. You will regain access on 2026-11-01'}}",
    "RateLimitError: Error code: 429 - {'error': {'code': 'insufficient_quota'}}",
    "ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'status': 'RESOURCE_EXHAUSTED'}}",
    "PermissionDeniedError: Error code: 403 - your team has no credits left",
    "AuthenticationError: Error code: 401 - invalid api key",
])  # fmt: skip
def test_provider_limits_and_keys_leave_a_request_pending(error):
    assert harness.TRANSIENT.search(error)


@pytest.mark.parametrize(("error", "account"), [
    ("HTTPStatusError: 401", True),  # a status code alone: waiting won't fix a key
    ("Error code: 402 - payment required", True),
    ("HTTPStatusError: 403 for url https://api.x.ai/v1/responses", True),
    ("RateLimitError: Error code: 429 - too many requests", False),  # capacity: wait longer
    ("RateLimitError: Error code: 429 - retry in 4031 ms", False),  # 403 inside a number
    # Azure's ordinary rate limit links its quota-increase form: capacity, not the account
    ("RateLimitError: Error code: 429 - {'error': {'code': '429', 'message': 'Requests to the "
     "Responses_Create Operation under Azure OpenAI API version v1 have exceeded token rate "
     "limit of your current AIServices S0 pricing tier. Please retry after 6 seconds. Please go "
     "here: https://aka.ms/oai/quotaincrease if you would like to further increase the default "
     "rate limit.'}}", False),
    ("RateLimitError: Error code: 429 - {'error': {'message': 'You exceeded your current quota, "
     "please check your plan and billing details.', 'code': 'insufficient_quota'}}", True),
])  # fmt: skip
def test_account_errors_are_told_from_capacity_limits(error, account):
    assert harness.TRANSIENT.search(error)
    assert bool(harness.ACCOUNT.search(error)) is account


@pytest.mark.parametrize(
    "error",
    [
        "APIConnectionError: Connection error.",
        "BadRequestError: Error code: 400 - the image is too large",
    ],
)
def test_other_errors_count_as_attempts(error):
    assert not harness.TRANSIENT.search(error)


def test_a_blocked_prompt_reporting_no_usage_costs_nothing(run):
    model, meta = run(), {"custom_id": "c", "model": "fake", "rung": "R0"}
    assert harness.record("pilot", model, Reply("c", "refusal"), meta, 1)["cost_usd"] == 0.0
    assert harness.record("pilot", model, Reply("c", "request_error"), meta, 1)["cost_usd"] is None


def test_a_batch_failing_part_way_is_collected_not_sent_again(run):
    run(cells=[plans.Cell("fake", f"R{i}", "C2") for i in range(3)], batch=True,
        fail_batch_after=1)  # fmt: skip
    with pytest.raises(PartialBatch):
        harness.submit("pilot")
    (batch,) = harness.open_batches("pilot")
    assert len(batch["custom_ids"]) == 1
    run(fail_batch_after=99)
    assert harness.submit("pilot") == 2  # the two the first batch never took
    assert harness.collect("pilot") == 2 and len(fake.CALLS) == 3


def test_the_status_reported_is_what_the_scorer_reads_now(run):
    run(fake_text="** --- **")  # markup alone: recorded as an answer, read as nothing
    harness.submit("pilot")
    (atts,) = responses().values()
    assert atts[-1]["status"] == "ok"
    (row,) = harness.score("pilot", partial=True)
    assert row["status"] == "parse_error" and row["failed"]


def test_a_model_id_mismatch_stops_the_stage_until_resolved(run):
    run(reported_model="fake-2")
    with pytest.raises(harness.ModelMismatch):
        harness.submit("pilot")
    (atts,) = responses().values()
    assert atts[-1]["status"] == "model_mismatch"
    for again in (lambda: harness.submit("pilot"), lambda: harness.score("pilot")):
        with pytest.raises(harness.ModelMismatch):
            again()
    assert len(fake.CALLS) == 1
    run(expect="fake-2")  # re-pinned to what was served: the billed answer counts, unsent again
    assert harness.submit("pilot") == 0 and len(fake.CALLS) == 1
    assert responses()[atts[0]["custom_id"]][-1]["status"] == "ok"
    (row,) = harness.score("pilot", partial=True)
    assert row["status"] == "ok" and not row["failed"]


def test_a_refusal_naming_no_model_is_not_a_mismatch(run):
    run(status="refusal", reported_model=None)
    harness.submit("pilot")
    (atts,) = responses().values()
    assert atts[-1]["status"] == "refusal"


def test_the_paid_stages_wait_for_their_tags(run, monkeypatch):
    run()
    monkeypatch.setattr(harness, "tags_in_force", lambda: [])
    with pytest.raises(PermissionError, match="pilot waits for the prereg-1 tag"):
        harness.submit("pilot")
    assert harness.submit("smoke") == 1  # the smoke tests come before either tag
    monkeypatch.setattr(harness, "tags_in_force", lambda: ["prereg-1"])
    assert harness.submit("pilot") == 1
    for stage in ("canary-1", "b4"):  # a canary opens a model's window: not before prereg-2
        with pytest.raises(PermissionError, match="waits for the prereg-2 tag"):
            harness.submit(stage)
    assert len(fake.CALLS) == 2


def test_an_unpinned_model_is_refused_outside_the_smoke_test(run):
    run(expect="TBD (B1)")
    with pytest.raises(ValueError, match="no pinned model ID"):
        harness.submit("pilot")
    assert harness.submit("smoke") == 1


def test_a_json_answer_that_fails_to_parse_is_final(run):
    run(cells=[plans.Cell("fake", "R4", "C2")], fake_text="not json")
    harness.submit("pilot")
    (atts,) = responses().values()
    assert atts[-1]["status"] == "parse_error" and len(fake.CALLS) == 1
    (row,) = harness.score("pilot", partial=True)
    assert row["failed"] and row["live_absent"] == 1  # worst case


@pytest.mark.parametrize("status", ["length", "refusal"])
def test_an_output_limit_stop_or_a_refusal_scores_worst_case(run, status):
    run(status=status)
    harness.submit("pilot")
    (row,) = harness.score("pilot", partial=True)
    assert row["failed"] and row["status"] == status and row["live_absent"] == 1


def test_an_empty_answer_is_a_final_failure(run):
    run(fake_text="  ")
    harness.submit("pilot")
    (atts,) = responses().values()
    assert atts[-1]["status"] == "parse_error"
    (row,) = harness.score("pilot", partial=True)
    assert row["failed"]


def test_batches_are_submitted_then_collected_once(run):
    run(batch=True)
    harness.submit("pilot")
    assert not responses() and len(harness.open_batches("pilot")) == 1
    assert harness.submit("pilot") == 0  # in flight, not pending
    assert harness.collect("pilot") == 1
    assert not harness.open_batches("pilot")
    (atts,) = responses().values()
    assert atts[0]["batch_id"].startswith("fake-batch-") and atts[0]["mode"] == "batch"
    assert harness.submit("pilot") == 0 and len(fake.CALLS) == 1


def test_a_collect_rerun_after_a_crash_records_nothing_twice(run, tmp_path):
    run(batch=True)
    harness.submit("pilot")
    (batch,) = harness.open_batches("pilot")
    harness.collect("pilot")
    # the crash: replies recorded, the batch never marked collected
    lines = (tmp_path / "pilot" / "batches.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "pilot" / "batches.jsonl").write_text(lines[0] + "\n", encoding="utf-8")
    assert harness.collect("pilot") == 1
    (atts,) = responses().values()
    assert len(atts) == 1


def test_batches_split_by_size(run, monkeypatch):
    cells = [plans.Cell("fake", f"R{i}", "C2") for i in range(4)]
    run(cells=cells, batch=True)
    monkeypatch.setattr(harness, "BATCH_REQUESTS", 3)
    harness.submit("pilot")
    assert sorted(len(b["custom_ids"]) for b in harness.open_batches("pilot")) == [1, 3]


def test_a_held_lock_refuses_a_second_run(run, tmp_path):
    run()
    (tmp_path / "pilot").mkdir()
    (tmp_path / "pilot" / ".lock").write_text("123")
    (tmp_path / ".shared.lock").write_text("123")  # the repeats and the ablation share one lock
    for stage in ("pilot", "repeats", "ablation"):
        with pytest.raises(RuntimeError, match="another run"):
            harness.submit(stage)
    assert not fake.CALLS


def test_batch_cost_takes_the_batch_price(run):
    run(batch=True)
    harness.submit("pilot")
    harness.collect("pilot")
    (atts,) = responses().values()
    inp, out = harness.tokens(atts[0]["usage"])
    assert atts[0]["cost_usd"] == pytest.approx((inp * 2.0 + out * 10.0) / 1e6 * 0.5)


def test_input_written_to_a_prompt_cache_bills_at_its_multiplier(monkeypatch):
    monkeypatch.setattr(harness, "price", lambda model: {"input": 10.0, "output": 50.0,
                                                         "flex": 0.5, "cache_write": 1.25})  # fmt: skip
    usage = {"input_tokens": 4000, "input_tokens_details": {"cache_write_tokens": 3000},
             "output_tokens": 800, "total_tokens": 4800}  # fmt: skip
    usd = (1000 * 10.0 + 3000 * 12.5 + 800 * 50.0) / 1e6  # as Azure bills GPT-6
    assert harness.cost("gpt", usage) == pytest.approx(usd)
    assert harness.cost("gpt", usage, "flex") == pytest.approx(usd / 2)  # Flex halves writes too


def test_dry_run_prices_all_input_as_written_where_the_cache_bills_more(run):
    run(cells=[plans.Cell("fake", "R0", "C2")])
    harness.submit("smoke")
    harness.config("prices")["fake"]["cache_write"] = 1.25
    (atts,) = responses("smoke").values()
    inp, out = harness.tokens(atts[0]["usage"])
    assert harness.dry_run(["pilot"]) == pytest.approx((inp * 2.0 * 1.25 + out * 10.0) / 1e6)


def test_flex_is_billed_at_the_tier_served_and_timed_only_in_the_latency_sample(run):
    run(service_tier="flex")
    harness.submit("pilot")
    harness.submit("latency")  # one call at a time, at the standard tier
    (flex,) = responses("pilot").values()
    (timed,) = responses("latency").values()
    assert flex[0]["mode"] == "flex" and flex[0]["settings"]["service_tier"] == "flex"
    assert timed[0]["mode"] == "sequential" and "service_tier" not in timed[0]["settings"]
    inp, out = harness.tokens(flex[0]["usage"])
    assert flex[0]["cost_usd"] == pytest.approx((inp * 2.0 + out * 10.0) / 1e6 * 0.5)
    assert timed[0]["cost_usd"] == pytest.approx(2 * flex[0]["cost_usd"])
    run(cells=[plans.Cell("fake", "R0", "C2"), plans.Cell("fake", "R1", "C2")])
    assert harness.dry_run(["pilot"]) == pytest.approx(flex[0]["cost_usd"])  # R1, at Flex
    (row,) = harness.score("pilot", partial=True)
    assert row["latency_s"] is None  # Flex's queueing is not latency
    (row,) = harness.score("latency", partial=True)
    assert row["latency_s"] == 0.0


def test_the_latency_sample_runs_with_power_throttling_off(run, monkeypatch):
    run()
    calls = []
    monkeypatch.setattr(harness.arms, "full_speed", lambda: calls.append(True))
    harness.submit("pilot")
    assert not calls  # an untimed stage leaves the process as it was launched
    harness.submit("latency")
    assert calls == [True]


def test_dry_run_prices_pending_requests_from_usage_seen(run):
    run(cells=[plans.Cell("fake", "R0", "C2")])
    harness.submit("smoke")  # the usage the dry run prices from
    run(cells=[plans.Cell("fake", "R0", "C2"), plans.Cell("fake", "R1", "C2")])
    (atts,) = responses("smoke").values()
    inp, out = harness.tokens(atts[0]["usage"])
    one = (inp * 2.0 + out * 10.0) / 1e6
    assert harness.dry_run(["pilot"]) == pytest.approx(2 * one)  # R1 priced from R0's output
    assert harness.dry_run(["pilot"], smoke=True) == pytest.approx(
        2 * (inp * 2.0 + 2 * out * 10.0) / 1e6
    )


def test_a_rung_not_yet_answered_is_priced_from_r2_unless_it_answers_in_json(run):
    run(cells=[plans.Cell("fake", "R2", "C2")], fake_text="a short answer")
    harness.submit("smoke")
    run(cells=[plans.Cell("fake", "R5", "C2")], fake_text=" ".join(["word"] * 40))  # JSON, long
    harness.submit("smoke")
    inp = harness.tokens(next(iter(responses("smoke").values()))[0]["usage"])[0]
    run(cells=[plans.Cell("fake", "R1", "C2")])
    assert harness.dry_run(["pilot"]) == pytest.approx((inp * 2.0 + 3 * 10.0) / 1e6)  # R2's
    run(cells=[plans.Cell("fake", "R6", "C2")])
    pooled = harness._p95([3, 40])  # every rung's: R6 answers in JSON
    assert harness.dry_run(["pilot"]) == pytest.approx((inp * 2.0 + pooled * 10.0) / 1e6)


def test_a_model_s_prompt_cache_setting_spares_the_stages_sent_with_it_on(run):
    run(prompt_cache="off")
    harness.config("prices")["fake"]["cache_write"] = 1.25
    harness.submit("smoke")  # sent before the probe: as it was, cache on
    harness.submit("cache-probe")
    (cached,) = responses("smoke").values()
    (probe,) = responses("cache-probe").values()
    assert "prompt_cache" not in cached[0]["settings"]
    assert probe[0]["settings"]["prompt_cache"] == "off"
    assert probe[0]["custom_id"] != cached[0]["custom_id"]
    inp, out = harness.tokens(probe[0]["usage"])
    no_writes = (inp * 2.0 + out * 10.0) / 1e6
    assert harness.dry_run(["latency"]) == pytest.approx(no_writes)


def test_dry_run_refuses_an_unpriced_model(run, monkeypatch):
    run()
    harness.submit("smoke")
    monkeypatch.setattr(harness, "price", lambda model: None)
    with pytest.raises(ValueError, match="can't price"):
        harness.dry_run(["pilot"])


def test_dry_run_charges_searches_to_pdf_requests_alone(run, monkeypatch):
    run(cells=[plans.Cell("fake", "R0", "C1"), plans.Cell("fake", "R0", "C2")])
    harness.config("prices")["fake"]["per_search"] = 0.005
    monkeypatch.setattr(harness, "searches", lambda usage: 4)  # as xAI's PDF route reports

    def prompt(model, item):
        part = Part("pdf", path="p.pdf", sha256="x") if item.cell.input == "C1" else Part("text")
        return None, (part,)

    monkeypatch.setattr(harness, "_prompt", prompt)
    harness.submit("smoke")
    seen = [harness.tokens(a[0]["usage"]) for a in responses("smoke").values()]
    tokens_usd = sum((i * 2.0 + o * 10.0) / 1e6 for i, o in seen)
    assert harness.dry_run(["pilot"]) == pytest.approx(tokens_usd + 4 * 0.005)  # C1's searches


def test_dry_run_counts_what_is_spent_against_the_cap(run):
    run()
    harness.submit("smoke")
    harness.config("budget")["cap_b2_usd"] = harness.spent()  # nothing more fits
    with pytest.raises(ValueError, match="passes the"):
        harness.dry_run(["pilot"])


def test_a_free_arm_bills_the_cost_pass_and_reports_the_latency_pass(run, monkeypatch):
    run()
    cost = {"words": [["page", []]], "cpu_s": 1.8, "wall_s": 40.0, "pass": "cost"}
    timed = {"cpu_s": 9.0, "wall_s": 0.5, "pass": "latency"}
    monkeypatch.setattr(harness.arms, "cached", lambda path=None: {
        ("P-native", PAGE): timed if path == harness.arms.LATENCY else cost})  # fmt: skip
    (row,) = harness.score("pilot", partial=True)  # no vCPU price yet: unknown, never $0
    assert not row["priced"] and math.isnan(report.metrics([row])["usd_per_1000"])
    harness.config("prices")["cpu"] = {"usd_per_vcpu_hour": 0.04}
    (row,) = harness.score("pilot", partial=True)
    m = report.metrics([row])
    assert m["usd_per_1000"] == pytest.approx(1000 * 1.8 * 0.04 / 3600) and m["s_per_page"] == 0.5
    cost["workers"] = 6  # timed in a parallel run, sharing the machine: never billed
    (row,) = harness.score("pilot", partial=True)
    assert not row["priced"]
    del cost["workers"], cost["pass"]  # timed before the cost pass pinned onnxruntime: never billed
    (row,) = harness.score("pilot", partial=True)
    assert not row["priced"]


def test_a_page_the_package_refused_is_a_failed_page_of_its_own_kind(run, monkeypatch):
    run()
    refused = {"words": [], "cnn_prob": [], "status": "error", "pass": "cost",
               "error": "ValueError: Word.bbox must be normalized page fractions"}  # fmt: skip
    monkeypatch.setattr(harness.arms, "cached", lambda path=None: {
        ("P-scan", PAGE): refused} if path is None else {})  # fmt: skip
    (row,) = harness.score("pilot", partial=True)
    assert row["status"] == "error" and row["failed"]


def test_tokens_from_every_provider_shape():
    assert harness.tokens({"input_tokens": 10, "output_tokens": 5}) == (10, 5)
    assert harness.tokens({"input_tokens": 10, "output_tokens": 5, "total_tokens": 18}) == (10, 8)
    assert harness.tokens({"promptTokenCount": 10, "totalTokenCount": 25}) == (10, 15)
    assert harness.tokens(
        {"promptTokenCount": 10, "candidatesTokenCount": 3, "thoughtsTokenCount": 4}
    ) == (10, 7)
    assert harness.tokens({"prompt_tokens": 10, "total_tokens": 30, "completion_tokens": 12}) == (
        10,
        20,
    )
    assert harness.tokens(
        {"input_tokens": 3, "cache_read_input_tokens": 4, "output_tokens": 1}
    ) == (7, 1)


def test_stale_inputs_are_refused_before_anything_is_sent(run, monkeypatch):
    run()

    def stale(pages):
        raise ValueError(f"{len(pages)} pages' inputs are missing or stale")

    monkeypatch.setattr(harness.render, "require_current", stale)
    with pytest.raises(ValueError, match="stale"):
        harness.submit("pilot")
    assert not fake.CALLS


def test_the_ablation_reads_the_repeats_answers_to_identical_requests(run):
    run()
    assert harness.submit("repeats") == 1
    assert harness.submit("ablation") == 0 and len(fake.CALLS) == 1  # the repeats' answer stands
    assert harness.status("ablation") == {"fake": {"ok, from repeats": 1}}
    (row,) = harness.score("ablation", partial=True)
    assert row["reused_from"] == "repeats" and row["cost_usd"] == harness.spent() > 0
    for stage in ("pilot", "ladder", "canary-1", "latency", "smoke"):  # each sends its own
        assert harness.submit(stage) == 1
    assert len(fake.CALLS) == 6


def test_a_request_in_the_other_stages_batch_waits_for_it(run):
    run(batch=True)
    harness.submit("repeats")
    assert harness.submit("ablation") == 0
    assert harness.status("ablation") == {"fake": {"in batch (repeats)": 1}}
    harness.collect("repeats")
    assert harness.submit("ablation") == 0 and len(fake.CALLS) == 1


def test_dry_run_prices_a_shared_request_once(run):
    run()
    harness.submit("smoke")
    (atts,) = responses("smoke").values()
    inp, out = harness.tokens(atts[0]["usage"])
    one = (inp * 2.0 + out * 10.0) / 1e6
    assert harness.dry_run(["repeats", "ablation"]) == pytest.approx(one)
    assert harness.dry_run(["pilot", "ladder"]) == pytest.approx(2 * one)


def test_a_shared_answer_comes_only_from_the_model_now_pinned(run):
    run()
    harness.submit("repeats")  # answered by fake-1
    run(expect="fake-2")  # the provider moved on, and the pin with it
    assert harness.submit("ablation") == 1  # fake-1's answer is not borrowed
    (row,) = harness.score("ablation", partial=True)
    assert "reused_from" not in row


def test_either_sharing_stage_sends_both_stages_requests_together(run, monkeypatch):
    run()
    high = plans.Cell("fake", "R0", "C2", repeat=1)
    cells = {"repeats": [high], "ablation": [high, plans.Cell("fake", "R0", "C2", effort="low")]}
    monkeypatch.setattr(harness.plans, "items",
                        lambda stage, guard=True: [plans.Item(c, PAGE) for c in cells.get(stage, [high])])  # fmt: skip
    harness.submit("smoke")  # usage for the dry run to price from
    assert harness.dry_run(["repeats"]) == harness.dry_run(["repeats", "ablation"]) > 0
    assert harness.submit("repeats") == 2 and harness.submit("ablation") == 0
    assert harness.status("ablation") == {"fake": {"ok, from repeats": 2}}


def test_after_a_re_pin_the_sharing_stages_answer_again_from_the_model_now_pinned(run):
    run(cells=[plans.Cell("fake", "R0", "C2", repeat=1),
               plans.Cell("fake", "R0", "C2", effort="low")])  # fmt: skip
    assert harness.submit("ablation") == 2 and harness.submit("pilot") == 2  # fake-1 answers
    run(expect="fake-2")  # the provider moved on, and the pin with it
    assert harness.submit("ablation") == 2  # its own answers from fake-1 no longer count either
    assert all(a[-1]["reported_model"] == "fake-2" for a in responses("ablation").values())
    assert harness.submit("pilot") == 0  # a stage sharing nothing keeps them


def test_errors_naming_a_model_still_count_toward_the_limit_in_the_sharing_stages(run):
    run(fail_times=99, fail_model="some-version")  # Vertex names a model on a failed answer
    for _ in range(3):
        harness.submit("repeats")
    assert len(fake.CALLS) == harness.MAX_ATTEMPTS and harness.submit("repeats") == 0


def test_a_refusal_naming_no_model_is_sent_again_after_a_re_pin(run):
    run(status="refusal", reported_model=None)  # Azure's content filter names none
    assert harness.submit("repeats") == 1
    assert harness.submit("repeats") == 0  # final while its pin holds
    run(expect="fake-2")
    assert harness.submit("ablation") == 1  # sent under fake-1: sent again


def test_either_sharing_stage_collects_the_others_batches(run):
    run(batch=True)
    harness.submit("repeats")
    assert harness.collect("ablation") == 1  # the batch the repeats sent
    assert harness.status("ablation") == {"fake": {"ok, from repeats": 1}}
    assert harness.collect("repeats") == 0  # read once


def test_a_promotional_price_stops_sending_once_past(run, monkeypatch):
    run()
    harness.config("prices")["fake"]["until"] = "2000-01-01"
    for call in (lambda: harness.submit("pilot"), lambda: harness.dry_run(["pilot"])):
        with pytest.raises(ValueError, match="expired for fake"):
            call()
    assert not fake.CALLS


def test_a_requests_attempts_in_both_sharing_stages_count_toward_its_limit(run, monkeypatch):
    run(fail_times=99, fail_error="APIConnectionError: down")
    monkeypatch.setattr(harness, "TRIP", 3)  # the repeats stop after three errors in a row
    with pytest.raises(harness.ProviderDown):
        harness.submit("repeats")
    assert harness.submit("ablation") == 1 and len(fake.CALLS) == 4  # one attempt was left
    (atts,) = responses("ablation").values()
    assert atts[0]["attempt"] == 4
    assert harness.submit("repeats") == 0  # four attempts in all: a final failure
    (row,) = harness.score("repeats", partial=True)
    assert row["failed"] and row["status"] == "request_error" and row["reused_from"] == "ablation"


def test_a_mismatch_in_one_sharing_stage_stops_the_other(run):
    run(reported_model="fake-2")
    with pytest.raises(harness.ModelMismatch):
        harness.submit("repeats")
    calls = (lambda: harness.submit("ablation"), lambda: harness.score("ablation", partial=True))
    for again in calls:
        with pytest.raises(harness.ModelMismatch, match=r"\(in repeats\)"):
            again()
    assert len(fake.CALLS) == 1


def test_the_words_p_scan_leaves_uncertain_reach_every_arms_score(run, monkeypatch):
    run(fake_text="page")
    pscan = {"words": [["page", []]], "cnn_prob": [0.5], "cpu_s": 1.0, "pass": "cost",
             "package": {"model_meta": {"p_lo": 0.15, "p_hi": 0.85}}}  # fmt: skip
    monkeypatch.setattr(harness.arms, "cached", lambda path=None: {
        ("P-scan", PAGE): pscan} if path is None else {})  # fmt: skip
    harness.submit("pilot")
    model, free = harness.score("pilot", partial=True)
    assert free["band"] == [0.15, 0.85] and free["probs"] == [[0, 0, 0.5]]
    word = {"live": 1, "live_live": 1}
    bands = ("uncertain", "uncertain 0.05-0.95", "uncertain 0.30-0.70", "uncertain 0.40-0.60")
    for row in (model, free):  # 0.5 lies inside the shipped band and every band of the grid
        assert row["subsets"] == dict.fromkeys(bands, word)
    pscan["cnn_prob"] = [0.85]  # decided at the shipped band's p_hi, and outside the narrower ones
    model, _ = harness.score("pilot", partial=True)
    assert model["subsets"] == {"uncertain": {}, "uncertain 0.05-0.95": word,
                                "uncertain 0.30-0.70": {}, "uncertain 0.40-0.60": {}}  # fmt: skip


@pytest.fixture
def guarded(run, monkeypatch):
    """The fixture's stage with the cap's guard in place, priced from a smoke test's usage."""
    monkeypatch.setattr(harness, "_check_cap", CHECK_CAP)
    monkeypatch.setattr(harness, "_room", ROOM)
    run()
    harness.submit("smoke")  # exempt from the cap: it is what later stages are priced from
    (atts,) = responses("smoke").values()
    inp, out = harness.tokens(atts[0]["usage"])
    return (inp * 2.0 + out * 10.0) / 1e6  # one request's price


CHECK_CAP, ROOM = harness._check_cap, harness._room


def test_the_cap_is_checked_before_anything_is_sent(guarded, run):
    budget = harness.config("budget")
    budget["cap_b2_usd"] = harness.spent() + 0.5 * guarded  # half a request's room left
    with pytest.raises(ValueError, match="passes the"):
        harness.submit("pilot")
    assert len(fake.CALLS) == 1  # the smoke test's alone
    budget["cap_b2_usd"] = harness.spent() + 1.5 * guarded
    assert harness.submit("pilot") == 1


def test_a_flex_request_sent_at_the_standard_tier_is_priced_at_full_price(guarded, run, tmp_path):
    run(service_tier="flex")
    day0 = (datetime.now(UTC) - timedelta(days=6)).isoformat()
    canary = {"model": "fake", "started": day0, "status": "ok", "custom_id": "c", "cost_usd": 0.0}
    (tmp_path / "canary-1").mkdir()
    (tmp_path / "canary-1" / "responses.jsonl").write_text(json.dumps(canary) + "\n", "utf-8")
    budget = harness.config("budget")
    budget["cap_b3_usd"] = harness.spent() + 0.75 * guarded  # room at Flex's price, not at full
    with pytest.raises(ValueError, match="passes the"):
        harness.submit("test", standard_tier=True)
    budget["cap_b3_usd"] = harness.spent() + 1.25 * guarded
    assert harness.submit("test", standard_tier=True) == 1


def test_a_cap_not_set_or_quoted_refuses_to_send(guarded):
    budget = harness.config("budget")
    budget["cap_b2_usd"] = "TBD (B1)"
    with pytest.raises(ValueError, match="not set"):
        harness.submit("pilot")
    budget["cap_b2_usd"] = "340"  # a quoted number would turn the cap off: refused
    with pytest.raises(ValueError, match="not a number"):
        harness.submit("pilot")
    assert len(fake.CALLS) == 1


def test_batches_in_flight_in_any_stage_count_against_the_cap(guarded, run):
    run(batch=True)  # each request in a batch, at half price
    budget = harness.config("budget")
    budget["cap_b2_usd"] = harness.spent() + 0.75 * guarded
    assert harness.submit("ladder") == 1  # in flight, so not yet spent
    with pytest.raises(ValueError, match="passes the"):
        harness.submit("pilot")  # its half and the ladder's half in flight: no room
    with pytest.raises(ValueError, match="passes the"):
        harness.dry_run(["pilot"])  # the dry run counts the ladder's batch too
    budget["cap_b2_usd"] = harness.spent() + 1.1 * guarded
    assert harness.dry_run(["pilot"]) == pytest.approx(0.5 * guarded)  # the stage's own total


def test_a_model_sent_request_by_request_stops_when_the_cap_leaves_no_room(guarded, run):
    answer = " ".join(["word"] * 300)  # each answer costs far more than its price
    run(cells=[plans.Cell("fake", r, "C2") for r in ("R0", "R1", "R2", "R3")], fake_text=answer)
    budget = harness.config("budget")
    budget["cap_b2_usd"] = harness.spent() + 4.5 * guarded  # room for the four at their price
    with pytest.raises(harness.CapReached, match="fake: the cap in force leaves no room"):
        harness.submit("pilot", workers=1)
    assert len(responses()) == 1 and len(fake.CALLS) == 2  # one sent, after the smoke test's
    with pytest.raises(ValueError, match="passes the"):
        harness.submit("pilot")  # the other three wait, and no room is left for them


def second_model(**options):
    """A second fake model, `fake2`, priced like the first."""
    harness.config("models")["fake2"] = {"provider": "fake", "id": "fake-2", "expect": "fake-2",
                                         "role": "flagship", "batch": False, **options}  # fmt: skip
    harness.config("prices")["fake2"] = harness.config("prices")["fake"]


def slow(monkeypatch, model, seconds):
    """Every send to `model` takes `seconds`, so a pool's attempts are all out at once."""
    send = fake.Fake.send
    monkeypatch.setattr(fake.Fake, "send", lambda self, req: (
        time.sleep(seconds) if req.model == model else None) or send(self, req))  # fmt: skip


@pytest.mark.parametrize("in_turn", [False, True])
def test_an_overrun_never_takes_the_price_set_aside_for_the_other_models(guarded, run, in_turn):
    second_model()
    run(cells=[plans.Cell("fake2", "R0", "C2")])
    harness.submit("smoke")  # fake2's own usage, priced like fake's
    cells = [plans.Cell("fake", "R0", "C2"), plans.Cell("fake", "R1", "C2"),
             plans.Cell("fake2", "R0", "C2")]  # fmt: skip
    run(cells=cells, fake_text="one two three four five")  # fake's answers: 1.6 times the price
    harness.config("budget")["cap_b2_usd"] = harness.spent() + 3.2 * guarded
    with pytest.raises(harness.CapReached, match="fake: "):
        harness.submit("pilot", workers=1, in_turn=in_turn)
    # fake's second request would fit in the cap's room, but only by taking fake2's price
    sent = sorted((a[0]["model"], a[0]["rung"]) for a in responses().values())
    assert sent == [("fake", "R0"), ("fake2", "R0")]


@pytest.mark.parametrize("in_turn", [False, True])
@pytest.mark.parametrize("workers", [1, 8])
def test_attempts_already_out_pass_the_cap_rather_than_take_a_later_models_price(
    guarded, run, monkeypatch, workers, in_turn
):
    second_model()
    run(cells=[plans.Cell("fake2", "R0", "C2")])
    harness.submit("smoke")
    # with 8 workers, all 8 of fake's attempts are out before any of them comes back
    cells = [plans.Cell("fake", "R0", "C2", repeat=i) for i in range(1, workers + 1)]
    cells += [plans.Cell("fake2", "R0", "C2", repeat=i) for i in range(1, 5)]
    run(cells=cells, fake_text="one two three four five")  # fake's answers: 1.6 times the price
    spent = harness.spent()
    harness.config("budget")["cap_b2_usd"] = spent + (workers + 4) * guarded * (1 + 1e-9)
    slow(monkeypatch, "fake", 0.5 if workers > 1 else 0)
    # fake2's four all went
    assert harness.submit("pilot", workers=workers, in_turn=in_turn) == workers + 4
    sent = [a[-1]["model"] for a in responses().values()]
    assert sent.count("fake") == workers and sent.count("fake2") == 4
    assert harness.spent() - spent > (workers + 4) * guarded  # past the cap by fake's overrun


def test_a_mismatch_still_out_when_the_cap_stops_a_model_stops_the_run(run, monkeypatch):
    run(cells=[plans.Cell("fake", r, "C2") for r in ("R0", "R1")])

    def send(stage, provider, req, meta, done, stop, trip, room=None):
        if meta["rung"] == "R1":
            raise harness.CapReached("fake: no room")
        time.sleep(0.3)  # still out when R1 finds no room
        raise harness.ModelMismatch("fake: reported another model")

    monkeypatch.setattr(harness, "_send", send)
    with pytest.raises(harness.ModelMismatch):
        harness.submit("pilot", workers=2)


def test_a_calls_own_answers_never_reprice_what_is_in_flight(run, monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "_check_cap", CHECK_CAP)
    monkeypatch.setattr(harness, "_room", ROOM)
    twenty = " ".join(["w"] * 20)
    run(batch=True, fake_text=twenty)
    second_model(batch=True, fake_text=twenty)
    for model in ("fake", "fake2"):  # usage to price from: the 95th percentile of [1, 21] is 20
        for k, out in enumerate((1, 21)):
            harness.append_jsonl(tmp_path / "seed" / "responses.jsonl", {
                "custom_id": f"seed-{model}-{k}", "status": "ok", "model": model, "rung": "R0",
                "input": "C2", "page_id": PAGE, "fewshot": False, "cost_usd": 0.0,
                "usage": {"input_tokens": 2, "output_tokens": out}})  # fmt: skip
    cells = {"test": [plans.Cell("fake", "R0", "C2", repeat=i) for i in range(1, 41)],
             "latency": [plans.Cell("fake", "R0", "C2"), plans.Cell("fake2", "R0", "C2")]}  # fmt: skip
    monkeypatch.setattr(harness.plans, "items",
                        lambda stage, guard=True: [plans.Item(c, PAGE) for c in cells[stage]])  # fmt: skip
    budget = harness.config("budget")
    budget["cap_b3_usd"] = 1.0
    harness.submit("test")  # 40 of fake's requests in a batch, not yet collected
    _, meta = harness.build("latency")
    by, _ = harness._priced([("latency", m) for m in meta.values()] + harness._in_flight_meta())
    one = (2 * 2.0 + 20 * 10.0) / 1e6  # a latency request's price, and what each one costs
    budget["cap_b3_usd"] = harness.spent() + sum(by.values()) + 0.5 * one
    # fake's answer moves its 95th percentile, which must not reprice the batch in flight mid-call
    assert harness.submit("latency") == 2


def test_spending_from_outside_the_call_is_never_handed_on_as_an_overrun(
    guarded, run, monkeypatch, tmp_path
):
    second_model()
    harness.config("models")["fake3"] = harness.config("models")["fake2"] | {
        "id": "fake-3", "expect": "fake-3"}  # fmt: skip
    harness.config("prices")["fake3"] = harness.config("prices")["fake"]
    run(cells=[plans.Cell(m, "R0", "C2") for m in ("fake2", "fake3")])
    harness.submit("smoke")  # their own usage, priced like fake's
    run(cells=[plans.Cell(m, "R0", "C2") for m in ("fake", "fake2", "fake3")])
    harness.config("budget")["cap_b2_usd"] = harness.spent() + 3 * guarded * (1 + 1e-9)  # all fit
    send = fake.Fake.send

    def meanwhile(self, req):  # another stage's batch, collected while fake is out, costs 2 more
        if req.model == "fake":
            harness.append_jsonl(tmp_path / "ladder" / "responses.jsonl", {
                "custom_id": "x", "status": "ok", "model": "fake", "cost_usd": 2 * guarded})  # fmt: skip
        return send(self, req)

    monkeypatch.setattr(fake.Fake, "send", meanwhile)
    with pytest.raises(harness.CapReached):
        harness.submit("pilot", workers=1, in_turn=True)  # in turn: each room set as it starts
    # fake2 starts with no room, and must not hand that on to fake3 as an overrun of its own
    assert [a[-1]["model"] for a in responses().values()] == ["fake"]


@pytest.mark.parametrize(("stage", "in_turn", "most"), [
    ("pilot", False, 2),  # the two models at once, each one call at a time
    ("pilot", True, 1),  # one model, then the other
    ("latency", False, 1),  # timed: one call at a time across every model, whatever is asked
])  # fmt: skip
def test_the_models_sent_request_by_request_go_at_once_but_the_latency_sample_one_call_at_a_time(
    run, monkeypatch, stage, in_turn, most
):
    second_model()
    run(cells=[plans.Cell(m, r, "C2") for m in ("fake", "fake2") for r in ("R0", "R1")])
    out, guard, send = {"now": 0, "most": 0}, threading.Lock(), fake.Fake.send

    def counted(self, req):
        with guard:
            out["now"] += 1
            out["most"] = max(out["most"], out["now"])
        time.sleep(0.1)
        with guard:
            out["now"] -= 1
        return send(self, req)

    monkeypatch.setattr(fake.Fake, "send", counted)
    assert harness.submit(stage, workers=1, in_turn=in_turn) == 4
    assert out["most"] == most


def test_in_turn_sends_the_models_one_after_another(run):
    second_model()
    run(cells=[plans.Cell(m, r, "C2") for m in ("fake", "fake2") for r in ("R0", "R1", "R2")])
    _, meta = harness.build("pilot")
    assert harness.submit("pilot", in_turn=True) == 6
    assert [meta[c]["model"] for c in fake.CALLS] == ["fake"] * 3 + ["fake2"] * 3


def test_a_mismatch_in_one_model_stops_the_models_sent_with_it(run, monkeypatch):
    second_model()
    cells = [plans.Cell("fake", "R0", "C2")]
    cells += [plans.Cell("fake2", "R0", "C2", repeat=i) for i in range(1, 21)]
    run(cells=cells, reported_model="fake-9")  # fake's answer names another model
    slow(monkeypatch, "fake2", 0.05)
    with pytest.raises(harness.ModelMismatch):
        harness.submit("pilot", workers=1)
    sent = [a[-1]["model"] for a in responses().values()]
    assert sent.count("fake") == 1 and sent.count("fake2") < 20


def test_ctrl_c_stops_every_model_once_the_attempts_out_have_finished(run, monkeypatch):
    second_model()
    cells = [plans.Cell(m, "R0", "C2", repeat=i) for m in ("fake", "fake2") for i in range(1, 11)]
    run(cells=cells)
    send, real = fake.Fake.send, harness.as_completed
    monkeypatch.setattr(fake.Fake, "send", lambda self, req: time.sleep(0.05) or send(self, req))

    def interrupted(jobs):
        for i, job in enumerate(real(jobs)):
            if i == 2:
                raise KeyboardInterrupt
            yield job

    monkeypatch.setattr(harness, "as_completed", interrupted)
    with pytest.raises(KeyboardInterrupt):
        harness.submit("pilot", workers=1)
    assert len(fake.CALLS) < 20  # the rest never went
    # nothing was still out when `submit` raised: every call made has its attempt recorded
    assert len(fake.CALLS) == sum(len(a) for a in responses().values())


def test_a_provider_that_fails_to_start_stops_the_call_before_any_model_sends(run, monkeypatch):
    second_model()
    run(cells=[plans.Cell(m, "R0", "C2") for m in ("fake", "fake2")])
    get = harness.providers.get

    def broken(model):
        if model["id"] == "fake-2":
            raise RuntimeError("fake2: no key")
        return get(model)

    monkeypatch.setattr(harness.providers, "get", broken)
    with pytest.raises(RuntimeError, match="no key"):
        harness.submit("pilot")
    assert not fake.CALLS


def test_a_model_that_stops_hands_its_part_of_the_room_to_the_models_still_sending(
    guarded, run, monkeypatch
):
    second_model()
    run(cells=[plans.Cell("fake2", "R0", "C2")])
    harness.submit("smoke")  # fake2's own usage, priced like fake's
    harness.config("models")["fake2"] |= {"fail_times": 99, "fail_error": "APIConnectionError: x"}
    monkeypatch.setattr(harness, "TRIP", 1)  # fake2 stops at its first error
    cells = [plans.Cell("fake", r, "C2") for r in ("R0", "R1", "R2")]
    cells += [plans.Cell("fake2", r, "C2") for r in ("R0", "R1", "R2", "R3")]
    run(cells=cells, fake_text="one two three four five")  # fake's answers: 1.6 times the price
    harness.config("budget")["cap_b2_usd"] = harness.spent() + 7 * guarded * (1 + 1e-9)
    slow(monkeypatch, "fake", 0.3)  # fake2 has stopped before fake's first answer is back
    with pytest.raises(harness.ProviderDown, match="fake2"):
        harness.submit("pilot", workers=1)
    # fake's overruns spend its own part by its third request, which goes in the part fake2 left
    assert harness.status("pilot") == {"fake": {"ok": 3}, "fake2": {"pending": 4}}


def test_a_stopped_models_part_goes_to_the_others_at_once_not_when_its_last_attempt_is_back(
    guarded, run, monkeypatch
):
    second_model()
    run(cells=[plans.Cell("fake2", "R0", "C2")])
    harness.submit("smoke")  # fake2's own usage, priced like fake's
    harness.config("models")["fake2"] |= {"fail_times": 99, "fail_error": "APIConnectionError: x"}
    monkeypatch.setattr(harness, "TRIP", 1)  # fake2 stops at its first error
    cells = [plans.Cell(m, r, "C2") for m in ("fake", "fake2") for r in ("R0", "R1", "R2")]
    run(cells=cells, fake_text="one two three four five")  # fake's answers: 1.6 times the price
    harness.config("budget")["cap_b2_usd"] = harness.spent() + 6 * guarded * (1 + 1e-9)
    send = fake.Fake.send

    def timed(self, req):  # fake2's R1 is still out long after its R0 stopped fake2
        late = req.model == "fake2" and req.parts[0].text.startswith("R1")
        time.sleep(0.3 if req.model == "fake" else 1.5 if late else 0)
        return send(self, req)

    monkeypatch.setattr(fake.Fake, "send", timed)
    with pytest.raises(harness.ProviderDown, match="fake2"):
        harness.submit("pilot", workers=2)
    # fake's third request goes at 0.3 s, in the part fake2 left when it stopped
    assert harness.status("pilot")["fake"] == {"ok": 3}


@pytest.mark.parametrize("in_turn", [False, True])
def test_a_model_that_has_sent_everything_hands_what_it_left_unspent_to_the_others(
    guarded, run, monkeypatch, in_turn
):
    second_model(fake_text="a b c d e f g h i j")  # priced from ten words
    run(cells=[plans.Cell("fake2", "R0", "C2")])
    harness.submit("smoke")
    harness.config("models")["fake2"]["fake_text"] = "a"  # its answers: a seventh of the price
    cells = [plans.Cell("fake2", r, "C2") for r in ("R0", "R1")]
    cells += [plans.Cell("fake", r, "C2") for r in ("R0", "R1", "R2")]
    run(cells=cells, fake_text="one two three four five")  # fake's answers: 1.6 times the price
    total = harness.dry_run(["pilot"])
    harness.config("budget")["cap_b2_usd"] = harness.spent() + total * (1 + 1e-9)  # all five fit
    slow(monkeypatch, "fake", 0.3)  # fake2 has sent everything before fake's first answer is back
    # fake's third request fits only in what fake2 left unspent
    assert harness.submit("pilot", workers=1, in_turn=in_turn) == 5
    assert harness.status("pilot") == {"fake2": {"ok": 2}, "fake": {"ok": 3}}


def test_a_rooms_parts_are_each_models_own_and_an_overrun_takes_its_part_then_the_slack():
    price, model_of = {"a1": 2.0, "a2": 2.0, "b1": 3.0}, {"a1": "a", "a2": "a", "b1": "b"}
    room = harness.Room(10.0, price, model_of)
    assert (room.part, room.slack) == ({"a": 4.0, "b": 3.0}, 3.0)
    assert room.take("a", "a1")
    room.settle("a", "a1", 7.0)  # 5 over its price: a's other 2, then the slack's 3
    assert (room.part, room.slack) == ({"a": 0.0, "b": 3.0}, 0.0)
    assert not room.take("a", "a2")  # nothing left for a
    assert room.take("b", "b1")  # b's part whole
    room.settle("b", "b1", 1.0)  # 2 under its price: kept in b's part
    assert room.part["b"] == 2.0
    room.close("b")  # b's 2 go back to the slack, where a may take them
    assert room.slack == 2.0 and room.take("a", "a2") and not room.take("b", "b1")
    room.settle("a", "a2", 5.0)  # 3 over, with no slack left to cover it
    room.close("a")
    assert (room.passed, room.slack) == (3.0, 0.0)  # it passes the cap


def test_a_stopped_models_part_and_late_answers_go_to_the_slack_which_covers_its_overrun():
    room = harness.Room(10.0, {"a1": 1.0, "b1": 1.0}, {"a1": "a", "b1": "b"})
    assert room.take("a", "a1")
    room.close("a")  # stopped with a1 still out
    assert not room.take("a", "a1")  # a stopped model takes nothing, whatever room is left
    room.settle("a", "a1", 0.0)  # its late answer cost nothing: the price goes to the slack
    assert (room.part["a"], room.slack) == (0.0, 9.0)
    room = harness.Room(4.0, {"a1": 2.0, "b1": 2.0}, {"a1": "a", "b1": "b"})
    assert room.take("a", "a1")
    room.settle("a", "a1", 5.0)  # 3 over, with no slack to cover it yet
    assert room.part["a"] == -3.0
    room.close("b")  # b's 2 come back to the slack
    room.close("a")  # which covers 2 of a's overrun; the last 1 passes the cap
    assert (room.passed, room.slack) == (1.0, 0.0)


def test_an_answer_still_out_when_its_model_stopped_pays_back_what_the_model_passed_the_cap_by():
    room = harness.Room(4.0, {"a1": 2.0, "a2": 2.0}, {"a1": "a", "a2": "a"})
    assert room.take("a", "a1") and room.take("a", "a2")
    room.settle("a", "a1", 5.0)  # 3 over, no slack
    room.close("a")  # stopped: the 3 pass the cap
    assert room.passed == 3.0
    room.settle("a", "a2", 0.5)  # 1.5 under its price: it pays back half of it, not the slack
    assert (room.passed, room.slack) == (1.5, 0.0)  # as the running total ends: 4 - 5.5


def test_one_models_room_with_attempts_out_when_it_stops_ends_as_its_running_total_did():
    rng = random.Random(20261018)
    for _ in range(500):
        price = {f"r{i}": rng.uniform(0.1, 2.0) for i in range(rng.randint(2, 8))}
        start = rng.uniform(0.0, 8.0)
        room, left = harness.Room(start, price, dict.fromkeys(price, "m")), start
        out, todo = [], list(price)
        while todo or out:
            if todo and len(out) < 4 and (not out or rng.random() < 0.5):  # a worker is free
                cid = todo.pop(0)
                fits = left >= price[cid] - 1e-9
                assert room.take("m", cid) == fits
                if not fits:  # the model stops; its attempts already out still come back
                    room.close("m")
                    todo = []
                    continue
                left -= price[cid]
                out.append(cid)
            else:
                cid = out.pop(rng.randrange(len(out)))
                cost = rng.uniform(0.0, 3.0) * price[cid]
                room.settle("m", cid, cost)
                left += price[cid] - cost
        room.close("m")
        assert room.passed == pytest.approx(max(0.0, -left), abs=1e-12)
        assert room.slack == pytest.approx(max(0.0, left), abs=1e-12)


def test_a_room_short_of_the_parts_takes_from_the_first_models_and_never_starts_below_nothing():
    price, model_of = {"a1": 2.0, "b1": 2.0}, {"a1": "a", "b1": "b"}
    room = harness.Room(3.0, price, model_of)
    assert (room.part, room.slack) == ({"a": 1.0, "b": 2.0}, 0.0)
    room = harness.Room(-1.0, price, model_of)
    assert (room.part, room.slack) == ({"a": 0.0, "b": 0.0}, 0.0)


def test_one_models_room_decides_as_its_running_total_did():
    rng = random.Random(20261016)
    for _ in range(300):
        price = {f"r{i}": rng.uniform(0.1, 2.0) for i in range(rng.randint(1, 6))}
        start = rng.uniform(-2.0, 12.0)
        room, left = harness.Room(start, price, dict.fromkeys(price, "m")), max(0.0, start)
        stopped = False
        for cid in price:
            for _ in range(rng.randint(1, 3)):  # an attempt, then retries
                fits = left >= price[cid] - 1e-9
                assert room.take("m", cid) == fits
                if not fits:
                    stopped = True
                    break
                left -= price[cid]  # set aside while the attempt is out
                assert room.part["m"] + room.slack == pytest.approx(left)
                cost = rng.uniform(0.0, 3.0) * price[cid]
                room.settle("m", cid, cost)
                left += price[cid] - cost
                assert room.part["m"] + room.slack == pytest.approx(left)
            if stopped:
                break
        room.close("m")
        assert room.passed == pytest.approx(max(0.0, -left))


def test_no_interleaving_lets_a_model_touch_anothers_part_or_pass_the_cap_but_by_overruns():
    rng = random.Random(20261017)

    def others(room, model):
        return {k: v for k, v in room.part.items() if k != model}

    for _ in range(300):
        models = ["a", "b", "c"][: rng.randint(2, 3)]
        price = {f"{m}{i}": rng.uniform(0.1, 2.0) for m in models for i in range(rng.randint(1, 4))}
        start = rng.uniform(0.0, 1.2) * sum(price.values())
        room = harness.Room(start, price, {cid: cid[0] for cid in price})
        out, todo, spent, overrun = [], list(price), 0.0, 0.0
        for _ in range(60):
            roll = rng.random()
            if roll < 0.5 and todo:
                cid = rng.choice(todo)
                before = others(room, cid[0])
                if room.take(cid[0], cid):
                    out.append(cid)
                    if rng.random() < 0.7:  # else it goes again, as a retry
                        todo.remove(cid)
            elif roll < 0.9 and out:
                cid = out.pop(rng.randrange(len(out)))
                before, cost = others(room, cid[0]), rng.uniform(0.0, 2.0) * price[cid]
                room.settle(cid[0], cid, cost)
                spent, overrun = spent + cost, overrun + max(0.0, cost - price[cid])
            else:
                cid = rng.choice(list(price))
                before = others(room, cid[0])
                room.close(cid[0])
            assert others(room, cid[0]) == before
            assert room.slack >= -1e-9
        for cid in out:  # the attempts still out finish
            cost = rng.uniform(0.0, 2.0) * price[cid]
            room.settle(cid[0], cid, cost)
            spent, overrun = spent + cost, overrun + max(0.0, cost - price[cid])
        for model in models:
            room.close(model)
        assert spent <= start + room.passed + 1e-9  # past the cap only by what passed it
        assert room.passed <= overrun + 1e-9  # and only overruns pass it


def test_a_batch_refused_outright_leaves_its_requests_pending_and_the_others_go(run, monkeypatch):
    second_model(batch=True)  # batched, so it goes first
    run(cells=[plans.Cell("fake", "R0", "C2"), plans.Cell("fake2", "R0", "C2")])
    submit_batch = fake.Fake.submit_batch

    def refuse(self, reqs):
        if reqs[0].model == "fake2":
            raise RuntimeError("Batch API is not available for zero-data-retention accounts.")
        return submit_batch(self, reqs)

    monkeypatch.setattr(fake.Fake, "submit_batch", refuse)
    with pytest.raises(RuntimeError, match="zero-data-retention"):
        harness.submit("pilot")
    status = harness.status("pilot")
    assert status["fake"] == {"ok": 1} and status["fake2"] == {"pending": 1}
    assert not harness.open_batches("pilot")


def test_a_batch_failing_part_way_still_lets_the_other_models_go(run):
    second_model(batch=True, fail_batch_after=1)  # takes one of its two requests, then fails
    run(cells=[plans.Cell("fake", "R0", "C2"), plans.Cell("fake2", "R0", "C2"),
               plans.Cell("fake2", "R1", "C2")])  # fmt: skip
    with pytest.raises(PartialBatch):
        harness.submit("pilot")
    status = harness.status("pilot")
    assert status["fake"] == {"ok": 1} and status["fake2"] == {"in batch": 1, "pending": 1}


def test_every_batch_goes_before_the_models_sent_request_by_request(run):
    second_model(batch=True)
    run(cells=[plans.Cell("fake", "R0", "C2"), plans.Cell("fake2", "R0", "C2")])
    _, meta = harness.build("pilot")
    assert harness.submit("pilot") == 2
    assert [meta[c]["model"] for c in fake.CALLS] == ["fake2", "fake"]


def test_batches_in_flight_shrink_the_room_of_a_model_sent_request_by_request(guarded, run):
    run(batch=True)
    harness.submit("ladder")  # in flight, at half a request's price
    cells = [plans.Cell("fake", r, "C2") for r in ("R0", "R1")]
    run(batch=False, cells=cells, fake_text="one two three four five")  # 1.6 times the price
    harness.config("budget")["cap_b2_usd"] = harness.spent() + 2.6 * guarded
    with pytest.raises(harness.CapReached):
        harness.submit("pilot", workers=1)
    assert len(responses()) == 1  # the ladder's batch left no room for the second


def test_the_smoke_tests_come_before_any_cap(guarded, run):
    harness.config("budget")["cap_b2_usd"] = "TBD (B1)"
    run(cells=[plans.Cell("fake", "R1", "C2")])
    assert harness.submit("smoke") == 1


def test_dry_run_names_the_groups_and_reports_a_cap_not_yet_set(guarded, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(harness, "build", lambda stage, write=False, guard=True:
                        seen.append((stage, guard)) or ([], {}))  # fmt: skip
    harness.config("budget")["cap_b2_usd"] = "TBD (B1)"
    assert harness.dry_run(["b2"]) == 0
    assert [s for s, _ in seen] == list(plans.GROUPS["b2"])
    assert all(guard is False for _, guard in seen)  # priced from local files alone
    assert "no cap is set yet for b2" in capsys.readouterr().out


def test_a_choice_not_yet_made_is_priced_at_its_costliest(monkeypatch):
    monkeypatch.setattr(plans, "config", {"choices": {"a": {"rbest": "TBD (B2)"}}}.__getitem__)
    with pytest.raises(ValueError, match="not chosen yet"):
        plans.choice("a", "rbest")
    with plans.costliest_choices() as assumed:
        assert plans.choice("a", "rbest") == "R6"
    assert assumed == {("a", "rbest")}
    with pytest.raises(ValueError, match="not chosen yet"):  # only while pricing
        plans.choice("a", "rbest")


def test_a_late_flex_request_may_go_at_the_standard_tier_billed_as_served(run, tmp_path):
    run(service_tier="flex")
    with pytest.raises(ValueError, match="only a test request"):
        harness.submit("pilot", standard_tier=True)
    with pytest.raises(ValueError, match="not 5 days into the window yet: fake"):
        harness.submit("test", standard_tier=True)  # no canary sent yet
    day0 = (datetime.now(UTC) - timedelta(days=4)).isoformat()
    canary = {"model": "fake", "started": day0, "status": "ok", "custom_id": "c"}
    (tmp_path / "canary-1").mkdir()
    (tmp_path / "canary-1" / "responses.jsonl").write_text(json.dumps(canary) + "\n", "utf-8")
    with pytest.raises(ValueError, match="not 5 days"):
        harness.submit("test", standard_tier=True)
    canary["started"] = (datetime.now(UTC) - timedelta(days=5, minutes=1)).isoformat()
    (tmp_path / "canary-1" / "responses.jsonl").write_text(json.dumps(canary) + "\n", "utf-8")
    assert harness.submit("test", standard_tier=True) == 1
    (atts,) = responses("test").values()
    row = atts[0]
    assert row["sent_tier"] == "standard" and row["settings"]["service_tier"] == "flex"
    inp, out = harness.tokens(row["usage"])
    assert row["cost_usd"] == pytest.approx((inp * 2.0 + out * 10.0) / 1e6)  # full price
    assert harness.submit("test") == 0  # its answer stands for the Flex request it replaced


def test_answers_from_two_versions_of_a_model_are_refused_or_split(run):
    run(cells=[plans.Cell("fake", "R0", "C2")])
    harness.submit("pilot")  # answered by fake-1
    run(cells=[plans.Cell("fake", "R0", "C2"), plans.Cell("fake", "R1", "C2")],
        reported_model="fake-2", expect="fake-2")  # fmt: skip  # the provider moved on
    harness.submit("pilot")
    with pytest.raises(ValueError, match="more than one version"):
        harness.score("pilot", partial=True)
    arms = sorted(r["arm"] for r in harness.score("pilot", partial=True, split_versions=True))
    assert arms == ["fake:R0:C2@fake-1", "fake:R1:C2"]  # the old version's answer, apart


def test_a_test_answer_from_a_version_no_longer_pinned_is_sent_again_on_request(run):
    run()
    assert harness.submit("test") == 1  # answered by fake-1
    run(reported_model="fake-2", expect="fake-2")  # the provider moved on, and the pin with it
    assert harness.submit("test") == 0  # a final answer stands
    with pytest.raises(ValueError, match="only test requests"):
        harness.submit("canary-1", resend_versions=True)  # the drift test's start runs stay
    assert harness.submit("test", resend_versions=True) == 1
    assert harness.submit("test", resend_versions=True) == 0  # answered under the pin now
    (atts,) = responses("test").values()
    assert [a["reported_model"] for a in atts] == ["fake-1", "fake-2"]
    (row,) = harness.score("test", partial=True)  # one version: nothing to refuse
    assert row["reported_model"] == "fake-2" and row["attempts"] == 1
    assert row["cost_usd"] == pytest.approx(atts[1]["cost_usd"])  # its own cost, not fake-1's


def test_a_resend_that_has_not_ended_leaves_the_request_pending(run, monkeypatch):
    run()
    harness.submit("test")  # answered by fake-1
    run(reported_model="fake-2", expect="fake-2", fail_times=99,
        fail_error="RateLimitError: Error code: 429 - too many requests")  # fmt: skip
    fake.reset()
    assert harness.submit("test", resend_versions=True) == 1  # only busy signals: still pending
    assert "fake: 1 requests not final" in harness.unfinished("test")
    assert harness.score("test", partial=True) == []  # left out, never a crash
    run(fail_error="APIConnectionError: down")
    monkeypatch.setattr(harness, "TRIP", 99)
    fake.reset()
    harness.submit("test")  # the re-send's own errors: it now runs to its four attempts
    (row,) = harness.score("test", partial=True)
    assert (row["status"], row["attempts"], row["failed"]) == ("request_error", 4, True)


def test_a_resend_never_runs_past_four_attempts_nor_reopens_a_final_failure(run, monkeypatch):
    run(fail_times=99, fail_error="APIConnectionError: down")
    monkeypatch.setattr(harness, "TRIP", 2)  # the run stops after two errors, still pending
    with pytest.raises(harness.ProviderDown):
        harness.submit("test", workers=1)
    run(expect="fake-2", reported_model="fake-2")  # the provider moved on, and the pin with it
    monkeypatch.setattr(harness, "TRIP", 99)
    fake.reset()
    harness.submit("test", resend_versions=True)  # not a stale answer: its two attempts count
    assert harness.submit("test") == 0 and harness.submit("test", resend_versions=True) == 0
    (atts,) = responses("test").values()
    assert [a["status"] for a in atts] == ["request_error"] * 4  # four in all, never more
    (row,) = harness.score("test", partial=True)
    assert (row["status"], row["attempts"], row["failed"]) == ("request_error", 4, True)


def test_a_resend_in_a_batch_is_out_until_collected(run, monkeypatch):
    run(batch=True)
    harness.submit("test")
    harness.collect("test")  # answered by fake-1
    harness.submit("smoke")  # usage for the dry run to price from
    run(expect="fake-2", reported_model="fake-2")
    assert harness.submit("test", resend_versions=True) == 1
    assert harness.status("test") == {"fake": {"in batch": 1}}  # not fake-1's old answer
    assert "fake: 1 requests not final" in harness.unfinished("test")
    assert harness.score("test", partial=True) == []  # nothing scored from the old version
    assert harness.dry_run(["test"]) > 0  # priced: not billed yet
    harness.collect("test")
    (row,) = harness.score("test", partial=True)
    assert row["reported_model"] == "fake-2"


def test_a_resend_meeting_another_version_shows_as_a_mismatch(run, monkeypatch):
    run(fail_times=3, fail_error="APIConnectionError: down")
    monkeypatch.setattr(harness, "TRIP", 99)
    harness.submit("test")  # three errors, then fake-1's answer
    run(expect="fake-2", reported_model="fake-3", fail_times=0)  # re-pinned; another serves
    with pytest.raises(harness.ModelMismatch):
        harness.submit("test", resend_versions=True)
    assert harness.status("test") == {"fake": {"model_mismatch": 1}}


def test_a_split_keeps_a_failure_naming_no_model_with_the_version_it_was_sent_to(run, monkeypatch):
    r0, r1, r2 = (plans.Cell("fake", f"R{i}", "C2") for i in range(3))
    run(cells=[r0])
    harness.submit("pilot")  # answered by fake-1
    run(cells=[r0, r1], reported_model="fake-2", expect="fake-2")
    harness.submit("pilot")  # R1 answered by fake-2
    run(cells=[r0, r1, r2], fail_times=99, fail_error="APIConnectionError: down")
    monkeypatch.setattr(harness, "TRIP", 99)
    fake.reset()
    harness.submit("pilot")  # R2 fails four times under the fake-2 pin, naming no model
    arms = sorted(r["arm"] for r in harness.score("pilot", partial=True, split_versions=True))
    assert arms == ["fake:R0:C2@fake-1", "fake:R1:C2", "fake:R2:C2"]  # not `@None`


def test_list_prices_after_a_promotion_cost_each_answer_again(run):
    run(batch=True)
    harness.config("prices")["fake"] |= {"list_input": 4.0, "list_output": 20.0}
    harness.submit("pilot")
    harness.collect("pilot")
    (row,) = harness.score("pilot", partial=True)
    assert row["cost_list_usd"] == pytest.approx(2 * row["cost_usd"])  # batch at list price


def test_scoring_records_the_code_and_tags_it_ran_at(run, tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "provenance", lambda: {"commit": "abc", "tags": []})
    run()
    harness.submit("pilot")
    harness.score("pilot", partial=True)
    made = json.loads((tmp_path / "pilot" / "provenance.json").read_text(encoding="utf-8"))
    assert made["commit"] == "abc" and made["partial"] is True and "scored" in made
