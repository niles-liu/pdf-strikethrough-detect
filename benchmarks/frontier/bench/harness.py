"""Running a stage (PREREG.md, Operations): requests, the response cache, batches, the model-ID
check, resubmission, the dry run and scoring.

    python -m bench harness dry-run <stage or group>... [--smoke]   # groups: b2, b3
    python -m bench harness submit <stage> [--model M ...] [--workers N] [--standard-tier]
                                           [--resend-versions] [--in-turn]
    python -m bench harness collect <stage>
    python -m bench harness status <stage>
    python -m bench harness score <stage> [--partial] [--split-versions]

A stage's requests are rebuilt from `plans.py` every time, each keyed by a hash of everything
sent; `submit` lists them in `runs/<stage>/requests.jsonl`. Every attempt is appended to
`runs/<stage>/responses.jsonl` and never rewritten, so a rerun calls no API for a request that
already has a final answer: `ok`, a refusal, an output-limit stop, an answer that fails to parse
(an empty one included), or a request error on its fourth attempt (three resubmissions).

Two things are not attempts and leave a request pending: a `transient` error (rate limit,
authentication, quota or spend limit), and a request in a batch not yet collected. Five request
or account errors in a row (an account error being one waiting won't fix: a key, a quota, a
spend limit) stop sending to a model; a capacity limit (a 429 asking the client to wait, as Flex
gives when busy) waits longer instead, and never counts toward that stop. `submit` refuses stage
B2 before the prereg-1 tag and stage B3 before prereg-2 (TAG_NEEDED), and a stage whose price,
with what is spent and what is in flight, passes the cap in force (`plans.cap`), the smoke tests
apart; a model sent request by request stops once its next request's price no longer fits in what
the cap leaves it, its other requests left pending. A response reporting a model other than the
pinned one is recorded as `model_mismatch`, and the stage then refuses to submit, collect or
score until `expect` is re-pinned to the model that served (its answers then count, unsent
again). A stage holding answers from two versions of a model is then refused at scoring: `submit
--resend-versions` sends its test requests the earlier version answered again, within the cap,
or `score --split-versions` reports each version apart (PREREG, Operations). `submit` and
`collect` hold a lock on the stage, so two stages can be sent at once, but each `submit` then
prices against the cap without the other's requests still to go: send one stage at a time. The
models a stage sends request by request go at once, each `--workers` at a time at its own
provider's rate limits (`--in-turn`: one after another). The latency sample goes one call at a
time, with the SDKs' own retries off, so each timed call is one attempt, and Windows' power
throttling off, as the free arms' timing passes do: launched with no window, a process may
otherwise be moved to slow cores.

The repeats and the effort ablation share answers (`plans.SHARED`): a request one of them holds
a final answer to from the model now pinned, or holds in a batch not yet collected, is not sent
by the other, which reads that answer as its own (`reused_from` names the stage in its scores). A
request's attempts in both count toward its four, they hold one lock, a model mismatch in one
stops both, and a dry run prices a shared request once.
"""

import functools
import json
import re
import statistics
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from bench import arms, plans, prompts, providers, render
from bench.common import (
    GT, ROOT, RUNS, SETS, append_jsonl, config, harness_commit, load_pages, lock, provenance,
    read_jsonl, sha256_bytes, tags_in_force, write_jsonl,
)  # fmt: skip
from bench.providers import OUTPUT_LIMIT, Reply, Request, now
from bench.score import (
    CASCADE_GRID, ParseError, normalize, parse, score_page, tokens_from_words, uncertain_bands,
)  # fmt: skip

MAX_ATTEMPTS = 4
FINAL = {"ok", "refusal", "length", "parse_error"}
TRANSIENT = re.compile(
    r"RateLimit|rate_limit|Authentication|PermissionDenied|InsufficientQuota|\b(?:401|402|403|429)\b"
    r"|quota|billing|credit balance|spend(?:ing)? limit|usage limit",
    re.I,
)
# a transient error waiting won't fix, which counts toward TRIP as a request error does; never a
# bare "quota", which Azure's ordinary rate limit names (a link to its quota-increase form)
ACCOUNT = re.compile(
    r"Authentication|PermissionDenied|InsufficientQuota|insufficient_quota"
    r"|exceeded your current quota|\b(?:401|402|403)\b|billing|credit balance"
    r"|spend(?:ing)? limit|usage limit",
    re.I,
)
TRIP = 5  # request errors (or account errors) in a row that stop sending to a model
MAX_TRANSIENT = 3  # transient retries per request per run
BACKOFF_S = 1.0  # the wait before resubmission n is BACKOFF_S * 2**n, at most a minute
CAPACITY_BACKOFF = 15  # after a capacity limit, 15 times as long, at most CAPACITY_WAIT_S
CAPACITY_WAIT_S = 300
FLEX_GRACE = timedelta(days=5)  # into a model's window, a test request may leave Flex
TEST_STAGES = ("test", "fable", "qa", "latency")  # a model's test requests, between its canaries
# the tag each group of stages waits for: the pilot comes after prereg-1, and stage B3's canaries,
# which open each model's window, after prereg-2 (the smoke tests come before either)
TAG_NEEDED = {"b2": "prereg-1", "b3": "prereg-2", "later": "prereg-2"}
BATCH_BYTES, BATCH_REQUESTS = 100_000_000, 10_000  # per batch: under every provider's limit


class ModelMismatch(RuntimeError):
    """A response reported a model other than the pinned one (PREREG, Operations)."""


class ProviderDown(RuntimeError):
    """Request errors in a row: the provider is failing, so its requests are left pending."""


class CapReached(RuntimeError):
    """The cap in force leaves a model sent request by request no room for its next request at
    its price, so its other requests are left pending (PREREG, Operations)."""


# ------------------------------------------------------------------------------------ requests


@functools.cache
def _qa_items() -> dict[str, dict]:
    return {q["qa_id"]: q for q in read_jsonl(GT / "G0-qa.jsonl")}


def _prompt(model: dict, item: plans.Item):
    rung, page = item.cell.rung, item.page_id
    if rung in prompts.RUNGS:
        return prompts.assemble(model, rung, item.cell.input, page, item.cell.para)
    if rung.startswith("QA-"):
        question = _qa_items()[item.qa_id]["question"]
        return prompts.qa(model, page, question, rung.removeprefix("QA-"))
    if rung == "M+P":
        return prompts.m_plus_p(model, page)
    if rung == "P→M":
        return prompts.text_only(model, page)
    raise ValueError(f"unknown rung {rung!r}")


def settings(model: dict, cell: plans.Cell) -> dict:
    keys = ("image_detail", "media_resolution")
    return {
        "effort": cell.effort,
        "max_tokens": min(OUTPUT_LIMIT, model.get("max_output", OUTPUT_LIMIT)),
        **{k: model[k] for k in keys if k in model},
    }


def build(
    stage: str, write: bool = False, guard: bool = True
) -> tuple[list[Request], dict[str, dict]]:
    """The stage's requests and, per custom ID, what the response records carry about it. `guard`
    False builds test pages' requests before prereg-2, for the dry run alone: from local files,
    nothing sent."""
    models = config("models")
    reqs, meta, items = [], {}, plans.items(stage, guard=guard)
    wanted = {i.page_id for i in items}
    render.require_current([p for p in load_pages(*SETS) if p["page_id"] in wanted])
    for item in items:
        cell = item.cell
        model = {**models[cell.model], "key": cell.model}
        system, parts = _prompt(model, item)
        how = plans.mode(stage, model)
        sets = settings(model, cell) | ({"service_tier": "flex"} if how == "flex" else {})
        if "prompt_cache" in model and stage not in plans.CACHE_ON:
            sets["prompt_cache"] = model["prompt_cache"]
        texts = [system, *(p.text for p in parts if p.kind == "text")]
        key = {"id": model["id"], "settings": sets, "system": system,
               "parts": [[p.kind, p.sha256 or p.text] for p in parts], "repeat": cell.repeat}  # fmt: skip
        cid = sha256_bytes(json.dumps(key, sort_keys=True, ensure_ascii=False).encode())[:32]
        if cid in meta:
            raise ValueError(
                f"{stage}: two items build the same request ({cell.arm}, {item.page_id})"
            )
        # the latency sample times one call each: no SDK retry, and no back-off, inside it
        retries = 0 if stage == "latency" else 2
        reqs.append(Request(cid, cell.model, model, system, parts, effort=cell.effort,
                            max_tokens=sets["max_tokens"], service_tier=sets.get("service_tier"),
                            retries=retries, prompt_cache=sets.get("prompt_cache")))  # fmt: skip
        meta[cid] = {
            "custom_id": cid, "arm": cell.arm, "page_id": item.page_id, "qa_id": item.qa_id,
            **asdict(cell), "mode": how, "settings": sets,
            "fewshot": any(p.path.startswith("prompts/") for p in parts),
            "prompt_sha256": sha256_bytes(json.dumps(texts, ensure_ascii=False).encode()),
            "input_sha256": sha256_bytes("".join(p.sha256 for p in parts if p.kind != "text").encode()),
        }  # fmt: skip
    if write:
        write_jsonl(RUNS / stage / "requests.jsonl", meta.values())
    return reqs, meta


def attempts(stage: str) -> dict[str, list[dict]]:
    """Every attempt, per request. An answer recorded as another model's counts as what it was
    once `expect` is re-pinned to the model that served it: it is billed, and not sent again."""
    models = config("models")
    out: dict[str, list[dict]] = {}
    for row in read_jsonl(RUNS / stage / "responses.jsonl"):
        if row["status"] == "model_mismatch" and row.get("answer_status"):
            if row["reported_model"] == _expected(models.get(row["model"], {"expect": "TBD"})):
                row = row | {"status": row["answer_status"]}
        out.setdefault(row["custom_id"], []).append(row)
    return out


def _sharing(stage: str) -> tuple[str, ...]:
    """The stage and those whose answers it shares (plans.SHARED), itself first."""
    return (stage, *(s for s in plans.SHARED if s != stage)) if stage in plans.SHARED else (stage,)


def _lock(stage: str) -> Path:
    """The lock a stage's submit and collect hold: one for the stages sharing answers, so they
    never send the same request at once."""
    return RUNS / ".shared.lock" if stage in plans.SHARED else RUNS / stage / ".lock"


def current(atts: list[dict] | None) -> list[dict]:
    """A request's attempts since the last final answer that more attempts followed: a request
    sent again under a new version (`submit --resend-versions`) starts afresh, pending until its
    new answer is final, so an answer's attempts and cost are its own."""
    rows = list(atts or ())
    start = 0
    for i in range(1, len(rows)):
        if is_final(rows[start:i]):
            start = i
    return rows[start:]


def answers(stage: str) -> dict[str, list[dict]]:
    """attempts(stage), each request's `current` ones, shared: a request without a final answer
    here takes a sharing stage's final one (an identical request is answered once); failing that,
    its attempts in every sharing stage are its history, so the attempt limit counts them all. In
    the sharing stages an answer counts only while its model is the one pinned, a stage's own
    included, so after a re-pin both send again rather than set one model's answers against
    another's."""
    if stage not in plans.SHARED:
        return {cid: current(rows) for cid, rows in attempts(stage).items()}
    models = config("models")
    out, *others = ({cid: current(rows) for cid, rows in _pinned(attempts(s), models).items()}
                    for s in _sharing(stage))  # fmt: skip
    for cid in set().union(*others):
        if is_final(out.get(cid)):
            continue
        theirs = [a[cid] for a in others if cid in a]
        lent = [x for x in theirs if is_final(x)]
        history = out.get(cid, []) + [row for x in theirs if not is_final(x) for row in x]
        if lent or history:
            out[cid] = lent[0] if lent else sorted(history, key=lambda r: r.get("recorded", ""))
    return out


def _pinned(atts: dict[str, list[dict]], models: dict) -> dict[str, list[dict]]:
    """The attempts that count while the model now pinned is the one compared: its answers (one
    naming no model counts by the pin it was sent under), the other attempts sent under that
    pin, so the attempt limit holds whatever an error names, and any mismatch, which stops the
    stage until resolved."""
    out = {}
    for cid, rows in atts.items():
        if mine := [r for r in rows if _now_pinned(r, models)]:
            out[cid] = mine
    return out


def _now_pinned(row: dict, models: dict) -> bool:
    pin = _expected(models.get(row["model"], {"expect": "TBD"}))
    if row["status"] == "model_mismatch":
        return True
    if row["status"] in FINAL and row.get("reported_model") is not None:
        return row["reported_model"] == pin
    return row.get("expected_model") == pin


def counted(atts: list[dict] | None) -> list[dict]:
    """The attempts that count: a transient error is not one."""
    return [a for a in atts or () if a["status"] != "transient"]


def is_final(atts: list[dict] | None) -> bool:
    c = counted(atts)
    return bool(c) and (c[-1]["status"] in FINAL or len(c) >= MAX_ATTEMPTS)


def open_batches(stage: str) -> list[dict]:
    rows = read_jsonl(RUNS / stage / "batches.jsonl")
    closed = {r["batch_id"] for r in rows if "collected" in r}
    return [r for r in rows if "collected" not in r and r["batch_id"] not in closed]


def in_flight(stage: str) -> dict[str, str]:
    """The requests in a batch not yet collected, by this stage or one sharing its answers, each
    with the stage whose batch holds it."""
    return {cid: s for s in reversed(_sharing(stage)) for b in open_batches(s)
            for cid in b["custom_ids"]}  # fmt: skip


def _stale(atts: list[dict] | None, models: dict) -> bool:
    """A final answer from a version no longer pinned: what `submit --resend-versions` sends again,
    its attempts counted afresh. Anything else keeps its attempts, the old pin's included."""
    return is_final(atts) and not _now_pinned(counted(atts)[-1], models)


def _resent(atts: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """`answers` as `submit --resend-versions` reads them: a stale answer as no attempt yet."""
    models = config("models")
    return {cid: [] if _stale(rows, models) else rows for cid, rows in atts.items()}


def pending(stage: str, reqs: list[Request], resend: bool = False) -> list[Request]:
    """The requests without a final answer, or in a batch not yet collected; with `resend`, an
    answer from a version no longer pinned is not final either (`submit`)."""
    flying, atts = in_flight(stage), answers(stage)  # in that order: collect records, then closes
    if resend:
        atts = _resent(atts)
    return [r for r in reqs if not is_final(atts.get(r.custom_id)) and r.custom_id not in flying]


def mismatched(stage: str) -> dict[str, list[str]]:
    """Models with a recorded mismatch the current pin still disagrees with, in this stage or one
    sharing its answers: re-pinning `expect` to what was served clears them (their requests are
    then sent again)."""
    models = config("models")
    out: dict[str, list[str]] = {}
    for s in _sharing(stage):
        for atts in attempts(s).values():
            for a in atts:
                if a["status"] != "model_mismatch":
                    continue
                expect = _expected(models[a["model"]])
                if expect is not None and a["reported_model"] != expect:
                    where = "" if s == stage else f" (in {s})"
                    out.setdefault(a["model"], []).append(
                        f"reported {a['reported_model']!r}, expected {expect!r}{where}"
                    )
    return out


def _refuse_mismatches(stage: str, models: set[str] | None = None) -> None:
    bad = {m: v for m, v in mismatched(stage).items() if models is None or m in models}
    if bad:
        detail = "; ".join(f"{m}: {len(v)} ({v[0]})" for m, v in bad.items())
        raise ModelMismatch(f"{stage}: responses from another model: {detail}")


# ------------------------------------------------------------------------------------ recording

_LOCK = threading.Lock()


@functools.cache
def _run_meta() -> dict:
    return {"commit": harness_commit(), "package": arms.fingerprint(), "tags": tags_in_force()}


def _expected(model: dict) -> str | None:
    """The model ID a response must report: the pinned `expect` (GPT-6's served-model header)."""
    return None if model["expect"].startswith("TBD") else model["expect"]


def _discount(reply: Reply, batch_id: str | None) -> str | None:
    """The half-price route that served an answer: a batch, or the Flex tier Azure reports."""
    if batch_id:
        return "batch"
    return "flex" if reply.extra.get("service_tier") == "flex" else None


def record(stage: str, model: dict, reply: Reply, meta: dict, attempt: int, batch_id=None) -> dict:
    """Append one attempt and return it. An empty answer, or a JSON one that fails to parse, is a
    parse failure now, so it is final; the reported model is checked."""
    status = reply.status
    if status == "request_error" and TRANSIENT.search(str(reply.extra.get("error", ""))):
        status = "transient"
    if status == "ok" and not reply.text.strip():
        status = "parse_error"
    elif status == "ok" and meta["rung"] in prompts.RUNGS and prompts.json_answer(meta["rung"]):
        try:
            parse(reply.text, json_answer=True)
        except ParseError:
            status = "parse_error"
    expect = _expected(model)
    answered = status in ("ok", "length", "parse_error") or reply.reported_model is not None
    answer = None  # what a mismatched response was, kept for a re-pin to the model that served
    if answered and status not in ("request_error", "transient") and expect is not None:
        if reply.reported_model != expect:  # a refusal naming no model is not a mismatch
            answer, status = status, "model_mismatch"
    usd = cost(meta["model"], reply.usage, _discount(reply, batch_id))
    if usd is None and status == "refusal" and not reply.usage and price(meta["model"]):
        usd = 0.0  # a prompt a content filter blocked: no tokens reported, none billed
    row = {
        "run_id": stage, **meta, "pinned_model": model["id"], "expected_model": expect,
        "reported_model": reply.reported_model, "started": reply.started,
        "finished": reply.finished, "recorded": now(), "latency_s": reply.latency_s,
        "batch_id": batch_id, "attempt": attempt, "status": status, "answer_status": answer,
        "text": reply.text, "usage": reply.usage, "extra": reply.extra, "cost_usd": usd,
        **_run_meta(),
    }  # fmt: skip
    with _LOCK:
        append_jsonl(RUNS / stage / "responses.jsonl", row)
    return row


# ---------------------------------------------------------------------------------- submitting


def _send(stage: str, provider, req: Request, meta: dict, done: int, stop, trip: dict,
          room: Room | None = None) -> None:  # fmt: skip
    """One request, resubmitted after a request error until it has had four attempts, and after a
    transient error up to MAX_TRANSIENT times this run (then it waits for the next run). The wait
    doubles with every retry, a capacity limit's from a longer start, and ends early when the run
    stops. Request errors and account errors count toward TRIP; a capacity limit neither counts
    nor clears the count. Under a cap, `room` holds what the cap leaves the models sent with this
    one (`Room`): an attempt goes only if its price fits in its model's part of the room, which
    sets that price aside until the attempt's cost replaces it, so an attempt costing more than
    its price shrinks the room for the rest. The attempts already out finish."""
    n, waits = done, 0
    while not stop.is_set():
        if room is not None and not room.take(req.model, req.custom_id):
            if stop.is_set():  # its model stopped meanwhile: this request waits, as its others do
                return
            stop.set()
            raise CapReached(f"{req.model}: the cap in force leaves no room for its next "
                             "request; its others wait (see `harness dry-run`)")  # fmt: skip
        row = record(stage, req.model_config, provider.send(req), meta, n + 1)
        if room is not None:
            room.settle(req.model, req.custom_id, row["cost_usd"] or 0.0)
        status = row["status"]
        if status == "model_mismatch":
            stop.set()
            raise ModelMismatch(
                f"{meta['arm']}: reported {row['reported_model']!r}, expected {row['expected_model']!r}"
            )
        capacity = status == "transient" and not ACCOUNT.search(str(row["extra"].get("error", "")))
        with _LOCK:
            if status in ("request_error", "transient") and not capacity:
                trip["n"] += 1
            elif not capacity:
                trip["n"] = 0
            tripped = trip["n"] >= TRIP
        if tripped:
            stop.set()
            raise ProviderDown(f"{req.model}: {TRIP} request or account errors in a row; last: "
                               f"{row['extra'].get('error')}")  # fmt: skip
        n += status != "transient"
        waits += 1
        if (
            status not in ("request_error", "transient")
            or n >= MAX_ATTEMPTS
            or waits > MAX_TRANSIENT
        ):
            return
        wait = BACKOFF_S * 2**waits
        stop.wait(min(wait * CAPACITY_BACKOFF, CAPACITY_WAIT_S) if capacity else min(wait, 60))


def _size(req: Request) -> int:
    """A request's rough size on the wire: base64 files plus text."""
    return sum(
        len(p.text) if p.kind == "text" else (ROOT / p.path).stat().st_size * 4 // 3
        for p in req.parts
    )


def _chunks(reqs: list[Request]) -> list[list[Request]]:
    out, cur, size = [], [], 0
    for r in reqs:
        n = _size(r)
        if cur and (size + n > BATCH_BYTES or len(cur) >= BATCH_REQUESTS):
            out.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += n
    return out + [cur] * bool(cur)


def _send_together(stage: str, group: list[str], mine: dict[str, list[Request]], meta: dict,
                   atts: dict, room: Room | None, workers: int) -> dict[str, Exception]:  # fmt: skip
    """Send the models of `group` request by request, all at once, each `workers` at a time, and
    return what stopped each model that stopped. A model whose provider fails (ProviderDown), or
    whose next request the cap leaves no room for (CapReached), stops alone: its requests not yet
    begun return unsent and wait, and its part of the room goes to the others, as a model's part
    does once it has sent everything. Anything else (a model mismatch, an error, Ctrl+C) stops
    every model. Either way the attempts already out finish before this returns or raises."""
    stops = {m: threading.Event() for m in group}
    stopped: dict[str, Exception] = {}
    provider_of = {m: providers.get(mine[m][0].model_config) for m in group}  # before any goes
    with ExitStack() as pools:  # leaving it waits for every attempt already out
        try:
            jobs = {}
            for model in group:
                pool, trip = pools.enter_context(ThreadPoolExecutor(workers)), {"n": 0}
                for r in mine[model]:
                    job = pool.submit(_send, stage, provider_of[model], r, meta[r.custom_id],
                                      len(counted(atts.get(r.custom_id))), stops[model], trip,
                                      room)  # fmt: skip
                    jobs[job] = model
            unfinished = Counter(jobs.values())
            for job in as_completed(jobs):
                error, model = job.exception(), jobs[job]
                unfinished[model] -= 1
                if error is not None:
                    if not isinstance(error, (ProviderDown, CapReached)):
                        raise error  # an attempt still out after its model stopped is seen here
                    if model not in stopped:  # the first reason; its other workers may raise too
                        stopped[model] = error
                        stops[model].set()
                if room is not None and (model in stopped or not unfinished[model]):
                    room.close(model)  # stopped, or sent everything: its part goes to the others
        except BaseException:
            for stop in stops.values():
                stop.set()
            raise
    return stopped


def submit(stage: str, models: list[str] | None = None, workers: int = 8,
           standard_tier: bool = False, resend_versions: bool = False,
           in_turn: bool = False) -> int:  # fmt: skip
    """Send the stage's pending requests: batches per model where the stage batches it, all of
    them first, then the models sent request by request, all at once, each `workers` at a time
    at its own provider's rate limits (`_send_together`). With `in_turn`, those models go one
    after another, as the latency sample always does, one call at a time at full speed, since it
    times each call. A stage sharing answers sends the others' pending requests with its own,
    page by page. A batch the provider refuses, wholly or part way, leaves that model's other
    requests pending while the other models go, and is raised once they have gone. Refused past
    the cap in force; a model sent request by request stops once its next request's price no
    longer fits in what the cap leaves it, the price of the models sent with it and after it in
    this call kept for them (`Room`). With `standard_tier`, a test request bound for Flex goes at
    the standard tier instead, once its model is 5 days into its window (PREREG, Operations): the
    request is the same, so its answer is cached as Flex's would be, and billed as served. With
    `resend_versions`, a test request answered by a version of its model no longer pinned is sent
    again, its attempts counted afresh (PREREG, Operations: Model IDs); never a canary's, whose
    start runs the end run is compared with."""
    if (need := TAG_NEEDED.get(plans.group(stage))) and need not in tags_in_force():
        raise PermissionError(f"{stage} waits for the {need} tag (PREREG.md, What each tag fixes)")
    if resend_versions and stage not in TEST_STAGES:
        raise ValueError(f"only test requests are sent again under a new version, not {stage}'s")
    if stage == "latency":  # timed: one call at a time, at full speed however it was launched
        workers = 1
        arms.full_speed()
    with lock(_lock(stage)):
        _refuse_mismatches(stage)
        reqs, meta = build(stage, write=True)
        if stage in plans.SHARED:  # sent together, each page's requests at once
            for other in _sharing(stage)[1:]:
                more, theirs = build(other, write=True)
                reqs += [r for r in more if r.custom_id not in meta]
                meta = theirs | meta
            reqs.sort(key=lambda r: meta[r.custom_id]["page_id"])  # stable: cells keep their order
        if unknown := sorted(set(models or ()) - {r.model for r in reqs}):
            raise ValueError(f"{stage} sends nothing to {', '.join(unknown)}")
        todo = [r for r in pending(stage, reqs, resend_versions)
                if models is None or r.model in models]  # fmt: skip
        _refuse_expired({r.model for r in todo})
        if standard_tier:
            _refuse_early_standard(stage, {r.model for r in todo if r.service_tier == "flex"})
            meta = {c: m | {"sent_tier": "standard"} if m["mode"] == "flex" else m
                    for c, m in meta.items()}  # fmt: skip
            todo = [replace(r, service_tier=None) if r.service_tier == "flex" else r for r in todo]
        # one look at the usage seen so far prices the whole call, so the cap's arithmetic holds
        # still while this call's own answers come in
        seen = None if plans.group(stage) == "b1" else _observed()
        _check_cap(stage, [meta[r.custom_id] for r in todo], seen)
        if not stage.startswith("smoke"):
            unpinned = sorted({r.model for r in todo if r.model_config["expect"].startswith("TBD")})
            if unpinned:
                raise ValueError(
                    f"no pinned model ID to check for {', '.join(unpinned)} (models.toml)"
                )
        atts = answers(stage)  # a request's attempts in every sharing stage count
        if resend_versions:  # a stale answer's re-send counts its attempts afresh
            atts = _resent(atts)
        _run_meta()  # once, before any worker thread records
        down, capped, refused, over = [], [], [], 0.0
        # every batch first, as it goes in seconds; then the models sent request by request
        mine = {m: [r for r in todo if r.model == m] for m in dict.fromkeys(r.model for r in todo)}
        batched = [m for m, rs in mine.items() if meta[rs[0].custom_id]["mode"] == "batch"]
        for model in batched:
            provider = providers.get(mine[model][0].model_config)
            for chunk in _chunks(mine[model]):
                try:
                    batch_id, ids, error = provider.submit_batch(chunk), None, None
                except providers.PartialBatch as e:  # recorded, so never sent again
                    batch_id, ids, error = e.batch_id, e.custom_ids, e
                except Exception as e:  # nothing taken: its requests wait, the others go
                    print(f"{model}: batch refused: {type(e).__name__}: {e}")
                    refused.append(e)
                    break
                ids = [r.custom_id for r in chunk] if ids is None else ids
                append_jsonl(RUNS / stage / "batches.jsonl", {
                    "batch_id": batch_id, "model": model, "custom_ids": ids,
                    "meta": {c: meta[c] for c in ids}, "submitted": now(),
                })  # fmt: skip
                print(f"{model}: batch {batch_id} with {len(ids)} requests")
                if error:
                    refused.append(error)
                    break
        online = [m for m in mine if m not in batched]
        one_by_one = in_turn or stage == "latency"  # the latency sample: one call at a time
        groups = [[m] for m in online] if one_by_one else [online] * bool(online)
        for i, group in enumerate(groups):
            later = {m for g in groups[i + 1 :] for m in g}  # their price stays set aside for them
            ahead = [meta[r.custom_id] for m in later for r in mine[m]]
            sent = [meta[r.custom_id] for m in group for r in mine[m]]
            room = _room(stage, sent, ahead, seen, over)
            stopped = _send_together(stage, group, mine, meta, atts, room, workers)
            if room is not None:  # what attempts already out cost past the room passes the cap,
                over += room.passed  # rather than take the later models' price
            for model in group:
                if model in stopped:
                    error = stopped[model]
                    (capped if isinstance(error, CapReached) else down).append(str(error))
                else:
                    print(f"{model}: {len(mine[model])} requests sent")
        if capped:
            raise CapReached("; ".join(capped + down))
        if down:
            raise ProviderDown("; ".join(down))
        if refused:  # the first; each was printed as it came
            raise refused[0]
    return len(todo)


def _refuse_early_standard(stage: str, models: set[str]) -> None:
    """The standard-tier fallback (PREREG, Operations): only for a test request, and only once its
    model is FLEX_GRACE into its window, counted from its first start canary."""
    if not models:
        return
    if stage not in TEST_STAGES:
        raise ValueError(f"only a test request may leave Flex, not {stage}'s")
    first: dict[str, datetime] = {}
    for n in (1, 2, 3):
        for r in read_jsonl(RUNS / f"canary-{n}" / "responses.jsonl"):
            if r["model"] in models and (when := r.get("started") or r.get("recorded")):
                t = datetime.fromisoformat(when)
                first[r["model"]] = min(first.get(r["model"], t), t)
    now_ = datetime.now(UTC)
    if early := sorted(m for m in models if m not in first or now_ - first[m] < FLEX_GRACE):
        raise ValueError(f"not {FLEX_GRACE.days} days into the window yet: {', '.join(early)}")


def collect(stage: str) -> int:
    """Read every finished batch, once: a reply already recorded for this batch is skipped, so a
    rerun after a crash records nothing twice. Request errors go back to pending for the next
    submit; a model-ID mismatch stops the stage once the batch is recorded. A stage sharing
    answers reads the others' batches too, each reply recorded under the stage that sent it."""
    models = config("models")
    done = 0
    with lock(_lock(stage)):
        blocked = set(mismatched(stage))  # their batches wait; other models' are read
        for holder, batch in [(s, b) for s in _sharing(stage) for b in open_batches(s)]:
            model = {**models[batch["model"]], "key": batch["model"]}
            provider = providers.get(model)
            if batch["model"] in blocked or not provider.batch_done(batch["batch_id"]):
                continue
            own, atts = attempts(holder), answers(holder)
            seen = {
                a["custom_id"]
                for rows in own.values()
                for a in rows
                if a["batch_id"] == batch["batch_id"]
            }
            for reply in provider.batch_results(batch["batch_id"], batch["custom_ids"]):
                if reply.custom_id in seen or reply.custom_id not in batch["meta"]:
                    continue
                n = len(counted(atts.get(reply.custom_id))) + 1
                record(holder, model, reply, batch["meta"][reply.custom_id], n, batch["batch_id"])
                seen.add(reply.custom_id)  # a reply the provider repeats is recorded once
            append_jsonl(
                RUNS / holder / "batches.jsonl", {"batch_id": batch["batch_id"], "collected": now()}
            )
            done += 1
    _refuse_mismatches(stage)
    return done


def status(stage: str) -> dict[str, dict[str, int]]:
    """Per model, its requests by state: final (by status, naming the stage an answer is shared
    from), in a batch (naming another stage's), a model mismatch, or pending."""
    reqs, _ = build(stage)
    flying, own, atts = in_flight(stage), attempts(stage), answers(stage)
    out: dict[str, dict[str, int]] = {}
    for r in reqs:
        mine, a = current(own.get(r.custom_id)), atts.get(r.custom_id)
        # this stage's last attempt, else the request's history's
        tail = (mine or a or [{}])[-1].get("status")
        if tail == "model_mismatch" and not is_final(mine):
            state = "model_mismatch"
        elif r.custom_id in flying:  # a re-send in a batch is out, whatever answered before
            holder = flying[r.custom_id]
            state = "in batch" if holder == stage else f"in batch ({holder})"
        elif is_final(a):
            last = counted(a)[-1]
            state = last["status"] + (f", from {last['run_id']}" if last["run_id"] != stage else "")
        else:
            state = "pending"
        row = out.setdefault(r.model, {})
        row[state] = row.get(state, 0) + 1
    return out


# ------------------------------------------------------------------------------------ cost


def tokens(usage: dict) -> tuple[int, int]:
    """(input, output) tokens from any provider's usage, thinking counted as output."""
    if "promptTokenCount" in usage:  # Gemini
        inp = usage["promptTokenCount"]
        if "totalTokenCount" in usage:
            return inp, usage["totalTokenCount"] - inp
        return inp, (usage.get("candidatesTokenCount") or 0) + (
            usage.get("thoughtsTokenCount") or 0
        )
    if "prompt_tokens" in usage:  # Chat Completions (xAI's image route)
        inp = usage["prompt_tokens"]
        return inp, usage.get("total_tokens", inp + usage.get("completion_tokens", 0)) - inp
    inp = sum(
        usage.get(k) or 0
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    if "total_tokens" in usage:  # Responses (Azure, xAI): reasoning wherever it is counted
        return inp, usage["total_tokens"] - inp
    return inp, usage.get("output_tokens") or 0


def cache_writes(usage: dict) -> int:
    """Input tokens a provider reports writing to its prompt cache, which Azure bills for GPT-6
    at `cache_write` times the input price (prices.toml)."""
    return (usage.get("input_tokens_details") or {}).get("cache_write_tokens") or 0


def searches(usage: dict) -> int:
    """xAI's billed document searches on C1, by either usage field it reports (stage B1: none on
    one PDF, one on R6's several)."""
    return usage.get("num_sources_used") or usage.get("num_server_side_tools_used") or 0


def price(model: str) -> dict | None:
    return config("prices").get(model)


def _refuse_expired(models: set[str]) -> None:
    """A promotional price (`until` in prices.toml, its last day) stops sending and dry runs
    once it has passed, until prices.toml holds the price list then in force (PREREG,
    Operations: Cost)."""
    today = date.today().isoformat()
    gone = sorted(f"{m} (until {p['until']})" for m in models
                  if (p := price(m)) and "until" in p and p["until"] < today)  # fmt: skip
    if gone:
        raise ValueError(f"prices.toml's price has expired for {', '.join(gone)}: update it")


def _usd(p: dict, inp: float, out: float, discount: str | None, n_searches: float,
         writes: float = 0.0) -> float:  # fmt: skip
    """USD at price table `p`: tokens at the `batch` or `flex` multiplier where one applies, the
    `writes` of them written to a prompt cache at `cache_write` times the input price, and xAI's
    document searches at their own price."""
    rate = p.get(discount, 1.0) if discount else 1.0
    written = writes * p["input"] * (p.get("cache_write", 1.0) - 1.0)
    tokens_usd = (inp * p["input"] + written + out * p["output"]) / 1e6 * rate
    return tokens_usd + p.get("per_search", 0.0) * n_searches


def cost(
    model: str, usage: dict, discount: str | None = None, at_list: bool = False
) -> float | None:
    """USD for one answer; `discount` is `batch` or `flex`, its multiplier in prices.toml. With
    `at_list`, at the list price once a promotion ends (`list_input`, `list_output`), where a
    model has one."""
    p = price(model)
    if p is None or not usage:
        return None
    if at_list:
        p = p | {
            "input": p.get("list_input", p["input"]),
            "output": p.get("list_output", p["output"]),
        }
    return _usd(p, *tokens(usage), discount, searches(usage), cache_writes(usage))


BILLED = {"ok", "refusal", "length", "parse_error", "model_mismatch"}
PDF_INPUTS = ("C1", "C1-scan")  # where xAI bills its document searches
GROUP_ORDER = ("b2", "b3", "later")


def spent() -> float:
    """USD recorded so far, over every stage."""
    return sum(r["cost_usd"] or 0.0 for path in RUNS.glob("*/responses.jsonl")
               for r in read_jsonl(path) if "custom_id" in r)  # fmt: skip


def _observed() -> tuple[dict, dict, dict]:
    """From every billed answer so far (the smoke test, then dev): input tokens by route and
    image size (few-shot requests apart), output tokens by model and rung and by model, and xAI's
    searches by model and input. Output-limit stops count: they are the costliest answers."""
    sizes = render.manifest()
    inp: dict[tuple, list[int]] = {}
    out: dict[tuple, list[int]] = {}
    found: dict[tuple, list[int]] = {}
    for path in RUNS.glob("*/responses.jsonl"):
        for row in read_jsonl(path):
            if "custom_id" not in row or row["status"] not in BILLED or not row.get("usage"):
                continue
            i, o = tokens(row["usage"])
            px = tuple(sizes.get(row["page_id"], {}).get("c2_px", ()))
            few = row.get("fewshot", False)
            for key in ((row["model"], row["input"], px, few), (row["model"], row["input"], few)):
                inp.setdefault(key, []).append(i)
            for key in ((row["model"], row["rung"]), (row["model"],)):
                out.setdefault(key, []).append(o)
            found.setdefault((row["model"], row["input"]), []).append(searches(row["usage"]))
    return inp, out, found


def _p95(xs: list[float]) -> float:
    return statistics.quantiles(xs, n=20, method="inclusive")[-1] if len(xs) > 1 else float(xs[0])


def _outs(out_seen: dict, model: str, rung: str) -> list[int] | None:
    """Output tokens seen for a model at a rung. For a rung it hasn't answered yet (a flagship's
    R1, the QA probe): its R2's, else its R0's, where the rung answers in text; every rung's,
    JSON's included, where it answers in JSON (R4–R6)."""
    if outs := out_seen.get((model, rung)):
        return outs
    if not (rung in prompts.RUNGS and prompts.json_answer(rung)):
        for near in ("R2", "R0"):
            if outs := out_seen.get((model, near)):
                return outs
    return out_seen.get((model,))


def _priced(rows: list[tuple[str, dict]], smoke: bool = False, each: dict | None = None,
            seen: tuple | None = None) -> tuple[dict, set]:  # fmt: skip
    """USD for requests, each given by its stage and its meta, from the usage seen so far: input
    tokens at the median seen for the same model, route and image size (few-shot requests
    apart), all of it written to a prompt cache where that bills more unless the request turns
    the cache off; output and thinking at the 95th percentile seen per model and rung (`_outs`
    for a rung not yet answered; per model with `smoke`, the smoke test having run few rungs,
    and doubled); a PDF's searches at the 95th percentile seen on its input. Nothing is sent,
    not even to count tokens. Returns USD per (stage, model) and the cells it can't price;
    `each`, if given, gets each request's USD by custom ID. `seen` is a look at the usage
    (`_observed`) taken earlier, to price from instead."""
    inp_seen, out_seen, searches_seen = seen if seen is not None else _observed()
    sizes = render.manifest()
    by: dict[tuple, float] = {}
    missing: set[str] = set()
    for stage, m in rows:
        model, p = m["model"], price(m["model"])
        px = tuple(sizes.get(m["page_id"], {}).get("c2_px", ()))
        few = m["fewshot"]
        ins = inp_seen.get((model, m["input"], px, few), inp_seen.get((model, m["input"], few)))
        outs = out_seen.get((model,)) if smoke else _outs(out_seen, model, m["rung"])
        pdf = m["input"] in PDF_INPUTS and p and p.get("per_search")
        found = searches_seen.get((model, m["input"])) if pdf else [0]
        if p is None or not ins or not outs or not found:
            missing.add(f"{model}:{m['rung']}:{m['input']}" + ("" if p else " (no price)"))
            continue
        # a Flex request sent at the standard tier (`submit --standard-tier`) bills at full price
        flex_sent = m["mode"] == "flex" and m.get("sent_tier") != "standard"
        discount = "batch" if m["mode"] == "batch" else "flex" if flex_sent else None
        out = _p95(outs) * (2 if smoke else 1)
        inp = statistics.median(ins)
        writes = 0.0 if m.get("settings", {}).get("prompt_cache") == "off" else inp
        usd = _usd(p, inp, out, discount, _p95(found), writes=writes)
        by[(stage, model)] = by.get((stage, model), 0.0) + usd
        if each is not None:
            each[m["custom_id"]] = usd
    return by, missing


def _in_flight_meta() -> list[tuple[str, dict]]:
    """Every request in a batch not yet collected, in any stage, with that stage: billed once
    collected, so `spent` doesn't count it yet."""
    return [(path.parent.name, b["meta"][cid]) for path in sorted(RUNS.glob("*/batches.jsonl"))
            for b in open_batches(path.parent.name) for cid in b["custom_ids"]
            if cid in b["meta"]]  # fmt: skip


def _cap_for(stages: list[str]) -> tuple[float | None, str | None]:
    """The cap in force over `stages`: their latest group's (plans.cap), and that group; None and
    None for the smoke tests alone."""
    groups = [plans.group(s) for s in stages if plans.group(s) != "b1"]
    if not groups:
        return None, None
    top = max(groups, key=GROUP_ORDER.index)
    return plans.cap(next(s for s in stages if plans.group(s) == top), config("budget")), top


def _check_cap(stage: str, metas: list[dict], seen: tuple | None = None) -> None:
    """`submit`'s guard (PREREG, Operations): the stage's requests about to go, priced as the dry
    run prices them (from `seen`, the usage `submit` looked at, if given), plus what is in flight
    in any stage, plus what is spent, within the cap in force; the smoke tests apart. A request
    it can't price, or a cap not yet set, refuses too."""
    if plans.group(stage) == "b1" or not metas:
        return
    limit = plans.cap(stage, config("budget"))
    if limit is None:
        raise ValueError(f"{stage}: its group's cap is not set yet (config/budget.toml)")
    by, missing = _priced([(stage, m) for m in metas] + _in_flight_meta(), seen=seen)
    if missing:
        raise ValueError(f"{stage}: can't price {', '.join(sorted(missing))}, so nothing is sent")
    so_far, ahead = spent(), sum(by.values())
    if so_far + ahead > limit:
        raise ValueError(f"{stage}: ${ahead:.2f} to send or in flight, with ${so_far:.2f} spent, "
                         f"passes the ${limit:.2f} cap; see `harness dry-run`")  # fmt: skip


class Room:
    """What the cap in force leaves the models a `submit` sends request by request together
    (PREREG, Operations: the caps), shared by their workers. Each model starts with a part of
    its own, the price of its requests, and the rest of the room is common slack. An attempt goes
    only if its price fits in its model's part and the slack, and sets that price aside, from the
    part first, until the attempt's cost replaces it in the part. So a model's next request goes
    only if it fits in what the cap leaves after what is spent, what is in flight and the parts
    kept for the other models; and an attempt costing more than its price takes its model's part,
    then the slack, never another model's part. What a model's attempts come in under their price
    stays in its part until the model stops or has sent everything, when its part goes back to
    the slack: a tight cap may leave another model's requests for the next call, never spend past
    it. A part overspent past the slack passes the cap (`passed`), as attempts already out finish
    rather than take the price kept for the models still to go; an answer still out when its
    model stopped pays that back first. Every change holds the lock: the workers share it."""

    def __init__(self, left: float, price: dict[str, float], model_of: dict[str, str]) -> None:
        """`left`: the room as the models start; `price`: each request's, by custom ID;
        `model_of`: each request's model, the models in the order they go."""
        self.price, self.passed, self.closed = price, 0.0, set()
        self.part: dict[str, float] = {}
        for cid, model in model_of.items():
            self.part[model] = self.part.get(model, 0.0) + price[cid]
        self.owed = dict.fromkeys(self.part, 0.0)  # what each model has passed the cap by
        self.slack = left - sum(self.part.values())
        for model in self.part:  # a room short of the parts: the first models give way, as a
            give = min(self.part[model], max(0.0, -self.slack))  # model sent earlier did
            self.part[model] -= give
            self.slack += give
        self.slack = max(0.0, self.slack)  # a room never starts below nothing
        self._lock = threading.Lock()

    def take(self, model: str, cid: str) -> bool:
        """Set an attempt's price aside, or say it doesn't fit (or its model has stopped)."""
        p = self.price[cid]
        with self._lock:
            # float sums: a nano-dollar's slack
            if model in self.closed or self.part[model] + self.slack < p - 1e-9:
                return False
            own = min(max(self.part[model], 0.0), p)
            self.part[model] -= own
            self.slack -= p - own
            return True

    def settle(self, model: str, cid: str, cost: float) -> None:
        """An attempt's cost replaces its price in its model's part; an overrun past the part
        takes the slack next."""
        with self._lock:
            self.part[model] += self.price[cid] - cost
            if self.part[model] < 0:
                cover = min(-self.part[model], self.slack)
                self.part[model] += cover
                self.slack -= cover
            if model in self.closed:  # an attempt out when its model stopped
                self._release(model)

    def close(self, model: str) -> None:
        """A model that stops, or has sent everything: its part goes back to the slack."""
        with self._lock:
            self.closed.add(model)
            self._release(model)

    def _release(self, model: str) -> None:
        left, self.part[model] = self.part[model], 0.0
        if left >= 0:  # what passed the cap is paid back first
            back = min(left, self.owed[model])
            self.owed[model] -= back
            self.passed -= back
            self.slack += left - back
        else:  # overspent: the slack covers what it can, and the rest passes the cap
            cover = min(-left, self.slack)
            self.slack -= cover
            self.owed[model] += -left - cover
            self.passed += -left - cover


def _room(stage: str, mine: list[dict], ahead: list[dict], seen: tuple | None = None,
          over: float = 0.0) -> Room | None:  # fmt: skip
    """What the cap in force leaves the models sent request by request together (`mine`, their
    requests' meta), as they start: the cap, less what is spent, what is in flight in any stage's
    batches, and the price of the requests to go after them in the same `submit` (`ahead`), all
    priced from `seen` as `_check_cap` priced them. Models sent earlier in the call may have
    spent past their room by what their attempts already out cost over their price: that is
    `over`, given back here, so it passes the cap rather than take these models' price. A room
    never starts below nothing, so spending from outside the call (another stage's batch
    collected meanwhile) is never handed on as `over`; spending from outside meets the models
    sent together at the next call, those sent `in_turn` as each starts. None for the smoke
    tests, which come before any cap."""
    if plans.group(stage) == "b1":
        return None
    limit = plans.cap(stage, config("budget"))
    price_of: dict[str, float] = {}
    _, unpriced = _priced([(stage, m) for m in mine], each=price_of, seen=seen)
    by, missing = _priced([(stage, m) for m in ahead] + _in_flight_meta(), seen=seen)
    if limit is None or unpriced or missing:  # `_check_cap` refused all three before any send
        raise ValueError(f"{stage}: the cap can't be kept as requests go, so nothing more is sent")
    left = limit - spent() - sum(by.values()) + over
    return Room(left, price_of, {m["custom_id"]: m["model"] for m in mine})


def dry_run(stages: list[str], smoke: bool = False) -> float:
    """Price every request of the stages without a final answer, pending or in a batch, before
    more is spent (PREREG, Operations), and what other stages hold in batches not yet collected,
    by `_priced`: nothing is sent, so test pages can be priced before prereg-2. `b2` and `b3`
    name their groups (plans.GROUPS). A choice not yet made is priced at its costliest
    (plans.COSTLIEST), so stage B2 is priced before R-best is chosen. A request it can't price is
    an error, as the total would be too low, and so is a total that passes the cap in force with
    what is spent and in flight; a cap still TBD is reported, not checked. A sharing stage is
    priced with the others, as it sends their requests too. Returns the stages' total."""
    names = [x for s in stages for x in plans.GROUPS.get(s, (s,))]
    stages = list(dict.fromkeys(y for x in names for y in _sharing(x)))
    rows: list[tuple[str, dict]] = []
    shared: set[str] = set()  # SHARED stages send an identical request once: price it once
    with plans.costliest_choices() as assumed:
        for stage in stages:
            (reqs, meta), atts = build(stage, guard=False), answers(stage)
            flying = in_flight(stage)  # not yet billed, a re-send's old answer whatever
            _refuse_expired({r.model for r in reqs})
            for req in (r for r in reqs
                        if not is_final(atts.get(r.custom_id)) or r.custom_id in flying):  # fmt: skip
                if stage in plans.SHARED:
                    if req.custom_id in shared:
                        continue
                    shared.add(req.custom_id)
                rows.append((stage, meta[req.custom_id]))
    elsewhere = [x for x in _in_flight_meta() if x[0] not in stages]
    by, missing = _priced(rows + elsewhere, smoke)
    for (stage, model), usd in sorted(by.items()):
        print(f"  {stage:<16} {model:<18} ${usd:>9.2f}" + ("" if stage in stages else " in flight"))
    total = sum(v for (s, _), v in by.items() if s in stages)
    flying, so_far = sum(by.values()) - total, spent()
    print(
        f"total ${total:.2f}, with ${flying:.2f} in other stages' batches and ${so_far:.2f} spent"
    )
    if assumed:
        print("  priced at their costliest, not yet chosen: "
              + ", ".join(f"{p}'s {k} as {plans.COSTLIEST[k]}" for p, k in sorted(assumed)))  # fmt: skip
    if missing:
        raise ValueError(
            f"can't price (no price, or no usage seen yet): {', '.join(sorted(missing))}"
        )
    limit, top = _cap_for(stages)
    if top is not None and limit is None:
        print(f"  no cap is set yet for {top} (config/budget.toml)")
    elif limit is not None and so_far + flying + total > limit:
        hint = (
            ": add drop_order's next item to `dropped` in config/budget.toml" if top == "b3" else ""
        )
        raise ValueError(f"${so_far + flying + total:.2f} in all passes the ${limit:.2f} cap{hint}")
    return total


# ------------------------------------------------------------------------------------ scoring


def outputs(stage: str) -> list[dict]:
    """Every final answer of the stage, a shared one included, as tokens (None for a failed page),
    parsed now with the scorer as it stands, and its status as that parse reads it (an answer the
    scorer now reads is `ok`, one it can't is a `parse_error`); then the free arms on the stage's
    pages: P-scan's tokens carry its `cnn_prob`, and its output the model meta's band (`band`).
    `probabilities` marks the arms giving words a probability of being struck: JSON answers and
    P-scan."""
    _refuse_mismatches(stage)
    _, meta = build(stage)
    flying, atts = in_flight(stage), answers(stage)
    out = []
    for cid, m in meta.items():
        if not is_final(a := atts.get(cid)) or cid in flying:  # a re-send out: not yet answered
            continue
        last = counted(a)[-1]
        source = {"reused_from": last["run_id"]} if last["run_id"] != stage else {}
        tokens_, status = None, last["status"]
        json_answer = m["rung"] in prompts.RUNGS and prompts.json_answer(m["rung"])
        if status in ("ok", "parse_error") and not m["rung"].startswith("QA-"):
            try:
                tokens_ = parse(last["text"], json_answer=json_answer)
            except ParseError:
                tokens_ = None
            usable = tokens_ is not None and any(normalize(t.text)[0] for t in tokens_)
            status = "ok" if usable else "parse_error"
        out.append({
            "arm": m["arm"], "page_id": m["page_id"], "model": m["model"], "rung": m["rung"],
            "input": m["input"], "qa_id": m["qa_id"], "status": status,
            "attempts": len(counted(a)), "cost_usd": sum(x["cost_usd"] or 0 for x in a),
            "cost_list_usd": sum(_list_cost(x) or 0 for x in a),
            "priced": all(x["cost_usd"] is not None for x in a if x["status"] in BILLED),
            "reported_model": last.get("reported_model"),
            "expected_model": last.get("expected_model"),  # the pin it was sent under
            # a model's seconds per page come from the latency sample alone: elsewhere calls run
            # in parallel, batched or queued at Flex
            "latency_s": last["latency_s"] if stage == "latency" else None,
            "tokens": tokens_, "text": last["text"], "probabilities": json_answer, **source,
        })  # fmt: skip
    cache, timed = arms.cached(), arms.cached(arms.LATENCY)
    rate = config("prices").get("cpu", {}).get("usd_per_vcpu_hour")
    for pid in dict.fromkeys(m["page_id"] for m in meta.values()):
        for arm in arms.ARMS:
            if (rec := cache.get((arm, pid))) is not None:
                if "words" in rec:
                    toks = tokens_from_words(rec["words"], rec.get("cnn_prob"))
                else:  # B-naive outputs every word live (PREREG, Arms and inputs)
                    toks = parse(rec["text"], marks=arm != "B-naive")
                # billed: CPU time from the pinned cost pass, run in one process (a parallel
                # run's shares the machine: unpriced); latency: the default-thread pass
                one = rec.get("pass") == "cost" and rec.get("workers", 1) == 1
                cpu = rec.get("cpu_s") if one else None
                usd = None if cpu is None or rate is None else cpu * rate / 3600
                meta_ = (rec.get("package") or {}).get("model_meta") or {}
                band = {"band": [meta_["p_lo"], meta_["p_hi"]]} if "p_lo" in meta_ else {}
                out.append({"arm": arm, "page_id": pid, "model": arm,
                            "status": rec.get("status", "ok"),  # `error`: the package refused it
                            "cpu_s": cpu, "cost_usd": usd, "cost_list_usd": usd,
                            "priced": usd is not None,
                            "latency_s": timed.get((arm, pid), {}).get("wall_s"),
                            "tokens": toks, "probabilities": arm in arms.SCAN,
                            **(band if arm == "P-scan" else {})})  # fmt: skip
    return out


def _list_cost(row: dict) -> float | None:
    """An attempt's cost at list price, by the route that served it (`cost`, `at_list`)."""
    via = "batch" if row.get("batch_id") else None
    if not via and (row.get("extra") or {}).get("service_tier") == "flex":
        via = "flex"
    return cost(row["model"], row.get("usage") or {}, via, at_list=True)


def _versions(outs: list[dict], split: bool) -> None:
    """Refuse answers from more than one version of a model in one stage (PREREG, Operations),
    or with `split`, name each answer from a version other than the one now pinned as its own
    arm, `<arm>@<version>`. An answer naming no model (a failure, a filtered prompt) belongs to
    the version pinned when it was sent."""
    models = config("models")
    seen: dict[str, set[str]] = {}
    for o in outs:
        if o.get("reported_model"):
            seen.setdefault(o["model"], set()).add(o["reported_model"])
    mixed = {m: sorted(v) for m, v in seen.items() if len(v) > 1}
    if mixed and not split:
        raise ValueError(f"answers from more than one version of a model: {mixed}; send its "
                         "test requests again under the version now pinned (submit "
                         "--resend-versions), or score with --split-versions to report them by "
                         "version")  # fmt: skip
    for o in outs:
        version = o.get("reported_model") or o.get("expected_model")
        if o["model"] in mixed and version != _expected(models[o["model"]]):
            o["arm"] = f"{o['arm']}@{version}"


def unfinished(stage: str) -> list[str]:
    """What keeps a stage from being complete: requests without a final answer, and free arms
    missing on its pages. Scoring around them would drop those pages from both sides of every
    comparison instead of scoring them as failures."""
    reqs, meta = build(stage)
    flying, atts, cache = in_flight(stage), answers(stage), arms.cached()
    # a request in a batch not yet collected is not final, a re-send's old answer whatever
    todo = Counter(r.model for r in reqs
                   if not is_final(atts.get(r.custom_id)) or r.custom_id in flying)  # fmt: skip
    out = [f"{m}: {n} requests not final" for m, n in sorted(todo.items())]
    pages = dict.fromkeys(m["page_id"] for m in meta.values())
    missing = Counter(a for p in pages for a in arms.ARMS
                      if arms.applies(a, p) and (a, p) not in cache)  # fmt: skip
    return out + [f"{a}: {n} pages not run" for a, n in sorted(missing.items())]


def score(stage: str, partial: bool = False, split_versions: bool = False) -> list[dict]:
    """Score every final answer of the stage, and the free arms on the stage's pages. An
    incomplete stage is refused unless `partial` (while iterating on dev), and so is one where a
    model's answers report more than one version, unless `split_versions` (`_versions`). The
    commit, the tags and what changed since prereg-2 go to the stage's provenance.json."""
    _refuse_mismatches(stage)
    if (gaps := unfinished(stage)) and not partial:
        raise ValueError(
            f"{stage} is incomplete ({'; '.join(gaps)}); score with --partial to see it anyway"
        )
    pages = {p["page_id"]: p for p in load_pages(*SETS)}
    outs = outputs(stage)
    _versions(outs, split_versions)
    # the cascade's words: per page, those P-scan leaves uncertain (PREREG, Scoring), in the model
    # meta's band and in each of the grid's
    unsure = {}
    for o in (o for o in outs if o["arm"] == "P-scan" and "band" in o):
        bands = {"uncertain": tuple(o["band"]),
                 **{f"uncertain {lo:.2f}-{hi:.2f}": (lo, hi) for lo, hi in CASCADE_GRID}}  # fmt: skip
        unsure[o["page_id"]] = uncertain_bands(pages[o["page_id"]], o["tokens"], bands)
    rows = []
    for o in outs:
        page = pages[o["page_id"]]
        base = {k: v for k, v in o.items() if k not in ("tokens", "text", "probabilities")}
        base |= {"run_id": stage, "set": page["set"], "cluster": page["cluster"]}
        base["partial"] = partial  # the endpoints refuse a stage scored with --partial
        if "strict" in page:  # G2's bills introduced after the cutoff
            base["strict"] = page["strict"]
        if o.get("qa_id"):
            rows.append({**base, "answer": qa_outcome(o["qa_id"], o)})
        else:
            subsets = unsure.get(page["page_id"])
            given = o["probabilities"]
            rows.append({**base, **score_page(page, o["tokens"], subsets=subsets,
                                              probabilities=given)})  # fmt: skip
    write_jsonl(RUNS / stage / "scores.jsonl", rows)
    made = provenance() | {"scored": now(), "partial": partial}
    (RUNS / stage / "provenance.json").write_text(json.dumps(made, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")  # fmt: skip
    return rows


def qa_outcome(qa_id: str, response: dict) -> str:
    """Whether the answer gives the live value, the struck one, both or neither."""
    if response["status"] != "ok":
        return "failed"
    item = _qa_items()[qa_id]
    said = " " + " ".join(normalize(t)[0] for t in response["text"].split()) + " "

    def gives(value: str) -> bool:
        return " " + " ".join(normalize(t)[0] for t in value.split()) + " " in said

    live, struck = gives(item["live"]), gives(item["struck"])
    return "both" if live and struck else "live" if live else "struck" if struck else "neither"


# ------------------------------------------------------------------------------------ commands


def register(sub) -> None:
    p = sub.add_parser("harness", help="dry-run, submit, collect, status and score a stage")
    cmd = p.add_subparsers(required=True, metavar="action")
    d = cmd.add_parser("dry-run", help="price the pending requests")
    d.add_argument("stage", nargs="+", help="stages, or the groups b2 and b3")
    d.add_argument("--smoke", action="store_true", help="price from the smoke test, output doubled")
    d.set_defaults(func=lambda a: dry_run(a.stage, a.smoke))
    s = cmd.add_parser("submit", help="send the pending requests")
    s.add_argument("stage")
    s.add_argument("--model", nargs="+")
    s.add_argument("--workers", type=int, default=8, help="concurrent requests, unbatched models")
    s.add_argument("--standard-tier", action="store_true",
                   help="test requests not final 5 days into the window: Flex's at the standard tier")  # fmt: skip
    s.add_argument("--resend-versions", action="store_true",
                   help="test requests answered by a version no longer pinned: sent again")  # fmt: skip
    s.add_argument("--in-turn", action="store_true",
                   help="the unbatched models one after another, not all at once")  # fmt: skip
    s.set_defaults(func=lambda a: print(
        f"{submit(a.stage, a.model, a.workers, a.standard_tier, a.resend_versions, a.in_turn)} "
        "requests submitted"
    ))  # fmt: skip
    c = cmd.add_parser("collect", help="read finished batches")
    c.add_argument("stage")
    c.set_defaults(func=lambda a: print(f"{collect(a.stage)} batches collected"))
    t = cmd.add_parser("status", help="final, pending and in-batch requests per model")
    t.add_argument("stage")
    t.set_defaults(func=lambda a: [print(f"{k:<18} {v}") for k, v in status(a.stage).items()])
    r = cmd.add_parser("score", help="score the final answers and the free arms")
    r.add_argument("stage")
    r.add_argument("--partial", action="store_true", help="score an incomplete stage as it stands")
    r.add_argument("--split-versions", action="store_true",
                   help="a model's answers from another version than the pinned one as their own arm")  # fmt: skip
    r.set_defaults(func=lambda a: print(
        f"{len(score(a.stage, a.partial, a.split_versions))} scores -> {RUNS / a.stage}"
    ))  # fmt: skip
