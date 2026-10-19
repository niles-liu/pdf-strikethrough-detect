"""A deterministic in-memory provider for the harness's tests: no network, no keys.

Behaviour comes from the model's table in the config (`provider = "fake"`):

- `fake_text`: the answer (default: one naming the request's custom ID).
- `fail_times`: the first N calls for a custom ID, by any `Fake`, return `request_error`, with
  the error text `fail_error` (e.g. a rate limit's), naming the model `fail_model` if given (as
  Vertex's errors can).
- `status`: force `refusal` or `length`.
- `reported_model`: the model ID to report in place of `expect`, to trip the model-ID check.
- `counts_tokens`: answer `count_tokens()` as Anthropic does, rather than with None.
- `batch = false`: `submit_batch()` raises NotImplementedError, as for Grok 4.7.
- `fail_batch_after`: a batch takes this many requests, then fails (PartialBatch).
- A request's `service_tier` comes back in `extra`, as Azure reports the tier that served it.

`CALLS` lists the custom ID of every request any `Fake` has sent, batched or not, so a test can
assert that a rerun calls nothing even when the harness builds its own `Fake` through
`providers.get()`. Batches live beside it in `BATCHES`, as on a real server, so one `Fake` can
collect what another submitted; `reset()` clears both. Each instance also keeps its own `calls`.
"""

import uuid

from bench.providers import PartialBatch, Reply, Request, now

CALLS: list[str] = []
BATCHES: dict[str, list[Reply]] = {}


def reset() -> None:
    CALLS.clear()
    BATCHES.clear()


def _input_tokens(req: Request) -> int:
    return sum(len(p.text.split()) if p.kind == "text" else 1000 for p in req.parts)


class Fake:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def body(self, req: Request) -> dict:
        parts = [
            {"kind": p.kind, "text": p.text, "path": p.path, "sha256": p.sha256} for p in req.parts
        ]
        return {
            "model": req.model_config["id"],
            "system": req.system,
            "parts": parts,
            "effort": req.effort,
            "max_tokens": req.max_tokens,
        }

    def _reply(self, req: Request) -> Reply:
        config = req.model_config
        CALLS.append(req.custom_id)
        self.calls.append(req.custom_id)
        at = now()
        if CALLS.count(req.custom_id) <= config.get("fail_times", 0):
            error = {"error": config.get("fail_error", "FakeError: told to fail")}
            return Reply(req.custom_id, "request_error", reported_model=config.get("fail_model"),
                         extra=error, started=at, finished=at)  # fmt: skip
        text = config.get("fake_text", f"Transcription of {req.custom_id}.")
        return Reply(
            req.custom_id,
            config.get("status", "ok"),
            text=text,
            usage={"input_tokens": _input_tokens(req), "output_tokens": len(text.split())},
            reported_model=config.get("reported_model", config["expect"]),
            extra={"finish_reason": "stop", "service_tier": req.service_tier},
            started=at,
            finished=at,
        )

    def send(self, req: Request) -> Reply:
        reply = self._reply(req)
        reply.latency_s = 0.0
        return reply

    def submit_batch(self, reqs: list[Request]) -> str:
        if not reqs[0].model_config.get("batch", True):
            raise NotImplementedError(f"{reqs[0].model} has no batch API")
        batch_id = f"fake-batch-{uuid.uuid4().hex[:8]}"
        n = reqs[0].model_config.get("fail_batch_after", len(reqs))
        BATCHES[batch_id] = [self._reply(r) for r in reqs[:n]]
        if n < len(reqs):
            error = RuntimeError("FakeError: the batch failed")
            raise PartialBatch(batch_id, [r.custom_id for r in reqs[:n]]) from error
        return batch_id

    def batch_done(self, batch_id: str) -> bool:
        return batch_id in BATCHES

    def batch_results(self, batch_id: str, custom_ids: list[str]) -> list[Reply]:
        return list(BATCHES[batch_id])

    def count_tokens(self, req: Request) -> int | None:
        return _input_tokens(req) if req.model_config.get("counts_tokens") else None
