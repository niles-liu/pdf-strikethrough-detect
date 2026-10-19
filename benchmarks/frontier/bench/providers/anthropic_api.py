"""Anthropic: the Claude models through the Messages API and Message Batches (PREREG.md, Models).

Effort goes in `output_config.effort`, beside adaptive thinking, the one thinking mode every Claude
5.x model takes. No `fallbacks` are sent, so a refusal is never answered by another model. A page
is a base64 `image` block, or a base64 `document` block for a PDF (C1).
"""

import base64
import os
import time

import anthropic
import httpx2

from bench.providers import Part, Reply, Request, failed, now

ERRORS = (anthropic.APIError, httpx2.HTTPError)
LENGTH = {"max_tokens", "model_context_window_exceeded"}


def _source(part: Part, media_type: str) -> dict:
    data = base64.b64encode(part.data()).decode()
    return {"type": "base64", "media_type": media_type, "data": data}


def _block(part: Part) -> dict:
    match part.kind:
        case "text":
            return {"type": "text", "text": part.text}
        case "image":
            return {"type": "image", "source": _source(part, "image/png")}
        case "pdf":
            return {"type": "document", "source": _source(part, "application/pdf")}
    raise ValueError(f"unknown part kind {part.kind!r}")


def reply_from_message(custom_id: str, message: dict) -> Reply:
    """A Messages API response, as a dict, as a Reply: its text blocks, never its thinking."""
    stop = message.get("stop_reason")
    status = "refusal" if stop == "refusal" else "length" if stop in LENGTH else "ok"
    extra = {"stop_reason": stop, "stop_details": message.get("stop_details"), "id": message["id"]}
    return Reply(
        custom_id,
        status,
        text="".join(b["text"] for b in message["content"] if b["type"] == "text"),
        usage=message.get("usage") or {},
        reported_model=message.get("model"),
        extra=extra,
    )


class Anthropic:
    name = "anthropic"

    def __init__(self) -> None:
        self._client: anthropic.Anthropic | None = None

    def _api(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=2
            )
        return self._client

    def body(self, req: Request) -> dict:
        body = {"model": req.model_config["id"], "max_tokens": req.max_tokens}
        if req.system is not None:
            body["system"] = req.system
        return body | {
            "messages": [{"role": "user", "content": [_block(p) for p in req.parts]}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": req.effort},
        }

    def send(self, req: Request) -> Reply:
        body = self.body(req)
        client = self._api().with_options(max_retries=req.retries)
        started, t0 = now(), time.perf_counter()
        try:
            # Streamed: the SDK refuses a plain call whose max_tokens could outlast its timeout.
            with client.messages.stream(**body) as stream:
                message = stream.get_final_message()
        except ERRORS as e:
            return failed(req.custom_id, e, started)
        latency, finished = time.perf_counter() - t0, now()
        reply = reply_from_message(req.custom_id, message.model_dump(mode="json"))
        # The SDK's own retries happen inside the timed call (none in the latency sample).
        reply.extra["retries"] = int(
            stream.response.request.headers.get("x-stainless-retry-count", 0)
        )
        reply.started, reply.finished, reply.latency_s = started, finished, latency
        return reply

    def submit_batch(self, reqs: list[Request]) -> str:
        requests = [{"custom_id": r.custom_id, "params": self.body(r)} for r in reqs]
        return self._api().messages.batches.create(requests=requests).id

    def batch_done(self, batch_id: str) -> bool:
        return self._api().messages.batches.retrieve(batch_id).processing_status == "ended"

    def batch_results(self, batch_id: str, custom_ids: list[str]) -> list[Reply]:
        client = self._api()
        batch = client.messages.batches.retrieve(batch_id)
        started = batch.created_at.isoformat()
        finished = batch.ended_at.isoformat() if batch.ended_at else ""
        replies = {}
        for item in client.messages.batches.results(batch_id):
            result = item.result
            if result.type == "succeeded":
                reply = reply_from_message(item.custom_id, result.message.model_dump(mode="json"))
            else:
                error = result.error.error if result.type == "errored" else None
                text = f"{error.type}: {error.message}" if error else result.type
                reply = Reply(item.custom_id, "request_error", extra={"error": text})
            replies[item.custom_id] = reply
        missing = {"error": f"no result; the batch ended {batch.processing_status}"}
        out = [replies.get(c) or Reply(c, "request_error", extra=missing) for c in custom_ids]
        for reply in out:
            reply.started, reply.finished = started, finished
        return out

    def count_tokens(self, req: Request) -> int | None:
        body = self.body(req)
        del body["max_tokens"]
        return self._api().messages.count_tokens(**body).input_tokens
