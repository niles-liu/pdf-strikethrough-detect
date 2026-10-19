"""xAI: Grok through Chat Completions for images, the Responses API for a PDF (C1), and the Batch
API for Grok 4.3 (PREREG.md, Models), with the OpenAI SDK at api.x.ai.

A PDF goes up through the Files API and is referenced by ID. xAI's docs describe a search tool over
it, but in stage B1 the model read the page itself: it transcribed an image-only PDF with no search
call, and searched once only when R6's worked examples brought several PDFs. The file is deleted once the answer is in: after a sequential call, or, for a batch,
when its results are read, whichever process reads them (a batch's files carry its ID in their
names). The Batch API is xAI's own (create a batch, add requests, page through results), called
through the SDK's low-level `get` and `post`; it answers a Responses request in Chat Completions
form too (stage B1).
"""

import base64
import json
import os
import time
from collections.abc import Iterator

import httpx2
import openai

from bench.providers import PartialBatch, Part, Reply, Request, failed, now
from bench.providers.azure import responses_outcome

ERRORS = (openai.APIError, httpx2.HTTPError)
BASE_URL = "https://api.x.ai/v1"
# A 64,000-token answer after long reasoning can outlast the SDK's 10-minute default.
TIMEOUT_S = 3600
ADD_LIMIT_BYTES = 20_000_000  # per add-requests call; xAI caps a payload at 25 MB


def _data_url(part: Part) -> str:
    return f"data:image/png;base64,{base64.b64encode(part.data()).decode()}"


def _chat_part(part: Part, config: dict) -> dict:
    match part.kind:
        case "text":
            return {"type": "text", "text": part.text}
        case "image":
            image = {"url": _data_url(part), "detail": config["image_detail"]}
            return {"type": "image_url", "image_url": image}
    raise ValueError(f"part kind {part.kind!r} has no Chat Completions form")


def _responses_item(part: Part, config: dict) -> dict:
    match part.kind:
        case "text":
            return {"type": "input_text", "text": part.text}
        case "image":
            image = _data_url(part)
            return {"type": "input_image", "image_url": image, "detail": config["image_detail"]}
        case "pdf":
            return {"type": "input_file", "file_id": None}  # filled in after the upload
    raise ValueError(f"unknown part kind {part.kind!r}")


def reply_from_chat(custom_id: str, completion: dict) -> Reply:
    """A chat completion, as a dict, as a Reply: the message content, never its reasoning."""
    choice = completion["choices"][0]
    message, finish = choice.get("message") or {}, choice.get("finish_reason")
    if message.get("refusal") or finish == "content_filter":
        status = "refusal"
    else:
        status = "length" if finish == "length" else "ok"
    extra = {
        "finish_reason": finish,
        "refusal": message.get("refusal"),
        "system_fingerprint": completion.get("system_fingerprint"),
        "id": completion.get("id"),
    }
    return Reply(
        custom_id,
        status,
        text=message.get("content") or "",
        usage=completion.get("usage") or {},
        reported_model=completion.get("model"),
        extra=extra,
    )


def reply_from_responses(custom_id: str, resp: dict) -> Reply:
    """A Responses API response, as a dict, as a Reply."""
    status, text, reason = responses_outcome(resp)
    extra = {
        "response_status": resp.get("status"),
        "incomplete_reason": reason,
        "system_fingerprint": resp.get("system_fingerprint"),
        "id": resp.get("id"),
    }
    if status == "request_error":
        extra["error"] = str(resp.get("error") or f"response {resp.get('status')}")
    return Reply(
        custom_id,
        status,
        text=text,
        usage=resp.get("usage") or {},
        reported_model=resp.get("model"),
        extra=extra,
    )


def _reply_from_result(result: dict) -> Reply:
    """One entry of a batch's results."""
    custom_id = result["batch_request_id"]
    outcome = result.get("batch_result") or {}
    # the error field, unseen in stage B1 as no request failed: the guide says `error_message`,
    # the reference `batch_result.error`, so both are read
    error = result.get("error_message") or outcome.get("error")
    response = outcome.get("response")
    if error or not isinstance(response, dict):
        return Reply(custom_id, "request_error", extra={"error": str(error or "no response")})
    # stage B1: a Responses request's answer comes back under `chat_get_completion` too; any
    # other key is read as a Responses result
    if "chat_get_completion" in response:
        return reply_from_chat(custom_id, response["chat_get_completion"])
    return reply_from_responses(custom_id, next(iter(response.values())))


def _chunks(items: list[dict]) -> Iterator[list[dict]]:
    chunk, size = [], 0
    for item in items:
        n = len(json.dumps(item))
        if chunk and size + n > ADD_LIMIT_BYTES:
            yield chunk
            chunk, size = [], 0
        chunk.append(item)
        size += n
    if chunk:
        yield chunk


class XAI:
    name = "xai"

    def __init__(self) -> None:
        self._client: openai.OpenAI | None = None

    def _api(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(
                base_url=BASE_URL,
                api_key=os.environ["XAI_API_KEY"],
                max_retries=2,
                timeout=TIMEOUT_S,
            )
        return self._client

    def body(self, req: Request) -> dict:
        """Chat Completions for text and images; the Responses API once a PDF is in the request,
        each PDF an `input_file` whose `file_id` is set when the file is uploaded."""
        config, model = req.model_config, req.model_config["id"]
        if any(p.kind == "pdf" for p in req.parts):
            items = [_responses_item(p, config) for p in req.parts]
            body = {"model": model, "input": [{"role": "user", "content": items}]}
            if req.system is not None:
                body["instructions"] = req.system
            return body | {"max_output_tokens": req.max_tokens, "reasoning": {"effort": req.effort}}
        messages = [] if req.system is None else [{"role": "system", "content": req.system}]
        messages.append({"role": "user", "content": [_chat_part(p, config) for p in req.parts]})
        return {
            "model": model,
            "messages": messages,
            "max_completion_tokens": req.max_tokens,
            "reasoning_effort": req.effort,
        }

    def _upload(self, req: Request, body: dict, file_ids: list[str], name: str = "page") -> None:
        """Upload the request's PDFs as `<name>.pdf`, appending each ID to `file_ids` and setting
        it in `body`."""
        if "input" not in body:
            return
        items = [i for i in body["input"][0]["content"] if i["type"] == "input_file"]
        pdfs = [p for p in req.parts if p.kind == "pdf"]
        for item, part in zip(items, pdfs, strict=True):
            upload = (f"{name}.pdf", part.data(), "application/pdf")
            file = self._api().files.create(file=upload, purpose="assistants")
            file_ids.append(file.id)
            item["file_id"] = file.id

    def _delete(self, file_ids: list[str]) -> list[str]:
        """Delete uploaded files; returns those that could not be deleted."""
        left = []
        for file_id in file_ids:
            try:
                self._api().files.delete(file_id)
            except ERRORS:
                left.append(file_id)
        return left

    def send(self, req: Request) -> Reply:
        body = self.body(req)
        client = self._api().with_options(max_retries=req.retries)
        file_ids: list[str] = []
        started = now()
        try:
            self._upload(req, body, file_ids)
            started, t0 = now(), time.perf_counter()
            if "messages" in body:
                raw = client.chat.completions.with_raw_response.create(**body)
            else:
                raw = client.responses.with_raw_response.create(**body)
            payload = raw.parse().model_dump(mode="json")
            latency, finished = time.perf_counter() - t0, now()
        except ERRORS as e:
            reply = failed(req.custom_id, e, started)
        else:
            parse = reply_from_chat if "messages" in body else reply_from_responses
            reply = parse(req.custom_id, payload)
            reply.extra["retries"] = raw.retries_taken
            reply.started, reply.finished, reply.latency_s = started, finished, latency
        if undeleted := self._delete(file_ids):
            reply.extra["undeleted_files"] = undeleted
        return reply

    def submit_batch(self, reqs: list[Request]) -> str:
        if not reqs[0].model_config.get("batch"):
            raise NotImplementedError(f"{reqs[0].model} has no batch API")
        client = self._api()
        stamp = time.strftime("%Y%m%dt%H%M%S", time.gmtime())
        name = f"bench-{reqs[0].model}-{stamp}"
        batch_id = client.post("/batches", body={"name": name}, cast_to=dict)["batch_id"]
        file_ids, added = [], []
        try:
            items = []
            for r in reqs:
                body = self.body(r)
                self._upload(r, body, file_ids, name=batch_id)
                # a batch takes a Responses request under `responses` (stage B1)
                endpoint = "chat_get_completion" if "messages" in body else "responses"
                items.append({"batch_request_id": r.custom_id, "batch_request": {endpoint: body}})
            for chunk in _chunks(items):
                client.post(
                    f"/batches/{batch_id}/requests", body={"batch_requests": chunk}, cast_to=dict
                )
                added += [i["batch_request_id"] for i in chunk]
        except ERRORS as e:
            if added:  # those run and are billed: the harness must collect them, not resend
                raise PartialBatch(batch_id, added) from e
            self._delete(file_ids)  # the batch holds nothing
            raise
        return batch_id

    def _pages(self, path: str, key: str) -> Iterator[dict]:
        token = None
        while True:
            params = {"limit": 100} | ({"pagination_token": token} if token else {})
            page = self._api().get(path, cast_to=dict, options={"params": params})
            yield from page.get(key) or []
            if not (token := page.get("pagination_token")):
                return

    def batch_done(self, batch_id: str) -> bool:
        batch = self._api().get(f"/batches/{batch_id}", cast_to=dict)
        return batch["state"]["num_pending"] == 0

    def batch_results(self, batch_id: str, custom_ids: list[str]) -> list[Reply]:
        replies = {}
        for result in self._pages(f"/batches/{batch_id}/results", "results"):
            reply = _reply_from_result(result)
            replies[reply.custom_id] = reply
        out = []
        for meta in self._pages(f"/batches/{batch_id}/requests", "batch_request_metadata"):
            cid = meta["batch_request_id"]
            missing = {"error": f"no result; the request is {meta.get('state')}"}
            reply = replies.get(cid) or Reply(cid, "request_error", extra=missing)
            reply.started, reply.finished = (
                meta.get("create_time") or "",
                meta.get("finish_time") or "",
            )
            out.append(reply)
        listed = {r.custom_id for r in out}
        unlisted = {"error": "no result; the batch does not list the request"}
        out += [Reply(c, "request_error", extra=unlisted) for c in custom_ids if c not in listed]
        # the Files API lists the batch's uploads with their names (stage B1)
        try:
            uploads = self._api().files.list()
            left = self._delete([f.id for f in uploads if str(f.filename).startswith(batch_id)])
        except ERRORS as e:
            left = [f"listing the files failed: {e}"]
        if left:
            for reply in out:
                reply.extra["undeleted_files"] = left
        return out

    def count_tokens(self, req: Request) -> int | None:
        return None
