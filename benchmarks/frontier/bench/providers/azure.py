"""OpenAI's GPT-6 through Azure OpenAI's v1 API: the Responses API (PREREG.md, Models), with the
OpenAI SDK at `$AZURE_OPENAI_ENDPOINT`.

Requests name the model's deployment (`deployment_env` in models.toml), never the model. The
response body's `model` names the deployment too, so the served version is read from the
`x-ms-served-model` header, and no record keeps the deployment's name or the endpoint's host.
Azure offers no batch for GPT-6: its half price is the Flex tier (`service_tier: flex`) on the
same Global Standard deployment, one call per request, and each response reports the tier that
served it. A response Azure's content filter blocks is a refusal (PREREG.md, Models).
"""

import base64
import os
import time
from urllib.parse import urlsplit

import httpx2
import openai

from bench.providers import Part, Reply, Request, Status, failed, now

ERRORS = (openai.APIError, httpx2.HTTPError)
# A 64,000-token answer at high effort can outlast the SDK's 10-minute default, and Flex queues.
TIMEOUT_S = 3600
NO_BATCH = "Azure offers no batch for GPT-6; its half price is the Flex tier"
# reported without the header, so the model-ID check stops the run; the body's `model` names the
# account's deployment, which no record keeps (it stays in .env)
NO_HEADER = "unknown: no x-ms-served-model header"


def _data_url(part: Part, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(part.data()).decode()}"


def _item(part: Part, config: dict) -> dict:
    match part.kind:
        case "text":
            return {"type": "input_text", "text": part.text}
        case "image":
            image = _data_url(part, "image/png")
            return {"type": "input_image", "image_url": image, "detail": config["image_detail"]}
        case "pdf":
            pdf = _data_url(part, "application/pdf")
            return {"type": "input_file", "filename": "page.pdf", "file_data": pdf}
    raise ValueError(f"unknown part kind {part.kind!r}")


def _content_filtered(error: object) -> bool:
    """Whether an API error object is Azure's content filter blocking the prompt."""
    if not isinstance(error, dict):
        return False
    inner = error.get("innererror") or {}
    return (
        error.get("code") == "content_filter" or inner.get("code") == "ResponsibleAIPolicyViolation"
    )


def _error_text(error: object) -> str:
    if isinstance(error, dict):
        return f"{error.get('code')}: {error.get('message')}"
    return str(error)


def _failed(custom_id: str, error: Exception, started: str) -> Reply:
    blocked = isinstance(error, openai.APIStatusError) and _content_filtered(error.body)
    return failed(custom_id, error, started, "refusal" if blocked else "request_error")


def responses_outcome(resp: dict) -> tuple[Status, str, str | None]:
    """A Responses API response's status, answer text and incomplete reason (xAI's too)."""
    reason = (resp.get("incomplete_details") or {}).get("reason")
    texts, refused = [], False
    for item in resp.get("output") or []:
        if item.get("type") != "message":
            continue  # reasoning and tool calls
        for content in item.get("content") or []:
            if content["type"] == "output_text":
                texts.append(content["text"])
            elif content["type"] == "refusal":
                refused = True
    if refused or reason == "content_filter":
        status = "refusal"
    elif reason == "max_output_tokens":
        status = "length"
    else:
        status = "ok" if resp.get("status") == "completed" else "request_error"
    return status, "".join(texts), reason


def reply_from_response(custom_id: str, resp: dict, served_model: str | None) -> Reply:
    """A Responses API response, as a dict, as a Reply. `served_model` is the header's value."""
    status, text, reason = responses_outcome(resp)
    extra = {
        "served_model": served_model,
        "service_tier": resp.get("service_tier"),  # what is billed: `flex` at half price
        "response_status": resp.get("status"),
        "incomplete_reason": reason,
        "id": resp.get("id"),
    }
    if status == "request_error":
        extra["error"] = _error_text(resp.get("error") or f"response {resp.get('status')}")
    return Reply(
        custom_id,
        status,
        text=text,
        usage=resp.get("usage") or {},
        reported_model=served_model or NO_HEADER,
        extra=extra,
    )


def _scrubbed(reply: Reply, req: Request) -> Reply:
    """The reply with its error text free of the account's own names, which no record keeps:
    the deployment and the endpoint's host."""
    host = urlsplit(os.environ.get("AZURE_OPENAI_ENDPOINT", "")).hostname
    deployment = os.environ.get(req.model_config["deployment_env"])
    if isinstance(error := reply.extra.get("error"), str):
        for name, stand_in in ((deployment, "<deployment>"), (host, "<endpoint>")):
            if name:
                error = error.replace(name, stand_in)
        reply.extra["error"] = error
    return reply


class Azure:
    name = "openai"

    def __init__(self) -> None:
        self._client: openai.OpenAI | None = None

    def _api(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(
                base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                max_retries=2,
                timeout=TIMEOUT_S,
            )
        return self._client

    def body(self, req: Request) -> dict:
        body = {
            "model": os.environ[req.model_config["deployment_env"]],
            "input": [{"role": "user", "content": [_item(p, req.model_config) for p in req.parts]}],
        }
        if req.system is not None:
            body["instructions"] = req.system
        # Flex takes images and PDFs and answers with service_tier "flex" (stage B1). "default" is
        # the standard tier, even on a deployment set to Priority.
        if req.prompt_cache == "off":  # no implicit breakpoint and none given: nothing cached
            body["prompt_cache_options"] = {"mode": "explicit"}
        return body | {
            "max_output_tokens": req.max_tokens,
            "reasoning": {"effort": req.effort},
            "service_tier": req.service_tier or "default",
        }

    def send(self, req: Request) -> Reply:
        body = self.body(req)
        client = self._api().with_options(max_retries=req.retries)
        started, t0 = now(), time.perf_counter()
        try:
            raw = client.responses.with_raw_response.create(**body)
            resp = raw.parse()
        except ERRORS as e:
            return _scrubbed(_failed(req.custom_id, e, started), req)
        latency, finished = time.perf_counter() - t0, now()
        served = raw.headers.get("x-ms-served-model")
        reply = reply_from_response(req.custom_id, resp.model_dump(mode="json"), served)
        reply.extra["retries"] = raw.retries_taken
        reply.started, reply.finished, reply.latency_s = started, finished, latency
        return _scrubbed(reply, req)

    def submit_batch(self, reqs: list[Request]) -> str:
        raise NotImplementedError(NO_BATCH)

    def batch_done(self, batch_id: str) -> bool:
        raise NotImplementedError(NO_BATCH)

    def batch_results(self, batch_id: str, custom_ids: list[str]) -> list[Reply]:
        raise NotImplementedError(NO_BATCH)

    def count_tokens(self, req: Request) -> int | None:
        return None
