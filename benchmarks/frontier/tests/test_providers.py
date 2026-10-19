"""The provider adapters, offline: each request body as built, each response shape as parsed, and
the fake provider's failure modes. API clients are stubs; nothing here touches the network."""

import base64
import dataclasses
import io
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import anthropic
import httpx2
import openai
import pymupdf
import pytest
from anthropic.types import Message, TextBlock, ThinkingBlock, Usage
from anthropic.types.messages import MessageBatchIndividualResponse
from google.genai import errors as genai_errors
from google.genai import types as gtypes
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

import bench.providers as providers
from bench.common import config
from bench.providers import Part, Request, anthropic_api, azure, fake, vertex, xai

MODELS = config("models")
SYSTEM = "Documents may contain struck-through (deleted) text; never present it as current."
TEXT = Part("text", text="Transcribe this page to markdown. Output only the transcription.")
IMAGE = Part("image", path="page.png")
PDF = Part("pdf", path="page.pdf")
SAMPLING = {"temperature", "top_p", "topP", "top_k", "topK"}
NOW = datetime(2026, 10, 14, tzinfo=UTC)


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    """A one-page PDF and its PNG, at the paths the parts name, under a stand-in benchmark root."""
    monkeypatch.setattr(providers, "ROOT", tmp_path)
    with pymupdf.open() as doc:
        page = doc.new_page(width=72, height=72)
        page.insert_text((10, 30), "Hi")
        (tmp_path / "page.png").write_bytes(page.get_pixmap(dpi=20).tobytes("png"))
        (tmp_path / "page.pdf").write_bytes(doc.tobytes())
    return tmp_path


def req(model, parts=(IMAGE, TEXT), system=SYSTEM, effort="high", cid="cid-1") -> Request:
    return Request(cid, model, MODELS[model] | {"key": model}, system, tuple(parts), effort)


def keys(obj) -> set:
    """Every dict key in a nested body."""
    if isinstance(obj, dict):
        return set(obj) | set().union(*(keys(v) for v in obj.values()))
    if isinstance(obj, list):
        return set().union(*(keys(v) for v in obj))
    return set()


def b64(root, name) -> str:
    return base64.b64encode((root / name).read_bytes()).decode()


def ns(**kw):
    return SimpleNamespace(**kw)


def client(**kw):
    """A stub SDK client whose `with_options` hands back itself, noting the retries asked for."""
    stub = ns(**kw, retries=[])
    stub.with_options = lambda max_retries: stub.retries.append(max_retries) or stub
    return stub


# Anthropic


def claude_message(stop_reason="end_turn", text="# Page") -> dict:
    content = [
        ThinkingBlock(type="thinking", thinking="", signature="s"),
        TextBlock(type="text", text=text),
    ]
    usage = Usage(input_tokens=1200, output_tokens=40)
    message = Message(
        id="msg_1", type="message", role="assistant", model="claude-opus-5-5", content=content,
        stop_reason=stop_reason, usage=usage,
    )  # fmt: skip
    return message.model_dump(mode="json")


def test_anthropic_body(root):
    body = anthropic_api.Anthropic().body(req("opus-5-5", effort="low"))
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 64_000
    assert body["system"] == SYSTEM
    assert body["thinking"] == {"type": "adaptive"} and body["output_config"] == {"effort": "low"}
    image, text = body["messages"][0]["content"]
    assert image == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": b64(root, "page.png")},
    }
    assert text == {"type": "text", "text": TEXT.text}
    assert not keys(body) & (SAMPLING | {"fallbacks"})


def test_anthropic_body_r0_and_pdf(root):
    body = anthropic_api.Anthropic().body(req("sonnet-5-5", parts=(PDF, TEXT), system=None))
    assert "system" not in body and body["output_config"] == {"effort": "high"}
    document = body["messages"][0]["content"][0]
    assert document["type"] == "document"
    assert document["source"] == {
        "type": "base64", "media_type": "application/pdf", "data": b64(root, "page.pdf"),
    }  # fmt: skip


@pytest.mark.parametrize(
    "stop, status",
    [("end_turn", "ok"), ("refusal", "refusal"), ("max_tokens", "length")],
)
def test_anthropic_parse(stop, status):
    reply = anthropic_api.reply_from_message("c", claude_message(stop))
    assert (reply.status, reply.text, reply.reported_model) == (status, "# Page", "claude-opus-5-5")
    assert reply.usage["output_tokens"] == 40 and reply.extra["stop_reason"] == stop


def test_anthropic_send_and_errors():
    provider, sent = anthropic_api.Anthropic(), []
    message = Message.model_validate(claude_message())
    headers = {"x-stainless-retry-count": "1"}

    class Stream:
        response = ns(request=ns(headers=headers))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return message

    def stream(**body):
        sent.append(body)
        return Stream()

    provider._client = client(messages=ns(stream=stream))
    reply = provider.send(req("haiku-5-5", parts=(TEXT,)))
    assert sent[0]["model"] == "claude-haiku-5-5" and reply.status == "ok"
    assert reply.extra["retries"] == 1 and reply.latency_s is not None and reply.started
    provider.send(dataclasses.replace(req("haiku-5-5", parts=(TEXT,)), retries=0))
    assert provider._client.retries == [2, 0]  # the latency sample's calls: no SDK retries

    def broken(**body):
        raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))

    provider._client = client(messages=ns(stream=broken))
    reply = provider.send(req("haiku-5-5", parts=(TEXT,)))
    assert reply.status == "request_error" and reply.extra["error"].startswith("APIConnection")


def test_anthropic_batch_and_count_tokens():
    provider, created, counted = anthropic_api.Anthropic(), [], []
    results = [
        {"custom_id": "a", "result": {"type": "succeeded", "message": claude_message()}},
        {"custom_id": "b", "result": {"type": "errored", "error": {"type": "error", "error": {
            "type": "invalid_request_error", "message": "bad image"}}}},
        {"custom_id": "c", "result": {"type": "expired"}},
    ]  # fmt: skip
    batches = ns(
        create=lambda requests: created.extend(requests) or ns(id="msgbatch_1"),
        retrieve=lambda batch_id: ns(processing_status="ended", created_at=NOW, ended_at=NOW),
        results=lambda batch_id: [
            MessageBatchIndividualResponse.model_validate(r) for r in results
        ],
    )
    count = ns(input_tokens=1234)
    messages = ns(batches=batches, count_tokens=lambda **body: counted.append(body) or count)
    provider._client = ns(messages=messages)
    assert (
        provider.submit_batch([req("opus-5-5", cid="a"), req("opus-5-5", cid="b")]) == "msgbatch_1"
    )
    assert [r["custom_id"] for r in created] == ["a", "b"]
    assert created[0]["params"] == provider.body(req("opus-5-5", cid="a"))
    assert provider.batch_done("msgbatch_1")
    replies = {r.custom_id: r for r in provider.batch_results("msgbatch_1", ["a", "b", "c"])}
    assert replies["a"].status == "ok" and replies["a"].started == NOW.isoformat()
    assert replies["b"].status == "request_error" and "bad image" in replies["b"].extra["error"]
    assert replies["c"].extra["error"] == "expired"
    assert provider.count_tokens(req("opus-5-5")) == 1234 and "max_tokens" not in counted[0]


# Google (Vertex)


def gemini_response(finish="STOP", block=None, candidates=True) -> dict:
    parts = [gtypes.Part(text="thinking", thought=True), gtypes.Part(text="# Page")]
    content = gtypes.Content(role="model", parts=parts)
    response = gtypes.GenerateContentResponse(
        candidates=[gtypes.Candidate(content=content, finish_reason=finish)] if candidates else [],
        prompt_feedback=gtypes.GenerateContentResponsePromptFeedback(block_reason=block),
        model_version="gemini-4-argon",
        usage_metadata=gtypes.GenerateContentResponseUsageMetadata(prompt_token_count=2300),
    )
    return response.model_dump(mode="json", by_alias=True, exclude_none=True)


def test_vertex_body(root):
    body = vertex.Vertex().body(req("gemini-4-argon", parts=(TEXT, IMAGE), effort="low"))
    text, image = body["contents"][0]["parts"]
    assert text == {"text": TEXT.text}
    assert image == {
        "inlineData": {"mimeType": "image/png", "data": b64(root, "page.png")},
        "mediaResolution": {"level": "MEDIA_RESOLUTION_ULTRA_HIGH"},
    }
    assert body["systemInstruction"] == {"parts": [{"text": SYSTEM}]}
    assert body["generationConfig"] == {
        "maxOutputTokens": 64_000, "thinkingConfig": {"thinkingLevel": "LOW"},
    }  # fmt: skip
    assert not keys(body) & SAMPLING


def test_vertex_body_pdf_and_r0(root):
    body = vertex.Vertex().body(req("gemini-3-8-flash", parts=(TEXT, PDF), system=None))
    assert "systemInstruction" not in body
    assert body["contents"][0]["parts"][1] == {
        "inlineData": {"mimeType": "application/pdf", "data": b64(root, "page.pdf")},
    }  # fmt: skip


@pytest.mark.parametrize(
    "kw, status",
    [
        ({}, "ok"),
        ({"finish": "SAFETY"}, "refusal"),
        ({"finish": "PROHIBITED_CONTENT"}, "refusal"),
        ({"block": "OTHER", "candidates": False}, "refusal"),
        ({"finish": "MAX_TOKENS"}, "length"),
        ({"candidates": False}, "request_error"),
    ],
)
def test_vertex_parse(kw, status):
    reply = vertex.reply_from_response("c", gemini_response(**kw))
    assert reply.status == status and reply.reported_model == "gemini-4-argon"
    assert reply.usage == {"promptTokenCount": 2300}
    assert reply.text == ("" if "candidates" in kw else "# Page")


def test_vertex_send(root):
    provider, calls = vertex.Vertex(), []

    def generate_content(**kw):
        calls.append(kw)
        return gtypes.GenerateContentResponse.model_validate(gemini_response())

    provider._clients = {(2, None): ns(models=ns(generate_content=generate_content))}
    reply = provider.send(req("gemini-4-argon", parts=(TEXT, IMAGE)))
    assert reply.status == "ok" and reply.text == "# Page" and reply.latency_s is not None
    call = calls[0]
    assert call["model"] == "gemini-4-argon"
    image = call["contents"][0].parts[1]
    assert image.inline_data.data == (root / "page.png").read_bytes()
    assert image.media_resolution.level == "MEDIA_RESOLUTION_ULTRA_HIGH"
    assert call["config"].system_instruction.parts[0].text == SYSTEM
    assert call["config"].thinking_config.thinking_level == "HIGH"
    assert call["config"].max_output_tokens == 64_000 and call["config"].temperature is None

    def broken(**kw):
        raise genai_errors.ServerError(503, {"error": {"message": "overloaded"}})

    provider._clients = {(2, None): ns(models=ns(generate_content=broken))}
    reply = provider.send(req("gemini-4-argon", parts=(TEXT, IMAGE)))
    assert reply.status == "request_error" and reply.extra["error"].startswith("ServerError")


class Bucket:
    """Cloud Storage, in a dict."""

    name = "bench-bucket"

    def __init__(self):
        self.store = {}

    def blob(self, name):
        store = self.store

        class Writer(io.StringIO):
            def close(self):
                store[name] = self.getvalue()
                super().close()

        return ns(
            name=name,
            upload_from_string=lambda data, content_type=None: store.__setitem__(name, data),
            open=lambda mode, encoding=None: Writer(),
            download_as_text=lambda: store[name],
        )

    def list_blobs(self, prefix):
        return [self.blob(n) for n in list(self.store) if n.startswith(prefix)]


def test_vertex_batch_round_trip():
    provider, bucket, created = vertex.Vertex(), Bucket(), []
    job_name = "projects/p/locations/us/batchPredictionJobs/1"
    provider._bucket = lambda: bucket
    batches = ns(create=lambda **kw: created.append(kw) or ns(name=job_name))
    # a batch goes to the `us` multi-region, whatever $GOOGLE_CLOUD_LOCATION says
    provider._clients = {(2, vertex.BATCH_LOCATION): ns(batches=batches)}
    reqs = [req("gemini-4-argon", parts=(TEXT, IMAGE), cid=c) for c in ("a", "b", "c")]
    assert provider.submit_batch(reqs) == job_name
    name = created[0]["config"].display_name
    assert created[0]["model"] == "gemini-4-argon"
    assert created[0]["src"] == f"gs://bench-bucket/{name}/input.jsonl"
    assert created[0]["config"].dest == f"gs://bench-bucket/{name}/output"
    lines = [json.loads(line) for line in bucket.store[f"{name}/input.jsonl"].splitlines()]
    # not "1": Vertex reads a label that looks like a number as one, then refuses it
    assert lines[1]["request"] == provider.body(reqs[1]) | {"labels": {"bench_line": "line-1"}}

    # Vertex echoes each request beside its response, in any order; "a" never came back.
    out = f"{name}/output/prediction-model-1/predictions.jsonl"
    bucket.store[out] = "\n".join([
        json.dumps({"status": "Bad image", "request": lines[2]["request"], "response": {}}),
        json.dumps({"status": "", "request": lines[1]["request"], "response": gemini_response()}),
    ])  # fmt: skip
    job = ns(
        display_name=name,
        state=gtypes.JobState.JOB_STATE_SUCCEEDED,
        output_info=ns(gcs_output_directory=f"gs://bench-bucket/{name}/output/prediction-model-1"),
        create_time=NOW,
        end_time=NOW,
    )
    provider._clients = {(2, vertex.BATCH_LOCATION): ns(batches=ns(get=lambda name: job))}
    assert provider.batch_done(job_name)
    a, b, c = provider.batch_results(job_name, ["a", "b", "c"])
    assert (a.custom_id, a.status) == ("a", "request_error")
    assert (b.custom_id, b.status, b.text) == ("b", "ok", "# Page")
    assert (c.custom_id, c.status, c.extra["error"]) == ("c", "request_error", "Bad image")
    assert b.started == NOW.isoformat()


# OpenAI (Azure)


@pytest.fixture
def deployments(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_GPT6_ASTRA", "astra-rt")


def gpt_response(status="completed", reason=None, refusal=False, model="astra-rt") -> Response:
    content = (
        {"type": "refusal", "refusal": "I can't help with that."}
        if refusal
        else {"type": "output_text", "text": "# Page", "annotations": []}
    )
    return Response.model_validate({
        "id": "resp_1", "created_at": 0, "model": model, "object": "response", "status": status,
        "service_tier": "flex", "incomplete_details": {"reason": reason} if reason else None,
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
             "content": [content]},
        ],
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": {"input_tokens": 900, "output_tokens": 50, "total_tokens": 950,
                  "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 20}},
    })  # fmt: skip


def test_azure_body(root, deployments):
    body = azure.Azure().body(req("gpt-6-astra"))
    assert body["model"] == "astra-rt" and body["instructions"] == SYSTEM
    image, text = body["input"][0]["content"]
    assert image == {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{b64(root, 'page.png')}",
        "detail": "original",
    }
    assert text == {"type": "input_text", "text": TEXT.text}
    assert body["max_output_tokens"] == 64_000 and body["reasoning"] == {"effort": "high"}
    assert not keys(body) & SAMPLING
    assert body["service_tier"] == "default"  # the standard tier, never a deployment's Priority
    flex = azure.Azure().body(dataclasses.replace(req("gpt-6-astra"), service_tier="flex"))
    assert flex == body | {"service_tier": "flex"}
    uncached = azure.Azure().body(dataclasses.replace(req("gpt-6-astra"), prompt_cache="off"))
    assert uncached == body | {"prompt_cache_options": {"mode": "explicit"}}  # no breakpoint


def test_azure_has_no_batch_for_gpt6():
    with pytest.raises(NotImplementedError, match="Flex"):
        azure.Azure().submit_batch([req("gpt-6-astra")])


def test_azure_body_pdf(root, deployments):
    body = azure.Azure().body(req("gpt-6-astra", parts=(PDF, TEXT), system=None, effort="low"))
    assert "instructions" not in body and body["reasoning"] == {"effort": "low"}
    assert body["input"][0]["content"][0] == {
        "type": "input_file",
        "filename": "page.pdf",
        "file_data": f"data:application/pdf;base64,{b64(root, 'page.pdf')}",
    }


@pytest.mark.parametrize(
    "kw, status",
    [
        ({}, "ok"),
        ({"refusal": True}, "refusal"),
        ({"status": "incomplete", "reason": "content_filter"}, "refusal"),
        ({"status": "incomplete", "reason": "max_output_tokens"}, "length"),
        ({"status": "failed"}, "request_error"),
    ],
)
def test_azure_parse(kw, status):
    resp = gpt_response(**kw).model_dump(mode="json")
    reply = azure.reply_from_response("c", resp, served_model="gpt-6-astra-2026-09-03")
    assert reply.status == status and reply.usage["output_tokens"] == 50
    assert reply.text == ("" if "refusal" in kw else "# Page")


@pytest.mark.parametrize(
    "headers, reported",
    [
        ({"x-ms-served-model": "gpt-6-astra-2026-09-03"}, "gpt-6-astra-2026-09-03"),
        ({}, azure.NO_HEADER),
    ],  # without the header, a mismatch: never the deployment's name
)
def test_azure_send_reads_the_served_model_header(deployments, headers, reported):
    provider, sent = azure.Azure(), []
    raw = ns(headers=headers, parse=gpt_response, retries_taken=0)
    create = lambda **body: sent.append(body) or raw  # noqa: E731
    provider._client = client(responses=ns(with_raw_response=ns(create=create)))
    reply = provider.send(req("gpt-6-astra"))
    assert sent[0]["model"] == "astra-rt" and reply.status == "ok"
    assert reply.reported_model == reported
    assert reply.extra["served_model"] == headers.get("x-ms-served-model")
    assert reply.extra["service_tier"] == "flex"
    assert "astra-rt" not in json.dumps([reply.reported_model, reply.extra])  # nor anywhere


def test_azure_errors_keep_no_deployment_or_endpoint_name(deployments, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://acct.openai.azure.com/openai/v1")
    provider = azure.Azure()

    def create(**body):
        raise httpx2.ConnectError("deployment astra-rt at acct.openai.azure.com is down")

    provider._client = client(responses=ns(with_raw_response=ns(create=create)))
    reply = provider.send(req("gpt-6-astra"))
    assert reply.status == "request_error"
    assert reply.extra["error"].endswith("deployment <deployment> at <endpoint> is down")


@pytest.mark.parametrize(
    "body, status",
    [
        ({"code": "content_filter", "message": "filtered"}, "refusal"),
        ({"code": None, "innererror": {"code": "ResponsibleAIPolicyViolation"}}, "refusal"),
        ({"code": "invalid_value", "message": "bad image"}, "request_error"),
    ],
)
def test_azure_content_filter_is_a_refusal(deployments, body, status):
    response = httpx2.Response(400, request=httpx2.Request("POST", "https://x"))

    def create(**kw):
        raise openai.BadRequestError("Error code: 400", response=response, body=body)

    provider = azure.Azure()
    provider._client = client(responses=ns(with_raw_response=ns(create=create)))
    reply = provider.send(req("gpt-6-astra"))
    assert reply.status == status and reply.extra["error"].startswith("BadRequestError")


# xAI


def grok_completion(finish="stop", refusal=None) -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "c1", "object": "chat.completion", "created": 0, "model": "grok-4.7",
        "system_fingerprint": "fp_1",
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": "# Page", "refusal": refusal}}],
        "usage": {"prompt_tokens": 800, "completion_tokens": 30, "total_tokens": 830,
                  "cost_in_usd_ticks": 123},
    })  # fmt: skip


def test_xai_chat_body(root):
    body = xai.XAI().body(req("grok-4-7", effort="low"))
    system, user = body["messages"]
    assert system == {"role": "system", "content": SYSTEM}
    image, text = user["content"]
    assert image == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64(root, 'page.png')}", "detail": "high"},
    }
    assert text == {"type": "text", "text": TEXT.text}
    assert body["max_completion_tokens"] == 64_000 and body["reasoning_effort"] == "low"
    assert not keys(body) & (SAMPLING | {"max_tokens"})


def test_xai_pdf_goes_through_responses_and_the_file_is_deleted(root):
    provider, sent, deleted = xai.XAI(), [], []
    body = provider.body(req("grok-4-3", parts=(PDF, TEXT)))
    assert body["instructions"] == SYSTEM and body["reasoning"] == {"effort": "high"}
    assert body["max_output_tokens"] == 64_000
    assert body["input"][0]["content"][0] == {"type": "input_file", "file_id": None}

    def upload(file, purpose):
        assert file == ("page.pdf", (root / "page.pdf").read_bytes(), "application/pdf")
        return ns(id="file-1")

    raw = ns(parse=lambda: gpt_response(model="grok-4.3"), retries_taken=0)
    provider._client = client(
        files=ns(create=upload, delete=deleted.append),
        responses=ns(with_raw_response=ns(create=lambda **b: sent.append(b) or raw)),
    )
    reply = provider.send(req("grok-4-3", parts=(PDF, TEXT)))
    assert sent[0]["input"][0]["content"][0]["file_id"] == "file-1"
    assert deleted == ["file-1"] and reply.status == "ok" and reply.reported_model == "grok-4.3"


@pytest.mark.parametrize(
    "kw, status",
    [({}, "ok"), ({"finish": "length"}, "length"), ({"refusal": "No."}, "refusal")],
)
def test_xai_chat_parse(kw, status):
    reply = xai.reply_from_chat("c", grok_completion(**kw).model_dump(mode="json"))
    assert (reply.status, reply.text, reply.reported_model) == (status, "# Page", "grok-4.7")
    assert reply.extra["system_fingerprint"] == "fp_1" and reply.usage["cost_in_usd_ticks"] == 123


def test_xai_batch(root):
    provider, posts, deleted = xai.XAI(), [], []

    def post(path, body, cast_to):
        posts.append((path, body))
        return {"batch_id": "b1"} if path == "/batches" else {}

    results = [
        {"batch_request_id": "a", "batch_result": {"error": "", "response": {
            "chat_get_completion": grok_completion().model_dump(mode="json")}}},
        {"batch_request_id": "b", "error_message": "image too large"},
    ]  # fmt: skip
    meta = [{"batch_request_id": c, "state": "succeeded", "create_time": "t0"} for c in "abc"]
    pages = {
        "/batches/b1": {"state": {"num_pending": 0}},
        "/batches/b1/results": {"results": results, "pagination_token": None},
        "/batches/b1/requests": {"batch_request_metadata": meta},
    }
    uploads = [ns(id="f1", filename="b1.pdf"), ns(id="f2", filename="page.pdf")]
    provider._client = ns(post=post, get=lambda path, cast_to, options=None: pages[path],
                          files=ns(list=lambda: uploads, delete=deleted.append))  # fmt: skip
    reqs = [req("grok-4-3", cid=c) for c in "abc"]
    assert provider.submit_batch(reqs) == "b1"
    items = posts[1][1]["batch_requests"]
    assert posts[1][0] == "/batches/b1/requests"
    assert [i["batch_request_id"] for i in items] == ["a", "b", "c"]
    assert items[0]["batch_request"] == {"chat_get_completion": provider.body(reqs[0])}
    assert provider.batch_done("b1")
    a, b, c, d = provider.batch_results("b1", ["a", "b", "c", "d"])
    assert [(r.custom_id, r.status) for r in (a, b, c, d)] == [
        ("a", "ok"), ("b", "request_error"), ("c", "request_error"), ("d", "request_error"),
    ]  # fmt: skip
    assert b.extra["error"] == "image too large" and a.started == "t0" and a.text == "# Page"
    assert deleted == ["f1"]  # the batch's own upload, found by its name
    with pytest.raises(NotImplementedError):
        provider.submit_batch([req("grok-4-7")])


def test_an_xai_batch_failing_part_way_names_the_requests_it_holds(root, monkeypatch):
    monkeypatch.setattr(xai, "ADD_LIMIT_BYTES", 1)  # one request per post

    def post(path, body, cast_to):
        if path == "/batches":
            return {"batch_id": "b1"}
        if body["batch_requests"][0]["batch_request_id"] == "b":
            raise openai.APIConnectionError(request=httpx2.Request("POST", "https://x"))
        return {}

    provider = xai.XAI()
    provider._client = ns(post=post)
    with pytest.raises(providers.PartialBatch) as held:
        provider.submit_batch([req("grok-4-3", cid=c) for c in "abc"])
    assert (held.value.batch_id, held.value.custom_ids) == ("b1", ["a"])


# The fake


FAKE = {"provider": "fake", "id": "fake-1", "expect": "fake-1", "batch": True}


def fake_req(cid="cid-1", **config) -> Request:
    return Request(cid, "fake", FAKE | config, SYSTEM, (IMAGE, TEXT))


def test_fake_registry_and_reply():
    fake.reset()
    provider = providers.get(FAKE)
    reply = provider.send(fake_req(fake_text="# Page"))
    assert (reply.status, reply.text, reply.reported_model) == ("ok", "# Page", "fake-1")
    assert fake.CALLS == ["cid-1"] and provider.calls == ["cid-1"]
    assert provider.body(fake_req())["parts"][0] == {
        "kind": "image", "text": "", "path": "page.png", "sha256": "",
    }  # fmt: skip
    assert provider.count_tokens(fake_req()) is None
    assert provider.count_tokens(fake_req(counts_tokens=True)) == reply.usage["input_tokens"]


def test_fake_failure_modes():
    fake.reset()
    provider = fake.Fake()
    statuses = [provider.send(fake_req(fail_times=2)).status for _ in range(3)]
    assert statuses == ["request_error", "request_error", "ok"]
    assert provider.send(fake_req("x", status="refusal")).status == "refusal"
    assert provider.send(fake_req("y", reported_model="other")).reported_model == "other"
    # A new instance shares the count, as a real API would: the third call still succeeds.
    assert fake.Fake().send(fake_req(fail_times=2)).status == "ok"


def test_fake_batch():
    fake.reset()
    batch_id = fake.Fake().submit_batch([fake_req("a"), fake_req("b", status="length")])
    collector = fake.Fake()  # collected by another instance, as by a later harness run
    assert collector.batch_done(batch_id)
    assert [(r.custom_id, r.status) for r in collector.batch_results(batch_id, ["a", "b"])] == [
        ("a", "ok"), ("b", "length"),
    ]  # fmt: skip
    assert fake.CALLS == ["a", "b"]
    with pytest.raises(NotImplementedError):
        fake.Fake().submit_batch([fake_req(batch=False)])
