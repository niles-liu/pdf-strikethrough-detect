"""Google: Gemini on Vertex AI through `generateContent` and Vertex batch prediction (PREREG.md,
Models), with Application Default Credentials: a request sent alone at `$GOOGLE_CLOUD_LOCATION`
(`global`), a batch in the `us` multi-region (BATCH_LOCATION), since the project's organization
allows no batch job at `global` and `us-central1` has no batch for Gemini 4 Argon (stage B1).

`body()` is the REST `GenerateContentRequest` (camelCase, base64 data), the JSON a batch input line
carries; `send()` hands the same JSON to the SDK, and both routes parse the REST response JSON. A
batch lives under `gs://$BENCH_GCS_BUCKET/<batch name>/`: `input.jsonl` and `output/`. Output lines
carry no caller ID, only an echo of the request, so each request carries its line number as the
label `bench_line`, as `line-<n>`: Vertex reads a label that looks like a number as one, then
refuses it as a label (stage B1). The harness passes the custom IDs back in line order.
"""

import base64
import json
import os
import secrets
import time
from collections.abc import Iterator

import httpx
import pydantic
from google import genai
from google.cloud import storage
from google.genai import errors, types

from bench.providers import Part, Reply, Request, failed, now

ERRORS = (errors.APIError, httpx.HTTPError)
REFUSALS = {
    "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION",
}  # fmt: skip
DONE = {
    types.JobState.JOB_STATE_SUCCEEDED,
    types.JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
    types.JobState.JOB_STATE_FAILED,
    types.JobState.JOB_STATE_CANCELLED,
    types.JobState.JOB_STATE_EXPIRED,
}
LABEL, LINE = "bench_line", "line-"
BATCH_LOCATION = "us"
CONTENTS = pydantic.TypeAdapter(list[types.Content])


def _inline(part: Part, mime_type: str) -> dict:
    return {"inlineData": {"mimeType": mime_type, "data": base64.b64encode(part.data()).decode()}}


def _part(part: Part, config: dict) -> dict:
    match part.kind:
        case "text":
            return {"text": part.text}
        case "image":
            level = f"MEDIA_RESOLUTION_{config['media_resolution'].upper()}"
            return _inline(part, "image/png") | {"mediaResolution": {"level": level}}
        case "pdf":
            return _inline(part, "application/pdf")
    raise ValueError(f"unknown part kind {part.kind!r}")


def reply_from_response(custom_id: str, resp: dict) -> Reply:
    """A REST `GenerateContentResponse` as a Reply: the answer's parts, never its thoughts."""
    block = (resp.get("promptFeedback") or {}).get("blockReason")
    candidates = resp.get("candidates") or []
    finish = candidates[0].get("finishReason") if candidates else None
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    if block or finish in REFUSALS:
        status = "refusal"
    elif finish == "MAX_TOKENS":
        status = "length"
    else:  # anything but a clean stop (OTHER, LANGUAGE, unspecified, ...) is not an answer
        status = "ok" if finish == "STOP" else "request_error"
    extra = {"finish_reason": finish, "block_reason": block, "id": resp.get("responseId")}
    if status == "request_error":
        extra["error"] = (
            f"finish reason {finish}" if candidates else "no candidates, no block reason"
        )
    return Reply(
        custom_id,
        status,
        text="".join(p.get("text", "") for p in parts if not p.get("thought")),
        usage=resp.get("usageMetadata") or {},
        reported_model=resp.get("modelVersion"),
        extra=extra,
    )


def _reply_from_line(ids: list[str], line: dict) -> Reply:
    custom_id = ids[int(str(line["request"]["labels"][LABEL]).removeprefix(LINE))]
    if line.get("status"):  # a failed line's error; its `response` is empty
        return Reply(custom_id, "request_error", extra={"error": line["status"]})
    return reply_from_response(custom_id, line["response"])


def _output_lines(bucket: storage.Bucket, job: types.BatchJob) -> Iterator[dict]:
    """Every line of a finished job's output files; none if it wrote none."""
    directory = job.output_info and job.output_info.gcs_output_directory
    if not directory:
        return
    prefix = directory.removeprefix("gs://").partition("/")[2].rstrip("/") + "/"
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith(".jsonl"):
            yield from (json.loads(x) for x in blob.download_as_text().splitlines() if x.strip())


class Vertex:
    name = "google"

    def __init__(self) -> None:
        self._clients: dict[int, genai.Client] = {}
        self._storage: storage.Client | None = None

    def _genai(self, retries: int = 2, location: str | None = None) -> genai.Client:
        """A client at `location`, by default `$GOOGLE_CLOUD_LOCATION`."""
        if (retries, location) not in self._clients:
            self._clients[retries, location] = genai.Client(
                vertexai=True,
                project=os.environ["GOOGLE_CLOUD_PROJECT"],
                location=location or os.environ["GOOGLE_CLOUD_LOCATION"],
                # attempts count the first call
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=retries + 1)
                ),
            )
        return self._clients[retries, location]

    def _bucket(self) -> storage.Bucket:
        if self._storage is None:
            self._storage = storage.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"])
        return self._storage.bucket(os.environ["BENCH_GCS_BUCKET"].removeprefix("gs://"))

    def body(self, req: Request) -> dict:
        parts = [_part(p, req.model_config) for p in req.parts]
        body = {"contents": [{"role": "user", "parts": parts}]}
        if req.system is not None:
            body["systemInstruction"] = {"parts": [{"text": req.system}]}
        body["generationConfig"] = {
            "maxOutputTokens": req.max_tokens,
            "thinkingConfig": {"thinkingLevel": req.effort.upper()},
        }
        return body

    def send(self, req: Request) -> Reply:
        body = self.body(req)
        client = self._genai(req.retries)
        # JSON-mode validation decodes the base64 `data`; Python mode would send the base64 text.
        contents = CONTENTS.validate_json(json.dumps(body["contents"]))
        config = {**body["generationConfig"], "systemInstruction": body.get("systemInstruction")}
        config = types.GenerateContentConfig.model_validate_json(json.dumps(config))
        started, t0 = now(), time.perf_counter()
        try:
            resp = client.models.generate_content(
                model=req.model_config["id"], contents=contents, config=config
            )
        except ERRORS as e:
            return failed(req.custom_id, e, started)
        latency, finished = time.perf_counter() - t0, now()
        rest = resp.model_dump(mode="json", by_alias=True, exclude_none=True)
        reply = reply_from_response(req.custom_id, rest)
        reply.started, reply.finished, reply.latency_s = started, finished, latency
        return reply

    def submit_batch(self, reqs: list[Request]) -> str:
        stamp = time.strftime("%Y%m%dt%H%M%S", time.gmtime())
        name = f"bench-{reqs[0].model}-{stamp}-{secrets.token_hex(3)}"
        bucket = self._bucket()
        with bucket.blob(f"{name}/input.jsonl").open("w", encoding="utf-8") as f:
            for i, r in enumerate(reqs):
                request = self.body(r) | {"labels": {LABEL: f"{LINE}{i}"}}
                f.write(json.dumps({"request": request}) + "\n")
        job = self._genai(location=BATCH_LOCATION).batches.create(
            model=reqs[0].model_config["id"],
            src=f"gs://{bucket.name}/{name}/input.jsonl",
            config=types.CreateBatchJobConfig(
                display_name=name, dest=f"gs://{bucket.name}/{name}/output"
            ),
        )
        return job.name

    def batch_done(self, batch_id: str) -> bool:
        return self._genai(location=BATCH_LOCATION).batches.get(name=batch_id).state in DONE

    def batch_results(self, batch_id: str, custom_ids: list[str]) -> list[Reply]:
        job = self._genai(location=BATCH_LOCATION).batches.get(name=batch_id)
        bucket, ids = self._bucket(), custom_ids
        replies = {}
        for line in _output_lines(bucket, job):
            reply = _reply_from_line(ids, line)
            replies[reply.custom_id] = reply
        started = job.create_time.isoformat() if job.create_time else ""
        finished = job.end_time.isoformat() if job.end_time else ""
        missing = f"no result; the job ended {job.state.value}"
        out = [replies.get(c) or Reply(c, "request_error", extra={"error": missing}) for c in ids]
        for reply in out:
            reply.started, reply.finished = started, finished
        return out

    def count_tokens(self, req: Request) -> int | None:
        return None
