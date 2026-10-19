"""Provider adapters: one per API, all speaking the same `Request` and `Reply`.

The harness builds a `Request` (prompts.py decides the parts and their order); an adapter turns it
into the provider's own call, sequential or batched, and its answer back into a `Reply`. Adapters
never retry a failed request themselves beyond the SDK's transport retries (`Request.retries`,
none in the latency sample, where each call is timed alone): the harness counts attempts
(PREREG.md, Operations).
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

from bench.common import ROOT

Status = Literal["ok", "refusal", "length", "request_error"]
OUTPUT_LIMIT = 64_000  # thinking included, xAI apart; a model's own `max_output` if lower


@dataclass(frozen=True)
class Part:
    """One piece of the user turn: text, or a file sent as an image (PNG) or a document (PDF)."""

    kind: Literal["text", "image", "pdf"]
    text: str = ""
    path: str = ""  # relative to the benchmark root
    sha256: str = ""

    def data(self) -> bytes:
        return (ROOT / self.path).read_bytes()


@dataclass(frozen=True)
class Request:
    custom_id: str  # stable across reruns; the cache key
    model: str  # the key in config/models.toml
    model_config: dict = field(hash=False, compare=False)
    system: str | None
    parts: tuple[Part, ...]
    effort: str = "high"
    max_tokens: int = OUTPUT_LIMIT
    service_tier: str | None = None  # Azure's `flex` (half price); None is the standard tier
    retries: int = 2  # the SDK's transport retries, inside a timed call
    prompt_cache: str | None = None  # "off": Azure's implicit prompt cache off for GPT-6


@dataclass
class Reply:
    custom_id: str
    status: Status
    text: str = ""
    usage: dict = field(default_factory=dict)  # the provider's own usage object, as a dict
    reported_model: str | None = None  # what the provider says served the request
    extra: dict = field(default_factory=dict)  # finish reason, fingerprints, error message, ...
    started: str = ""  # ISO timestamps, UTC
    finished: str = ""
    latency_s: float | None = None  # sequential calls only


def now() -> str:
    return datetime.now(UTC).isoformat()


def failed(
    custom_id: str, error: Exception, started: str, status: Status = "request_error"
) -> Reply:
    """A call that raised, as a Reply."""
    error_text = f"{type(error).__name__}: {error}"
    return Reply(custom_id, status, extra={"error": error_text}, started=started, finished=now())


class PartialBatch(Exception):
    """A batch the provider created that took only some of its requests before an error: the
    harness records the ones it holds, so they are collected rather than sent twice."""

    def __init__(self, batch_id: str, custom_ids: list[str]) -> None:
        super().__init__(f"batch {batch_id} took {len(custom_ids)} requests, then failed")
        self.batch_id, self.custom_ids = batch_id, custom_ids


class Provider(Protocol):
    name: str

    def body(self, req: Request) -> dict:
        """The provider-native request body, exactly as sent."""

    def send(self, req: Request) -> Reply:
        """One sequential call, timed from request to last byte."""

    def submit_batch(self, reqs: list[Request]) -> str:
        """Submit a batch; returns its ID. Raises PartialBatch if it holds only some of them."""

    def batch_done(self, batch_id: str) -> bool:
        """True once the batch has finished (completed, expired or failed)."""

    def batch_results(self, batch_id: str, custom_ids: list[str]) -> list[Reply]:
        """One Reply per request in a finished batch, `custom_ids` in submission order; a request
        with no result comes back as a request_error."""

    def count_tokens(self, req: Request) -> int | None:
        """Exact input tokens where the provider counts them for free; None otherwise."""


def get(model_config: dict) -> Provider:
    match model_config["provider"]:
        case "anthropic":
            from bench.providers.anthropic_api import Anthropic

            return Anthropic()
        case "google":
            from bench.providers.vertex import Vertex

            return Vertex()
        case "openai":
            from bench.providers.azure import Azure

            return Azure()
        case "xai":
            from bench.providers.xai import XAI

            return XAI()
        case "fake":
            from bench.providers.fake import Fake

            return Fake()
    raise ValueError(f"unknown provider {model_config['provider']!r}")
