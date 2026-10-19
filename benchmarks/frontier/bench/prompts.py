"""Prompts (PREREG.md, Prompts): the ladder's texts, R6's worked examples, the QA probe and stage
B4's prompts, and the order of a request's parts.

    python -m bench prompts fewshot   # R6's three worked examples, cropped from G0-dev
    python -m bench prompts hash      # the sha256 of every assembled prompt, for prereg-2

Placement: from R1 on, the system line goes in each API's system field. The page (image, bands
or PDF) comes before the instruction text for every provider, as Vertex AI's guidance for a
single image asks.
"""

import functools
import io
import json
import tomllib

import pdf_strikethrough as st
from PIL import Image, ImageDraw

from bench import render
from bench.common import PROMPTS, load_pages, rel, sha256_bytes, sha256_file
from bench.plans import FEWSHOT_PAGE, dev_pages
from bench.providers import Part

RUNGS = tuple(f"R{i}" for i in range(7))
PARAPHRASES = 4  # the base wording (0) and three paraphrases
FEWSHOT = PROMPTS / "fewshot"


@functools.cache
def _toml(name: str) -> dict:
    with (PROMPTS / f"{name}.toml").open("rb") as f:
        return tomllib.load(f)


def text(rung: str, para: int = 0) -> tuple[str | None, str]:
    """A rung's system line (None for R0) and instruction, in the wording of one paraphrase."""
    recipe, words = _toml("ladder")["rungs"][rung], _toml("ladder")["components"]
    system = " ".join(words[c][para] for c in recipe["system"])
    return system or None, " ".join(words[c][para] for c in recipe["user"])


def json_answer(rung: str) -> bool:
    return "json" in _toml("ladder")["rungs"][rung]["user"]


def _file(page_id: str, name: str, kind: str) -> Part:
    path = render.path(page_id, name)
    return Part(kind, path=rel(path), sha256=render.manifest()[page_id]["files"][name])


def page_parts(page_id: str, input_: str) -> list[Part]:
    """The page as one input condition: C1, C1′ (image, then the page's text), C2 or C2-tiled."""
    if input_ == "C1":
        return [_file(page_id, "c1.pdf", "pdf")]
    if input_ == "C1-scan":  # the smoke test's image-only PDF, to see what xAI passes on
        return [_file(page_id, "scan.pdf", "pdf")]
    if input_ == "C2":
        return [_file(page_id, "c2.png", "image")]
    if input_ == "C1p":
        page_text = render.path(page_id, "text.txt").read_text(encoding="utf-8")
        return [_file(page_id, "c2.png", "image"), Part("text", text=page_text)]
    if input_ == "C2-tiled":
        bands = len(render.manifest()[page_id]["bands"])
        return [_file(page_id, f"band-{k}.png", "image") for k in range(bands)]
    raise ValueError(f"unknown input {input_!r}")


def _place(instruction: str, before: list[Part], page: list[Part]) -> tuple[Part, ...]:
    """A request's parts: R6's examples, the page, then the instruction, for every provider."""
    return tuple([*before, *page, Part("text", text=instruction)])


def assemble(model: dict, rung: str, input_: str, page_id: str, para: int = 0):
    """(system, parts) for one cell on one page."""
    system, instruction = text(rung, para)
    page = page_parts(page_id, input_)
    if input_ == "C2-tiled":
        instruction += " " + _toml("ladder")["fixed"]["tiled"].format(n=len(page))
    before = fewshot_parts() if _toml("ladder")["rungs"][rung].get("fewshot") else []
    return system, _place(instruction, before, page)


def qa(model: dict, page_id: str, question: str, framing: str):
    """The QA probe on C2, with R0's framing or R1's system line."""
    system = text("R1")[0] if framing == "R1" else None
    user = _toml("probes")["qa"]["user"].format(question=question)
    return system, _place(user, [], page_parts(page_id, "C2"))


@functools.cache
def _p_scan() -> dict[tuple[str, str], dict]:
    from bench.arms import cached  # the free arms' outputs, as current as their inputs

    return cached()


def m_plus_p(model: dict, page_id: str):
    """Stage B4's M+P context: R2's prompt and the page's image, plus P-scan's reading of the
    page's scan, the app's second look (P-scan's cached output, so the free arms run first): its
    struck words with their struck characters, and its clean text, the live characters of its
    words in its reading order. On a page P-scan failed, the context says it gave no reading."""
    rec = _p_scan().get(("P-scan", page_id))
    if rec is None:
        raise FileNotFoundError(f"no P-scan output for {page_id}; run `python -m bench arms run`")
    system, instruction = text("R2")
    p = _toml("probes")["m_plus_p"]
    if rec.get("error"):
        context = p["none"]
    else:
        struck, live = [], []
        for word, spans in rec["words"]:
            mask = [any(s <= i < e for s, e in spans) for i in range(len(word))]
            if chars := "".join(c for c, m in zip(word, mask) if m):
                struck.append(f"{word} [{chars}]")
            if rest := "".join(c for c, m in zip(word, mask) if not m):
                live.append(rest)
        listed = "\n".join(struck) or "(none)"
        context = f"{p['intro']}\n\n{p['struck']}\n{listed}\n\n{p['clean']}\n{' '.join(live)}"
    return system, _place(instruction, [], [*page_parts(page_id, "C2"), Part("text", text=context)])


def text_only(model: dict, page_id: str):
    """Stage B4's P→M: no image, the package's provenance_text and one instruction."""
    res = st.detect_pdf(str(render.path(page_id, "c1.pdf")), method="vector")
    user = _toml("probes")["text_only"]["user"]
    return None, (Part("text", text=st.provenance_text(res)), Part("text", text=user))


# ------------------------------------------------------------------------------------- few-shot

# R6's examples: (file name, the stratum shown, vertical padding in line heights, whether to
# blank the neighbouring lines' glyphs the crop catches; blanking would cut the table's rules)
EXAMPLES = (
    ("struck", frozenset({"E1", "S1", "S6"}), 0.15, True),  # a plain black single line
    ("underline", frozenset({"N2"}), 0.15, True),  # below the baseline: N2b is left out
    ("table-rule", frozenset({"N3"}), 0.1, False),
)
WINDOW = 6


def fewshot_parts() -> list[Part]:
    rows = json.loads((FEWSHOT / "examples.json").read_text(encoding="utf-8"))
    parts = []
    for row in rows:
        path = FEWSHOT / row["image"]
        parts += [
            Part("text", text=_toml("ladder")["fixed"]["example"]),
            Part("image", path=rel(path), sha256=sha256_file(path)),
            Part("text", text=json.dumps(row["answer"])),
        ]
    return parts


def _example(strata: frozenset[str]) -> tuple[dict, list[int]]:
    """On R6's example page, which no dev list draws (so R6 is never scored on its own example's
    page), a run of up to six words on one line around a word whose strata all lie in `strata`,
    every other word in the run plain or the same."""
    drawn = {p for pages in dev_pages().values() for p in pages}
    if FEWSHOT_PAGE in drawn:
        raise ValueError(f"{FEWSHOT_PAGE}, R6's example page, is in a dev list")
    for page in load_pages("G0", split="dev"):
        if page["page_id"] != FEWSHOT_PAGE:
            continue
        words = page["words"]
        for i, w in enumerate(words):
            if not w.get("strata") or not set(w["strata"]) <= strata or w["furniture"]:
                continue
            line = [j for j, x in enumerate(words) if x["line"] == w["line"]]
            k = line.index(i)
            run = line[max(0, k - WINDOW // 2) : max(0, k - WINDOW // 2) + WINDOW]
            if len(run) >= 3 and all(set(words[j].get("strata", ())) <= strata for j in run):
                return page, run
    raise ValueError(f"no line on {FEWSHOT_PAGE} shows {sorted(strata)}")


def build_fewshot() -> list[dict]:
    """R6's three worked examples from G0-dev: a struck word, an underlined word and a table row
    with its rules, each a run of whole words on one line cropped from C2, with the answer for
    every word in it."""
    FEWSHOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, stratum, pad, blank in EXAMPLES:
        page, run = _example(stratum)
        words = [page["words"][j] for j in run]
        image = Image.open(render.path(page["page_id"], "c2.png")).convert("RGB")
        scale = image.size * 2
        h = max(w["bbox"][3] - w["bbox"][1] for w in words)
        box = (min(w["bbox"][0] for w in words) - 0.003, min(w["bbox"][1] for w in words) - pad * h,
               max(w["bbox"][2] for w in words) + 0.003, max(w["bbox"][3] for w in words) + pad * h)  # fmt: skip
        px = [round(v * s) for v, s in zip(box, scale)]
        crop = image.crop(px)
        draw = ImageDraw.Draw(crop)
        for j, w in enumerate(page["words"]):
            if blank and j not in run:
                x0, y0, x1, y1 = (round(v * s) for v, s in zip(w["bbox"], scale))
                draw.rectangle([x0 - px[0], y0 - px[1], x1 - px[0], y1 - px[1]], fill="white")
        buf = io.BytesIO()
        crop.save(buf, "PNG")
        (FEWSHOT / f"{name}.png").write_bytes(buf.getvalue())
        answer = {"words": [{"text": w["text"], "chars": "".join(w["text"][a:b] for a, b in w["struck"]),
                             "verdict": "struck" if w["struck"] else "clean", "confidence": 1.0}
                            for w in words]}  # fmt: skip
        rows.append({"image": f"{name}.png", "page_id": page["page_id"], "answer": answer})
    text = json.dumps(rows, indent=2) + "\n"
    (FEWSHOT / "examples.json").write_text(text, encoding="utf-8", newline="\n")
    return rows


def hashes() -> dict[str, str]:
    """sha256 of every assembled prompt (rung × paraphrase), the fixed lines, the probes and the
    few-shot files: what prereg-2 tags."""
    out = {}
    for rung in RUNGS:
        for para in range(PARAPHRASES):
            system, user = text(rung, para)
            out[f"{rung}.p{para}"] = sha256_bytes(json.dumps([system, user]).encode())
    for name in ("ladder.toml", "probes.toml"):
        out[name] = sha256_file(PROMPTS / name)
    for path in sorted(FEWSHOT.glob("*")):
        out[f"fewshot/{path.name}"] = sha256_file(path)
    return out


def register(sub) -> None:
    p = sub.add_parser("prompts", help="R6's worked examples, and the prompts' hashes")
    cmd = p.add_subparsers(required=True, metavar="action")
    cmd.add_parser("fewshot", help="crop R6's examples from G0-dev").set_defaults(
        func=lambda a: print("\n".join(f"{r['image']}: {r['page_id']}" for r in build_fewshot()))
    )
    cmd.add_parser("hash", help="sha256 of every assembled prompt").set_defaults(
        func=lambda a: print("\n".join(f"{v}  {k}" for k, v in hashes().items()))
    )
