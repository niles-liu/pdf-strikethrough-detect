"""G5, BRESSAY's handwritten pages (PREREG.md, G5): ground truth from the pages' transcriptions.

    python -m bench g5 build    # data/gt/G5.jsonl
    python -m bench g5 check    # stage B1 step 4

A transcription marks struck text `--…--`, and text added above or below the line `##…##` and
`$$…$$`; the markers pair up within a line, and nest. Its words, split on whitespace once the
markers are removed, are the page's words in order: the characters inside `--…--` struck, the
additions live where they stand. A word reading `xxx`, or holding a struck `xxx` (`--xxx--`,
struck and unreadable) or `@@???@@` (illegible), is masked whole, with the scorer's UNREAD as its
text, so it absorbs one output token and is never scored. A transcription with an unpaired
marker, or an `@` left over, does not parse, and its page is not sampled. G5 has no furniture
and no boxes; a word's `line` is `[paragraph, line]`, paragraphs being separated by blank lines.

Pages, from BRESSAY's test split: a struck page holds at least one readable struck sub-token, a
strike-free page no strike at all; a page whose only strikes are unreadable is neither. G5 holds
every struck page (strikes are sparse: a median of 3 a page) and 10 strike-free ones. One
generator, random.Random(20261023), orders the struck pages with `.sample(sorted(struck),
len(struck))`; the first in that order, until there are at least 40 holding at least 200 struck
sub-tokens, are the core (`core`), the pages R0 runs on; then it draws the 10 strike-free pages,
also core, with `.sample(sorted(free), 10)`.
"""

import random
import re
import sys
from pathlib import Path

from bench.common import GT, page_id, read_jsonl, rel, seed, sha256_file, write_jsonl
from bench.fetch import BRESSAY
from bench.score import UNREAD, normalize, score_page

MIN_PAGES, MIN_STRUCK = 40, 200  # the core's struck pages, and the struck sub-tokens they hold
N_FREE = 10
MARKER = re.compile(r"--|##|\$\$|@@\?\?\?@@")


class ParseError(ValueError):
    """A transcription with an unpaired marker, or an `@` left over."""


def parse(text: str) -> list[dict]:
    """A transcription's words, in order, as ground-truth word records."""
    words, para, row = [], 0, 0
    for k, line in enumerate(text.splitlines()):
        if not line.strip():  # a blank line ends a paragraph
            para, row = para + (row > 0), 0
            continue
        chars: list[tuple[str, bool]] = []
        open_ = dict.fromkeys(("--", "##", "$$"), False)
        pos = 0
        for m in MARKER.finditer(line):
            chars += [(c, open_["--"]) for c in line[pos : m.start()]]
            if m.group() in open_:
                open_[m.group()] = not open_[m.group()]
            else:  # @@???@@
                chars.append((UNREAD, open_["--"]))
            pos = m.end()
        chars += [(c, open_["--"]) for c in line[pos:]]
        if unpaired := [mark for mark, on in open_.items() if on]:
            raise ParseError(f"line {k + 1}: unpaired {unpaired[0]}")
        if any(c == "@" for c, _ in chars):
            raise ParseError(f"line {k + 1}: an @ outside @@???@@")
        word: list[tuple[str, bool]] = []
        for c, struck in [*chars, (" ", False)]:
            if not c.isspace():
                word.append((c, struck))
            elif word:
                words.append(_word("".join(c for c, _ in word), [s for _, s in word], [para, row]))
                word = []
        row += 1
    return words


def _word(text: str, struck: list[bool], line: list[int]) -> dict:
    spans, start = [], None
    for i, s in enumerate([*struck, False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            spans.append([start, i])
            start = None
    parts = [text, *(text[s:e] for s, e in spans)]  # the word, and each struck run in it
    unread = UNREAD in text or "xxx" in {normalize(t)[0] for t in parts}
    if unread:
        text, spans = UNREAD, [[0, 1]] if spans else []
    return {"text": text, "bbox": None, "struck": spans, "masked": unread, "furniture": False,
            "line": line}  # fmt: skip


def n_struck(words: list[dict]) -> int:
    """Readable struck sub-tokens, as the scorer counts them."""
    return score_page({"words": words}, None)["struck"]


def kind(words: list[dict]) -> str | None:
    """`struck`, `free` (no strike at all), or None: its only strikes are unreadable."""
    if n_struck(words):
        return "struck"
    return None if any(w["struck"] for w in words) else "free"


def build() -> tuple[list[dict], dict]:
    """G5's pages, and the counts behind the draw."""
    files = {r["file"]: r["sha256"] for r in read_jsonl(BRESSAY / "manifest.jsonl")}
    if not files:
        raise FileNotFoundError("no BRESSAY pages; run `python -m bench fetch bressay`")

    def checked(path: Path) -> Path:
        if sha256_file(path) != files.get(path.relative_to(BRESSAY).as_posix()):
            raise ValueError(f"{path.name} differs from data/bressay/manifest.jsonl")
        return path

    names = sorted(checked(BRESSAY / "test.txt").read_text(encoding="utf-8").split())
    pools: dict[str | None, list[str]] = {"struck": [], "free": [], None: []}
    pages, failed = {}, []
    for name in names:
        png, txt = (checked(BRESSAY / "pages" / f"{name}.{ext}") for ext in ("png", "txt"))
        try:
            words = parse(txt.read_text(encoding="utf-8"))
        except ParseError as e:
            failed.append(f"{name}: {e}")
            continue
        pools[kind(words)].append(name)
        pages[name] = {
            "page_id": page_id("G5", name, 0), "set": "G5", "split": "test", "doc": name,
            "page": 0, "cluster": name, "pdf": rel(png), "pdf_sha256": files[f"pages/{png.name}"],
            "transcription": rel(txt), "transcription_sha256": files[f"pages/{txt.name}"],
            "words": words,
        }  # fmt: skip
    drawn, core = draw(
        {name: n_struck(pages[name]["words"]) for name in pools["struck"]}, pools["free"]
    )
    counts = {"test pages": len(names), "unparsed": failed,
              **{f"{k or 'neither'} pool": len(v) for k, v in pools.items()}}  # fmt: skip
    return [pages[name] | {"core": name in core} for name in drawn], counts


def draw(struck: dict[str, int], free: list[str]) -> tuple[list[str], set[str]]:
    """Every struck page and N_FREE strike-free pages, sorted; and the core: the struck pages
    (each with its struck sub-tokens) in seeded order until there are at least MIN_PAGES holding
    at least MIN_STRUCK, with the strike-free pages."""
    rng = random.Random(seed("g5_pages"))
    core, n = [], 0
    for name in rng.sample(sorted(struck), len(struck)):
        if len(core) >= MIN_PAGES and n >= MIN_STRUCK:
            break
        core.append(name)
        n += struck[name]
    clean = rng.sample(sorted(free), N_FREE)
    return sorted([*struck, *clean]), {*core, *clean}


def check() -> bool:
    """Stage B1 step 4: G5 rebuilds identically, and every transcription in the test split
    parses; G5 holds every struck page and 10 strike-free ones; the core's struck pages, at least
    40, hold at least 200 struck sub-tokens."""
    pages = read_jsonl(GT / "G5.jsonl")
    fresh, counts = build()
    kinds = [kind(p["words"]) for p in pages]
    struck = [p for p, k in zip(pages, kinds) if k == "struck"]
    core = [p for p in struck if p["core"]]
    n, free = sum(n_struck(p["words"]) for p in core), kinds.count("free")
    held = f"the core's {len(core)} struck pages (at least {MIN_PAGES}) hold {n} struck sub-tokens"
    checks = [
        ("G5 rebuilds identically", pages == fresh),
        (f"all {counts['test pages']} test transcriptions parse", not counts["unparsed"]),
        (f"all {counts['struck pool']} struck pages are in", len(struck) == counts["struck pool"]),
        (f"{held} (at least {MIN_STRUCK})", len(core) >= MIN_PAGES and n >= MIN_STRUCK),
        (f"{free} strike-free pages ({N_FREE}), all in the core, and no others",
         free == N_FREE and len(struck) + free == len(pages)
         and all(p["core"] for p, k in zip(pages, kinds) if k == "free")),
    ]  # fmt: skip
    for text, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return all(ok for _, ok in checks)


def _summary(pages: list[dict], counts: dict) -> None:
    masked = [w for p in pages for w in p["words"] if w["masked"]]
    struck = sum(bool(w["struck"]) for w in masked)
    print(f"G5: {len(pages)} pages, {sum(p['core'] for p in pages)} of them the core, from "
          f"{counts['test pages']} test pages (pools: "
          f"{counts['struck pool']} struck, {counts['free pool']} strike-free, "
          f"{counts['neither pool']} neither; {len(counts['unparsed'])} do not parse); "
          f"{len(masked)} masked words, {struck} of them struck and unreadable")  # fmt: skip
    for line in counts["unparsed"]:
        print(f"  not parsed: {line}")


def _build(args) -> None:
    pages, counts = build()
    write_jsonl(GT / "G5.jsonl", pages)
    _summary(pages, counts)


def register(sub) -> None:
    p = sub.add_parser("g5", help="G5 ground truth (BRESSAY) and checks")
    cmd = p.add_subparsers(required=True, metavar="action")
    cmd.add_parser("build", help="parse the transcriptions and draw G5's pages").set_defaults(
        func=_build
    )
    cmd.add_parser("check", help="stage B1 step 4").set_defaults(
        func=lambda a: sys.exit(0 if check() else 1)
    )
