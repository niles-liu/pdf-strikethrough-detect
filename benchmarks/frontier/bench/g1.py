"""G1, the public redline corpus (PREREG.md, G1): ground truth, the page sample and its checks.

    python -m bench g1 build    # data/gt/G1.jsonl and the error-rate sample
    python -m bench g1 check    # stage B1 step 3

Ground truth per page: the package's `vector` and `flag` detectors (0.12.1, as frozen) run
separately and are matched to PyMuPDF's words by box. A word both report is struck, with the
vector detector's character span; a word only one reports is masked; every other word is live.
The manifest's scanned pages are left out, which removes the Copyright Office document.

Furniture is found per document. A line within 15% of the top or bottom edge is furniture when
its text, digits ignored, recurs at the same height (to 1% of the page) on at least a fifth of
the document's pages, and on two at least; a line with no letters must also recur at the same
horizontal position, or a table's column of figures would count. A line with letters is also
furniture when it is like one of those at its height (Indel similarity at least 0.6: a running
foot whose date changes). And a line holding only a number, near any edge, is furniture when the
number is the page's own: its index plus the offset most of the document's page numbers show
(which catches the page number of a rotated or differently laid-out page).

After the test run, `data/gt/G1-corrections.jsonl` (written by `audit ingest` from the relabel)
corrects the labels of the words the annotators relabelled; it never changes which pages are
drawn.
"""

import random
import re
import sys
from collections import Counter, defaultdict

import pymupdf
from pdf_strikethrough import detect, native
from rapidfuzz.distance import Indel

from bench.common import (
    GT, ROOT, load_pages, page_id, read_jsonl, rel, seed, sha256_file, write_jsonl,
)  # fmt: skip
from bench.corpus import iter_corpus
from bench.score import normalize, parse, score_page
from bench.stats import largest_remainder

STRUCK_PAGES, FREE_PAGES = 10, 3
EDGE, SHARE, NEAR = 0.15, 0.2, 0.01  # furniture: edge band, share of pages, same position
SIMILAR = 0.6  # a line this like a recurring one at its height is furniture too
DIGITS = re.compile(r"\d+")
ERROR_SAMPLE = 200  # struck words, and as many live words on struck lines
EXPECTED = (55_171, 0.9985)  # vector detections, share the flag detector confirms
CORRECTIONS = GT / "G1-corrections.jsonl"


def _key(rec: dict) -> tuple:
    """confirmation_rate's key: float noise in the boxes doesn't break the match."""
    return rec["page"], rec["text"], round(rec["bbox_frac"][0], 2), round(rec["bbox_frac"][1], 2)


def page_truth(page: pymupdf.Page, pno: int) -> tuple[list[dict], list, list]:
    """Ground-truth words for one page, furniture not yet marked (see `mark_furniture`), plus
    the raw vector and flag records."""
    words = page.get_text("words")
    vec = native.page_strikes(page, pno, "vector", words=words)
    flag = native.page_strikes(page, pno, "flag", words=words)
    by_vec = detect._match_native_seq(page, vec, words)
    by_flag = detect._match_native_seq(page, flag, words)
    lines = [(w[5], w[6]) for w in words if w[4].strip()]
    out = []
    for (text, bbox, v), (_, _, f), line in zip(by_vec, by_flag, lines, strict=True):
        out.append(
            {
                "text": text,
                "bbox": [round(x, 5) for x in bbox],
                "struck": [list(v["char_span"])] if v and f else [],
                "masked": (v is None) != (f is None),
                "furniture": False,
                "line": list(line),
            }
        )
    return out, vec, flag


def mark_furniture(pages: dict[int, list[dict]]) -> None:
    """Mark the furniture of one document's pages (page index -> words) in place, by the three
    rules in the module docstring."""
    lines = {}  # (page, line) -> raw text, masked text, centre x, centre y
    for pno, words in pages.items():
        by_line = defaultdict(list)
        for w in words:
            by_line[tuple(w["line"])].append(w)
        for key, ws in by_line.items():
            x0, y0 = min(w["bbox"][0] for w in ws), min(w["bbox"][1] for w in ws)
            x1, y1 = max(w["bbox"][2] for w in ws), max(w["bbox"][3] for w in ws)
            raw = " ".join(w["text"] for w in ws)
            lines[pno, key] = raw, DIGITS.sub("#", raw).casefold(), (x0 + x1) / 2, (y0 + y1) / 2
    edge = {k: v for k, v in lines.items() if v[3] < EDGE or v[3] > 1 - EDGE}  # top or bottom
    by_text = defaultdict(list)
    for (pno, _), (_, text, x, y) in edge.items():
        by_text[text].append((pno, x, y))
    need = max(2, SHARE * len(pages))
    found = set()
    for k, (_, text, x, y) in edge.items():
        letters = any(c.isalpha() for c in text)
        recurs = {q for q, qx, qy in by_text[text]
                  if abs(qy - y) <= NEAR and (letters or abs(qx - x) <= NEAR)}  # fmt: skip
        if len(recurs) >= need:
            found.add(k)
    heights = defaultdict(set)  # each recurring line with letters -> the heights it sits at
    for k in found:
        if any(c.isalpha() for c in edge[k][1]):
            heights[edge[k][1]].add(round(edge[k][3], 3))
    for k, (_, text, _, y) in edge.items():
        if k in found or not any(c.isalpha() for c in text):
            continue
        if any(Indel.normalized_similarity(text, f) >= SIMILAR
               for f, ys in heights.items() if any(abs(fy - y) <= NEAR for fy in ys)):  # fmt: skip
            found.add(k)
    offsets = Counter(int(edge[k][0]) - k[0] for k in found if edge[k][0].isdecimal())
    if offsets:  # page numbers: a lone number near any edge that is the page's own
        offset = offsets.most_common(1)[0][0]
        for (pno, key), (raw, _, x, y) in lines.items():
            if min(x, y, 1 - x, 1 - y) < EDGE and raw.isdecimal() and int(raw) == pno + offset:
                found.add((pno, key))
    for pno, key in found:
        for w in pages[pno]:
            w["furniture"] = w["furniture"] or tuple(w["line"]) == key


def _body(words: list[dict]) -> list[dict]:
    return [w for w in words if not w["furniture"]]


def _kind(words: list[dict]) -> str | None:
    """Struck, strike-free or neither, from the words left after furniture."""
    if any(w["struck"] for w in _body(words)):
        return "struck"
    return None if any(w["masked"] for w in _body(words)) else "strike-free"


def _tertiles(pages: list[dict]) -> list[list[dict]]:
    """Struck pages ranked by struck body words (ties by page index), split as evenly as
    possible."""
    ranked = sorted(
        pages, key=lambda p: (sum(bool(w["struck"]) for w in _body(p["words"])), p["page"])
    )
    q, r = divmod(len(ranked), 3)
    sizes = [q + (i < r) for i in range(3)]
    return [ranked[sum(sizes[:i]) : sum(sizes[: i + 1])] for i in range(3)]


def _partial(p: dict) -> bool:
    """A body word struck in part: some of the characters the scorer keeps are struck and some
    not, so a strike stopping short of a word's edge punctuation (`Board.`) is a whole-word one."""
    for w in _body(p["words"]):
        struck = {i for s, e in w["struck"] for i in range(s, e)}
        kept = set(normalize(w["text"])[1])
        if kept & struck and kept - struck:
            return True
    return False


def sample_struck(pages: list[dict], rng: random.Random) -> list[dict]:
    """Up to 10 struck pages, from six cells (tertile × partial-word strike) in proportion to
    their sizes; with more than 10 pages no cell's share exceeds its size."""
    if len(pages) <= STRUCK_PAGES:
        return pages
    cells = [
        [p for p in t if _partial(p) == part] for t in _tertiles(pages) for part in (False, True)
    ]
    share = largest_remainder([len(c) for c in cells], STRUCK_PAGES)
    return [p for c, k in zip(cells, share) for p in rng.sample(c, k)]


def build() -> list[dict]:
    """Every page of the two dev documents, and the sampled pages of the seven test documents.
    Writes nothing; `write` does."""
    docs = []
    for entry, path in iter_corpus():
        scanned = set(entry.get("scanned_pages", ()))
        with pymupdf.open(path) as doc:
            if len(scanned) == doc.page_count:
                continue
            truth = {pno: page_truth(doc[pno], pno)[0]
                     for pno in range(doc.page_count) if pno not in scanned}  # fmt: skip
        mark_furniture(truth)
        pages = [{"page": pno, "kind": kind, "words": words}
                 for pno, words in truth.items() if (kind := _kind(words))]  # fmt: skip
        docs.append((entry, path, pages))

    rng = random.Random(seed("g1_dev"))
    dev = set(rng.sample(sorted(e["file"] for e, _, _ in docs), 2))
    out = []
    for entry, path, pages in docs:
        split = "dev" if entry["file"] in dev else "test"
        if split == "test":
            struck = sample_struck([p for p in pages if p["kind"] == "struck"], rng)
            free = sorted(p["page"] for p in pages if p["kind"] == "strike-free")
            keep = {p["page"] for p in struck} | set(rng.sample(free, min(FREE_PAGES, len(free))))
            pages = [p for p in pages if p["page"] in keep]
        doc_id = path.stem
        digest = sha256_file(path)
        for p in pages:
            out.append(
                {
                    "page_id": page_id("G1", doc_id, p["page"]),
                    "set": "G1",
                    "split": split,
                    "doc": doc_id,
                    "page": p["page"],
                    "cluster": doc_id,
                    "pdf": rel(path),
                    "pdf_sha256": digest,
                    "kind": p["kind"],
                    "words": p["words"],
                }
            )
    _correct(out)
    return out


def _correct(pages: list[dict]) -> None:
    """The post-test relabel (PREREG, The audit): each corrected word becomes struck (all of it,
    or the characters the annotators gave), live, or masked where they could not tell."""
    by_id = {p["page_id"]: p for p in pages}
    for c in read_jsonl(CORRECTIONS) if CORRECTIONS.exists() else ():
        w = by_id[c["page_id"]]["words"][c["word"]]
        if w["text"] != c["text"]:
            raise ValueError(f"{CORRECTIONS.name}: word {c['word']} of {c['page_id']} is "
                             f"{w['text']!r}, not {c['text']!r}")  # fmt: skip
        k = w["text"].find(c["chars"]) if c["chars"] else -1
        if c["label"] == "struck" and c["chars"] and k < 0:
            raise ValueError(f"{CORRECTIONS.name}: {c['chars']!r} is not in word {c['word']} of "
                             f"{c['page_id']}, {w['text']!r}")  # fmt: skip
        span = [k, k + len(c["chars"])] if k >= 0 else [0, len(w["text"])]
        w["struck"] = [span] if c["label"] == "struck" else []
        w["masked"] = c["label"] == "cannot tell"


def write(pages: list[dict]) -> None:
    write_jsonl(GT / "G1.jsonl", pages)
    if not CORRECTIONS.exists():  # drawn before the test; the relabel must not redraw it
        test = [p for p in pages if p["split"] == "test"]
        write_jsonl(GT / "G1-error-sample.jsonl", error_sample(test))


def error_sample(test_pages: list[dict]) -> list[dict]:
    """PREREG, The audit: 200 struck G1-test words and 200 live words on lines carrying a strike."""
    struck, live = [], []
    for p in test_pages:
        scored = [(i, w) for i, w in enumerate(p["words"]) if not (w["masked"] or w["furniture"])]
        struck_lines = {tuple(w["line"]) for _, w in scored if w["struck"]}
        for i, w in scored:
            if w["struck"]:
                struck.append((p["page_id"], i))
            elif tuple(w["line"]) in struck_lines:
                live.append((p["page_id"], i))
    rng = random.Random(seed("g1_error_sample"))
    picks = [("struck", x) for x in rng.sample(struck, min(ERROR_SAMPLE, len(struck)))]
    picks += [("live", x) for x in rng.sample(live, min(ERROR_SAMPLE, len(live)))]
    return [{"page_id": pid, "word": i, "kind": kind} for kind, (pid, i) in picks]


def check() -> bool:
    """Stage B1 step 3: the confirmation numbers, the sample's reproducibility, and B-naive's
    alignment on every G1-dev page."""
    n_vec = n_conf = 0
    for _, path in iter_corpus():
        with pymupdf.open(path) as doc:
            for pno in range(doc.page_count):
                _, vec, flag = page_truth(doc[pno], pno)
                n_vec += len(vec)
                n_conf += len({_key(r) for r in vec} & {_key(r) for r in flag})
    rate = n_conf / n_vec if n_vec else 0.0
    ok = n_vec == EXPECTED[0] and round(rate, 4) == EXPECTED[1]
    print(f"{'ok  ' if ok else 'FAIL'} {n_vec:,} vector detections, {rate:.2%} confirmed")

    same = read_jsonl(GT / "G1.jsonl") == build()
    ok &= same
    print(f"{'ok  ' if same else 'FAIL'} the page sample rebuilds identically")

    worst = (2.0, "")  # above any share, so the first page sets it
    for p in load_pages("G1", split="dev"):
        with pymupdf.open(ROOT / p["pdf"]) as doc:
            text = doc[p["page"]].get_text()
        rows = score_page(p, parse(text), details=True)["details"]
        words = {r["word"] for r in rows}
        seen = {r["word"] for r in rows if r["outcome"] != "absent"}
        worst = min(worst, (len(seen) / len(words) if words else 1.0, p["page_id"]))
    ok &= worst[0] >= 0.99
    print(f"{'ok  ' if worst[0] >= 0.99 else 'FAIL'} B-naive aligns to >= {worst[0]:.2%} of words "
          f"on every G1-dev page (lowest: {worst[1]})")  # fmt: skip
    return ok


def _summary(pages: list[dict]) -> None:
    for split in ("dev", "test"):
        ps = [p for p in pages if p["split"] == split]
        docs = sorted({p["doc"] for p in ps})
        struck = sum(bool(w["struck"]) for p in ps for w in p["words"])
        print(f"G1-{split}: {len(docs)} documents, {len(ps)} pages, {struck:,} struck words")


def register(sub) -> None:
    p = sub.add_parser("g1", help="G1 ground truth and checks")
    cmd = p.add_subparsers(required=True, metavar="action")
    cmd.add_parser("build", help="ground truth, page sample, error-rate sample").set_defaults(
        func=lambda a: _summary(write(build()) or load_pages("G1"))
    )
    cmd.add_parser("check", help="stage B1 step 3").set_defaults(
        func=lambda a: sys.exit(0 if check() else 1)
    )
