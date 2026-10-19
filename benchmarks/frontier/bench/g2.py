"""G2, US Congress bills (PREREG.md, G2): ground truth from GovInfo's bill XML, aligned to the PDF.

    python -m bench g2 build    # data/gt/G2.jsonl and data/gt/G2-coverage.jsonl
    python -m bench g2 check    # stage B1 step 4

The XML's words in document order, each character flagged struck (inside `<deleted-phrase>`, or
under an element whose nearest `changed` says "deleted"), are aligned to the PDF's body words
(PyMuPDF, extraction order, furniture removed) with the scorer's alignment, a page at a time
against a window of the XML that follows the previous page's, equal-cost ties going to the
earliest copy (a committee substitute can repeat the struck text it replaces). A PDF character
aligned to a struck XML character is struck; a body word left unaligned is masked. A page's
coverage is the share of its body words aligned, and pages under 98% are dropped.

`check` holds every kept page to the strikes GPO drew: each letter and digit of a body word is
struck in the ground truth exactly when a drawn horizontal line crosses it.

Selection: bills issued after the cutoff, the latest training cutoff any model states
(config/models.toml); bills introduced after it first, then the most recently introduced others,
until 45 bills have a usable page, the bill with the most inline strikes (`<deleted-phrase>`)
taking the last place if it isn't in by then, since most bills strike whole sections. Per bill,
one page carrying struck text, the first with at least 10% of its words live, else the first; and
for a strict bill (introduced after the cutoff), its first page without. One page per bill keeps
G2's clusters many and alike in size, so no single bill decides a test. The three bills that
follow in the same order are G2-dev, each with its struck page and its first page without.
"""

import calendar
import datetime as dt
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from itertools import groupby
from pathlib import Path

import pymupdf

from bench.common import GT, ROOT, config, page_id, read_jsonl, rel, sha256_file, write_jsonl
from bench.fetch import BILLS
from bench.score import _is_punct, match, normalize, tokens_from_text

N_BILLS, N_DEV_BILLS = 45, 3  # G2-test's bills, then G2-dev's: the bills after them in order
MIN_COVERAGE = 0.98
MIN_LIVE = 0.10  # a struck page with this share of its words live goes before a wholly struck one
LOOKAHEAD = 30  # the next page's first words, see align_bill
# Furniture, by position on GPO's US Letter pages (page fractions) and by rotation.
TOP, BOTTOM, LINE_NUMBERS = 0.085, 0.93, 0.235
# GPO's authentication seal on a bill's first page: drawn, not text, in the top band's left part
SEAL, SEAL_RIGHT, SEAL_PATHS = (
    ("AUTHENTICATED", "U.S.", "GOVERNMENT", "INFORMATION", "GPO"),
    0.35,
    10,
)
SEAL_LINE = [-1, 0]  # the seal's words' line: none of the text layer's
INLINE = {
    "deleted-phrase", "added-phrase", "term", "quote", "short-title", "external-xref",
    "internal-xref", "act-name", "italic", "bold", "header-in-text", "sponsor", "cosponsor",
    "committee-name", "fraction",
}  # fmt: skip
ENACTING = (
    "Be it enacted by the Senate and House of Representatives of the United States of America "
    "in Congress assembled,"
)
LABELS = {
    "title": "TITLE", "subtitle": "Subtitle", "part": "PART", "subpart": "Subpart",
    "chapter": "CHAPTER", "subchapter": "SUBCHAPTER", "division": "DIVISION",
    "subdivision": "Subdivision",
}  # fmt: skip  # printed before the enum by GPO's stylesheet, as "SEC." is for a section


def cutoff() -> dt.date:
    """The latest stated training cutoff in the run; a month-only cutoff is its last day."""
    dates = []
    for m in config("models").values():
        y, mo, *d = map(int, m["cutoff"].split("-"))
        dates.append(dt.date(y, mo, d[0] if d else calendar.monthrange(y, mo)[1]))
    return max(dates)


def xml_text(xml: bytes) -> tuple[str, list[bool]]:
    """The printed text of a bill version, with a struck flag per character. Adds what GPO's
    typesetting prints but the XML leaves to the stylesheet: the official title on the cover
    under the bill number, the enacting clause, and the label before each section, title, part
    and so on's number ("SEC. 2.", "TITLE I")."""
    chars: list[str] = []
    marks: list[bool] = []

    def put(text: str, struck: bool) -> None:
        chars.extend(text)
        marks.extend([struck] * len(text))

    def walk(el: ET.Element, struck: bool, parent: ET.Element | None) -> None:
        if el.tag == "metadata":
            return
        if el.get("changed"):
            struck = el.get("changed") == "deleted"
        struck = struck or el.tag == "deleted-phrase"
        block = el.tag not in INLINE
        if block:
            put(" ", False)
        if el.tag == "legis-body" and not el.get("display-enacting-clause", "").startswith("no"):
            put(ENACTING + " ", False)
        if el.tag == "enum" and parent is not None and parent.tag == "section":
            put("SECTION " if parent.get("section-type") == "section-one" else "SEC. ", struck)
        elif el.tag == "enum" and parent is not None and parent.tag in LABELS:
            put(LABELS[parent.tag] + " ", struck)
        put(el.text or "", struck)
        if (
            el.tag == "legis-num"
            and parent is not None
            and (title := parent.find("official-title")) is not None
        ):
            put(" " + "".join(title.itertext()) + " ", struck)
        for child in el:
            walk(child, struck, el)
            put(child.tail or "", struck)
        if block:
            put(" ", False)

    walk(ET.fromstring(xml), False, None)
    return "".join(chars), marks


def page_words(page: pymupdf.Page) -> list[dict]:
    """The page's words with the furniture rule applied: rotated lines, the top and bottom bands
    (page number, running foot, GPO's slug), the margin line numbers, each a number alone on its
    line, and the words of GPO's seal, drawn on a bill's first page, which the text layer lacks."""
    rotated = [
        pymupdf.Rect(line["bbox"])
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", ())
        if abs(line["dir"][1]) > 0.01
    ]
    w, h = page.rect.width, page.rect.height
    words = [x for x in page.get_text("words") if x[4].strip()]
    per_line = Counter((x[5], x[6]) for x in words)
    out = []
    for x0, y0, x1, y1, text, block, line, _ in words:
        centre = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
        furniture = (
            any(centre in r for r in rotated)
            or y1 <= TOP * h
            or y0 >= BOTTOM * h
            or (text.isdigit() and x1 <= LINE_NUMBERS * w and per_line[block, line] == 1)
        )
        out.append(
            {
                "text": text,
                "bbox": [round(v, 5) for v in (x0 / w, y0 / h, x1 / w, y1 / h)],
                "struck": [],
                "masked": False,
                "furniture": furniture,
                "line": [block, line],
            }
        )
    drawn = [d["rect"] for d in page.get_drawings()
             if d["rect"].x1 <= SEAL_RIGHT * w and d["rect"].y1 <= TOP * h]  # fmt: skip
    if len(drawn) >= SEAL_PATHS:
        box = pymupdf.Rect(drawn[0])
        for r in drawn[1:]:
            box |= r
        frac = [round(v, 5) for v in (box.x0 / w, box.y0 / h, box.x1 / w, box.y1 / h)]
        out[:0] = [{"text": t, "bbox": frac, "struck": [], "masked": False, "furniture": True,
                    "line": SEAL_LINE} for t in SEAL]  # fmt: skip
    return out


def _spans(text: str, marked: dict[int, bool]) -> list[list[int]]:
    """Struck spans from a word's aligned characters (index -> struck?). A run of struck
    characters bridges the unaligned ones inside it, such as the `.—` GPO adds between a header and
    its text; a word whose every aligned character is struck is struck to its edges, punctuation
    included (`(a)`, `annex.`), as GPO's line runs through it."""
    if marked and all(marked.values()):
        lo, hi = min(marked), max(marked) + 1
        while lo and _is_punct(text[lo - 1]):
            lo -= 1
        while hi < len(text) and _is_punct(text[hi]):
            hi += 1
        return [[lo, hi]]
    spans: list[list[int]] = []
    prev = False
    for p in sorted(marked):
        if marked[p] and prev:
            spans[-1][1] = p + 1
        elif marked[p]:
            spans.append([p, p + 1])
        prev = marked[p]
    return spans


def _pieces(text: str) -> list[tuple[int, str]]:
    """A PDF word split at em dashes, for the alignment only: GPO runs a header into its text
    ("GENERAL.—Section") where the XML has two words."""
    out, start = [], 0
    for part in text.split("—"):
        if part:
            out.append((start, part))
        start += len(part) + 1
    return out


def align_bill(xml: bytes, pdf: Path) -> list[dict]:
    """Every page of a bill version: its words with struck spans and masks, and its coverage.

    Each page aligns against a window of the XML starting after the previous page's last aligned
    word. The next page's first words ride along as lookahead, so a word hyphenated across the
    page break joins its XML word, and its second half takes that alignment onto the next page.
    A page with no body words (a blank or back cover) has no coverage."""
    text, marks = xml_text(xml)
    tokens = tokens_from_text(text, marks)
    with pymupdf.open(pdf) as doc:
        all_words = [page_words(doc[pno]) for pno in range(doc.page_count)]
    pieces = [
        [(i, off, part) for i, w in enumerate(words) if not w["furniture"]
         for off, part in _pieces(w["text"])]
        for words in all_words
    ]  # fmt: skip
    pages, start, carried = [], 0, {}
    for pno, words in enumerate(all_words):
        own = [x for x in pieces[pno] if x[:2] not in carried]
        ahead = pieces[pno + 1][:LOOKAHEAD] if pno + 1 < len(pieces) else []
        window = tokens[start : start + 2 * len(own) + 100]
        hits, used = match(window, [part for _, _, part in own + ahead], early=True)
        chars: dict[int, dict[int, bool]] = {}
        for (i, off), got in carried.items():
            chars.setdefault(i, {}).update({off + p: m for p, m in got.items()})
        for j, (i, off, _) in enumerate(own):
            for p, m in hits.get(j, {}).items():
                chars.setdefault(i, {})[off + p] = m
        carried = {
            ahead[j - len(own)][:2]: hits[j]
            for js in used.values() if js and js[0] < len(own) <= js[-1]
            for j in js if j >= len(own) and hits.get(j)
        }  # fmt: skip
        counted = masked = 0
        for i, w in enumerate(words):
            if w["furniture"] or not normalize(w["text"])[0]:
                continue  # lone punctuation is never scored, so never counted
            counted += 1
            if i in chars:
                w["struck"] = _spans(w["text"], chars[i])
            else:
                w["masked"] = True
                masked += 1
        mine = [k for k, js in used.items() if js and js[0] < len(own)]
        start += max(mine) + 1 if mine else 0
        coverage = 1 - masked / counted if counted else None
        pages.append({"page": pno, "words": words, "coverage": coverage})
    return pages


def _n_struck(page: dict) -> int:
    return sum(bool(w["struck"]) for w in page["words"] if not w["furniture"])


def _live_share(page: dict) -> float:
    """The share of a page's scored words that are live (no furniture, masks or lone punctuation)."""
    words = [w for w in page["words"] if not (w["furniture"] or w["masked"])]
    words = [w for w in words if normalize(w["text"])[0]]
    return sum(not w["struck"] for w in words) / len(words) if words else 0.0


def pick(pages: list[dict], free: bool = False) -> list[dict]:
    """A bill's G2 pages, in page order: the first page carrying struck text with at least 10% of
    its words live, else the first carrying struck text; with `free`, also the first page without.
    None if no usable page carries struck text."""
    usable = [p for p in pages if (p["coverage"] or 0) >= MIN_COVERAGE]
    struck = sorted(
        (p for p in usable if _n_struck(p)), key=lambda p: (_live_share(p) < MIN_LIVE, p["page"])
    )[:1]
    clean = [p for p in usable if not _n_struck(p)][:1] if free else []
    return sorted(struck + clean, key=lambda p: p["page"]) if struck else []


def _bill(r: dict, split: str) -> tuple[list[dict], list[dict]]:
    """A bill's G2 records in `split` and its coverage-report rows: a strike-free page for a
    strict bill in G2-test, and for every bill in G2-dev."""
    pdf, xml = BILLS / "pdf" / f"{r['package']}.pdf", BILLS / "xml" / f"{r['package']}.xml"
    if sha256_file(pdf) != r["pdf_sha256"] or sha256_file(xml) != r["xml_sha256"]:
        raise ValueError(f"{r['package']}: files differ from data/bills/manifest.jsonl")
    pages = align_bill(xml.read_bytes(), pdf)
    kept = {p["page"] for p in pick(pages, free=r["strict"] or split == "dev")}
    report = [{"package": r["package"], "page": p["page"], "coverage": p["coverage"],
               "struck_words": _n_struck(p), "kept": p["page"] in kept,
               "split": split} for p in pages]  # fmt: skip
    records = [
        {
            "page_id": page_id("G2", r["package"], p["page"]),
            "set": "G2",
            "split": split,
            "doc": r["package"],
            "page": p["page"],
            "cluster": r["bill"],
            "pdf": rel(pdf),
            "pdf_sha256": r["pdf_sha256"],
            "xml": rel(xml),  # the ground truth's source, frozen with the PDF
            "xml_sha256": r["xml_sha256"],
            "bill": r["bill"],
            "issued": r["issued"],
            "introduced": r["introduced"],
            "strict": r["strict"],
            "coverage": round(p["coverage"], 4),
            "words": p["words"],
        }
        for p in pages
        if p["page"] in kept
    ]
    return records, report


def build() -> tuple[list[dict], list[dict]]:
    """G2's pages, G2-test's then G2-dev's, and the coverage report for every page of every bill
    tried. Bills go in order until N_BILLS have a usable page, the bill with the most inline
    strikes taking the last place if it isn't in by then; G2-dev takes the next N_DEV_BILLS."""
    limit = cutoff()
    bills = [r for r in read_jsonl(BILLS / "manifest.jsonl") if r["issued"] > limit.isoformat()]
    for r in bills:
        r["strict"] = (r["introduced"] or "") > limit.isoformat()
    # strict first, each newest first; the package name breaks ties
    bills.sort(key=lambda r: (r["strict"], r["introduced"] or "", r["package"]), reverse=True)
    inline = max(bills, key=lambda r: (r["struck_phrases"], r["package"]), default=None)
    if inline is not None and not inline["struck_phrases"]:
        inline = None
    out, report, tried = [], [], set()
    for split, n in (("test", N_BILLS), ("dev", N_DEV_BILLS)):
        kept = 0
        while kept < n and (todo := [r for r in bills if r["package"] not in tried]):
            last = split == "test" and kept == n - 1
            r = inline if last and inline and inline["package"] not in tried else todo[0]
            tried.add(r["package"])
            records, rows = _bill(r, split)
            out, report = out + records, report + rows
            kept += bool(records)
    return out, report


def drawn(page: pymupdf.Page) -> list[tuple[str, list[bool]]]:
    """The page's words, in `page_words` order, each with a flag per character: does a drawn
    horizontal line cross it between 10% and 75% of the font size above the baseline, both its
    line's main text's (so a raised numerator, as in 1⁄4, is judged by its line)? A check on the
    alignment that shares nothing with it, or with the package."""
    segments = []
    for d in page.get_drawings():
        for item in d["items"]:
            if item[0] == "l" and abs(item[1].y - item[2].y) < 0.5:
                a, b = sorted((item[1].x, item[2].x))
                segments.append((a, b, item[1].y))
            elif item[0] == "re" and item[1].height < 3:
                segments.append((item[1].x0, item[1].x1, (item[1].y0 + item[1].y1) / 2))
    segments = [s for s in segments if s[1] - s[0] > 0.5]

    def crossed(ch: dict, base: float, size: float) -> bool:
        x = (ch["bbox"][0] + ch["bbox"][2]) / 2
        return any(a <= x <= b and base - 0.75 * size <= y <= base - 0.1 * size
                   for a, b, y in segments)  # fmt: skip

    words: list[tuple[str, list[bool]]] = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", ()):
            main = max(line["spans"], key=lambda span: len(span["chars"]))
            if not main["chars"]:
                continue
            base, size = main["chars"][0]["origin"][1], main["size"]
            chars = [ch for span in line["spans"] for ch in span["chars"]]
            for space, run in groupby(chars, key=lambda ch: ch["c"].isspace()):
                if not space:
                    run = list(run)
                    words.append(("".join(ch["c"] for ch in run),
                                  [crossed(ch, base, size) for ch in run]))  # fmt: skip
    return words


def check() -> bool:
    """Stage B1 step 4: G2 rebuilds identically; on every kept page, G2-dev's too, each letter
    and digit of a body word is struck in the ground truth exactly when a drawn line crosses it;
    the furniture rule shown on two bills; 45 bills in G2-test, with the strict subset counted,
    and 3 others in G2-dev."""
    pages = read_jsonl(GT / "G2.jsonl")
    ok = pages == build()[0]
    print(f"{'ok  ' if ok else 'FAIL'} G2 rebuilds identically")
    checked = 0
    for p in pages:
        with pymupdf.open(ROOT / p["pdf"]) as doc:
            lines = drawn(doc[p["page"]])
        words = [w for w in p["words"] if w["line"] != SEAL_LINE]  # the text layer's
        if [t for t, _ in lines] != [w["text"] for w in words]:
            ok = False
            print(f"FAIL {p['page_id']}: the drawn-line check reads other words than the page's")
            continue
        off = []
        for w, (_, crossed) in zip(words, lines):
            if w["furniture"] or w["masked"]:
                continue
            for i, line in enumerate(crossed):
                if w["text"][i].isalnum():
                    checked += 1
                    if line != any(s <= i < e for s, e in w["struck"]):
                        off.append(w["text"])
        if off:
            ok = False
            print(f"FAIL {p['page_id']}: {len(off)} characters disagree with the drawn lines, "
                  f"in {sorted(set(off))[:6]}")  # fmt: skip
    print(f"{'ok  ' if ok else 'FAIL'} {checked:,} letters and digits on {len(pages)} pages "
          f"against the drawn lines")  # fmt: skip
    for doc in sorted({p["doc"] for p in pages})[:2]:
        for p in (p for p in pages if p["doc"] == doc):
            furniture = " | ".join(w["text"] for w in p["words"] if w["furniture"])
            print(f"  {doc} p{p['page']} furniture: {furniture[:150]}")
    test = [p for p in pages if p["split"] == "test"]
    bills = {p["cluster"] for p in test}
    dev = {p["cluster"] for p in pages if p["split"] == "dev"}
    strict = {p["cluster"] for p in test if p["strict"]}
    sized = len(bills) == N_BILLS and len(dev) == N_DEV_BILLS and not bills & dev
    ok &= sized
    print(f"{'ok  ' if sized else 'FAIL'} G2-test: {len(bills)} bills, {len(test)} pages, "
          f"{len(strict)} bills in the strict subset (cutoff {cutoff()}); G2-dev: {len(dev)} "
          f"other bills, {len(pages) - len(test)} pages")  # fmt: skip
    return ok


def _build(args) -> None:
    pages, report = build()
    write_jsonl(GT / "G2.jsonl", pages)
    write_jsonl(GT / "G2-coverage.jsonl", report)
    blank = sum(r["coverage"] is None for r in report)
    dropped = sum(r["coverage"] is not None and r["coverage"] < MIN_COVERAGE for r in report)
    for split in ("test", "dev"):
        mine = [p for p in pages if p["split"] == split]
        print(f"G2-{split}: {len({p['cluster'] for p in mine})} bills, {len(mine)} pages")
    print(f"of {len(report)} pages tried, {dropped} fell under {MIN_COVERAGE:.0%} coverage and "
          f"{blank} have no body words")  # fmt: skip


def register(sub) -> None:
    p = sub.add_parser("g2", help="G2 ground truth and checks")
    cmd = p.add_subparsers(required=True, metavar="action")
    cmd.add_parser("build", help="align the bills and choose G2's pages").set_defaults(func=_build)
    cmd.add_parser("check", help="stage B1 step 4").set_defaults(
        func=lambda a: sys.exit(0 if check() else 1)
    )
