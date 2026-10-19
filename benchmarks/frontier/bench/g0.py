"""G0, the seeded synthetic redline pages (PREREG.md, Ground truth, "G0 synthetic redlines").

Page i of a split is a function of (seed, i) alone: a contract excerpt from a template grammar,
laid out word by word in one font (Helvetica, Times or Courier, by i % 3) and size (9-13 pt), then
struck, underlined and ruled by a fixed recipe with random counts. A build adds pages until every
quota holds (dev holds 40% of each) and writes one single-page PDF per page, the ground truth
(`data/gt/G0.jsonl`, schema in `common.py`) and the S7 questions (`data/gt/G0-qa.jsonl`).

Word strata: E1-E6 the strike's encoding; S1-S7 its style (on E1); N2 underlined; N2b an
underline at the baseline, Courier only (a declared limitation: P-native reports it); N3 a table
cell between rules; [] a plain live word. The ground truth is the finished page's
`get_text("words")`, matched in order to the planned words.

`g0 heldback` draws the private 128-bit seed with `secrets` into `data/private/` and prints the
sha256 of its decimal string, the value PREREG publishes; `g0 build --heldback` builds pages from
it (or from $G0_HELDBACK_SEED, never the command line) under `data/g0/heldback/`, outside
G0.jsonl. `g0 check` is stage B1 step 2.
"""

import hashlib
import math
import os
import random
import re
import secrets
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pymupdf

from bench.common import DATA, GT, ROOT, page_id, read_jsonl, rel, seed, sha256_file, write_jsonl

W, H, M = 612.0, 792.0, 72.0  # US Letter, 1-inch margins
FONTS = ("helv", "tiro", "cour")  # page i is set in FONTS[i % 3]
BOLD = {"helv": "hebo", "tiro": "tibo", "cour": "cobo"}
SIZES = (9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13)
STRUCK = ("E1", "E2", "E3", "E4", "E5", "E6", "S2", "S3", "S4", "S5", "S6", "S7")
QUOTA = dict.fromkeys(STRUCK, 220) | {"N2+N3": 1500, "N1": 8}  # G0-test's; dev holds 40%
N1_P = 0.18
MIN_STROKE = 6.0  # pt; P-native misses a stroke shorter than this (MIN_STROKE_LEN, declared)
UNDERLINE = 0.12  # font sizes under the baseline: below every body font's mid band
STROKE = "\N{COMBINING LONG STROKE OVERLAY}"
INK = ((0, 0, 0), (0.8, 0.1, 0.1), (0.1, 0.25, 0.8))
OUT = DATA / "g0"
HELDBACK_SEED = DATA / "private" / "g0_heldback_seed.txt"
ALNUM = re.compile(r"[^\W_]")
SLOT = re.compile(r"\{(\w+)(?::([^}]+))?\}")
UNMARK = str.maketrans("", "", "\x01\x02\x03\x04")
STRATA = {s: ["E1", s] if s[0] == "S" else [s] for s in (*STRUCK, "S1")}
STRATA |= {"N2": ["N2"], "N2b": ["N2", "N2b"]}
WORD_KEYS = ("text", "bbox", "struck", "masked", "furniture", "line", "strata")  # common.py
PAGE_KEYS = ("page_id", "set", "split", "doc", "page", "cluster", "pdf", "pdf_sha256", "words")

# Per struck page, in placement order: (fewest units, most, fewest words a unit, most). S7 comes
# with the text instead, each correctable value struck at the page's rate; N1 pages draw N2 only.
RECIPE = {"S6": (1, 2, 3, 8), "S5": (5, 9, 1, 1), "E5": (3, 6, 1, 2), "S1": (0, 2, 1, 3)}
RECIPE |= dict.fromkeys(("S2", "S3", "S4", "E2", "E3", "E4", "E6"), (2, 5, 1, 3))
RECIPE["N2"] = (4, 8, 2, 6)

DEALS = (  # title, party A, party B, subject
    ("Master Supply Agreement", "Supplier", "Customer", "Products"),
    ("Software License Agreement", "Licensor", "Licensee", "Software"),
    ("Professional Services Agreement", "Provider", "Client", "Deliverables"),
    ("Equipment Lease Agreement", "Lessor", "Lessee", "Equipment"),
    ("Distribution Agreement", "Manufacturer", "Distributor", "Products"),
    ("Asset Purchase Agreement", "Seller", "Buyer", "Assets"),
)
NAMES = "Harbor Granite Bluewater Northfield Cedar Summit Ironwood Lakeshore Meridian".split()
TRADES = "Logistics Analytics Holdings Systems Industries Capital Energy Foods Robotics".split()
ENTITIES = (("LLC", "limited liability company"), ("Inc.", "corporation"), ("LP", "partnership"))
COUNTIES = {"Delaware": "New Castle", "New York": "New York", "Illinois": "Cook"}  # by state
COUNTIES |= {"Massachusetts": "Suffolk", "Texas": "Travis", "Colorado": "Denver"}
MONTHS = "January February March April May June July August September October November December"
COUNTS = {"days": (5, 10, 15, 20, 30, 45, 60, 90, 120), "months": (6, 12, 18, 24, 36)}
COUNTS["years"] = (2, 3, 4, 5, 7)
PCT = (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 8, 10, 12, 15)
USD = (2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 40, 50, 75, 100, 150, 250, 500, 1000)  # thousands
TAGS = ("Execution Version", "Draft for Discussion", "Confidential", "Redline Draft")

# One clause a paragraph, "Heading: text", its later lines indented. Slots: {A}, {B}, {X}, {state}
# and {county} come from the deal; {kind} is a plain value; {kind:attr} a value S7 may correct,
# asked about as "Under Section n, what is the <attr>?".
PREAMBLE = """This {title} (this "Agreement") is entered into as of {date} by and between {pa}, a
{pa_is} ("{A}"), and {pb}, a {pb_is} ("{B}"). In consideration of the mutual covenants set out
below, the parties agree as follows:"""
CLAUSES = """
Term: This Agreement begins on {date:commencement date} and continues for an initial term of
  {months:initial term}. It then renews automatically for successive terms of {months:renewal term}
  each, unless either party gives written notice of non-renewal at least
  {days:non-renewal notice period} before the end of the then-current term.
Fees: {B} shall pay {A} a monthly fee of {usd:monthly fee}, invoiced in arrears on the first
  business day of each month. Each invoice is payable within {days:payment period} after {B}
  receives it, by wire transfer to the account that {A} designates in writing.
Late Payment: Any amount not paid when due bears a late fee of {pct:late fee} per month from the
  due date until paid. If an undisputed amount remains unpaid for more than
  {days:suspension period} after written notice, {A} may suspend performance until it is paid.
Price Adjustments: {A} may increase its prices once in each contract year by no more than
  {pct:annual price increase cap}, on at least {days:price change notice period} prior written
  notice to {B}. An increase applies only to orders placed after its effective date.
Security Deposit: On signing, {B} shall deliver a security deposit of {usd:security deposit}. {A}
  shall hold the deposit in a segregated account and return it, less any amounts properly applied
  against unpaid obligations, within {days:deposit return period} after the end of the Term.
Limitation of Liability: Except for a breach of confidentiality or of the license granted, the
  aggregate liability of either party under this Agreement shall not exceed {usd:liability cap}.
  Neither party is liable for indirect, incidental, special or consequential damages, even if
  advised of their possibility.
Termination for Cause: Either party may terminate this Agreement by written notice if the other
  party materially breaches it and fails to cure the breach within {days:cure period} after
  receiving notice describing it in reasonable detail. Termination does not relieve {B} of its
  obligation to pay fees accrued before the termination date.
Confidentiality: Each party shall hold the Confidential Information of the other party in strict
  confidence, use it only to perform this Agreement, and disclose it only to employees and
  advisers who need to know it. These obligations survive for
  {years:confidentiality survival period} after this Agreement ends.
Notices: Every notice under this Agreement shall be in writing and delivered by hand, by
  recognized courier or by certified mail to the address of the receiving party set out above. A
  notice is effective on receipt, or {days:deemed receipt period} after posting by certified mail.
Governing Law: This Agreement is governed by the laws of the State of {state}, without regard to
  its conflict of laws rules. Each party submits to the exclusive jurisdiction of the state and
  federal courts located in {county} County, {state}, and waives any objection to venue there.
Warranty: {A} warrants that the {X} will conform in all material respects to the agreed
  specifications for {months:warranty period} after delivery. As its sole obligation for a breach
  of this warranty, {A} shall repair or replace the nonconforming {X} within
  {days:warranty repair period} after notice.
Delivery: {A} shall deliver the {X} to the facility that {B} designates no later than
  {date:delivery deadline}. Title and risk of loss pass to {B} on delivery. {B} may reject any
  shipment that arrives more than {days:late delivery allowance} after its scheduled date.
Early Payment Discount: Where {B} pays an invoice in full within ten days after its date, {B} may
  deduct a discount of {pct:early payment discount} from the invoiced amount. The discount does not
  apply to taxes, shipping charges or amounts in dispute.
Audit Rights: {A} shall keep complete and accurate records relating to this Agreement for
  {years:record retention period}. {B} may audit those records once in any calendar year on
  {days:audit notice period} prior notice. If an audit reveals an overcharge of more than
  {pct:audit cost threshold}, {A} shall bear the cost of the audit.
Force Majeure: Neither party is liable for a failure or delay in performance caused by events
  beyond its reasonable control, including fire, flood, labor disputes, epidemics and acts of
  government. If such an event continues for more than {days:force majeure period}, either party
  may terminate this Agreement on written notice.
"""
CLAUSES = [tuple(" ".join(c.split()).split(": ", 1)) for c in re.split(r"\n(?! )", CLAUSES.strip())]
# Tables: caption, header, the first column, then each other column's choices. No word or value of
# theirs is a clause's (no percentage or amount from PCT or USD, no days, months, years or dates),
# so a table never answers a QA question (`_conflicts` checks).
TABLES = (
    (
        "Service Levels",
        "Measure|Target|Service Credit",
        "System availability|Order accuracy|Invoice accuracy|First-call resolution|Help desk",
        ("99.9%|99.5%|99%|98%|95%", "$250|$500|$1,000|$1,500"),
    ),
    (
        "Support Hours",
        "Tier|Hours|Response",
        "Standard support|Premium support|Critical incidents|Weekend coverage|Holiday coverage",
        ("8:00-18:00|9:00-17:00|7:00-19:00|24/7", "1 hour|2 hours|4 hours|1 business day"),
    ),
    (
        "Key Personnel",
        "Role|Name|Location",
        "Account manager|Project lead|Technical contact|Escalation contact|Billing contact",
        ("J. Alvarez|M. Chen|R. Okafor|S. Patel|D. Nowak", "Boston|Chicago|Austin|Wilmington"),
    ),
    (
        "Equipment",
        "Item|Model|Quantity",
        "Rack server|Storage array|Network switch|Firewall appliance|Backup unit|Label printer",
        ("RS-210|SA-40|NS-24|FW-8|BU-12", "2|4|6|8|12|20"),
    ),
)
GENERIC = {"notice", "period", "date", "term"}  # attribute words too common to name a concept


@cache
def font(name: str) -> pymupdf.Font:
    """A base-14 font by PyMuPDF name, or "noto": Noto Serif, which has U+0336."""
    return pymupdf.Font(script=0) if name == "noto" else pymupdf.Font(name)


@dataclass
class Tok:
    """A planned word: its text and font, where it sits, and its ground truth."""

    text: str
    font: str
    size: float
    strata: list[str] = field(default_factory=list)
    struck: list[list[int]] = field(default_factory=list)
    furniture: bool = False
    lock: int = 0  # 0 free; 1 live beside a mark; 2 taken
    x: float = 0.0
    y: float = 0.0  # baseline
    w: float = 0.0
    line: int = -1  # body line, numbered so that a paragraph skips one; -1 off the body
    k: int = -1  # index among the body words


def _value(rng: random.Random, kind: str) -> tuple[str, str]:
    """A clause's filler as (its number, its unit): ("5%", ""), ("$12,500", ""), ("30", " days")."""
    if kind == "pct":
        return f"{rng.choice(PCT)}%", ""
    if kind == "usd":
        return f"${round(rng.choice(USD) * 1000):,}", ""
    if kind == "date":
        return f"{rng.choice(MONTHS.split())} {rng.randint(1, 28)}, {rng.randint(2027, 2030)}", ""
    return str(rng.choice(COUNTS[kind])), f" {kind}"


def _fill(rng, text: str, ctx: dict, p7: float, section: int, qa: list) -> str:
    """`text` with its slots filled. A correctable value is struck at rate p7 (S7), its correction
    beside it, marked \\x01old\\x02 \\x03new unit\\x04, and its question goes to `qa`."""

    def sub(m: re.Match) -> str:
        key, attr = m.groups()
        if key in ctx:
            return ctx[key]
        old, unit = _value(rng, key)
        if not attr or rng.random() >= p7:
            return old + unit
        new = old
        while new == old:
            new = _value(rng, key)[0]
        qa.append((f"Under Section {section}, what is the {attr}?", new + unit, old + unit))
        return f"\x01{old}\x02 \x03{new}{unit}\x04"

    return SLOT.sub(sub, text)


def _tokens(text: str, fname: str, size: float, strata=(), lock=0) -> list[Tok]:
    """The words of `text`; `_fill`'s markers make an S7 number struck, and it and its
    correction taken."""
    toks, mode = [], 0
    for word in text.split():
        mode = {"\x01": 1, "\x03": 2}.get(word[0], mode)
        t = Tok(word.translate(UNMARK), fname, size, list(strata), lock=2 if mode else lock)
        if mode == 1:
            t.strata, t.struck = ["E1", "S7"], [[0, len(t.text)]]
        mode = 0 if "\x02" in word or "\x04" in word else mode
        toks.append(t)
    return toks


def _flow(toks: list[Tok], x: float, y: float, lead: float = 0.0, line: int = -1):
    """Lay words out from (x, y), wrapping at the right margin when `lead` is given; returns the
    last baseline, the last line and the right edge."""
    for t in toks:
        f = font(t.font)
        t.w = f.text_length(t.text, t.size)
        if lead and x > M and x + t.w > W - M:
            x, y, line = M, y + lead, line + 1
        t.x, t.y, t.line = x, y, line
        x += t.w + f.text_length(" ", t.size)
    return y, line, toks[-1].x + toks[-1].w


def _furniture(text: str, fname: str, size: float, y: float, align: float) -> list[Tok]:
    """A running head or page number; `align` 0 is the left margin, 0.5 centred, 1 the right."""
    toks = _tokens(text, fname, size, lock=2)
    width = _flow(toks, 0, y)[2]
    for t in toks:
        t.x, t.furniture = t.x + M + align * (W - 2 * M - width), True
    return toks


def _table(rng, fname: str, fs: float, y: float):
    """A captioned table, ruled close above and below every row (N3): words, rules, bottom."""
    caption, header, firsts, choices = rng.choice(TABLES)
    rows = [header.split("|")] + [
        [first, *(rng.choice(c.split("|")) for c in choices)]
        for first in rng.sample(firsts.split("|"), rng.randint(3, 5))
    ]
    toks = _tokens(f"Table {rng.randint(1, 4)}. {caption}", BOLD[fname], fs, lock=2)
    _flow(toks, M, y)
    cells = [
        [_tokens(c, BOLD[fname] if r == 0 else fname, fs, ["N3"], 2) for c in row]
        for r, row in enumerate(rows)
    ]
    pad, xs = 0.6 * fs, [M]
    for c in range(3):
        xs.append(xs[-1] + max(_flow(row[c], 0, 0)[2] for row in cells) + 2 * pad)
    # A rule 0.88-0.98 font sizes above a baseline or 0.28-0.40 below it runs just outside every
    # font's mid band (0.79 above to 0.09 below the baseline, bold Courier the lowest).
    above, below = rng.uniform(0.88, 0.98) * fs, rng.uniform(0.28, 0.4) * fs
    ys = [y + 0.7 * fs]
    for row in cells:
        for x, cell in zip(xs, row):
            _flow(cell, x + pad, ys[-1] + above)
            toks += cell
        ys.append(ys[-1] + above + below)
    width = rng.uniform(0.4, 0.9)
    rules = [((M, ry), (xs[-1], ry), width) for ry in ys]
    if rng.random() < 0.5:
        rules += [((rx, ys[0]), (rx, ys[-1]), width) for rx in xs]
    return toks, rules, ys[-1]


def _layout(rng, fname: str, fs: float, p7: float):
    """Plan the page: its words in writing order, the body words, table rules and S7 questions."""
    title, a, b, x = rng.choice(DEALS)
    state = rng.choice(sorted(COUNTIES))
    ctx = {"title": title, "A": a, "B": b, "X": x, "state": state, "county": COUNTIES[state]}
    for p in ("pa", "pb"):
        suffix, kind = rng.choice(ENTITIES)
        ctx[p] = f"{rng.choice(NAMES)} {rng.choice(TRADES)} {suffix}"
        ctx[p + "_is"] = f"{rng.choice(sorted(COUNTIES))} {kind}"
    small, lead = 0.8 * fs, rng.uniform(1.18, 1.3) * fs
    toks = _furniture(title.upper(), fname, small, M / 2 + 8, 0)
    toks += _furniture(rng.choice(TAGS), fname, small, M / 2 + 8, 1)
    foot = rng.choice(("Page {}", "{}", "- {} -")).format(rng.randint(2, 40))
    n, target = rng.randint(1, 12), rng.randint(300, 340)
    paras = ([(None, PREAMBLE)] if n == 1 else []) + rng.sample(CLAUSES, len(CLAUSES))
    table_at = rng.randint(1, 3) if rng.random() < 0.6 else -1
    body, rules, qa, y, line = [], [], [], M + fs, 0
    for j, (heading, text) in enumerate(paras):
        if j == table_at:
            cells, lines, bottom = _table(rng, fname, fs, y)
            if bottom <= H - M:
                toks, rules, y = toks + cells, lines, bottom + lead + 0.5 * fs
        if sum(not t.furniture for t in toks) >= target:
            break
        asked = []
        head = _tokens(f"{n}. {heading}.", BOLD[fname], fs, lock=2) if heading else []
        para = head + _tokens(_fill(rng, text, ctx, p7, n, asked), fname, fs)
        last, end, _ = _flow(para, M, y, lead, line)
        if last > H - M - 0.3 * fs:
            break
        toks, body, qa = toks + para, body + para, qa + asked
        y, line, n = last + lead + 0.5 * fs, end + 2, n + bool(heading)
    return toks + _furniture(foot, fname, small, H - M / 2, 0.5), body, rules, qa


def _parts(run: list[Tok]) -> list[list[Tok]]:
    """A run's words split by line; each line's part takes its own mark."""
    parts = [[run[0]]]
    for t in run[1:]:
        if t.line == parts[-1][-1].line:
            parts[-1].append(t)
        else:
            parts.append([t])
    return parts


def _partial(rng, t: Tok) -> tuple[int, int] | None:
    """S5: a prefix or suffix of 2+ letters leaving 2+, as (c0, c1)."""
    if not t.text.isalpha() or len(t.text) < 5:
        return None
    k = rng.randint(2, len(t.text) - 2)
    return rng.choice(((0, k), (k, len(t.text))))


def _run(rng, body: list[Tok], s: str, n: int):
    """A random run of n free body words for stratum s, with its S5 span: on one line (S6: over
    one line break in a paragraph). Marks take the words' natural width, however narrow."""
    for _ in range(40 if len(body) >= n else 0):
        j = rng.randrange(len(body) - n + 1)
        run = body[j : j + n]
        parts = _parts(run)
        lines = len(parts) == 1 + (s == "S6") and parts[-1][0].line - run[0].line == len(parts) - 1
        if not lines or not all(t.lock == 0 and ALNUM.search(t.text) for t in run):
            continue
        if s == "S5":
            if span := _partial(rng, run[0]):
                return run, span
        elif s != "E5" or run[-1].x + run[-1].w - run[0].x <= 6 * run[0].size:  # rise stays on it
            return run, None
    return None, None


def _guard(body: list[Tok], run: list[Tok]) -> None:
    """Keep the words either side of a mark live, so marks never touch."""
    for k in (run[0].k - 1, run[-1].k + 1):
        if 0 <= k < len(body):
            body[k].lock = max(body[k].lock, 1)


def _select(rng, body: list[Tok], fname: str, n1: bool) -> list:
    """The page's marks as (stratum, words, S5 span): the S7 numbers set with the text, then
    RECIPE's draws."""
    units = []
    for k, t in enumerate(body):
        t.k = k
    for t in body:
        if t.lock == 2:  # a heading, an S7 number or its correction
            _guard(body, [t])
        if "S7" in t.strata and units and units[-1][1][-1].k == t.k - 1:
            units[-1][1].append(t)
        elif "S7" in t.strata:
            units.append(("S7", [t], None))
    for s, (u0, u1, n0, n_hi) in RECIPE.items():
        for _ in range(0 if n1 and s != "N2" else rng.randint(u0, u1)):
            run, span = _run(rng, body, s, rng.randint(n0, n_hi))
            if run is None:
                continue
            # on Courier, half the underlines sit at the baseline (N2b, a declared limitation)
            label = "N2b" if s == "N2" and fname == "cour" and rng.random() < 0.5 else s
            for t in run:
                t.lock, t.strata = 2, list(STRATA[label])
                t.text = "".join(c + STROKE for c in t.text) if s == "E6" else t.text
                t.struck = [] if s == "N2" else [list(span or (0, len(t.text)))]
            _guard(body, run)
            units.append((label, run, span))
    return units


def _write(page: pymupdf.Page, toks: list[Tok]) -> None:
    """Each word with a trailing space, so the text layer has real spaces; an E6 word letter by
    letter, each letter followed by U+0336."""
    tw = pymupdf.TextWriter(page.rect)
    for t in toks:
        f = font(t.font)
        if "E6" not in t.strata:
            tw.append((t.x, t.y), t.text + " ", font=f, fontsize=t.size)
            continue
        pos = pymupdf.Point(t.x, t.y)
        for ch in t.text[::2]:
            _, pos = tw.append(pos, ch, font=f, fontsize=t.size)
            # Noto's U+0336 has no advance and is centred on its origin: centre it on the letter
            mark = (pos.x - f.text_length(ch, t.size) / 2, pos.y)
            tw.append(mark, STROKE, font=font("noto"), fontsize=t.size)
        tw.append(pos, " ", font=f, fontsize=t.size)
    tw.write_text(page)


def _draw(page: pymupdf.Page, rng, s: str, run: list[Tok], span) -> None:
    """One mark (PREREG's encodings and styles) per line the unit touches."""
    f = font(run[0].font)
    xmid = f.glyph_bbox(ord("x")).y1 / 2  # the middle of the x-height, in font sizes
    for part in _parts(run):
        t, fs = part[0], part[0].size
        x0, x1 = t.x, part[-1].x + part[-1].w
        if span:  # S5: the struck characters' advances
            x0, x1 = (t.x + f.text_length(t.text[:c], fs) for c in span)
        y, ink = t.y - fs * (xmid + rng.uniform(-0.03, 0.03)), rng.choice(INK)
        if s in ("S1", "S3", "S4", "S5", "S6", "S7"):
            width = rng.uniform(2.0, 3.0) if s == "S4" else rng.uniform(0.5, 1.0)
            color = rng.choice(INK[1:]) if s == "S3" else INK[0]
            page.draw_line((x0, y), (x1, y), color=color, width=width)
        elif s == "S2":
            for dy in (-0.09 * fs, 0.09 * fs):
                page.draw_line((x0, y + dy), (x1, y + dy), width=rng.uniform(0.4, 0.7))
        elif s == "E2":
            h = rng.uniform(0.8, 2.0)
            page.draw_rect(pymupdf.Rect(x0, y - h / 2, x1, y + h / 2), color=None, fill=ink)
        elif s in ("E3", "E4"):
            if s == "E3":
                box = pymupdf.Rect(x0, t.y - f.ascender * fs, x1, t.y - f.descender * fs)
                annot = page.add_strikeout_annot(box)
            else:  # a pen stroke: overshooting a little, tilted under P-native's 1.5 pt
                over, tilt = rng.uniform(0, 0.8), rng.uniform(-0.4, 0.4)
                annot = page.add_ink_annot([[(x0 - over, y + tilt), (x1 + over, y - tilt)]])
                annot.set_border(width=rng.uniform(0.8, 1.5))
            annot.set_colors(stroke=ink)
            annot.update()
        elif s == "E5":  # leans 3 degrees or more and rises 1.8 pt or more, past both limits
            rise = (max(1.8, (x1 - x0) * math.tan(math.radians(3))) + rng.uniform(0, 0.5)) / 2
            rise *= rng.choice((1, -1))
            page.draw_line((x0, y + rise), (x1, y - rise), color=ink, width=rng.uniform(0.6, 1.0))
        elif s in ("N2", "N2b"):
            under = t.y + (0 if s == "N2b" else UNDERLINE * fs)
            page.draw_line((x0, under), (x1, under), width=rng.uniform(0.5, 0.8))


def make_page(seed_: int, idx: int):
    """Page `idx` drawn from `seed_` alone: (PDF bytes, word records, page extras, questions)."""
    rng = random.Random(f"{seed_}:{idx}")
    fname, fs, n1 = FONTS[idx % 3], rng.choice(SIZES), rng.random() < N1_P
    toks, body, rules, qa = _layout(rng, fname, fs, 0.0 if n1 else rng.uniform(0.35, 0.7))
    units = _select(rng, body, fname, n1)
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    _write(page, toks)
    for p0, p1, width in rules:
        page.draw_line(p0, p1, width=width)
    for unit in units:
        _draw(page, rng, *unit)
    doc.subset_fonts()
    data = doc.tobytes(garbage=3, deflate=True, no_new_id=True)
    found, words = pymupdf.open(stream=data, filetype="pdf")[0].get_text("words"), []
    for w, t in zip(found, toks, strict=True):
        if w[4] != t.text or abs(w[0] - t.x) > 0.05 or abs(w[2] - t.x - t.w) > 0.05:
            raise RuntimeError(f"G0 page {idx}: extracted {w[:5]} where {t.text!r} was planned")
        box = [round(v, 5) for v in (w[0] / W, w[1] / H, w[2] / W, w[3] / H)]
        rec = (t.text, box, t.struck, False, t.furniture, [w[5], w[6]], t.strata)
        words.append(dict(zip(WORD_KEYS, rec)))
    return data, words, {"font": font(fname).name, "fontsize": fs, "n1": n1}, qa


def page_record(split: str, idx: int, data: bytes, words: list, extras: dict, path: Path) -> dict:
    """A ground-truth page (schema in `common.py`) for page `idx` of a split, saved at `path`."""
    doc, pdf = f"{split}-{idx:03d}", path.resolve()
    pdf = rel(pdf) if pdf.is_relative_to(ROOT) else pdf.as_posix()
    pid, sha = page_id("G0", doc, 0), hashlib.sha256(data).hexdigest()
    return dict(zip(PAGE_KEYS, (pid, "G0", split, doc, 0, pid, pdf, sha, words))) | extras


def _tally(page: dict) -> Counter:
    """Struck sub-tokens per stratum, live sub-tokens under N2 or beside N3, and N1 pages."""
    c = Counter(N1=int(page["n1"]))
    for w in page["words"]:
        if w["struck"] and ALNUM.search(w["text"][slice(*w["struck"][0])]):
            c.update(s for s in w["strata"] if s in STRUCK)
        elif not w["struck"] and {"N2", "N3"} & set(w["strata"]):
            c["N2+N3"] += 1
    return c


def _quota(split: str) -> dict[str, int]:
    return {k: math.ceil(q * 0.4) if split == "dev" else q for k, q in QUOTA.items()}


def _clear(out: Path) -> None:
    """Empty an output folder so nothing stale survives, refusing one holding what G0 never
    writes (a mistyped `--out`)."""
    out.mkdir(parents=True, exist_ok=True)
    old = list(out.iterdir())
    if any(f.is_dir() or f.suffix not in (".pdf", ".jsonl") for f in old):
        sys.exit(f"{out} holds files G0 does not write; refusing to clear it")
    for f in old:
        f.unlink()


def build_split(split: str, seed_: int, out: Path) -> tuple[list[dict], list[dict]]:
    """Pages 0, 1, ... of a split until every quota holds, written to a cleared `out`; (pages,
    questions)."""
    _clear(out)
    pages, qa, got, need = [], [], Counter(), _quota(split)
    while any(got[k] < q for k, q in need.items()):
        idx = len(pages)
        data, words, extras, asked = make_page(seed_, idx)
        path = out / f"{split}-{idx:03d}.pdf"
        path.write_bytes(data)
        pages.append(page_record(split, idx, data, words, extras, path))
        pid = pages[-1]["page_id"]
        qa += [
            {"qa_id": f"{pid}.q{j}", "page_id": pid, "question": q, "live": live, "struck": struck}
            for j, (q, live, struck) in enumerate(asked)
        ]
        got += _tally(pages[-1])
    return pages, qa


def _summary(split: str, pages: list[dict]) -> bool:
    """Print a split's tallies against its quotas; True when all hold and every struck stratum
    appears in all three fonts."""
    got, need = sum(map(_tally, pages), Counter()), _quota(split)
    seen = {(s, p["font"]) for p in pages for w in p["words"] if w["struck"] for s in w["strata"]}
    short = [s for s in STRUCK if sum((s, font(f).name) in seen for f in FONTS) < 3]
    tallies = " ".join(f"{k} {got[k]}/{q}" for k, q in need.items())
    print(
        f"G0-{split}: {len(pages)} pages | {tallies}" + (f" | fonts short: {short}" * bool(short))
    )
    return not short and all(got[k] >= q for k, q in need.items())


def _heldback_seed() -> int:
    """The held-back seed, from $G0_HELDBACK_SEED or the private file `g0 heldback` wrote; never
    from the command line, which shell history keeps."""
    value = os.environ.get("G0_HELDBACK_SEED", "").strip()
    if not value and HELDBACK_SEED.exists():
        value = HELDBACK_SEED.read_text(encoding="utf-8").strip()
    if not value:
        sys.exit("no held-back seed: run `g0 heldback`, or set G0_HELDBACK_SEED")
    return int(value)


def cmd_build(args) -> None:
    if args.heldback:  # never in G0.jsonl
        out = args.out or OUT / "heldback"
        pages, qa = build_split("heldback", _heldback_seed(), out)
        write_jsonl(out / "G0-heldback.jsonl", pages)
        write_jsonl(out / "G0-heldback-qa.jsonl", qa)
        _summary("heldback", pages)
        return
    if args.out:
        sys.exit("--out goes with --heldback; dev and test always build into data/g0/")
    splits = ("dev", "test") if args.split == "both" else (args.split,)
    pages = [p for p in read_jsonl(GT / "G0.jsonl") if p["split"] not in splits]
    kept = {p["page_id"] for p in pages}
    qa = [q for q in read_jsonl(GT / "G0-qa.jsonl") if q["page_id"] in kept]
    for split in splits:
        new, asked = build_split(split, seed(f"g0_{split}"), OUT / split)
        pages, qa = pages + new, qa + asked
        _summary(split, new)
    write_jsonl(GT / "G0.jsonl", sorted(pages, key=lambda p: p["split"] == "test"))
    write_jsonl(GT / "G0-qa.jsonl", sorted(qa, key=lambda q: q["page_id"].startswith("G0.test")))


LABELS = ("E1", "E2", "E3", "E4", "E5", "E6", "S1", *STRUCK[6:], "N1", "N2", "N2b", "N3", "live")
COLUMNS = ("words", "found", "miss", "miss*", "false", "false*", "span")


def _match(words: list[dict], rec: dict) -> int | None:
    """The ground-truth word a P-native record reports: the same text, box centres within 0.1%."""
    b = rec["bbox_frac"]
    for j, w in enumerate(words):
        a = w["bbox"]
        near = abs(a[0] + a[2] - b[0] - b[2]) + abs(a[1] + a[3] - b[1] - b[3]) < 2e-3
        if near and w["text"] == rec["text"]:
            return j
    return None


def _pnative(pages: list[dict]) -> int:
    """P-native as frozen, per stratum; returns the undeclared misses, false reports and wrong
    spans. Starred columns are the declared limitations: E5 and E6 missed; a struck word, or
    an S5 word's struck part, narrower than 6 pt missed (its stroke is under MIN_STROKE_LEN;
    this covers the 3 pt rule); and Courier's baseline underlines (N2b) reported."""
    from pdf_strikethrough import detect_pdf  # only the check needs the package under test

    rows, bad, fonts = {k: Counter() for k in LABELS}, 0, {font(f).name: f for f in FONTS}
    for p in pages:
        found = {}
        for r in detect_pdf(str(ROOT / p["pdf"]), method="vector")["words"]:
            if (j := _match(p["words"], r)) is not None:
                found[j] = r
                continue
            print(f"  {p['page_id']}: unmatched report {r['text']!r}")
            bad += 1
        for j, w in enumerate(p["words"]):
            r, labels = found.get(j), set(w["strata"]) or {"live"}
            if w["struck"]:
                c0, c1 = w["struck"][0]
                stroke = font(fonts[p["font"]]).text_length(w["text"][c0:c1], p["fontsize"])
                declared = labels & {"E5", "E6"} or stroke < MIN_STROKE
                hit = r and list(r["char_span"]) == w["struck"][0]
                outcome = "found" if hit else "span" if r else "miss*" if declared else "miss"
            else:
                outcome = r and ("false*" if "N2b" in labels else "false")
            if outcome in ("miss", "span", "false"):
                bad += 1
                print(f"  {p['page_id']}: {outcome} {w['text']!r} {w['strata']}")
            for k in labels | ({"N1"} if p["n1"] else set()):
                rows[k].update(["words", outcome] if outcome else ["words"])
    print(f"{'stratum':8}" + "".join(f"{c:>8}" for c in COLUMNS))
    for k, row in rows.items():
        print(f"{k:8}" + "".join(f"{row[c]:>8}" for c in COLUMNS))
    return bad


def _concept(text: str) -> set[str]:
    """A phrase's content words, lowercased and singular, less the generic ones."""
    words = re.findall(r"[a-z]+", text.lower())
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words} - GENERIC


def _conflicts(pages: list[dict], qa: list[dict]) -> list[str]:
    """QA items their page could answer elsewhere: a question asked twice, or a table cell sharing
    a content word with the question's attribute or holding its live or struck value. (No other
    clause has the attribute: each clause appears once a page, and attributes are unique.)"""
    cells: dict[str, dict] = {}
    for p in pages:
        for w in p["words"]:
            if "N3" in w["strata"]:
                cells.setdefault(p["page_id"], {}).setdefault(tuple(w["line"]), []).append(
                    w["text"]
                )
    seen, bad = set(), []
    for q in qa:
        if (q["page_id"], q["question"]) in seen:
            bad.append(f"{q['qa_id']}: asked twice")
        seen.add((q["page_id"], q["question"]))
        attr = _concept(q["question"].partition("what is the ")[2])
        for cell in cells.get(q["page_id"], {}).values():
            text = f" {' '.join(cell)} "
            if attr & _concept(text) or any(f" {v} " in text for v in (q["live"], q["struck"])):
                bad.append(f"{q['qa_id']}: {text.strip()!r}")
    return bad


def cmd_check(args) -> None:
    """Stage B1 step 2: a byte-identical rebuild, the quotas, P-native on G0-dev, and no QA item
    answerable from elsewhere on its page."""
    pages, qa, ok = read_jsonl(GT / "G0.jsonl"), read_jsonl(GT / "G0-qa.jsonl"), True
    with tempfile.TemporaryDirectory() as tmp:
        for split in ("dev", "test"):
            disk = [p for p in pages if p["split"] == split]
            fresh, asked = build_split(split, seed(f"g0_{split}"), Path(tmp) / split)
            same = len(fresh) == len(disk) and all(
                a["pdf_sha256"] == b["pdf_sha256"] == sha256_file(ROOT / b["pdf"])
                and a["words"] == b["words"]
                for a, b in zip(fresh, disk)
            )
            same &= asked == [q for q in qa if q["page_id"].startswith(f"G0.{split}-")]
            print(f"rebuild G0-{split}: {len(fresh)} pages, {'identical' if same else 'DIFFERS'}")
            ok &= same
    for split in ("dev", "test"):
        ok &= _summary(split, [p for p in pages if p["split"] == split])
    conflicts = _conflicts(pages, qa)
    print(f"QA: {len(qa)} items, {len(conflicts)} answerable from elsewhere on their page")
    for c in conflicts:
        print(f"  {c}")
    ok &= not conflicts
    bad = _pnative([p for p in pages if p["split"] == "dev"])
    print(f"P-native on G0-dev: {bad} undeclared misses, false reports or wrong spans")
    if bad or not ok:
        sys.exit(1)


def cmd_heldback(args) -> None:
    """Draw the held-back seed; print the sha256 of its decimal string, never the seed."""
    if HELDBACK_SEED.exists():
        sys.exit(f"{HELDBACK_SEED} exists; refusing to overwrite it")
    value = str(secrets.randbits(128))
    HELDBACK_SEED.parent.mkdir(parents=True, exist_ok=True)
    HELDBACK_SEED.write_text(value + "\n", encoding="utf-8")
    print(hashlib.sha256(value.encode()).hexdigest())


def register(sub) -> None:
    p = sub.add_parser("g0", help="G0 synthetic redlines: build, check, heldback")
    g0 = p.add_subparsers(required=True, metavar="action")
    build = g0.add_parser("build", help="build G0 pages and their ground truth")
    build.add_argument("--split", choices=("dev", "test", "both"), default="both")
    build.add_argument("--heldback", action="store_true", help="build the held-back pages instead")
    build.add_argument("--out", type=Path, help="with --heldback: where (default data/g0/heldback)")
    build.set_defaults(func=cmd_build)
    g0.add_parser("check", help="stage B1 step 2's checks").set_defaults(func=cmd_check)
    g0.add_parser("heldback", help="draw the private held-back seed").set_defaults(
        func=cmd_heldback
    )
