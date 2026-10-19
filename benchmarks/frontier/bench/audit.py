"""The hand audit (PREREG.md, The audit): items to a self-contained HTML page per annotator, labels
back in as CSV, Cohen's κ; also G1's error-rate sample and relabel, and G2's span check.

    python -m bench audit build pilot|test    # runs/<stage>/audit/ (the pilot's: the author's)
    python -m bench audit g1                  # G1's 400-word error-rate sample -> runs/g1-audit/
    python -m bench audit g2                  # G2's 100-span check -> runs/g2-audit/ (the author's)
    python -m bench audit relabel             # after the test run -> runs/test/g1-relabel/
    python -m bench audit notations <stage>...  # dev outputs at R0 and R1 around every leak
    python -m bench audit ingest <dir> A.csv [B.csv] [--adjudicated FILE]

The author labels everything, each task on one page with no practice items and no κ, the labels
published: before prereg-2, the pilot audit, a triage of dev outputs that hunts scorer bugs before
the scorer is frozen, and G2's span check, which can change G2's labels; after the test run, the
test audit, the G1 sample and the relabel. The G1 sample is labelled from the crop alone, once the
test run's answers are in and before they are scored. Two annotators (`who="AB"`, practice items,
a shared quarter for κ) remain supported, for an addendum that adds them.

Ingesting the relabel writes `data/gt/G1-corrections.jsonl` once every item is decided; then
`g1 build` applies it and `harness score test` rescores every arm.

An item is a disagreement between an arm's output and the ground truth: a leak (struck, output
live), an over-deletion (live, output marked or absent), a missed strike (struck, output absent) or
a hallucinated token. Its page asks, in order: (1) on the crop alone, is the boxed word struck
through; (2) shown the ground-truth label, is it right; (3) shown the output in one neutral
rendering and the scorer's verdict, did the scorer read it correctly. A hallucinated token has no
word to label, so it asks only question 3, beside a crop of where it sits. The arm is never shown.
"""

import base64
import csv
import html
import io
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw

from bench import g1, harness, plans, render
from bench.common import (
    GT, PREREG2, RUNS, config, load_pages, read_jsonl, seed, sha256_file, write_jsonl,
)  # fmt: skip
from bench.score import normalize, score_page
from bench.stats import cohen_kappa, kappa_ci, largest_remainder, wilson

KINDS = ("leak", "over-deletion", "missed", "hallucination")
PILOT_PER_KIND, TOTAL, SHARED, PRACTICE = 5, 160, 0.25, 10
GAP = "▯"  # where the output holds nothing for the word


def _kind(row: dict) -> str | None:
    if row["struck"]:
        return {"live": "leak", "absent": "missed"}.get(row["outcome"])
    return "over-deletion" if row["outcome"] != "live" else None


def _render(tokens, focus: list[int], around: tuple[int, int] | None = None, width: int = 6) -> str:
    """The output near the focus tokens, the same for every arm: marked characters in ⟦ ⟧, the
    focus in bold. With no focus, the tokens either side of the word (`around`) and a gap mark."""
    anchor = focus or [k for k in around or () if k is not None]
    if not anchor:
        return GAP
    lo, hi = max(0, min(anchor) - width), min(len(tokens), max(anchor) + width + 1)
    out = []
    for k in range(lo, hi):
        t, word, run = tokens[k], "", False
        for c, m in zip(t.text, t.marks):
            word += ("⟦" if m and not run else "⟧" if run and not m else "") + c
            run = m
        word += "⟧" if run else ""
        out.append(f"<b>{html.escape(word)}</b>" if k in focus else html.escape(word))
        if not focus and around and k == around[0]:
            out.append(f"<b>{GAP}</b>")
    if not focus and around and around[0] is None:
        out.insert(0, f"<b>{GAP}</b>")
    return " ".join(out)


def _neighbours(tok_word: dict[int, int], word: int) -> tuple[int | None, int | None]:
    """The last output token aligned before `word` and the first one aligned after it."""
    before = [k for k, j in tok_word.items() if j < word]
    after = [k for k, j in tok_word.items() if j > word]
    return (max(before) if before else None), (min(after) if after else None)


def disagreements(stage: str, arms: list[str] | None = None) -> dict[tuple[str, str], list[dict]]:
    """Every disagreement in a stage, by arm and kind. Failed pages are left out: their scores are
    the failure rule's, not a reading of an output. So is G5, which is outside the audits."""
    pages = {p["page_id"]: p for p in load_pages("G0", "G1", "G2")}
    pool: dict[tuple[str, str], list[dict]] = {}
    for o in harness.outputs(stage):
        if o["tokens"] is None or o.get("qa_id") or (arms and o["arm"] not in arms):
            continue
        if o["page_id"] not in pages:  # G5's
            continue
        res = score_page(pages[o["page_id"]], o["tokens"], details=True)
        if res["failed"]:  # an empty answer: nothing to read
            continue
        tok_word = {k: r["word"] for r in res["details"] for k in r["tokens"]}
        base = {"stage": stage, "arm": o["arm"], "page_id": o["page_id"]}
        for r in res["details"]:
            if kind := _kind(r):
                around = None if r["tokens"] else _neighbours(tok_word, r["word"])
                output = _render(o["tokens"], r["tokens"], around)
                pool.setdefault((o["arm"], kind), []).append(
                    {
                        **base,
                        "kind": kind,
                        "words": [r["word"]],
                        "span": r["span"],
                        "outcome": r["outcome"],
                        "output": output,
                    }  # fmt: skip
                )
        for k in res["inserted_tokens"]:
            before = [j for t, j in tok_word.items() if t < k]
            after = [j for t, j in tok_word.items() if t > k]
            near = sorted({max(before)} if before else set()) + sorted(
                {min(after)} if after else set()
            )
            pool.setdefault((o["arm"], "hallucination"), []).append(
                {
                    **base,
                    "kind": "hallucination",
                    "token": k,
                    "words": near,
                    "outcome": "not on the page",
                    "output": _render(o["tokens"], [k]),
                }  # fmt: skip
            )
    return pool


def _primary_arms() -> list[str]:
    """The package's arms and each flagship's R0 C1, R0 C2, R1 C2 and R-best cells (its own
    rung, by the flagship check)."""
    out = ["P-native", "P-scan"]
    for k, m in config("models").items():
        if m["role"] == "flagship":
            best = plans.choice(m["provider"], "flagship")
            image = plans.choice(m["provider"], "image")
            out += [f"{k}:R0:C1", f"{k}:R0:C2", f"{k}:R1:C2", f"{k}:{best}:{image}"]
    return out


def _split(total: int, keys: list, rng: random.Random) -> dict:
    """`total` split evenly over `keys` by largest remainder, ties broken in a seeded order."""
    order = rng.sample(keys, len(keys))
    return dict(zip(order, largest_remainder([1] * len(order), total)))


def sample(stage: str) -> list[dict]:
    """The pilot: up to 5 per kind per pilot arm, seed 20261011. The test: 160 items across the
    primary arms, split evenly over the arms and then each arm's share over the kinds; a cell
    short of its share passes the rest to the arm's other kinds, then to the other arms, in a
    seeded order (seed 20261014)."""
    if stage != "test":
        # P-scan-shipped is P-scan with the OCR setting deviation 2 changed: its disagreements are
        # P-scan's, its garbled lines apart, which this audit has already triaged
        pool = {c: v for c, v in disagreements(stage).items() if c[0] != "P-scan-shipped"}
        rng = random.Random(seed("pilot_audit"))
        cells = [(a, k) for a in sorted({a for a, _ in pool}) for k in KINDS]
        cells = [pool.get(c, []) for c in cells]
        return [x for c in cells for x in rng.sample(c, min(PILOT_PER_KIND, len(c)))]
    arms = _primary_arms()
    pool = disagreements(stage, arms)
    rng = random.Random(seed("test_audit"))
    room = {(a, k): len(pool.get((a, k), [])) for a in arms for k in KINDS}
    share: dict[tuple, int] = {}
    spare = 0
    for arm, n in _split(TOTAL, arms, rng).items():
        mine = {(arm, k): v for k, v in _split(n, list(KINDS), rng).items()}
        extra = sum(max(0, v - room[c]) for c, v in mine.items())
        share |= {c: min(v, room[c]) for c, v in mine.items()}
        spare += _hand_out(extra, rng.sample(list(mine), len(mine)), share, room)
    _hand_out(spare, rng.sample(list(share), len(share)), share, room)
    return [x for c in sorted(share) for x in rng.sample(pool.get(c, []), share[c])]


def _hand_out(n: int, order: list, share: dict, room: dict) -> int:
    """Give n extra items, one at a time round the cells in `order`, to those with room left;
    returns what is left over."""
    while n and any(share[c] < room[c] for c in order):
        for c in order:
            if n and share[c] < room[c]:
                share[c] += 1
                n -= 1
    return n


# --------------------------------------------------------------------------------------- pages


BLUE, ORANGE = (0, 90, 255), (240, 130, 0)


def _turn(page: dict) -> int:
    """How far to turn the page's images, in degrees counter-clockwise, so its text reads upright:
    90 or 270 when most of its words of four letters or more stand taller than wide (a page
    printed sideways: such a word is several times taller than wide in page fractions, an
    upright one a fraction as tall), as their lines run down or up the page; else 0."""
    body = [w for w in page["words"] if len(w["text"]) >= 4 and not w.get("furniture")]
    tall = [w for w in body if w["bbox"][3] - w["bbox"][1] > 1.5 * (w["bbox"][2] - w["bbox"][0])]
    if 2 * len(tall) <= len(body):
        return 0
    down = sum((b["bbox"][1] > a["bbox"][1]) - (b["bbox"][1] < a["bbox"][1])
               for a, b in zip(tall, tall[1:]) if a.get("line") == b.get("line"))  # fmt: skip
    return 90 if down >= 0 else 270


def _png(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, fmt, **({"quality": 80} if fmt == "JPEG" else {}))
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def _boxed(image: Image.Image, boxes: list, colors: list | None, origin=(0.0, 0.0),
           width: int = 3) -> None:  # fmt: skip
    """Draw `boxes` (pixels of the image `origin` was cut from) on the image."""
    draw = ImageDraw.Draw(image)
    for b, color in zip(boxes, colors or [BLUE] * len(boxes)):
        draw.rectangle([b[0] - origin[0] - 4, b[1] - origin[1] - 4, b[2] - origin[0] + 4,
                        b[3] - origin[1] + 4], outline=color, width=width)  # fmt: skip


def _crop(page_id: str, boxes: list, colors: list | None = None, pad: float = 2.0,
          turn: int = 0) -> str:  # fmt: skip
    """A crop of the page around `boxes`, each boxed (blue unless `colors` says), as a data URL:
    15% of the page along the line either side, `pad` line heights across. On a page printed
    sideways (`turn`, `_turn`) the lines run down it, so the axes swap and the crop turns upright."""
    image = Image.open(render.path(page_id, "c2.png")).convert("RGB")
    w, h = image.size
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    if turn:
        lw = max(b[2] - b[0] for b in boxes)
        crop = (max(0, x0 - pad * lw) * w, max(0, y0 - 0.15) * h,
                min(1, x1 + pad * lw) * w, min(1, y1 + 0.15) * h)  # fmt: skip
    else:
        lh = max(b[3] - b[1] for b in boxes)
        crop = (max(0, x0 - 0.15) * w, max(0, y0 - pad * lh) * h,
                min(1, x1 + 0.15) * w, min(1, y1 + pad * lh) * h)  # fmt: skip
    out = image.crop(tuple(round(v) for v in crop))
    _boxed(out, [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in boxes], colors, crop[:2])
    return _png(out.rotate(turn, expand=True) if turn else out)


def _whole(page_id: str, boxes: list, colors: list | None = None, turn: int = 0) -> str:
    """The whole page, upright and scaled to 900 pixels across, `boxes` drawn: where an extra
    word's crop can't show what the output read (a chart, a photo, a heading drawn as a picture)."""
    image = Image.open(render.path(page_id, "c2.png")).convert("RGB")
    if turn:
        image = image.rotate(turn, expand=True)
    w, h = image.size
    small = image.resize((900, round(h * 900 / w)), Image.Resampling.LANCZOS)
    sw, sh = small.size
    _boxed(small, [[b[0] * sw, b[1] * sh, b[2] * sw, b[3] * sh]
                   for b in (_turned(b, turn) for b in boxes)], colors, width=2)  # fmt: skip
    return _png(small, "JPEG")


def _turned(b: list, turn: int) -> list:
    """A box in page fractions after the page turns `turn` degrees counter-clockwise."""
    if turn == 90:  # (x, y) -> (y, 1 - x)
        return [b[1], 1 - b[2], b[3], 1 - b[0]]
    if turn == 270:  # (x, y) -> (1 - y, x)
        return [1 - b[3], b[0], 1 - b[1], b[2]]
    return b


def _label(word: dict) -> str:
    struck = "".join(word["text"][s:e] for s, e in word["struck"])
    if not struck:
        return "not struck"
    return "struck" if struck == word["text"] else f"struck: “{struck}” only"


EXTRA = (  # an extra word's two answers: the scorer right, or not
    "yes: the output added it, or garbled or misread a word",
    "no: it is on the page, or it is formatting (LaTeX, markdown), not a word",
)
OUTCOME = {  # the scorer's verdict on a word, in plain words
    "live": "kept as ordinary text, not marked deleted",
    "marked": "marked as deleted",
    "absent": "left out",
}


def _choices(name: str, options: tuple[str, ...], shown: tuple[str, ...] | None = None) -> str:
    """Radio buttons whose values are `options`, labelled `shown` where given."""
    return " ".join(
        f'<label><input type="radio" name="{name}" value="{o}"> {s}</label>'
        for o, s in zip(options, shown or options)
    )


def _item_html(item: dict, pages: dict, q1_only: bool = False, n: str = "") -> str:
    iid, page = item["item_id"], pages[item["page_id"]]
    words = [page["words"][i] for i in item["words"]] or page["words"][:1]
    around = [page["words"][i] for i in item.get("neighbours", ())]
    colors = [BLUE] * len(words) + [ORANGE] * len(around)
    boxes, turn = [w["bbox"] for w in words + around], _turn(page)
    img = f'<img src="{_crop(item["page_id"], boxes, colors, turn=turn)}">'
    part = words[0]["text"][slice(*item["span"])] if "span" in item else ""
    what = (f"“{html.escape(part)}” (part of the boxed word)" if part and part != words[0]["text"]
            else "the boxed word")  # fmt: skip
    head = f'<section data-id="{iid}"><h3>{iid}' + (f" <small>{n}</small>" if n else "")
    out = f'<p class="out">{item.get("output", "")}</p>'
    if item.get("kind") == "hallucination":
        whole = f'<img src="{_whole(item["page_id"], boxes, colors, turn)}">'
        return (f"{head} · extra word</h3>{img}<p>The output has a word the scorer could not match "
                "to anything on the page (highlighted below). The blue boxes are the page words "
                "the output has just before and after it. The whole page is below the output, "
                "for what a crop can't show: a chart, a photo, a heading drawn as a picture.</p>"
                f"{out}<details><summary>The whole page</summary>{whole}</details><p>The scorer counts the "
                "highlighted word as one the output <b>added or misread</b>. Is that right?</p><p>"
                f"{_choices(f'{iid}-q3', ('yes', 'no'), EXTRA)}</p></section>")  # fmt: skip
    verdict = OUTCOME.get(item.get("outcome", ""), item.get("outcome", ""))
    q3 = (f"{out}<p>The scorer says the output has {what} <b>{html.escape(verdict)}</b>. Is that "
          f"what the output shows? {_choices(f'{iid}-q3', ('yes', 'no'))}</p>")  # fmt: skip
    question = item.get("question", "Is the word in the blue box struck through?")
    options = item.get("options", ("struck", "not struck", "cannot tell"))
    shown = tuple("can't tell" if o == "cannot tell" else o for o in options)
    body = (f"{head}</h3>{img}<p><b>1.</b> {question}</p><p>{_choices(f'{iid}-q1', options, shown)} "
            f'<input name="{iid}-chars" placeholder="only part struck? e.g. re~mark~s"></p>')  # fmt: skip
    if "answer" in item:
        body += f"<details><summary>Answer</summary><p><b>{html.escape(item['answer'])}</b></p></details>"
    elif not q1_only:
        body += (f"<details><summary><b>2.</b> Answer key</summary><p>The key says: "
                 f"<b>{html.escape(_label(words[0]))}</b>. Is the key right? "
                 f"{_choices(f'{iid}-q2', ('yes', 'no'))}</p></details>"
                 f"<details><summary><b>3.</b> Output</summary>{q3}</details>")  # fmt: skip
    return body + "</section>"


STRIKE = ("A strike is a line through the middle of the letters; underlines, table lines, borders "
          "and black bars are not.")  # fmt: skip
INTRO_AUDIT = f"""<p>Each item is a place where an output (a model's answer, or the package's)
disagrees with the answer key. You are checking whether the disagreement is real, or a mistake in
the key or in the scorer. The page never says which system wrote the output.</p><ol>
<li><b>Look at the blue box first.</b> Is that word struck through? {STRIKE} If only part of the
word is struck, type those letters. Answer this before opening 2 and 3.</li>
<li><b>Answer key:</b> is the key's label for the boxed word right?</li>
<li><b>Output:</b> the output's text around the word, the word highlighted, ⟦text⟧ where the output
marked text as deleted and {GAP} where it has nothing. Did the scorer read it right?</li></ol>
<p><b>Extra word</b> items ask only the last question: the output has a word the scorer could not
find on the page. Garbled output (stray letters where words should be) is the output's fault, so
the scorer is right to count it.</p>"""
INTRO_Q1 = f"""<p>Answer the question under each image from the image alone. {STRIKE} If only
part of a word is struck, type the word with each struck part between tildes: <b>re~mark~s</b> when
"mark" alone is struck, <b>~un~like~ly~</b> when it is struck in two places. (Typing just the struck
letters works too, when they occur once in the word.)</p>"""


PAGE = """<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>body{{font:15px/1.45 system-ui;max-width:900px;margin:2em auto;padding:0 16px;background:#fff;color:#111}}
section{{border-top:1px solid #ccc;padding:1em 0}}h3 small{{color:#666;font-weight:normal}}summary{{cursor:pointer;margin:.4em 0}}
img{{max-width:100%;border:1px solid #ddd}}.out{{font-family:ui-monospace,monospace;background:#f5f5f5;padding:.5em}}
.out b{{background:#ffd84d;padding:0 .15em;border-radius:2px}}input[name$="-chars"]{{width:17em}}label{{margin-right:.6em}}</style>
<h1>{title}</h1><p>Annotator {who}, {count} items.</p>{intro}
<p>Your answers stay in this tab only: <b>download the CSV before you close it.</b></p>
<button onclick="save()">Download my labels (CSV)</button>
{items}
<p><button onclick="save()">Download my labels (CSV)</button></p>
<script>function save(){{const ids=[...document.querySelectorAll('section[data-id]')].map(s=>s.dataset.id);
const v=n=>{{const e=document.querySelector(`[name="${{n}}"]:checked`)||document.querySelector(`input[name="${{n}}"]:not([type=radio])`);return e?e.value:''}};
const rows=[['annotator','item_id','q1','chars','q2','q3'],...ids.map(i=>['{who}',i,v(i+'-q1'),v(i+'-chars'),v(i+'-q2'),v(i+'-q3')])];
const csv=rows.map(r=>r.map(x=>'"'+String(x).replaceAll('"','""')+'"').join(',')).join('\\n');
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));a.download='{who}.csv';a.click()}}</script>"""


def _practice() -> list[dict]:
    """Ten G0-dev words with known answers, five struck and five not, for each annotator first."""
    rng = random.Random(seed("double_label"))
    words = [(p["page_id"], i, w) for p in load_pages("G0", split="dev")
             for i, w in enumerate(p["words"])
             if not w["furniture"] and len(w["text"]) > 2]  # fmt: skip
    struck = rng.sample(
        [x for x in words if x[2]["struck"] and "E5" not in x[2]["strata"]], PRACTICE // 2
    )
    live = rng.sample([x for x in words if not x[2]["struck"]], PRACTICE - PRACTICE // 2)
    return [{"item_id": f"practice-{n + 1}", "page_id": pid, "words": [i], "answer": _label(w)}
            for n, (pid, i, w) in enumerate(rng.sample(struck + live, PRACTICE))]  # fmt: skip


def write_pages(out: Path, title: str, items: list[dict], q1_only: bool = False,
                shared: float = SHARED, who: str = "AB") -> None:  # fmt: skip
    """Split the items between annotators A and B, a random 25% (seed 20261016) labelled by both,
    numbered in a second shuffle so the shared ones don't stand out; write each one's page
    (practice items first) and the key, which holds the arm and must not reach them. With `who`
    "A", one page holds every item and no practice item: the author's, for the pilot triage and
    G2's span check."""
    rng = random.Random(seed("double_label"))
    order = rng.sample(items, len(items))
    n_shared = round(shared * len(items)) if len(who) == 2 else 0
    for k, item in enumerate(order):
        item["annotators"] = who if k < n_shared or len(who) == 1 else who[k % 2]
    order = rng.sample(order, len(order))
    for k, item in enumerate(order):
        item["item_id"] = f"i{k + 1:03d}"
    pages = {p["page_id"]: p for p in load_pages("G0", "G1", "G2")}
    practice = _practice() if len(who) == 2 else []
    render.require_current([pages[i["page_id"]] for i in practice + order])
    out.mkdir(parents=True, exist_ok=True)
    intro = INTRO_Q1 if q1_only else INTRO_AUDIT
    if practice:
        intro += (
            f"<p>The first {len(practice)} items are practice: answer, then open the answer.</p>"
        )
    for annotator in who:
        mine = practice + [i for i in order if annotator in i["annotators"]]
        body = "".join(_item_html(i, pages, q1_only=q1_only, n=f"{k + 1} of {len(mine)}")
                       for k, i in enumerate(mine))  # fmt: skip
        page = PAGE.format(title=html.escape(title), who=annotator, count=len(mine), intro=intro,
                           items=body)  # fmt: skip
        (out / f"{annotator}.html").write_text(page, encoding="utf-8")
    write_jsonl(out / "key.jsonl", practice + order)
    print(f"{len(items)} items ({n_shared} shared) and {len(practice)} practice items -> {out}")


# -------------------------------------------------------------------------- G1 and G2 samples


def g1_items() -> list[dict]:
    """G1's error-rate sample, once prereg-2 has fixed it (PREREG, The audit): labelled alongside
    the test run, so its sha256 must be the one tagged first."""
    path = GT / "G1-error-sample.jsonl"
    frozen = json.loads(PREREG2.read_text(encoding="utf-8")) if PREREG2.exists() else {}
    if frozen.get("g1_error_sample") != sha256_file(path):
        raise ValueError(f"{path.name} differs from what {PREREG2.name} fixes, or prereg-2 has "
                         "not fixed it yet: freeze it first")  # fmt: skip
    return [{"page_id": r["page_id"], "words": [r["word"]], "kind": f"g1-{r['kind']}"}
            for r in read_jsonl(path)]  # fmt: skip


def relabel_items(stage: str = "test") -> list[dict]:
    """PREREG, G1's error rate: the G1-test words on which at least three of the four flagships'
    R-best arms disagree with the ground truth (a struck word output live, a live word output
    marked or absent), for both annotators; and the G1 words the test audit found its key wrong
    on (question 2), so their corrections reach the key the same way (deviation 5)."""
    best = {f"{k}:{plans.choice(m['provider'], 'flagship')}:{plans.choice(m['provider'], 'image')}"
            for k, m in config("models").items() if m["role"] == "flagship"}  # fmt: skip
    pages = {p["page_id"]: p for p in load_pages("G1", split="test")}
    against: dict[tuple[str, int], set[str]] = defaultdict(set)  # word -> the arms disputing it
    for o in harness.outputs(stage):
        if o["arm"] in best and o["page_id"] in pages and o["tokens"] is not None:
            res = score_page(pages[o["page_id"]], o["tokens"], details=True)
            if res["failed"]:  # a failure's worst case says nothing about any word
                continue
            for r in res["details"]:
                if _kind(r) in ("leak", "over-deletion"):
                    against[o["page_id"], r["word"]].add(o["arm"])
    words = {w for w, arms in against.items() if len(arms) >= 3}
    audit = RUNS / stage / "audit"
    if (audit / "ingest.json").exists():
        key = {k["item_id"]: k for k in read_jsonl(audit / "key.jsonl")}
        wrong = json.loads((audit / "ingest.json").read_text(encoding="utf-8"))["ground_truth_no"]
        words |= {
            (key[i]["page_id"], key[i]["words"][0]) for i in wrong if key[i]["page_id"] in pages
        }
    return [{"page_id": pid, "words": [i], "kind": "g1-relabel"} for pid, i in sorted(words)]


def _same_line(a: dict, b: dict) -> bool:
    """On one printed line: PyMuPDF splits a justified line into several of its own."""
    ha, hb = a["bbox"][3] - a["bbox"][1], b["bbox"][3] - b["bbox"][1]
    return abs(a["bbox"][1] + a["bbox"][3] - b["bbox"][1] - b["bbox"][3]) / 2 <= min(ha, hb) / 2


def g2_items(n: int = 100) -> list[dict]:
    """G2's span check: n spans, each a run of consecutive words on one printed line aligned as
    struck, boxed blue, with the live word just before or after it in reading order boxed orange
    where there is one. Every span at such a struck-live boundary is drawn, the rest at random
    (seed 20261021). Words that normalize to nothing (`‘‘;`) carry no label and are passed over."""
    boundary, inner = [], []
    for p in load_pages("G2", split="test"):
        words = p["words"]
        body = [i for i, w in enumerate(words) if not w["furniture"] and normalize(w["text"])[0]]

        def state(k: int) -> str:
            w = words[body[k]]
            return "masked" if w["masked"] else "struck" if w["struck"] else "live"

        k = 0
        while k < len(body):
            end = k + 1
            while (end < len(body) and state(end) == state(k)
                   and _same_line(words[body[end - 1]], words[body[end]])):  # fmt: skip
                end += 1
            if state(k) == "struck":
                side = [body[x] for x in (k - 1, end) if 0 <= x < len(body) and state(x) == "live"]
                span = {"page_id": p["page_id"], "words": body[k:end], "neighbours": side,
                        "kind": "g2-span"}  # fmt: skip
                (boundary if side else inner).append(span)
            k = end
    if len(boundary) + len(inner) < n:
        raise ValueError(f"G2 has {len(boundary) + len(inner)} spans, fewer than the {n} drawn")
    rng = random.Random(seed("g2_spans"))
    picks = rng.sample(boundary, min(n, len(boundary)))
    picks += rng.sample(inner, n - len(picks))
    q = ("Is every blue-boxed word struck through (in whole or in part), and no orange-boxed word "
         "struck?")  # fmt: skip
    return [{**s, "question": q, "options": ("yes", "no", "cannot tell")} for s in picks]


def notations(stages: list[str], width: int = 4) -> str:
    """PREREG, Scoring: before prereg-2, every struck sub-token scored live in the dev outputs at
    R0 and R1 (each stage named: the ladder, the paraphrases, the repeats), with the output
    around it in the audit's neutral rendering, to runs/<first stage>/notations.csv; and the
    tokens that sit beside those words most often, the way a notation the scorer doesn't read
    would show (`[struck: 5%]`, `5% (deleted)`). A notation that marks struck text, and wraps no
    live text on dev, joins the scorer's marker list before its hash is frozen."""
    pages = {p["page_id"]: p for p in load_pages("G0", "G1", "G2", split="dev")}
    rows, beside = [], Counter()
    for stage in stages:
        for o in harness.outputs(stage):
            if (
                o["tokens"] is None
                or o.get("rung") not in ("R0", "R1")
                or o["page_id"] not in pages
            ):
                continue
            res = score_page(pages[o["page_id"]], o["tokens"], details=True)
            if res["failed"]:  # no usable output: scored worst case, no notation to read
                continue
            for r in res["details"]:
                if not (r["struck"] and r["outcome"] == "live"):
                    continue
                near = [k for t in r["tokens"] for k in range(t - width, t + width + 1)
                        if 0 <= k < len(o["tokens"]) and k not in r["tokens"]]  # fmt: skip
                beside.update(o["tokens"][k].text for k in dict.fromkeys(near)
                              if not any(c.isalnum() for c in o["tokens"][k].text)
                              or o["tokens"][k].text[:1] in "[(<{~-")  # fmt: skip
                rows.append({"stage": stage, "arm": o["arm"], "page_id": o["page_id"],
                             "word": r["word"], "output": _render(o["tokens"], r["tokens"],
                                                                  width=width)})  # fmt: skip
    if not rows:
        return "no struck word scored live at R0 or R1 in " + ", ".join(stages)
    path = RUNS / stages[0] / "notations.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    common = ", ".join(f"{t!r} {n}" for t, n in beside.most_common(25))
    return f"{len(rows)} struck words left live -> {path}\nbeside them most often: {common}"


# ----------------------------------------------------------------------------------- labels


def _read(files: list[Path]) -> dict[str, dict[str, dict]]:
    """Labels by item and annotator, from the CSVs the pages download (Excel's BOM tolerated)."""
    labels: dict[str, dict[str, dict]] = {}
    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                who = row.get("annotator") or path.stem[:1].upper()
                if who not in ("A", "B"):
                    raise ValueError(f"{path}: annotator {who!r} is neither A nor B")
                labels.setdefault(row["item_id"], {})[who] = row
    return labels


def _answer(row: dict, relabel: bool) -> tuple[str, str]:
    """A question-1 answer; on a relabel item, a struck answer's characters are part of it."""
    chars = (row.get("chars") or "").strip()
    return row["q1"], chars if relabel and row["q1"] == "struck" else ""


def ingest(folder: Path, files: list[Path], adjudicated: Path | None = None) -> dict:
    """Merge the annotators' CSVs with the key: κ on question 1 over the shared items that ask it,
    both answered; each annotator's practice score; the items an assigned annotator didn't
    return or left blank; the question-1 disagreements for the author; the items with a "no" on
    question 2 (ground-truth errors) or 3 (scorer bugs) for the other annotator to check; G2's
    spans answered "no"; G1's error rates with Wilson intervals ("cannot tell" left out, and
    counted); and, once every relabel item is decided, G1's corrections. An item is decided by
    every annotator it went to giving the same answer; the author's adjudication (a CSV of
    item_id, q1, chars) decides the rest alone, and a row naming an item already decided is
    refused (PREREG, The audit)."""
    key = {k["item_id"]: k for k in read_jsonl(folder / "key.jsonl")}
    labels = _read(files)
    final = {}
    if adjudicated:
        with adjudicated.open(encoding="utf-8-sig", newline="") as f:
            rows = csv.DictReader(f)
            if not {"item_id", "q1", "chars"} <= set(rows.fieldnames or ()):
                raise ValueError(f"{adjudicated}: needs the columns item_id, q1 and chars")
            final = {r["item_id"]: r for r in rows}
    real = [i for i in key if not i.startswith("practice")]
    given = {i: [labels.get(i, {}).get(w) for w in key[i].get("annotators", "AB")] for i in real}
    q1 = [i for i in real if key[i]["kind"] != "hallucination"]
    if unknown := sorted(set(final) - set(q1)):
        raise ValueError(f"the adjudication names items with no question 1 to decide: {unknown}")
    shared = [i for i in q1 if len(given[i]) == 2 and all(r and r["q1"] for r in given[i])]
    a, b = [labels[i]["A"]["q1"] for i in shared], [labels[i]["B"]["q1"] for i in shared]

    def agreed(i: str) -> tuple[str, str] | None:
        if not all(r and r["q1"] for r in given[i]):
            return None
        answers = {_answer(r, key[i]["kind"] == "g1-relabel") for r in given[i]}
        return answers.pop() if len(answers) == 1 else None

    if overruled := sorted(i for i in final if i in key and i in q1 and agreed(i) is not None):
        raise ValueError(
            f"the adjudication names items the annotators already decided: {overruled}"
        )

    def decided(i: str) -> tuple[str, str] | None:
        if final.get(i, {}).get("q1"):
            return _answer(final[i], key[i]["kind"] == "g1-relabel")
        return agreed(i)

    def no(q: str) -> list[str]:
        return sorted(i for i in real if any(r and r[q] == "no" for r in given[i]))

    labelled = {i: decided(i) for i in q1}
    out = {
        "items": sum(any(given[i]) for i in real), "shared": len(shared),
        "kappa": cohen_kappa(a, b) if shared else None,
        "kappa_ci": kappa_ci(a, b) if shared else None,
        "practice_correct": {w: sum(labels.get(k, {}).get(w, {}).get("q1", "") ==
                                    ("not struck" if key[k]["answer"] == "not struck" else "struck")
                                    for k in key if k.startswith("practice")) for w in "AB"},
        "missing": sorted(i for i in real if not all(given[i])),
        "q1_blank": sorted(i for i in q1 if any(r and not r["q1"] for r in given[i])),
        "q1_disagree": sorted(i for i in q1 if labelled[i] is None
                              and all(r and r["q1"] for r in given[i])),
        "ground_truth_no": no("q2"),
        "scorer_no": no("q3"),
        "g2_span_no": sorted(i for i in no("q1") if key[i]["kind"] == "g2-span"),
        "unresolved": sorted(i for i, v in labelled.items() if v is None),
        "ingested": datetime.now(UTC).isoformat(),  # when these labels came in (PREREG, The audit)
    }  # fmt: skip
    for kind, wrong in (("g1-struck", "not struck"), ("g1-live", "struck")):
        known = [v[0] for i, v in labelled.items() if key[i]["kind"] == kind and v]
        out[f"{kind}_cannot_tell"] = known.count("cannot tell")
        if known := [v for v in known if v != "cannot tell"]:
            errors = sum(v == wrong for v in known)
            out[f"{kind}_error_rate"] = [errors / len(known), *wilson(errors, len(known))]
    relabel = [i for i in q1 if key[i]["kind"] == "g1-relabel"]
    if relabel and any(labelled[i] is None for i in relabel):
        out["corrections"] = "not written: every relabel item must be decided first"
    elif relabel:
        words = {p["page_id"]: p["words"] for p in load_pages("G1", split="test")}
        fixes = [{"page_id": key[i]["page_id"], "word": key[i]["words"][0],
                  "text": words[key[i]["page_id"]][key[i]["words"][0]]["text"],
                  "label": labelled[i][0], "chars": labelled[i][1]} for i in relabel]  # fmt: skip
        if bad := [
            i for i, f in zip(relabel, fixes) if g1.struck_spans(f["text"], f["chars"]) is None
        ]:
            out["corrections"] = (
                f"not written: struck characters missing from the word, or in it twice: {bad}"
            )
        else:
            write_jsonl(g1.CORRECTIONS, fixes)
            out["corrections"] = f"{len(fixes)} -> {g1.CORRECTIONS}; now `g1 build`, then rescore"
    write_jsonl(folder / "labels.jsonl", [{"item_id": i, **by} for i, by in labels.items()])
    return out


def register(sub) -> None:
    p = sub.add_parser("audit", help="the hand audit, G1's error-rate sample, G2's span check")
    cmd = p.add_subparsers(required=True, metavar="action")
    b = cmd.add_parser("build", help="sample a stage's disagreements into annotator pages")
    b.add_argument("stage", choices=["pilot", "test"])
    b.set_defaults(func=lambda a: write_pages(
        RUNS / a.stage / "audit", f"Audit: {a.stage}", sample(a.stage), who="A",  # the author's
    ))  # fmt: skip
    n = cmd.add_parser(
        "notations", help="dev outputs at R0 and R1 around every struck word left live"
    )
    n.add_argument("stage", nargs="+")
    n.set_defaults(func=lambda a: print(notations(a.stage)))
    cmd.add_parser("g1").set_defaults(  # the author's, before the test run is scored
        func=lambda a: write_pages(
            RUNS / "g1-audit", "G1 error-rate sample", g1_items(), q1_only=True, who="A"
        )
    )
    cmd.add_parser("g2").set_defaults(  # the author's, before prereg-2 freezes G2's labels
        func=lambda a: write_pages(
            RUNS / "g2-audit", "G2 span check", g2_items(), q1_only=True, who="A"
        )  # fmt: skip
    )
    r = cmd.add_parser("relabel", help="after the test run: words three flagships dispute")
    r.set_defaults(
        func=lambda a: write_pages(
            RUNS / "test" / "g1-relabel", "G1 relabel", relabel_items(), q1_only=True, who="A"
        )
    )
    i = cmd.add_parser("ingest", help="labels back in: κ, practice scores and the noes")
    i.add_argument("folder", type=Path)
    i.add_argument("csv", nargs="+", type=Path)
    i.add_argument("--adjudicated", type=Path, help="the author's CSV: item_id,q1,chars")
    i.set_defaults(
        func=lambda a: print(json.dumps(ingest(a.folder, a.csv, a.adjudicated), indent=2))
    )
