"""The scorer (PREREG.md, Scoring): one arm's output on one page against the ground truth.

Every arm goes through the same steps. An output is parsed into tokens whose characters carry a
mark (inside a strike marker, or carrying U+0334–U+0338); the package's arms arrive as tokens
already, marked from their records' character spans only. Tokens are aligned to the
ground-truth words by a global edit-distance alignment over normalized text, then characters
within each aligned pair, and each ground-truth sub-token (a word split at its struck/live
boundaries) is scored *marked*, *live* or *absent*.

Choices the PREREG text leaves open, fixed here:
- "List markers" are bullets (`-`, `*`, `+`, `•`); a numbered label such as `1.` is page text.
- GFM's single-tilde strikethrough marks too: `~` before a non-space opens, `~` after one closes,
  on one line, neither part of a `~~` nor escaped; so `~5%~` is marked and `~ 5 ~` is text.
- Code fences (three or more ` or ~) are markup, never a strike. HTML entities are decoded.
  Tags other than the strike markers become word breaks, and only real tag names count, so
  `<Party A>` is text.
  Marker delimiters vanish, so `~~semi-~~monthly` stays one token. Strike tags pair up by
  nesting, whichever of `<del>`, `<s>` and `<strike>` they are, and everything inside the
  outermost pair is marked; an unpaired one is a word break.
- Output tokens that normalize to nothing (leftover markup, lone punctuation) are dropped, and an
  output with no token left is no usable output: a failure.
- A JSON verdict is read case-insensitively. An item without a text gives no token, as a word left
  out would; a `words` list with no item that has one fails to parse.
- A JSON word's `confidence` written as a string counts as its number. A stray `unsure` (the
  prompts offer `struck` and `clean` alone) reads as live and gives 0.5, whatever the confidence;
  a boolean or any other non-number, or another verdict, gives none; a struck item's probability
  goes to its struck tokens alone.
- Joins go both ways: an output token equal to two adjacent ground-truth words joined aligns to
  both, and two adjacent output tokens equal to one ground-truth word joined align to it (hyphens
  ignored, edge punctuation kept inside the joined text).
- Characters align by an Indel (LCS) alignment, the cost model a substitution costing twice a gap
  gives; on its ties, unmatched characters between two matched runs pair up in order, so a
  misread character still counts as present.
- Ties in the word alignment go to a substitution, then a join, then a split, then an absent
  ground-truth word, then an inserted output token.
- CER and WER compare normalized live text: output tokens with marked characters removed, against
  the scored words' live characters, each joined by single spaces; words the alignment joined or
  split count as one.
- A masked ground-truth word whose text is UNREAD (G5's unreadable words) aligns to any one
  output token at a cost of 1.5: less than leaving both unaligned (2), more than any pair close
  enough to count as present (at most 1). So a model's reading of it is ignored, never counted as
  inserted, and never takes a token a readable word could have. A scored word is never a wildcard,
  whatever its text.
"""

import html
import itertools
import json
import re
import string
import unicodedata
from collections import Counter
from collections.abc import Collection
from typing import NamedTuple

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Indel, Levenshtein

COMBINING = frozenset("̴̵̶̷̸")
# a fence line (three or more backticks or tildes, as CommonMark allows), info string only
FENCE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})[^\s`~]*[ \t]*$", re.M)
MARKERS = re.compile(
    r"~~(.+?)~~|\[deleted:\s*(.*?)\]"
    r"|\[(?:red|blue) strikethrough:\s*(.*?)\]",  # from the notations audit (PREREG, Scoring)
    re.S | re.I,
)
TILDE = re.compile(r"(?<![~\\])~(?=[^\s~])([^~\n]*?[^\s~\\])~(?!~)")  # GFM's single tildes
STRIKE_TAG = re.compile(r"<(/?)(?:del|s|strike)\b[^>]*>", re.I)
DROP = re.compile(
    r"^[ \t]{0,3}#{1,6}(?=\s)"  # heading markers
    r"|^[ \t]*[-*+•](?=\s)"  # bullets
    r"|(?<![\w\\])[*_]+|(?<!\\)[*_]+(?!\w)"  # emphasis runs, unless escaped
    r"|\\(?=[!-/:-@\[-`{-~])",  # backslash escapes
    re.M,
)
TAGS = (
    "a|b|big|blockquote|br|caption|center|code|del|div|em|font|h[1-6]|hr|i|img|ins|li|mark|ol|p|pre|s|strike"
    "|section|small|span|strong|sub|sup|table|tbody|td|th|thead|tr|u|ul"
)
BREAK = re.compile(
    r"^[ \t]*(?=[|:\-=*_+ \t]*[-=*_]{3})[|:\-=*_+ \t]+$"  # table rule rows, horizontal rules
    rf"|</?(?:{TAGS})\b[^>\n]*>"  # other HTML tags
    r"|\|"  # table pipes
    r"|[{}]",  # LaTeX's braces (`\frac{4}{3}`, `10^{-3}`)
    re.M | re.I,
)
ENTITY = re.compile(r"&(?:#\d+|#[xX][0-9a-fA-F]+|[A-Za-z]+);")
LATEX = re.compile(r"\\([A-Za-z]+)")
LATEX_ENV = re.compile(r"\\(?:begin|end)\{[^{}\n]*\}")  # `\begin{aligned}`: markup, whole
GREEK = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi pi rho sigma tau "
         "upsilon phi chi psi omega").split()  # fmt: skip
LATEX_SYMBOLS = {  # a command a model writes for a printed symbol (PREREG, Scoring): the symbol
    # Unicode spells lambda "lamda"
    **{g: unicodedata.lookup(f"GREEK SMALL LETTER {g.replace('mb', 'm')}") for g in GREEK},
    **{g.capitalize(): unicodedata.lookup(f"GREEK CAPITAL LETTER {g.replace('mb', 'm')}")
       for g in "gamma delta theta lambda xi pi sigma upsilon phi psi omega".split()},
    "varepsilon": "ε", "vartheta": "ϑ", "varphi": "φ", "varrho": "ϱ", "varsigma": "ς",
    "cdot": "·", "times": "×", "div": "÷", "pm": "±", "mp": "∓", "le": "≤", "leq": "≤",
    "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠", "approx": "≈", "sim": "∼", "equiv": "≡",
    "propto": "∝", "infty": "∞", "circ": "°", "textdegree": "°", "partial": "∂", "nabla": "∇",
    "sum": "∑", "prod": "∏", "int": "∫", "sqrt": "√", "to": "→", "rightarrow": "→",
    "leftarrow": "←", "Rightarrow": "⇒", "ldots": "…", "cdots": "⋯", "dots": "…",
}  # fmt: skip
LATEX_MARKUP = frozenset(  # font, text and sizing commands: markup, like emphasis runs
    "mathrm mathbf mathit mathsf mathtt mathcal text textrm textbf textit operatorname rm "
    "left right big Big bigl bigr Bigl Bigr displaystyle frac dfrac tfrac quad qquad limits "
    "nonumber".split()
)
LATEX_NAMES = frozenset("max min log ln exp sin cos tan lim det arg sup inf".split())
TYPOGRAPHIC = str.maketrans(
    dict.fromkeys("‘’‚‛′", "'") | dict.fromkeys("“”„‟", '"') | dict.fromkeys("‐‑‒–—―−", "-")
)
PUNCT = frozenset(string.punctuation)
MAX_PAIR_DISTANCE = 0.5
UNREAD, UNREAD_COST = "�", 1.5  # a word no one could read, and its cost against any token


class Token(NamedTuple):
    text: str
    marks: tuple[bool, ...]  # per character: marked as struck
    p_struck: float | None = None  # the arm's probability that its word is struck, if it gives one


class ParseError(ValueError):
    """The output holds no usable answer: scored as a failure."""


# ---------------------------------------------------------------------------------- parsing


def parse(text: str, json_answer: bool = False, marks: bool = True) -> list[Token]:
    """Tokens of a raw output: markdown (R0–R3, B-pm4llm) or a JSON answer (R4–R6). `marks`
    False reads every word as live (B-naive)."""
    tokens = _parse_json(text) if json_answer else _parse_markdown(text)
    return tokens if marks else [Token(t.text, (False,) * len(t.text)) for t in tokens]


def _flag(flags: list[bool], start: int, end: int) -> None:
    flags[start:end] = [True] * (end - start)


def _parse_markdown(text: str) -> list[Token]:
    n = len(text)
    marked, drop, brk = [False] * n, [False] * n, [False] * n
    for m in FENCE.finditer(text):
        _flag(drop, m.start(), m.end())
    search = FENCE.sub(lambda m: " " * len(m.group()), text)  # a fence is never a marker
    for m in MARKERS.finditer(search):
        s, e = m.span(m.lastindex)
        _flag(marked, s, e)
        _flag(drop, m.start(), s)
        _flag(drop, e, m.end())
    outside = MARKERS.sub(lambda m: " " * len(m.group()), search)  # single tildes, outside those
    for m in TILDE.finditer(outside):
        s, e = m.span(1)
        _flag(marked, s, e)
        _flag(drop, m.start(), s)
        _flag(drop, e, m.end())
    opened: list[re.Match] = []
    for m in STRIKE_TAG.finditer(search):
        if not m.group(1):
            opened.append(m)
        elif opened:
            o = opened.pop()
            _flag(marked, o.end(), m.start())
            _flag(drop, o.start(), o.end())
            _flag(drop, m.start(), m.end())
    for regex, flags in ((DROP, drop), (BREAK, brk)):
        for m in regex.finditer(search):
            _flag(flags, m.start(), m.end())
    chars = list(text)
    for m in ENTITY.finditer(search):
        decoded = html.unescape(m.group())
        if len(decoded) == 1:
            chars[m.start()] = decoded
            _flag(drop, m.start() + 1, m.end())
    for m in LATEX_ENV.finditer(search):
        _flag(drop, m.start(), m.end())
    for m in LATEX.finditer(search):
        if (rep := _latex(m.group(1))) is not None:  # never longer than the command
            chars[m.start() : m.start() + len(rep)] = rep
            _flag(drop, m.start() + len(rep), m.end())
    return _tokenize([(c, marked[i], drop[i], brk[i]) for i, c in enumerate(chars)])


def _latex(name: str) -> str | None:
    """What a LaTeX command reads as: a symbol's character, nothing for markup, an operator's
    name; None for any other command, which stays text."""
    if name in LATEX_SYMBOLS:
        return LATEX_SYMBOLS[name]
    if name in LATEX_MARKUP:
        return ""
    return name if name in LATEX_NAMES else None


def _delatex(text: str) -> str:
    """A JSON word's text with its LaTeX read as the markdown parser reads it, braces breaking
    words."""
    text = LATEX.sub(lambda m: m.group() if (r := _latex(m.group(1))) is None else r,
                     LATEX_ENV.sub("", text))  # fmt: skip
    return text.replace("{", " ").replace("}", " ")


def _tokenize(chars: list[tuple[str, bool, bool, bool]]) -> list[Token]:
    """Split (char, marked, drop, break) on whitespace and breaks; a combining stroke marks the
    character before it and is itself dropped."""
    tokens, cur = [], []
    for c, mark, drop, brk in chars:
        if drop:
            continue
        if c in COMBINING:
            if cur:
                cur[-1] = (cur[-1][0], True)
            continue
        if brk or c.isspace():
            if cur:
                tokens.append(cur)
            cur = []
        else:
            cur.append((c, mark))
    if cur:
        tokens.append(cur)
    return [Token("".join(c for c, _ in t), tuple(m for _, m in t)) for t in tokens]


def _last_json_object(text: str) -> dict:
    decoder, pos, last = json.JSONDecoder(), 0, None
    while (start := text.find("{", pos)) != -1:
        try:
            obj, end = decoder.raw_decode(text, start)
        except ValueError:  # not JSON here, or a number too long to read
            pos = start + 1
            continue
        if isinstance(obj, dict):
            last = obj
        pos = end
    if last is None:
        raise ParseError("no JSON object in the output")
    return last


def _parse_json(text: str) -> list[Token]:
    words = _last_json_object(text).get("words")
    if not isinstance(words, list):
        raise ParseError("the last JSON object has no 'words' list")
    tokens = []
    for w in words:
        if not isinstance(w, dict) or not isinstance(w.get("text"), (str, int, float)):
            continue  # no text: no token, as for a word the answer left out
        word = _delatex(str(w["text"]))
        part = _delatex(w["chars"]) if isinstance(w.get("chars"), str) else ""
        start = word.find(part) if part else -1
        s, e = (start, start + len(part)) if start >= 0 else (0, len(word))
        verdict = str(w.get("verdict", "")).strip().lower()
        p = _p_struck(verdict, w.get("confidence"))
        # the verdict alone marks a JSON word: a combining stroke in its text is dropped, not read
        chars = [(c, verdict == "struck" and s <= i < e, c in COMBINING, False)
                 for i, c in enumerate(word)]  # fmt: skip
        # a struck item's probability goes to its struck tokens alone, not to a live word beside
        tokens += [t._replace(p_struck=p if verdict != "struck" or any(t.marks) else None)
                   for t in _tokenize(chars)]  # fmt: skip
    if not tokens:
        raise ParseError("the last JSON object has no word with a text")
    return tokens


def _p_struck(verdict: str, confidence) -> float | None:
    """A JSON word's probability of being struck: its confidence in a `struck` verdict, one minus
    its confidence in a `clean` one, 0.5 under `unsure` whatever the confidence; none for another
    verdict, or a confidence missing, outside 0 to 1, or neither a number nor a string holding
    one (PREREG, Scoring, the calibration)."""
    if verdict == "unsure":
        return 0.5
    if verdict not in ("struck", "clean") or isinstance(confidence, bool):
        return None
    try:
        c = float(confidence)
    except TypeError, ValueError, OverflowError:
        return None
    if not 0 <= c <= 1:  # NaN fails too
        return None
    return c if verdict == "struck" else 1 - c


def tokens_from_words(
    words: list[tuple[str, list[list[int]]]], probs: list[float | None] | None = None
) -> list[Token]:
    """Tokens from word records: each word's text with its marked character spans (the package's
    arms, P-native and P-scan), and with `probs`, each word's probability of being struck (P-scan's
    `cnn_prob`, None where the package scored none). Marks come from the spans alone: a U+0336 in
    the text is dropped, not read as a mark (PREREG, Arms and inputs)."""
    if probs is not None and len(probs) != len(words):
        raise ValueError(f"{len(probs)} probabilities for {len(words)} words")
    out = []
    for i, (text, spans) in enumerate(words):
        chars = [(c, any(s <= x < e for s, e in spans), c in COMBINING, False)
                 for x, c in enumerate(text)]  # fmt: skip
        p = probs[i] if probs is not None else None
        out += [t._replace(p_struck=p) for t in _tokenize(chars)]
    return out


def tokens_from_text(text: str, marks: list[bool]) -> list[Token]:
    """Tokens of plain text whose characters carry a mark (G2's bill XML)."""
    return _tokenize([(c, m, False, False) for c, m in zip(text, marks)])


# ---------------------------------------------------------------------------- normalization


def normalize(text: str) -> tuple[str, list[int]]:
    """For alignment only: NFKC, case-folded, typographic quotes and dashes to ASCII, U+0334–U+0338
    and format characters (the soft hyphen U+00AD, zero-width spaces) removed, leading and
    trailing punctuation and symbols stripped. Returns the normalized text and, for each of its
    characters, the index of the character it came from. A base character and the combining marks
    after it normalize together, so a decomposed accent composes."""
    out, src = [], []
    i = 0
    while i < len(text):
        if text[i] in COMBINING or unicodedata.category(text[i]) == "Cf":
            i += 1
            continue
        j = i + 1
        while j < len(text) and unicodedata.combining(text[j]) and text[j] not in COMBINING:
            j += 1
        for ch in unicodedata.normalize("NFKC", text[i:j]).casefold().translate(TYPOGRAPHIC):
            out.append(ch)
            src.append(i)
        i = j
    s, e = 0, len(out)
    while s < e and _is_punct(out[s]):
        s += 1
    while e > s and _is_punct(out[e - 1]):
        e -= 1
    return "".join(out[s:e]), src[s:e]


def _is_punct(c: str) -> bool:
    return (
        c in PUNCT
        or unicodedata.category(c) in ("Sc", "Sk", "Sm")
        or unicodedata.category(c).startswith("P")
    )


# -------------------------------------------------------------------------------- alignment

Step = tuple[tuple[int, ...], tuple[int, ...]]  # (output indices, ground-truth indices)


def align_words(out: list[str], gt: list[str], joins: list[str] | None = None,
                splits: list[str] | None = None, early: bool = False,
                wild: Collection[int] = ()) -> list[Step]:  # fmt: skip
    """Global alignment of normalized tokens: gaps cost 1, a substitution twice the pair's
    normalized edit distance, or UNREAD_COST against a ground-truth word in `wild`. At no cost,
    an output token aligns to two adjacent ground-truth words whose joined text (`joins[j]`, for
    words j and j+1) it equals, and two adjacent output tokens whose joined text (`splits[i]`)
    equals a ground-truth word align to it; hyphens are ignored in both. The path's steps:
    ((i,), (j,)) a pair, ((i,), (j, j+1)) a join, ((i, i+1), (j,)) a split, ((), (j,)) an
    unmatched ground-truth word, ((i,), ()) an inserted output token. Among equal-cost paths the
    traceback takes the latest copy of a repeated run; `early` takes the earliest, by aligning
    both sequences reversed."""
    n, m = len(out), len(gt)
    joins = joins if joins is not None else [gt[j] + gt[j + 1] for j in range(m - 1)]
    splits = splits if splits is not None else [out[i] + out[i + 1] for i in range(n - 1)]
    if early:  # word j's join with j+1 becomes reversed word m-2-j's with m-1-j: the same key
        steps = align_words(out[::-1], gt[::-1], joins[::-1], splits[::-1],
                            wild=[m - 1 - j for j in wild])  # fmt: skip
        return [(tuple(n - 1 - i for i in reversed(o)), tuple(m - 1 - j for j in reversed(g)))
                for o, g in reversed(steps)]  # fmt: skip
    sub = (2 * process.cdist(out, gt, scorer=Levenshtein.normalized_distance, dtype=np.float64)
           if n and m else np.zeros((n, m)))  # fmt: skip
    sub[:, sorted(wild)] = UNREAD_COST
    by_join, by_word = {}, {}
    for j, key in enumerate(joins):
        by_join.setdefault(key.replace("-", ""), []).append(j)
    for j, word in enumerate(gt):
        by_word.setdefault(word.replace("-", ""), []).append(j)
    join_at = [by_join.get(o.replace("-", ""), []) for o in out]  # j: out[i] = gt[j] + gt[j+1]
    split_at = [by_word.get(k.replace("-", ""), []) for k in splits]  # j: out[i] + out[i+1] = gt[j]
    cols = np.arange(m + 1, dtype=np.float64)
    d = np.empty((n + 1, m + 1))
    d[0] = cols
    for i in range(1, n + 1):
        best = np.empty(m + 1)
        best[0] = i
        best[1:] = np.minimum(d[i - 1, :-1] + sub[i - 1], d[i - 1, 1:] + 1)
        for j in join_at[i - 1]:
            best[j + 2] = min(best[j + 2], d[i - 1, j])
        if i >= 2:
            for j in split_at[i - 2]:
                best[j + 1] = min(best[j + 1], d[i - 2, j])
        d[i] = np.minimum.accumulate(best - cols) + cols

    def same(a: float, b: float) -> bool:
        return abs(a - b) <= 1e-9  # absolute: path costs grow, a relative tolerance would too

    path: list[Step] = []
    i, j = n, m
    while i or j:
        here = d[i, j]
        if i and j and same(d[i - 1, j - 1] + sub[i - 1, j - 1], here):
            path.append(((i - 1,), (j - 1,)))
            i, j = i - 1, j - 1
        elif i and j >= 2 and j - 2 in join_at[i - 1] and same(d[i - 1, j - 2], here):
            path.append(((i - 1,), (j - 2, j - 1)))
            i, j = i - 1, j - 2
        elif i >= 2 and j and j - 1 in split_at[i - 2] and same(d[i - 2, j - 1], here):
            path.append(((i - 2, i - 1), (j - 1,)))
            i, j = i - 2, j - 1
        elif j and same(d[i, j - 1] + 1, here):
            path.append(((), (j - 1,)))
            j -= 1
        else:
            path.append(((i - 1,), ()))
            i -= 1
    return path[::-1]


def align_chars(a: str, b: str) -> list[tuple[int, int]]:
    """Character pairs (index in a, index in b): the LCS alignment, plus in-order pairing of the
    unmatched characters between matched runs."""
    pairs, pend_a, pend_b = [], [], []

    def flush() -> None:
        pairs.extend(zip(pend_a, pend_b))
        pend_a.clear()
        pend_b.clear()

    for op in Indel.opcodes(a, b):
        if op.tag == "equal":
            flush()
            pairs.extend(zip(range(op.src_start, op.src_end), range(op.dest_start, op.dest_end)))
        else:
            pend_a.extend(range(op.src_start, op.src_end))
            pend_b.extend(range(op.dest_start, op.dest_end))
    flush()
    return pairs


def match(tokens: list[Token], texts: list[str], early: bool = False,
          wild: Collection[int] = (), leftovers: bool = False, furniture: Collection[int] = (),
          unscored: Collection[int] = ()) -> tuple[dict, dict]:  # fmt: skip
    """Align output tokens to ground-truth word texts, then characters within each aligned pair.
    Returns `hits`, per word, its aligned characters (original index -> aligned to a marked
    output character?), and `used`, per output token that normalizes to something, the words it
    aligned to (none: inserted, including a pair farther apart than 0.5, unless the word is in
    `wild`). `early` and `wild` as in `align_words`, `wild` indexing `texts`. `furniture` words
    align in a pass of their own, after the others, so a margin line number between the two
    halves of a word hyphenated at a line end no longer keeps them apart. `leftovers` pairs
    equal tokens and words the alignment left over, whatever their order, a scored word before an
    `unscored` one: the scorer's, never G2's ground truth, which aligns in order. With it, a token
    equal to a word left over and to no furniture word sits out the furniture's pass, so that pass
    can't take a word's own token typed out of order."""
    wild, furniture, unscored = set(wild), set(furniture), set(unscored)
    onorm = [normalize(t.text) for t in tokens]
    gnorm = [normalize(t) for t in texts]
    keep = [k for k, (o, _) in enumerate(onorm) if o]
    gkeep = [j for j, (g, _) in enumerate(gnorm) if g]
    hits: dict[int, dict[int, bool]] = {}
    used: dict[int, tuple[int, ...]] = dict.fromkeys(keep, ())

    def align(outs_at: list[int], gts_at: list[int]) -> None:
        joins = [normalize(texts[a] + texts[b])[0] for a, b in zip(gts_at, gts_at[1:])]
        splits = [normalize(tokens[a].text + tokens[b].text)[0] for a, b in zip(outs_at, outs_at[1:])]  # fmt: skip
        out_norm, gt_norm = [onorm[k][0] for k in outs_at], [gnorm[j][0] for j in gts_at]
        wild_at = [i for i, j in enumerate(gts_at) if j in wild]
        for outs, gts in align_words(out_norm, gt_norm, joins, splits, early, wild_at):
            if not outs or not gts:
                continue
            ks, js = [outs_at[i] for i in outs], [gts_at[j] for j in gts]
            o = "".join(onorm[k][0] for k in ks)
            marks = [tokens[k].marks[s] for k in ks for s in onorm[k][1]]
            target = "".join(gnorm[j][0] for j in js)
            far = Levenshtein.normalized_distance(o, target) > MAX_PAIR_DISTANCE
            if far and not wild.issuperset(js):
                continue
            for k in ks:
                used[k] = tuple(js)
            owner = [(j, p) for j in js for p in gnorm[j][1]]
            for a, b in align_chars(o, target):
                j, p = owner[b]
                got = hits.setdefault(j, {})
                got[p] = got.get(p, False) or marks[a]

    align(keep, [j for j in gkeep if j not in furniture])
    if furniture:  # PREREG, Scoring: the furniture's own pass, on the tokens still unaligned
        spare: set[str] = set()  # left over words' texts, for the leftover pass
        if leftovers:
            spare = {gnorm[j][0] for j in gkeep if j not in hits and j not in wild | furniture}
            spare -= {gnorm[j][0] for j in furniture}
        align([k for k in keep if not used[k] and onorm[k][0] not in spare],
              [j for j in gkeep if j in furniture])  # fmt: skip
    # PREREG, Scoring: a token left over that equals a word left over, both normalized, pairs with
    # it whatever the order, each in page order (a table read row by row against a ground truth
    # that lists a column first)
    left: dict[str, list[int]] = {}
    for j in sorted(gkeep, key=lambda j: j in unscored) if leftovers else ():  # stable
        if j not in hits and j not in wild:
            left.setdefault(gnorm[j][0], []).append(j)
    for k in keep:
        if not used[k] and (spare := left.get(onorm[k][0])):
            j = spare.pop(0)
            used[k] = (j,)
            got = hits.setdefault(j, {})
            for p, s in zip(gnorm[j][1], onorm[k][1]):  # equal texts: characters pair in order
                got[p] = got.get(p, False) or tokens[k].marks[s]
    return hits, used


# ---------------------------------------------------------------------------------- scoring

COUNTS = (
    "struck", "struck_live", "struck_marked", "struck_absent",
    "live", "live_live", "live_marked", "live_absent",
    "dropped", "inserted", "inserted_digit", "cer_num", "cer_den", "wer_num", "wer_den",
)  # fmt: skip


def sub_tokens(word: dict) -> list[tuple[int, int, bool]]:
    """A word split at its struck/live boundaries: (start, end, struck) runs."""
    mask = [any(s <= i < e for s, e in word["struck"]) for i in range(len(word["text"]))]
    out, i = [], 0
    for struck, run in itertools.groupby(mask):
        n = len(list(run))
        out.append((i, i + n, struck))
        i += n
    return out


def outcome(n_chars: int, n_aligned: int, n_marked: int) -> str:
    """Marked when at least half the characters align to marked output characters, absent when
    fewer than half align to any, live otherwise."""
    if 2 * n_marked >= n_chars:
        return "marked"
    return "absent" if 2 * n_aligned < n_chars else "live"


def score_page(
    page: dict,
    tokens: list[Token] | None,
    details: bool = False,
    subsets: dict[str, Collection[int]] | None = None,
    probabilities: bool = False,
) -> dict:
    """Counts for one arm on one page. `tokens` None, or none that normalize to anything, is a
    failed page, scored worst case: every struck sub-token live and every live one absent.
    `subsets` names sets of word indices whose sub-tokens are also counted apart (the cascade's
    words P-scan leaves uncertain). For an arm that gives probabilities of being struck
    (`probabilities`, or any token carrying one), `probs` lists [word, struck, probability] per
    scored word: whether it holds a struck sub-token, and the highest of its tokens'
    probabilities, as a word is struck when any part of it is, or None where none gives one, a
    failed page's words included (the calibration and its coverage). With `details`, also each
    sub-token's outcome, the output tokens aligned to its word, and the inserted tokens, for the
    audit."""
    words = page["words"]
    if tokens is not None and not any(normalize(t.text)[0] for t in tokens):
        tokens = None  # no usable output
    unread = unreadable(words)
    scored = [not (w["masked"] or w["furniture"] or j in unread) for j, w in enumerate(words)]
    wild = [j for j, w in enumerate(words) if w["masked"] and w["text"] == UNREAD]
    texts = [w["text"] for w in words]
    furniture = [j for j, w in enumerate(words) if w["furniture"]]
    unscored = {j for j, s in enumerate(scored) if not s}
    hits, used = (match(tokens, texts, wild=wild, leftovers=True, furniture=furniture,
                        unscored=unscored) if tokens is not None else ({}, {}))  # fmt: skip
    if tokens is not None:
        _furniture_out_of_order(words, tokens, used)
        _between_unreadable(used, unread, unscored, len(words))
    feeds: dict[int, list[int]] = {}  # per word, the output tokens aligned to it
    for k, js in used.items():
        for j in js:
            feeds.setdefault(j, []).append(k)
    counts = dict.fromkeys(COUNTS, 0)
    strata: dict[str, dict] = {}
    within = {name: set(js) for name, js in (subsets or {}).items()}
    apart: dict[str, dict] = {name: {} for name in within}
    with_p = probabilities or (tokens is not None and any(t.p_struck is not None for t in tokens))
    whole_page = ("N1",) if page.get("n1") else ()  # G0's strike-free pages, a stratum too
    rows, probs = [], []
    for j, w in enumerate(words):
        if not scored[j]:
            continue
        mapped, got = set(normalize(w["text"])[1]), hits.get(j, {})
        names = (*w.get("strata", ()), *whole_page)
        mine = [apart[name] for name, js in within.items() if j in js]
        holds_struck = None  # until one of the word's sub-tokens is scored
        for s, e, struck in sub_tokens(w):
            chars = [p for p in range(s, e) if p in mapped]
            if not chars:
                counts["dropped"] += 1
                continue
            if tokens is None:
                result = "live" if struck else "absent"
            else:
                n_marked = sum(got.get(p, False) for p in chars)
                result = outcome(len(chars), sum(p in got for p in chars), n_marked)
            holds_struck = bool(holds_struck) or struck
            kind = "struck" if struck else "live"
            for c in (counts, *(strata.setdefault(x, {}) for x in names), *mine):
                c[kind] = c.get(kind, 0) + 1
                c[f"{kind}_{result}"] = c.get(f"{kind}_{result}", 0) + 1
            if details:
                rows.append({"word": j, "span": [s, e], "struck": struck, "outcome": result,
                             "tokens": feeds.get(j, [])})  # fmt: skip
        ps = [tokens[k].p_struck for k in feeds.get(j, ())] if tokens is not None else []
        ps = [p for p in ps if p is not None]
        if with_p and holds_struck is not None:
            probs.append([j, int(holds_struck), round(max(ps), 4) if ps else None])

    inserted = [k for k, js in used.items() if not js]
    counts["inserted"] = len(inserted)
    counts["inserted_digit"] = sum(any(c.isdigit() for c in tokens[k].text) for k in inserted)
    gt_live, out_live = _live_text(words, scored, tokens, used)
    counts["cer_num"] = Levenshtein.distance(" ".join(out_live), " ".join(gt_live))
    counts["cer_den"] = len(" ".join(gt_live))
    counts["wer_num"] = Levenshtein.distance(out_live, gt_live)
    counts["wer_den"] = len(gt_live)

    result = {**counts, "failed": tokens is None}
    if strata:
        result["strata"] = strata
    if subsets is not None:
        result["subsets"] = apart
    if with_p:
        result["probs"] = probs
    if details:
        result["details"] = rows
        result["inserted_tokens"] = inserted
    return result


GARBLED = ("ETHIOPIC", "ORIYA", "MALAYALAM", "TAMIL", "TELUGU", "KANNADA", "SINHALA", "BENGALI",
           "GURMUKHI", "GUJARATI", "DEVANAGARI")  # fmt: skip


def garbled(text: str) -> bool:
    """A word a PDF's text layer garbles: an unassigned code point, or a character of the Ethiopic
    or an Indic script, which is what a math font's glyphs mapped to the wrong Unicode give (G1's
    equations: `⋅ሺ0.955ሻെ` for `·(0.955) −`). No document in the corpus is written in them."""
    return any(unicodedata.category(c) == "Cn" or unicodedata.name(c, "").startswith(GARBLED)
               for c in text)  # fmt: skip


def unreadable(words: list[dict]) -> set[int]:
    """PREREG, Scoring: the words on a printed line holding a garbled word (each word's own, when
    it carries no line), unscored as masked words are: what the text layer says there is not what
    the page shows."""
    lines = {tuple(w["line"]) for w in words if "line" in w and garbled(w["text"])}
    return {j for j, w in enumerate(words)
            if (tuple(w["line"]) in lines if "line" in w else garbled(w["text"]))}  # fmt: skip


def _between_unreadable(used: dict, unread: set[int], unscored: set[int], n: int) -> None:
    """An unaligned output token is taken as the reading of unreadable words, not as inserted,
    where no scored word lies between its nearest aligned tokens' words (the page's edge standing
    in for a side without one), and either an unreadable word does or both neighbours' words are
    unreadable: nothing on those lines can check it, while a readable word between stays a miss."""
    if not unread:
        return
    order = sorted(used)
    aligned = [k for k in order if used[k]]
    for k in (k for k in order if not used[k]):
        before = [used[i] for i in aligned if i < k][-1:]
        after = [used[i] for i in aligned if i > k][:1]
        if before and after and max(before[0]) >= min(after[0]):
            continue  # neighbours aligned out of order: what lies between them is unknown
        gap = range(max(before[0]) + 1 if before else 0, min(after[0]) if after else n)
        if not unscored.issuperset(gap):
            continue
        if inside := [j for j in gap if j in unread]:
            used[k] = (inside[0],)
        elif before and after and set(before[0]) | set(after[0]) <= unread:
            used[k] = before[0]


def _furniture_out_of_order(words: list[dict], tokens: list[Token], used: dict) -> None:
    """PREREG, Scoring: an inserted output token equal, once normalized, to a furniture word that
    no equal token aligned to aligns to it, each such word taking one token, in page order. So
    furniture typed outside the ground truth's reading order (margin line numbers beside their
    lines, a running foot at the top) is ignored as furniture aligned in order is, even where the
    alignment paired a furniture word with another number (`10` with `1`)."""
    text = [normalize(w["text"])[0] for w in words]
    taken = {j for k, js in used.items() for j in js if text[j] == normalize(tokens[k].text)[0]}
    free: dict[str, list[int]] = {}
    for j, w in enumerate(words):
        if w["furniture"] and j not in taken and text[j]:
            free.setdefault(text[j], []).append(j)
    for k, js in used.items():
        if not js and (spare := free.get(normalize(tokens[k].text)[0])):
            used[k] = (spare.pop(0),)


CASCADE_GRID = ((0.05, 0.95), (0.3, 0.7), (0.4, 0.6))  # the cascade's bands beside the shipped one


def uncertain(page: dict, tokens: list[Token] | None, lo: float, hi: float) -> set[int]:
    """The scored words the package leaves uncertain (PREREG, Scoring, the cascade): those with a
    token of `tokens` (P-scan's) aligned to them whose probability lies strictly between `lo` and
    `hi`. At either bound the package has decided: struck from `hi`, clean to `lo`."""
    return uncertain_bands(page, tokens, {"band": (lo, hi)})["band"]


def uncertain_bands(page: dict, tokens: list[Token] | None,
                    bands: dict[str, tuple[float, float]]) -> dict[str, set[int]]:  # fmt: skip
    """`uncertain` for several named bands, from one alignment."""
    rows = score_page(page, tokens, details=True)["details"]

    def inside(k: int, lo: float, hi: float) -> bool:
        p = tokens[k].p_struck
        return p is not None and lo < p < hi

    return {name: {r["word"] for r in rows if any(inside(k, lo, hi) for k in r["tokens"])}
            for name, (lo, hi) in bands.items()}  # fmt: skip


def _live_text(words, scored, tokens, used) -> tuple[list[str], list[str]]:
    """The live text on each side as normalized units, for CER and WER. Words one output token
    joined count as one unit, and so do tokens split from one word; such a unit is normalized as a
    whole and compared without hyphens, as the alignment compared it. Output tokens aligned only
    to masked or furniture words are left out."""

    def live(text: str, marked) -> str:
        return "".join(c for c, m in zip(text, marked) if not m)

    lead = {js[1]: js[0] for js in used.values() if len(js) == 2}  # word j+1 joins word j
    shared = Counter(used.values())
    joined = {lead.get(j, j) for js in used.values() for j in js if len(js) == 2 or shared[js] > 1}
    gt_units: dict[int, str] = {}
    for j, w in enumerate(words):
        if scored[j]:
            mask = [any(s <= i < e for s, e in w["struck"]) for i in range(len(w["text"]))]
            gt_units[lead.get(j, j)] = gt_units.get(lead.get(j, j), "") + live(w["text"], mask)
    out_units: dict[tuple, str] = {}
    out_joined: set[tuple] = set()
    for k, js in used.items():
        if js and not any(scored[j] for j in js):
            continue
        split = len(js) == 1 and shared[js] > 1
        unit = ("split", js) if split else ("token", k)
        if split or len(js) == 2:
            out_joined.add(unit)
        out_units[unit] = out_units.get(unit, "") + live(tokens[k].text, tokens[k].marks)

    def norm(text: str, hyphenless: bool) -> str:
        t = normalize(text)[0]
        return t.replace("-", "") if hyphenless else t

    gt = [norm(t, u in joined) for u, t in gt_units.items()]
    out = [norm(t, u in out_joined) for u, t in out_units.items()]
    return [t for t in gt if t], [t for t in out if t]
