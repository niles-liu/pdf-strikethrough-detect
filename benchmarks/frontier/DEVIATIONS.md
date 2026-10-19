# Deviations

Each change to a rule of [PREREG.md](PREREG.md) after `prereg-1`: the date, what changed, why,
and the results both ways (PREREG.md, Operations). The first three came out of stage B2's pilot,
on dev pages, before any test page was touched; the fourth out of the test run's reports, the fifth
out of its audit, and the sixth records audit answers changed after labelling. A run's folder,
`runs/<run>/`, is published as `frontier/<run>/` in the Hugging Face dataset
`niles-liu/strikethrough-benchmark`, without its logs and batch records and with two account IDs
scrubbed, as the dataset's card says.

## 1. How a page is scored

**2026-10-17, from triaging the pilot and the author's audit of it.** PREREG.md, Scoring and G2's
Furniture; `bench/score.py`, `bench/g2.py`.

**What changed.**
- **Words out of the ground truth's order.** After the in-order alignment, a token left over that
  equals a word left over, both normalized, pairs with it whatever the order, scored words first.
  A token equal to a furniture word no equal token took is ignored, as aligned furniture is.
- **Furniture's own pass.** Furniture aligns after the page's other words, leaving out a token
  equal to a word left over and to no furniture word, so a margin line number no longer keeps the
  two halves of a hyphenated word apart.
- **LaTeX.** A command for a printed symbol reads as the symbol (`\cdot` as `·`, `\Delta` as `Δ`);
  font, sizing and fraction commands and environments are markup; an operator (`\max`) reads as
  its name, and any other command stays text. Braces break words.
- **Garbled lines.** A line whose text layer is garbled (a math font mapping its glyphs to the
  wrong Unicode) is not scored, and an output token reading it, with nothing scored between, is
  not counted as inserted.
- **GPO's seal.** The authentication seal on each bill's first page is G2 furniture, with its five
  words. It is drawn, not text, so the ground truth had lacked them.

G1's ground truth and every page list and sample are unchanged (`g1 check`, and the dev pages
redraw as `config/prereg1.json` lists them). G2's gains the seal's 130 words on its 26 first pages and is
otherwise identical (`g2 check`).

**Why.** Each rule as written charged a reader for reading the page right, and P-native, which
reads the PDF's own words in the ground truth's order, met none of them. So they biased inserted
tokens, over-deletion and CER against every arm that reads the page:
- the bills' margin line numbers, typed beside their lines rather than in the ground truth's order
  (146 of G2-dev's 213 inserted tokens), which also split hyphenated words: a struck "Aviation"
  scored as one half marked and the other absent, 13 audit items;
- a G1 table whose ground truth lists its row numbers first, read row by row (75 to 90 tokens per
  arm), and a running foot;
- the models' LaTeX for G1's equations, counted as words;
- a math font's wrong Unicode on 70 of G1-dev's 477 pages (`⋅ሺ0.955ሻെ` for `·(0.955) −` on
  `epa-hd2027-nprm-reg-redline.p300`), which no reader could match: 15 of the 21 audit items the
  author answered "the scorer read it wrong". No test page has it;
- the seal, read by most image readers, on 23 of G2's 51 test pages.

Of the audit's 21 such items and the 13 hyphen items, 29 are no longer disagreements. Left as
they are: two equation pieces on p300 (a subscript's `,999`, and `$$v`), two of B-pm4llm's
fragments of words drawn with combining strokes (stratum E6, a declared limitation), and the one
ground-truth error, which goes through G1's error-rate sample and relabel. The audit was sampled
again after the first of these changes, and labels begun on the earlier sample were set aside.

**Results both ways.** The pilot: dev pages, the models at R2 on C2, under prereg-1's scoring →
now; P-scan is the call as frozen (now P-scan-shipped). Every row that changed; `=` is unchanged.
Leakage rose where a struck word typed out of order as live text had been scored absent: it is a
leak.

| Arm | Set | Leakage | Over-deletion (absent) | Unmarked | F1 | Inserted per 1,000 | CER |
|---|---|---|---|---|---|---|---|
| B-pm4llm | G0 | = | = | = | = | 23.0 → 19.6 | 0.096 → 0.089 |
| B-pm4llm | G1 | = | 2.2% → 0.3% | 3.1% → 1.5% | 0.983 → 0.990 | 32.2 → 7.1 | 0.033 → 0.026 |
| B-pm4llm | G2 | = | = | = | = | 52.5 → 3.2 | 0.034 → 0.001 |
| P-scan | G0 | 27.1% → 27.2% | = | = | 0.759 → 0.758 | 61.3 → 60.9 | = |
| P-scan | G1 | = | 3.3% → 1.5% | 16.4% → 13.0% | 0.751 → 0.770 | 36.0 → 14.4 | 0.065 → 0.058 |
| P-scan | G2 | = | = | = | = | 106.7 → 44.6 | 0.149 → 0.074 |
| Gemini 3.8 Flash | G1 | 25.2% → 25.6% | 2.0% → 0.1% | 29.8% → 25.6% | 0.676 → 0.703 | 30.3 → 3.3 | 0.074 → 0.054 |
| Gemini 3.8 Flash | G2 | = | = | = | = | 57.3 → 0.0 | 0.057 → 0.001 |
| GPT-6 Sol | G1 | = | 2.8% → 0.1% | 4.6% → 0.0% | 0.704 → 0.726 | 27.1 → 2.4 | 0.081 → 0.060 |
| GPT-6 Sol | G2 | = | = | 2.9% → 0.0% | 0.985 → 1.000 | 47.8 → 0.0 | 0.067 → 0.001 |
| Grok 4.3 | G1 | 48.1% → 51.5% | 1.9% → 0.3% | = | = | 20.5 → 0.9 | 0.060 → 0.053 |
| Grok 4.3 | G2 | 97.1% → 100% | = | = | = | 31.8 → 0.0 | 0.866 → 0.845 |
| Sonnet 5.5 | G1 | = | 1.7% → 0.0% | 3.4% → 0.0% | 0.983 → 1.000 | 26.7 → 2.1 | 0.031 → 0.018 |
| Sonnet 5.5 | G2 | = | 0.3% → 0.0% | 1.8% → 0.0% | 0.991 → 1.000 | 43.0 → 0.0 | 0.031 → 0.001 |

On G2 the struck halves of hyphenated words now count: marked for the readers that marked the
word, leaked for Grok 4.3, which passes struck text as live.

## 2. P-scan's OCR setting

**2026-10-17, while triaging the pilot.** PREREG.md, Arms and inputs; `bench/arms.py`.

**What changed.**
- P-scan calls `rapidocr_backend(params={"Global.use_cls": False})`, the package's own backend
  with RapidOCR's text-line orientation classifier off; everything else is the call as frozen.
- A page the call refuses is a failed page, scored worst case. With the classifier off, RapidOCR
  can return a word box past the page's edge, which the package's word check rejects: 1 of the
  504 dev pages (G1-dev `p111`), none of the pilot's.
- The call as released stays as an arm, **P-scan-shipped**, reported beside P-scan and in no
  endpoint or route. Each record carries its OCR settings; one without them is P-scan-shipped's.
  The pilot audit samples P-scan.

**Why.** P-scan's output held runs of stray letters ("r t t n r") where whole lines should be.
RapidOCR ships with the classifier on, and it reads a line carrying a strike or an underline as
upside down, so it turns the line over before reading it: the result is letters, or a line scored
below RapidOCR's cut-off and dropped. The package passes RapidOCR no settings, so its documented
call inherits this. It is a pipeline setting, not the strike detector: the CNN and every threshold
are as frozen, and it was found on dev pages, as each model's settings are chosen there.

**Results both ways.** The pilot, P-scan-shipped → P-scan, both under deviation 1's scoring.

| Set | Leakage | Over-deletion (absent) | Unmarked | F1 | Strikes found | Strikes read | Found of read | Inserted per 1,000 | CER |
|---|---|---|---|---|---|---|---|---|---|
| G0 | 27.2% → 28.4% | 9.4% → 0.2% | 37.5% → 29.3% | 0.758 → 0.817 | 62.5% → 70.7% | 89.7% → 99.1% | 69.7% → 71.3% | 60.9 → 8.2 | 0.135 → 0.064 |
| G1 | = | 1.5% → 0.1% | = | 0.770 → 0.765 | = | = | = | 14.4 → 1.7 | 0.058 → 0.046 |
| G2 | 13.3% → 14.0% | = | 19.4% → 14.0% | 0.893 → 0.925 | 80.6% → 86.0% | 93.9% → 100% | 85.9% → 86.0% | 44.6 → 0.0 | 0.074 → 0.056 |

Leakage rose a little where struck words that had been lost are now read, some as live. Strikes
found of those read barely moved: the setting fixes the reading, not the finding. The CPU-seconds
per page read 10.5 → 19.2, but from runs on different days; timed together in one process, the two
calls cost the same (6.4 to 7.7 s on two G0 pages). The test stage's cost pass times both in one
pass.

## 3. What is reported: the strike columns and the routing rule

**2026-10-17, before any route was computed.** PREREG.md, Scoring and Routing; `bench/report.py`
(`report routes`, new).

**What changed.**
- **Strike columns,** exploratory, for every arm: strikes found, strikes read, strikes found of
  those read, and mark precision. On a failed page each of the first three takes its worst.
- **Misses** are the unmarked share on every page kind. They had been leakage on born-digital
  pages and images of print.
- **Guards.** The paid option is guarded on misses, leakage and both over-deletions, each apart,
  against a reference: the system with the fewest misses among those not shown to be more than a
  point worse than the free default on either over-deletion. It had been guarded on misses and
  marked over-deletion only, against the system with the fewest misses, which passed against
  itself.
- **Systems** are named (cells covering every test page of the kind); a paid system's failed page
  scores as the free default's output, while its failure rate beside the route is its own.
- **Cascades** are scored on whole pages with the same rates (PREREG's cascade table gains the
  unmarked share and absent over-deletion), and one that passes at no more than half the price
  replaces the paid option.
- Smaller edits: the paid option is the cheapest *paid* system; each route shows its inserted
  tokens beside it; a limitation names what three clusters or fewer can't show. The rule is code,
  hashed at `prereg-2`.

**Why.**
- **Finding against reading.** F1 and the unmarked share count a struck word the OCR lost as a
  miss. Strikes found of those read does not, and strikes read sits beside it, so dropping a word
  never helps.
- **Lost text cost nothing in routing.** A struck word lost entirely (P-scan's garbled lines, or a
  clean copy that leaves struck text out) and live text lost entirely went unguarded, though the
  app audits struck values.
- **The reference was never guarded.** A system marking every word struck had no misses, became
  the reference, and won if cheapest.
- **The rule was not yet code,** so its details would have been settled by hand after the results.
- **Rejected:** weighting a drop at half a leak (a new number); one summed over-deletion guard (a
  system marking 9% of live text passed); two other references (each left the paid option empty,
  or the mark-everything system winning, in a simulation); an inserted-token guard (B-pm4llm,
  which cannot invent a word, scored 20 per 1,000, so the count was then mostly scorer artefacts);
  a guard on the free default (each kind has one real free reader so far).

**Results both ways.** On the pilot every route is the same under both rules: P-native for
born-digital pages, with no paid system on C1; for images of print, P-scan as the free default and
Sonnet 5.5 as the paid option. The change moves only against the package: P-scan's misses on images
of print are 26.0% as the unmarked share, against 25.3% as leakage.

## 4. When a batched xAI request was sent

**2026-10-19, running the test run's reports.** PREREG.md, Operations (the test window);
`bench/report.py` (`report canary`).

**What changed.** The window check times a request by its own start only when that records a
time; a bare date falls back to the batch's submission, as a batched request with no start
already did. A time with no UTC offset is read as UTC, the harness's own clock.

**Why.** xAI's batch metadata gives a request's start as a date with no time (`2026-10-18`), so
all 941 of Grok 4.3's batched canary and test answers recorded one, and the report stopped when
it compared those dates with Grok 4.3's latency-sample times. Read as midnight, all of Grok 4.3's
batched sends, canaries included, would fall on one instant a day, and the check would say
nothing about that model.

**Results both ways.** As frozen, `report canary` stops with an error and reports nothing. With
the fix, no model drifts (every p is 0.25 or more), every model's window lasts 0.96 days, and no
test request falls outside its canaries; Grok 4.3's first test request went 64 minutes after its
first canary. The drift tests read no times.

## 5. What the test audit found in the scorer

**2026-10-19, from the test audit.** PREREG.md, Scoring and The audit; `bench/score.py`,
`bench/audit.py`, `bench/g1.py`.

**What changed.** The scorer only: G0, G1 and G2 rebuild identically (`g0 check`, `g1 check`,
`g2 check`).
- **Words in pieces.** Up to eight adjacent output tokens that equal one word joined align to it,
  and one token that equals up to eight adjacent words joined aligns to them all, as two did: a
  subscript tagged apart (`ER<sub>Cl</sub>2` for ERCl2), a struck letter typed as its own token
  (`~~L~~ l ist`, where "List" became "list"), a URL or an OCR run glued to its neighbours.
- **Symbols.** Every Unicode symbol class is stripped from a token's edges, as PREREG's
  normalization says; the code stripped only math, currency and modifier symbols. So the
  box-drawing lines and arrows a model draws a chart with (`│`, `▼`) are no longer inserted words,
  and a tick on the page (`✓`) is dropped from scoring, as a struck dash is.
- **A word struck in part.** A token may match its live characters alone: a clean copy's "ms" for
  "t̶e̶r̶ms", or "Agr" for "Agr̶e̶e̶m̶e̶n̶t̶".
- **Ties.** Among alignments of equal cost, a token goes to the equal word its marks agree with.
  In "breach of ~~confidentiality or of~~ the", a clean copy's "of" is the live one; the traceback
  had taken the later, struck one, a leak and a dropped word for a right reading.
- **Out of order.** Deviation 1's pass over the tokens and words left over pairs a passage typed
  out of order (three or more tokens in a row equal to words in a row, all left over: a footnote
  typed at the end) wherever it lies, but a lone token only with a word whose neighbours' tokens
  surround it (those aligned to the words up to 100 either side). It had paired a lone token with
  the first equal word anywhere, so the "the" and "and" of a model's own summary, before or after
  its copy, became leaks of struck words far down the page.
- **Masked words** absorb one token at a cost of at most 1.5, as G5's unreadable words do: a
  masked word whose text layer orders its letters otherwise than the page ("(ii2)" for the printed
  "(2ii)") no longer leaves its reading inserted.
- **LaTeX accents** (`\bar{c}`) read as the letter and its combining mark (c̄), as deviation 1's
  symbol commands read as their symbols.
- **The audit page** shows a page printed sideways upright, and for an extra word the whole page
  under a fold, since a crop can't show a chart, a photo or a heading drawn as a picture.
- **G1's key errors** the test audit found (question 2) join the relabel page, so their
  corrections reach the key as the relabel's do. A word struck in part is typed with its struck
  parts between tildes (`CO2~CO2~`), which also says which of two equal parts is struck; struck
  letters typed alone, which the page allows when they occur once in the word, are refused when
  they occur twice.

**Why.** The test audit (`runs/test/audit/`; 24 answers changed after a check of each item
against the whole page and output, each listed in `review.txt`). On question 3, 52 of its 160
items say the scorer read the output wrong. 32 are the errors above, each now scored as the
author read it, and none of the 108 the author called right changed: words in pieces 20,
symbols 4, live parts 3, ties 2, out of order 1, a masked word 1, a LaTeX accent 1. Of the other
20, 18 read text the key, built from the text layer, doesn't hold (a chart, two button images, two
headings drawn as pictures), which the results state as a limitation of inserted tokens rather
than a change to the key; one is a struck "(I)" read as "(1)", which the misread rule (a pair
farther apart than 0.5 is absent plus an inserted token) still scores absent; one is P-scan
reading "§" as "S" inside a token glued to its neighbours, which a join must equal exactly. On
question 2 the key was wrong on two G1 words: "CO2CO2", whose struck "CO2" is the second, and
"Eaffects", struck in two places.

**Results both ways.** The test stage, scored as frozen (`75574f6`, the `prereg-2` code) and with
this change. G1's
relabel, in the rescore on `c29474f`, moves none of the figures below.
- **Endpoints:** no claim changes its verdict. On G2, endpoint 2's p before the Holm adjustment
  falls from 0.0058 to 0.0001 for Opus 5.5 and from 0.0027 to 0.0001 for GPT-6 Astra; endpoint 4's
  for Grok 4.7 rises from 0.50 to 1.
- **Routes:** each page kind keeps its systems. For handwriting, P-scan's leakage falls from 31.6%
  to 29.5% and Gemini 4 Argon's (R5, C2) from 6.0% to 5.6%, its misses from 17.8% to 17.6%, while
  their inserted tokens per 1,000 rise from 160.9 to 163.3 and from 6.4 to 6.6: the tokens no
  longer paired with a struck word far away. Opus 5.5's inserted tokens on born-digital pages fall
  from 0.11 to 0, and P-scan's on images of print from 0.83 to 0.71.
- **G1's relabel:** as frozen, two words three flagships dispute, one of them the split
  "(aRELHCl/aRELCl2)"; with the change, one, beside the audit's two key errors.
- **Cells,** every one whose leakage, unmarked share or either over-deletion moves by half a point
  or more; `=` is unchanged. On G5 the leaks taken away are lone common words and letters ("a",
  "e", "se") paired with an equal struck word elsewhere on the page: from a model's own preamble
  and notes, in Portuguese ("Aqui está a transcrição…", "O autor riscou…"), or from a garbled
  reading of the handwriting. On G0 and G1, Gemini 3.8 Flash's clean copies list the struck text
  after the copy, and only its passages of three words or more still count.

| Arm | Set | Leakage | Unmarked | Over-deletion (marked) | Over-deletion (absent) |
|---|---|---|---|---|---|
| Gemini 3.8 Flash R1 C2 | G0 | 42.4% → 28.8% | 46.5% → 48.0% | = | = |
| Opus 5.5 R1 C2 | G0 | 0.8% → 0.2% | = | = | 1.1% → 0.0% |
| Gemini 3.8 Flash R1 C2 | G1 | 22.1% → 18.8% | 29.0% → 28.5% | = | = |
| Gemini 4 Argon R1 C1 | G1 | 1.7% → 0.4% | = | = | = |
| Gemini 4 Argon R1 C2 | G1 | 1.6% → 0.3% | = | = | = |
| GPT-6 Astra R0 C2 | G1 | = | 0.9% → 0.3% | = | = |
| Haiku 5.5 R1 C2 | G1 | 3.4% → 2.4% | = | = | = |
| Opus 5.5 R0 C1 | G1 | = | 0.6% → 0.1% | = | = |
| Opus 5.5 R1 C2 | G1 | 3.4% → 2.1% | = | = | = |
| Gemini 4 Argon R1 C2, wording 1 | G2 | 4.2% → 3.6% | = | = | = |
| Gemini 4 Argon R1 C2, wording 3 | G2 | 8.0% → 6.1% | = | = | = |
| Gemini 4 Argon R1 C2, wording 1 | G2-strict | 10.5% → 7.7% | = | = | = |
| Gemini 4 Argon R1 C2, wording 3 | G2-strict | 11.0% → 8.3% | = | = | = |
| P-scan | G5 | 31.6% → 29.5% | = | = | = |
| P-scan-shipped | G5 | 31.8% → 29.5% | = | = | = |
| Gemini 3.8 Flash R0 C2 | G5 | 23.4% → 15.9% | = | = | = |
| Gemini 3.8 Flash R2 C2 | G5 | 11.2% → 10.3% | = | = | = |
| Gemini 3.8 Flash R2 C2-tiled | G5 | 10.4% → 9.5% | 26.0% → 25.4% | = | = |
| Gemini 4 Argon R0 C2 | G5 | 23.4% → 13.4% | 76.2% → 77.8% | = | = |
| GPT-6 Astra R0 C2 | G5 | 12.1% → 8.4% | = | = | = |
| GPT-6 Astra R2 C2 | G5 | 11.1% → 9.5% | = | = | = |
| GPT-6 Astra R2 C2-tiled | G5 | 11.5% → 10.3% | = | = | = |
| GPT-6 Sol R0 C2 | G5 | 22.2% → 18.0% | = | = | = |
| GPT-6 Sol R2 C2 | G5 | 14.9% → 12.0% | = | = | = |
| GPT-6 Sol R2 C2-tiled | G5 | 12.6% → 9.8% | = | = | = |
| Grok 4.3 R0 C2 | G5 | 46.0% → 41.8% | = | = | = |
| Grok 4.3 R2 C2 | G5 | 29.0% → 26.7% | 81.0% → 79.8% | = | = |
| Grok 4.7 R0 C2 | G5 | 36.4% → 33.1% | = | = | = |
| Grok 4.7 R2 C2 | G5 | 25.2% → 22.9% | = | = | = |
| Haiku 5.5 R0 C2 | G5 | 23.8% → 20.1% | = | = | = |
| Haiku 5.5 R2 C2 | G5 | 15.1% → 13.4% | = | = | = |
| Opus 5.5 R0 C2 | G5 | 7.1% → 5.4% | = | = | = |
| Opus 5.5 R2 C2 | G5 | 11.2% → 10.0% | 28.0% → 27.4% | = | = |
| Sonnet 5.5 R0 C2 | G5 | 17.6% → 15.1% | = | = | = |
| Sonnet 5.5 R2 C2 | G5 | 11.8% → 10.7% | = | = | = |

## 6. Audit answers changed after labelling

**2026-10-19, from the audits.** PREREG.md, The audit.

**What changed.**
- **G1's error-rate sample.** One answer, i055, a live word labelled "struck", was corrected to "not
  struck" as a misclick at 05:19 UTC, after every stage of the test run was scored (05:08 to
  05:16 UTC); PREREG has the sample labelled before any answer is scored. In `runs/g1-audit/`,
  `A.csv` is as labelled, and `A-corrected.csv` carries the fix, with `correction.txt`.
- **The test audit.** After labelling, the author checked each item against the whole page and the
  whole output, beyond the audit page PREREG names, and changed 24 answers: 3 blanks filled, 17
  answers to question 3 from no to yes, and 4 from yes to no. In `runs/test/audit/`, `A.csv` is as
  labelled, and `A-reviewed.csv` carries the changes, each listed in `review.txt`.

**Why.** The word i055 names is not struck on its page. The audit page shows a crop around each
word and an excerpt of the output: the crops did not show charts, photos or a model's own text,
and some excerpts were hard to read, so some answers judged what the page could not show.

**Results both ways.**
- **G1's error rate:** live words wrong in the key, 1 of 200 as labelled (0.5%, 95% interval 0.1%
  to 2.8%) and 0 of 200 corrected (up to 1.9%); struck words, 0 of 200 either way. Nothing else
  reads the sample.
- **The test audit:** question 3's "no" answers (the scorer read the output wrong) are 65 as
  labelled and 52 reviewed; question 2's (the key is wrong) are i077 and i106 either way, so the
  relabel is the same. Four of deviation 5's 32 fixes rest on answers the review changed to "no":
  the two ties (i048, i100) and two of the three live parts (i045, i060). PREREG has every "no"
  fixed at its source, but 20 of the 52 are not: the 18 words the key lacks, the misread the rule
  scores as written, and the token glued to its neighbours (deviation 5, Why).
