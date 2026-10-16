# Frontier-model benchmark: pre-registration

**Status: draft (stage B1), not yet pre-registered.** This file binds once tagged `prereg-1` (end
of stage B1), and again at `prereg-2` (end of stage B2), as [PLAN.md](PLAN.md#pre-registration)
describes; until `prereg-1` any value can change. `TBD (B1)` and `TBD (B2)` mark what that stage
fills in before its tag. A change after a tag is a deviation, logged in `DEVIATIONS.md` with the
results shown both ways.

**How to read it.** [PLAN.md](PLAN.md) gives the reasons and this file the rules; where they
differ, this file governs. **Part A** is the protocol. **Part B** is the build guide for stages B1
to B3; changing it is not a deviation unless the change alters a rule in Part A. The names (Q1–Q7,
G0–G5, the arms, C1–C4, R0–R6, stages B0–B7) are PLAN.md's.

# Part A: protocol

## Models

Exact IDs are pinned at stage B1 against each provider's live model list, and the per-model
settings at stage B2. A model released after `prereg-1` is an extra arm, never a replacement. A
pinned model retired before its test requests run is dropped and reported as dropped; nothing takes
its place in a primary endpoint.

- **Anthropic** (Messages API, Message Batches): Claude Opus 5.5 `claude-opus-5-5`, Claude
  Sonnet 5.5 `claude-sonnet-5-5` and Claude Haiku 5.5 `claude-haiku-5-5`, plus Claude Fable 5.1
  `claude-fable-5-1`.
- **Google** on Vertex AI (`generateContent` at location `global`, Vertex batch prediction in the
  `us` multi-region, as the project's organization allows no batch job at `global`): Gemini
  4 Argon `gemini-4-argon` and Gemini 3.8 Flash `gemini-3.8-flash`.
- **OpenAI** through Azure OpenAI (Responses API, Flex tier): GPT-6 Astra `gpt-6-astra` and
  GPT-6 Sol `gpt-6-sol`, each pinned to the dated version its Azure deployment serves (Azure lists
  `gpt-6-astra-2026-09-03` and `gpt-6-sol-2026-09-22`), on Global Standard deployments with
  automatic version upgrades off.
- **xAI** (Chat Completions for images, Responses for C1, Batch API for Grok 4.3 only): Grok 4.7
  `grok-4.7` and Grok 4.3 `grok-4.3`. If Grok 4.3 fails a stage B1 smoke test (images in a batch,
  or a PDF attachment), `grok-4.20-0309-reasoning` replaces it before `prereg-1`.
- **Stage B7's open-weights and OCR-to-markdown models** (PLAN.md, Arms) are fixed by neither tag
  and never enter a primary endpoint. Before they run, an addendum, logged like a deviation, fixes
  their IDs, hosts, settings and cells. Each open-weights model chooses its R-best and image
  condition on the dev pages by the rule under [Prompts](#prompts), then runs the test cells the
  other models ran; its G2 results say whether the bills postdate its knowledge cutoff.

Every response records what the provider reports about the version (Azure's served model,
Gemini's `modelVersion`, `system_fingerprint` where it exists), since the Grok and Gemini IDs are
undated and can change in place; the canary (see [Operations](#operations)) checks for drift, and
Operations says what happens when a version changes mid-run.

- **Flagships,** the models the primary endpoints test: Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra
  and Grok 4.7. The other models are reported in the same tables, as exploratory results.
- **Ladder models** choose R-best and the image condition for their provider (see
  [Prompts](#prompts)): Claude Sonnet 5.5, Gemini 3.8 Flash, GPT-6 Sol and Grok 4.3. Each flagship
  then keeps R-best or the runner-up rung, by the same rule.
- **Fable 5.1** runs on a ceiling subset of 50 test pages of G0–G2, stratified by set and drawn
  with seed `20261018`, at R0, R1 and Anthropic's R-best, on C2.

**Settings.** One row per model, fixed in `prereg-2` (TBD (B2)) and recorded with every
response:

| Model | Effort | Image | Discounted route |
|---|---|---|---|
| Opus, Sonnet and Haiku 5.5; Fable 5.1 | `effort: high` | as sent | Message Batches |
| Gemini 4 Argon and 3.8 Flash | `thinking_level: high` | `ultra_high` (see below) | Vertex batch |
| GPT-6 Astra and Sol | `reasoning.effort: high` | `detail: original` | Azure's Flex tier |
| Grok 4.7 | `reasoning_effort: high` | `detail: high` | none: sequential |
| Grok 4.3 | `reasoning_effort: high` | `detail: high` | Batch API |

- Images: a Gemini model's image parts take `media_resolution: ultra_high` if stage B1 shows
  Vertex batch prediction accepts it for that model, else `high` (on Gemini 3, 2,240 and 1,120
  tokens an image). GPT-6 Sol takes `detail: high` if stage B1 shows it refuses `original`.
- Effort is set explicitly at each model's documented `high` level, or its nearest equivalent. The
  effort ablation runs R-best at the `low` value of the same setting on the repeats' 20 dev pages,
  each page's request sent with the repeats' three at `high`. It is read per ladder model as its
  leakage, unmarked share, over-deletion, F1 and cost minus those of the three runs at `high`,
  pooled, beside the three runs' range; for leakage, unmarked share and marked over-deletion,
  with the canary's relabeling test (see [Operations](#operations)), low in the end run's place;
  and with each run's failures by kind, then again without the pages any run failed. Low's own
  run-to-run spread is not measured.
- Sampling parameters (temperature, top-p) are never sent: each model runs at its default.
- The output limit is 64,000 tokens, or the model's own maximum where that is lower (the smoke test
  shows it), thinking included everywhere except xAI, whose limit excludes reasoning tokens. A
  model that stops at the limit scores the page as a failure.
- Region and data retention stay at the account's defaults, stated in the write-up. GPT-6's
  deployments use Azure's content filter at its lowest blocking setting; a response it blocks
  counts as a refusal.

## Primary endpoints and decisions

Four primary endpoints, all tested on G2-test alone; everything else is exploratory. All are
paired comparisons on the same pages. "Points" are percentage points. Every arm's metrics, cost
and seconds per page are also reported per set and input, and the app routes pages from those
tables by the default under [Routing](#routing-an-engineering-default), which is not one of this
file's claims.

**The tested set.** G2's ground truth is independent of the package, and its bill versions were
published after every model's training cutoff, so G2 alone carries the tests. The other sets are
reported beside it, untested: G0 per stratum and pooled with its intervals, since its strata are
quotas fixed by design, two of them limitations the package declares; G1, whose ground truth is the
package's own detectors, and G2-strict, six bills, per cluster beside the pooled difference, with no
interval. G2-strict's p-values are shown, outside the Holm families.

1. **Default leakage (Q1).** Leakage at R0, per flagship, on C1 and on C2, against P-native: the
   package wins against a flagship when model minus package leakage lies above 0 on both.
2. **The one-line prompt (Q6).** At R1, per flagship, on the C2 pages, in each of R1's four
   wordings (its base wording and its three paraphrases): the model is no worse than the package
   when, in every wording, (i) model minus P-native leakage lies below +1 point, (ii) model minus
   P-native marked over-deletion lies below +1 point, and (iii) the model's absent over-deletion
   at R1 minus its own at R0 lies below +1 point, so that a model obeying R1 by leaving text out
   does not pass. One point is R-best's own tolerance (see [Prompts](#prompts)); on G2 it is about
   20 struck sub-tokens, roughly one struck phrase, so "no worse" means missing no passage the
   package catches. Each wording's differences are reported beside the claim, with intervals.
3. **Detection at its best (Q2).** Struck-word F1 at the flagship's R-best, on its provider's
   chosen image condition, against P-scan. Either side wins when the difference lies on its side
   of 0. P-native's F1 is reported beside it.
4. **Input route (Q3).** Leakage at R0, C1 minus C2, per flagship: C1 leaks more when the
   difference lies above 0. Where a provider's PDF route passes the model the text layer alone
   (`c1_route = "text"` in `config/models.toml`), the claim is tagged "by construction": that text
   still holds every struck word.

**The decision rule.** Each claim above is one or more one-sided hypotheses about a paired
difference, each with an exact p-value from a sign-flip test over clusters (see
[Statistics](#statistics)). A claim needing several parts to hold (both inputs in endpoint 1, three
bounds in each of four wordings in endpoint 2) takes the largest of their p-values. Each claim is
tested twice, with failed pages scored worst case (see [Scoring](#scoring)) and with them left out
of both sides, and takes the larger p-value: worst-case scoring keeps a model from gaining by
failing hard pages, and the second analysis keeps a model's failures from making package wins.
Each arm's failed pages, by kind, are reported beside each claim. Within each endpoint the
p-values are Holm-adjusted across the flagships (endpoint 3 counts each direction as its own
hypothesis), and a claim holds when its adjusted p-value is below 0.025. Each endpoint so holds
its family-wise error at 0.025, one-sided, with no correction across the four, which answer
separate questions. Reported intervals are unadjusted 95% bootstrap intervals.

**Insufficient data.** A claim whose tests, in either analysis, count fewer than 200 struck
sub-tokens or fewer than 6 clusters reads "insufficient data" and leaves its family before the Holm
adjustment.

**The package's README sentence.** The package's README reports the run in one sentence of this
form, every group named and an empty one left out, with a link to the full tables: "On {n} US
Senate bills published after the tested models' training cutoffs, asked only to transcribe a page,
{models} left more struck text live than pdf-strikethrough-detect 0.12.1 [endpoint 1 holds],
{models} left less [model minus package leakage has its 95% interval below 0 on C1 and on C2], and
{models} were not shown to differ; told in one line never to present struck text as current,
{models} were no worse than the package [endpoint 2 holds] and {models} were not shown to be. On
the {s} bills introduced after the cutoff: {the same differences, per bill}." Endpoint 1 alone is
the package's favourable test, so the sentence never quotes it without endpoint 2.

## Routing: an engineering default

The app sends each page to a system chosen from the results tables. This rule is an engineering
default, dated by the tag; it is not one of this file's claims, and the app may override it, with
a logged reason.

- **Page kinds:** born-digital pages are G0 and G2 on C1 or C1′, with P-native; images of print
  are G0 and G2 on C2 or C2-tiled, with P-scan; handwriting is G5; logbooks are G4.
- **Misses:** leakage on born-digital pages and images of print; on handwriting and logbooks, the
  unmarked share (struck sub-tokens left live or absent), since a system that cannot read a struck
  word leaves it absent rather than live.
- **Routes:** the free default is the free system with the fewest misses. The paid option is the
  cheapest system not shown to be more than 1 point worse than the one with the fewest misses (its
  difference's paired cluster-bootstrap 95% interval reaching below +1 point) on misses and on
  marked over-deletion, ties going to the faster.
- **Cascades:** a cascade (P-scan, then a model on the words P-scan leaves uncertain; see
  [Scoring](#scoring)) replaces a route when it passes the same test at no more than half the
  route's cost.
- **Reporting:** a route's accuracy is reported "as selected": it was chosen on the same pages.

## Ground truth

### G0 synthetic redlines

A seeded generator, `bench/g0.py`.

- **Text:** contract-style clauses (parties, dates, amounts, rates, obligations) from a seeded
  template grammar with random fillers; no LLM writes it.
- **Pages:** US Letter, 1-inch margins, one column, 9–13 pt body text, a running head and a page
  number (furniture), and some ruled tables; about 350 words a page.
- **Struck strata.** Each struck word has exactly one encoding:
  E1 a stroked vector line through the middle of the x-height, E2 a thin filled rectangle, E3 a
  `/StrikeOut` annotation with an appearance stream, E4 an ink annotation, E5 a line slanted over
  2° that rises more than 1.5 pt end to end (a declared limitation), E6 U+0336 combining strokes in
  the text (a declared limitation). Styles, on E1 unless noted: S1 single (the default), S2
  double, S3 coloured (red or blue), S4 thick (2 pt or more), S5 partial-word, S6 a passage
  running over a line break, S7 a struck number beside its written correction (`~~5%~~ 3%`).
  Fonts rotate through Helvetica, Times and Courier across every stratum.
- **Negatives:** N1 strike-free pages; N2 underlined words, including Courier with the underline
  at the baseline (a declared limitation); N3 table rules running close to text.
- **Quotas,** in G0-test: at least 220 struck sub-tokens in each of E1–E6 and S2–S7 (S2–S7 words
  also count toward E1), at least 1,500 live sub-tokens under N2 or beside N3, and 8 N1 pages.
  Each page is drawn independently of the others, and the generator packs strata onto pages until
  every quota is met; G0-dev holds 40% of each quota.
- **QA items:** for each S7 correction the generator also writes a question whose answer the
  deletion changed, with its live and struck answers, for the QA probe.
- **Seeds:** dev `20261008`, test `20261009`. The held-back set's seed is a random 128-bit number
  drawn at stage B1 and kept private; only the sha256 of its decimal string is published:
  `031ad201d07a0c38a4d01b3563a2e19a0d198cd823da3162cd72ddef4a0d0e38`.
- **Ground truth** is written by the generator from what it draws: each word's text, box and
  struck character spans. Each test PDF's sha256 is tagged in `prereg-1` (`config/prereg1.json`),
  with the generator's commit, and each test page's ground-truth record's sha256 in `prereg-2`, as
  for every set.

### G1 public redline corpus

- **Documents:** the nine in [`corpus_manifest.json`](corpus_manifest.json) left once its
  `scanned_pages` are removed (the Copyright Office document leaves entirely).
- **Ground truth:** per page, the package's `vector` and `flag` detectors (0.12.1, as frozen) run
  separately, and their records are matched to PyMuPDF's `page.get_text("words")` as the package
  matches them: by box and text, else by text with the record's centre inside the word's box. A
  word both report is struck, with the vector detector's character span; a word only one reports
  is masked for every arm and counted; every other word is live. Words keep PyMuPDF's extraction
  order.
- **Furniture,** found per document. A line within 15% of the top or bottom edge is furniture when
  its text, digits ignored, recurs at the same height (within 1% of the page) on at least a fifth
  of the document's pages, and on at least two; a line with no letters must also recur at the same
  horizontal position. A line with letters is also furniture when it is like one of those at its
  height (Indel similarity at least 0.6). And a line holding only a number, within 15% of any edge,
  is furniture when the number is the page's own: its index plus the offset most of the document's
  page numbers show.
- **Page kinds,** from the words left once furniture is removed: a struck page holds at least one
  struck word; a strike-free page has no report from either detector; a page whose only reports
  are masked words is neither, and is not sampled.
- **Dev documents:** one generator, `random.Random(20261010)`, first draws two of the nine with
  `.sample(sorted(files), 2)`, where `files` are the manifest's file names, and then each test
  document's pages, in manifest order. The other seven documents are G1-test.
- **Pages,** per test document, from that generator: up to 10 struck pages and up to 3
  strike-free ones. Struck pages fall into six cells: the document's tertiles of struck words per
  page (ranked by count, ties by page index, split as evenly as the count allows) crossed with
  whether the page holds a partial-word strike, one that leaves some of a word's scored characters
  live (a strike stopping short of edge punctuation is a whole-word one). Each cell's share of the
  10 is proportional to its size, rounded by largest remainder (with more than 10 pages, no share
  exceeds its cell). A document with 10 or fewer struck pages gives them all. The dev documents,
  the page list and every document's sha256 are tagged in `prereg-1`, in `config/prereg1.json`.
- **Its error rate** is measured on a sample of G1-test words (see [The audit](#the-audit)).

### G2 US Congress bills

- **Source:** GovInfo's `BILLS` collection (bulk data and package links, no API key), which
  publishes each bill version's XML and PDF. G2 uses Senate reported versions (`rs`) only: they
  mark struck phrases with `<deleted-phrase>` and wholly struck sections or titles with
  `changed="deleted"`, and the PDF prints that text struck through.
- **Selection:** reported versions carrying struck text and published after the cutoff date: the
  latest training cutoff any model in the run states (its release date where it states none; a
  cutoff given as a month is that month's last day): 2026-06-30 for the IDs above, pinned in
  stage B1, from the Claude models' June 2026 (each model's cutoff and its source are in
  `config/models.toml`). Bills introduced after the cutoff come first, then the most recently
  introduced others, until the slice is full; every result is also shown on the strict subset,
  the bills introduced after the cutoff (the set `G2-strict`, outside the Holm families). A House
  bill's Senate version opens with its receipt in the Senate, so its introduction date is unknown:
  it is never in the strict subset, and comes after every dated bill. Ties go by package name,
  the latest first.
- **Size:** 45 bills with a usable page, in that order, the bill with the most inline strikes
  (`<deleted-phrase>` elements) taking the last place if it is not among the first 44. Per bill,
  one page carrying struck text: the first with at least 10% of its scored words live, else the
  first; and for a strict bill, also its first page without. The three bills that follow in the
  same order, each with its struck page and its first page without, are G2-dev; no later slice
  takes them. The bill lists, each bill's PDF and XML sha256s and the cutoff date are tagged in
  `prereg-1`, in `config/prereg1.json`.
- **Alignment:** the XML's words in document order, each flagged struck or not, with what GPO's
  stylesheet prints added (the cover's official title, the enacting clause, and labels such as
  "SEC. 2." and "TITLE I"), are aligned a page at a time to the PDF's body words (PyMuPDF,
  extraction order, furniture removed, split at em dashes) with the scorer's edit-distance
  alignment. Equal-cost ties go to the earliest copy, and a word hyphenated across a page break
  aligns on both pages. A PDF word aligned to a struck XML word is struck, to its edge punctuation
  where every aligned character is. Coverage per page is the share of body words aligned; pages
  under 98%, and pages with no body words, are dropped. Stage B1 holds every kept page to the
  strikes GPO drew: each letter and digit is struck in the ground truth exactly when a drawn
  horizontal line crosses it. Before `prereg-2` the author checks 100 aligned spans of G2-test by
  eye. A span is a run of consecutive words on one printed line aligned as struck, shown with the
  live word just before or after it in reading order where there is one (words that normalize to
  nothing carry no label and are passed over); every span at such a boundary is drawn (up to 100),
  and the rest at random with seed `20261021`. A bad span fixes the rule and every G2 page is
  rebuilt, so the rule is tagged in `prereg-2`: TBD (B2).
- **Furniture,** found by position and rotation: GPO's typesetting slug (`VerDate … Jkt …`), the
  rotated margin line, the running foot (such as `•S 3923 RS`), margin line numbers (a number alone
  on its line, left of the text), running heads and page numbers. Stage B1 checks the rule on two
  bills.

### G3 and G4

Exploratory, each specified in an addendum before it runs, logged like a deviation.
- **G3:** crop level ("is this word struck?") on the HWG dataset's `written` and `synthetic`
  parts and the ASAP and GoBo crops of `collected`, by strike type: StrikeNet as shipped against
  the models. It runs once the crop input built for G4 exists.
- **G4:** logbook cells from NARA logbook pages the project holds out of all its training and
  tuning: about 10 G4-dev pages, on which the prompt is tuned, and about 50 G4-test pages holding
  about 100 struck cells. The pages are chosen by a person (never by an arm) with a quota of pages
  holding struck cells, and labelled by hand on the dry-bulb column with no arm's output in view:
  each cell's row, live value, struck value, correction and strike type. The unit is one dry-bulb
  cell, sent as a crop of its labelled cell box padded to the rows above and below and to the
  wet-bulb neighbour, the target cell boxed; one JSON prompt asks whether it is struck, its live
  and struck values, the strike type and a confidence. Measured per cell: missed strikes, false
  flags, value match, strike type, calibration, and $ and seconds per cell, on every cell and on
  the cells the package flags or leaves uncertain. Arms: the flagships, the ladder models and the
  package. It runs as soon as its labels exist.
- **Held out from training.** No model that later runs here as an arm (a retrained package, see
  [Arms and inputs](#arms-and-inputs)) is trained or tuned on G0's dev, test or held-back seeds
  (the generator with other seeds is fine), BRESSAY's test split, G4's pages, or bill versions
  published after G2's cutoff; G3's split is fixed before the first training run on HWG.

### G5 handwritten pages

Exploratory, in the test run.
- **Source:** BRESSAY (Zenodo 10.5281/zenodo.11637681, CC BY 4.0): Brazilian Portuguese essays,
  each page by a different writer, with page images and page transcriptions. Public since 2024, so
  a model may have seen it.
- **Ground truth:** the transcription's words, in its order. Text marked `--…--` is struck;
  `--xxx--` (struck and unreadable) and `@@???@@` (illegible) are masked for every arm and counted;
  additions above or below the line (`##…##`, `$$…$$`) are live text where they stand, struck
  where marked inside. A word reading `xxx`, or holding `--xxx--` or `@@???@@`, is masked whole; it
  has no text to align, so it takes any one output token (see [Scoring](#scoring)). Markers pair
  up within a line: a transcription with an unpaired marker, or an `@` left over, does not parse,
  and its page is not sampled; every one in the test split parses. G5 has no furniture.
- **Pages,** from the test split's 199: a struck page holds at least one readable struck sub-token,
  a strike-free page no strike at all, and a page whose only strikes are unreadable is neither
  (133, 35 and 31). G5 holds every struck page and 10 strike-free ones, 143 pages. One generator,
  `random.Random(20261023)`, orders the struck pages with `.sample(sorted(struck), len(struck))`;
  the first in that order until there are at least 40 holding at least 200 struck sub-tokens are
  the core (40 pages holding 239), where R0 runs. It then draws the 10 strike-free pages with
  `.sample(sorted(free), 10)`, all in the core, so the core holds 50 pages. Each page is its own
  cluster. The page list, the core and each page's image and transcription sha256s are tagged in
  `prereg-1`, in `config/prereg1.json`.
- **Inputs:** C2 is the page's scan in RGB, widened when narrower to 1,700 px, the width of every
  other set's C2 (US Letter at 200 dpi), with Lanczos resampling and its aspect kept; BRESSAY's
  scans are mostly about 500 px wide. It is cut into C2-tiled bands as any page is. There is no C1
  or C1′ and no born-digital arm; P-scan reads C2's pixels, as on every set. G5 is outside the
  audits' samples.

## Arms and inputs

**Free arms**, run by the harness on every page:
- **B-naive:** PyMuPDF `page.get_text()` on the born-digital page. Every word is output live.
- **B-pm4llm:** `pymupdf4llm.to_markdown(doc, pages=[p], use_ocr=False)` at the pinned version.
- **P-native:** `detect_pdf(path, method="vector")` on the single-page born-digital PDF.
- **P-scan:** `detect_pdf(scan, ocr=rapidocr_backend(), scan_config=ScanConfig.confidence_free(),
  dpi=200)`, where `scan` is C2's pixels as an image-only PDF (`scan.pdf`, built by `render.py`).

The package is scored from its word records, never its `~~` markdown. P-native's output is the
page's PyMuPDF words in extraction order, with the `char_span` of every `final` record marked.
P-scan's is the words the OCR backend returned, captured by a pass-through wrapper around
`rapidocr_backend()` that hands them on unchanged, put in the package's own reading order
(`markdown.ordered_rows`) and marked the same way. On every page the unmarked words must equal
the call's own `clean_text` as a multiset of tokens (the orders differ), or the harness stops.

**A later package** (exploratory): a later package version or a retrained model is an extra arm,
`P-app:<version>`, in its own pinned environment and cache, fingerprinted and reported, and never
used to choose a model.

**Hosted OCR arms** (exploratory): Datalab's hosted Chandra, as **D-track**, with its
track-changes extra, which marks deletions `<del>…</del>`, and **D-plain**, the same conversion
without it. They run after the test run, like stage B7's models: fixed by neither tag, they wait
for an addendum, logged like a deviation, that fixes their route, settings and cells.

**Model inputs**, all built by `render.py` and hashed:
- **C1 PDF upload:** the page as a single-page PDF (PyMuPDF `insert_pdf`, annotations kept), sent
  through the provider's own document input. Stage B1 checks GPT-6's and xAI's input with an
  image-only PDF beside the born-digital page, and what it finds is reported beside that
  provider's C1 and fixed in `prereg-1` as the model's `c1_route`: page images and text
  (`image+text`) for Anthropic's models, by their docs, for Gemini 4 Argon, whose usage counts
  both, and for GPT-6, which reads an image-only PDF and marks the underlines of a born-digital
  page, which its text layer does not hold; the page image alone (`image`) for Gemini 3.8 Flash,
  whose usage counts no text from a PDF, and for xAI's, which read an image-only PDF with no
  search call and count no more input tokens for a born-digital page than for it. The open vision
  models (stage B7) have no C1.
- **C1′ image plus text:** the C2 image, then a text block holding the page's `page.get_text()`.
- **C2 page image:** `page.get_pixmap(dpi=200)`, RGB PNG, annotations rendered: about
  1700 × 2200 px for US Letter. The same bytes for every model; the effective pixel size each
  provider's resizing leaves is reported per model, from its documented rule or from the image
  tokens its `usage` reports (every response records its `usage`).
- **C2-tiled:** the C2 image cut into horizontal bands of the full page width, each 1,000 px tall,
  starting every 850 px from the top; the last band ends at the bottom edge, overlapping its
  neighbour by more where the height requires. The bands go in order in one request, with one added
  prompt line: "The page is given as N overlapping horizontal bands, top to bottom. Transcribe the
  page once."
- **C3 degradation** (later, by addendum): five levels of one fixed recipe (blur, noise, JPEG and a
  small rotation, each stepped together), on G0-dev pages, up to the level where the author can no
  longer read 95% of a 100-word sample; plus 20 G0-test pages printed and scanned at 300 dpi, then
  resampled to 200 dpi. The recipe is fixed before the first C3 call.
- **C4 handwritten crops:** G3's crops as distributed.

**Cells.** On G0–G2's test pages, every model but Fable 5.1 (which runs its own subset) runs R0,
R1 and R-best on C2, R-best also on C2-tiled where its provider chose it, and R0 on C1. The
flagships also run R1 and R-best on C1, C1′ at R0 on G0's and G2's pages, and R1's three
paraphrases on G2's C2 pages. On G5, the same models run R0 on its core and R-best on every page,
on C2, and on C2-tiled where the provider chose it. A model without C1 skips the C1 cells. A
flagship's R-best is its own rung (see [Prompts](#prompts)), and endpoint 3 reads it on the
provider's chosen image condition, C2 or C2-tiled.

## Prompts

The texts live in `prompts/ladder.toml`, as parts each rung joins; every assembled rung and
paraphrase is hashed, as drafted in `prereg-1` and as tuned in `prereg-2` (TBD (B2)). Each rung
adds one thing to the rung before. Drafts, tuned on dev until `prereg-2`:

- **R0** (user turn): "Transcribe this page to markdown. Output only the transcription."
- **R1** = R0 plus a system line: "Documents may contain struck-through (deleted) text; never
  present it as current."
- **R2** = R1 plus, in the user turn: "Wrap any struck-through text in `~~`, keeping it where it
  stands, for example `~~old wording~~`."
- **R3** = R2 plus a definition: "A strike is a line drawn through the middle of characters:
  single, double, wavy or handwritten, through a whole word or only part of one (then wrap only the
  struck characters, as in `~~semi-~~monthly`). Underlines, table rules, borders and redaction bars
  are not strikes. A PDF's text layer still contains deleted words, so judge from the page's
  image, not its text."
- **R4** = R3 with JSON in place of markdown, its definition naming `chars` in place of the `~~`:
  "Transcribe this page. Output one JSON object, `{"words": [...]}`, listing every word on the
  page in reading order as `{"text": ..., "chars": ..., "verdict": ..., "confidence": ...}`:
  `chars` is the struck part of the word (empty unless struck), `verdict` is `struck` or `clean`,
  and `confidence` is how sure you are of the verdict, from 0 to 1."
- **R5** = R4 plus a procedure: "First transcribe the page. Then re-inspect it line by line for
  strokes through glyphs, and only then write the JSON."
- **R6** = R5 plus three worked examples: a struck word, an underlined word and a table rule, each
  as an image crop with its JSON answer, cut from one G0-dev page (`G0.dev-002.p0`) that no dev
  list draws.

**Placement.** From R1 on, the system line goes in each API's system field. The page (image, bands
or PDF) comes before the instruction text, for every provider.

**R-best** is chosen per provider on dev, on its ladder model, from the rungs that ask for marks
(R2–R6), against R0, on the ladder's pages. A rung that fails more of those pages than R0 does,
plus one, is out. R0 and the rest are compared on the pages none of them failed, where a rung is
eligible when its over-deletion, marked and absent, each stays no more than 1 point above R0's.
R-best is the cheapest eligible rung leaving at most 2 points more struck sub-tokens unmarked
(live or absent) than the eligible rung leaving fewest, and among equally cheap rungs the one
leaving fewer: R2 and R3 cost the same, and R4, R5 and R6 each cost more than the rung before.
The runner-up is the eligible rung, other than R-best, leaving fewest unmarked. If no rung is
eligible, R-best is the rung with the fewest failed pages, then the least excess over the limits,
and there is no runner-up. The rung is chosen on C2; the image condition is then chosen at R-best
by the same rule, against R0 on C2, C2 being the cheaper. Each flagship runs R-best and the
runner-up on the ladder's pages and keeps whichever the same rule chooses, with R-best in R0's
place: that rung is the flagship's R-best everywhere in this file. The choices are tagged in
`prereg-2`: TBD (B2).

**Dev pages:** one generator seeded with `20261013` draws the pilot's 34 pages, then from those the
ladder's 24, the repeats' 20 (the effort ablation's too) and the canary's 10, each on its own, so
they overlap. Each list takes half its G0 and G1 pages from G0-dev and half from G1-dev, G1-dev's
half split between its two documents (the first by name takes the odd page). In the pilot, the
ladder and the repeats, two of G0-dev's pages and one of each G1-dev document's are strike-free,
and the rest hold a struck sub-token as scored; all the canary's pages hold one. The pilot and the
ladder also take the same four G2-dev pages, three struck and one strike-free. Pages are drawn
uniformly within their part (G0-dev, a G1-dev document, G2-dev) and kind, and R6's example page
never is. All are listed in `prereg-1`, in `config/prereg1.json`.

**Paraphrases:** three of every rung, written and hashed before the pilot. Those of R0, R1 and
R-best run on dev, and the flagships' R1 paraphrases also on G2-test's C2 pages, for endpoint 2.

**The QA probe** (exploratory): 40 questions drawn with seed `20261020` from G0-test's S7 items,
each with a live answer and a struck one ("what is the rate?" where 5% is struck and 3% written
beside it), asked with R0's framing and again with R1's system line, on C2, to the flagships. An
answer is scored as giving the live value, the struck value, both, or neither.

**Stage B4's prompt,** a draft fixed in `prereg-2` (TBD (B2)): **M+P context**, R2's prompt and
the page's image (C2), plus P-scan's reading of the page's scan, the app's second look: its struck
words with their struck characters, and the live characters of its words in its reading order,
introduced as "A strikethrough detector reports the following for this page. Use it, but trust the
page where they disagree." It runs for the flagships on G0's and G2's test pages. P→M, the
package's `provenance_text` to a model with no image, waits for an addendum.

## Scoring

The scorer, `score.py`, and the statistics, `stats.py`, are built and golden-tested before any
paid call, and the sha256 of every code file, theirs and the endpoints' (`report.py`) among them,
is tagged in `prereg-2` (TBD (B2)).

- **Sub-tokens.** Each ground-truth word is split at its struck/live character boundaries, so
  `~~semi-~~monthly` is the struck sub-token `semi-` and the live sub-token `monthly`.
- **Parsing an output.** Characters inside a marker on the fixed list (`~~…~~`; GitHub's single
  `~…~`, opened before a non-space and closed after one on the same line, never part of a `~~`;
  `<del>`, `<s>`, `<strike>`; `[deleted: …]`) are marked, as is any character carrying
  U+0334–U+0338; strike tags pair up by nesting, so all of an outer pair's inside is marked. Then
  the markers and other markup are removed (heading `#`s, emphasis runs of `*` or `_`, list
  markers, table rule rows, code fences, backslash escapes), other HTML tags and table pipes
  become word breaks, HTML entities are decoded, and the rest is split on whitespace into tokens.
  Each kept character carries its mark. A JSON answer (R4–R6) is flattened word by word: a
  `struck` verdict marks the first occurrence of `chars` in `text`, or the whole word where
  `chars` is empty or not found; `clean` counts as live, and so does a stray `unsure` (the prompts
  offer `struck` and `clean` alone). The last JSON object in the output is the answer; an item
  without a text gives no token, and an answer with no item that has one fails to parse. The
  package's arms are marked from their records' character spans alone, and B-naive not at all,
  whatever their text carries.
- **Unlisted notations.** Before `prereg-2`, every struck sub-token scored live in the dev outputs
  at R0 and R1 is listed with the output around it. A notation that marks struck text there, and
  wraps no live text on dev, joins the marker list before the scorer is frozen.
- **Normalization,** for alignment only, through an offset map back to the original characters:
  NFKC, case-folding, typographic quotes and dashes to ASCII, U+0334–U+0338 and format characters
  (the soft hyphen U+00AD, zero-width spaces) removed, and leading and trailing punctuation and
  symbols stripped from each token. A ground-truth sub-token that normalizes to nothing (a struck
  dash, U+2014, or a semicolon) is dropped from scoring and counted.
- **Alignment.** Output tokens are aligned to ground-truth words by a global edit-distance
  alignment over normalized tokens: gaps cost 1, and a substitution costs twice the pair's
  normalized edit distance (Levenshtein distance ÷ the longer token's length). An output token
  that equals two adjacent ground-truth words joined (a hyphen split at a line end) may align to
  both, and two adjacent output tokens that equal one ground-truth word joined may align to it,
  hyphens ignored either way. Within each aligned pair, characters are aligned the same way,
  through the offset maps.
- **Outcomes.** A ground-truth sub-token is *marked* when at least half of its characters align to
  marked output characters, *absent* when fewer than half align to any output character, and
  *live* otherwise. A pair whose normalized edit distance exceeds 0.5 counts as absent plus an
  inserted token. A misread word (distance up to 0.5) is present.
- **Masked and furniture words** are removed from the ground truth before scoring, and output
  tokens aligned to them are ignored. Furniture is G0's running heads and page numbers, and G1's
  and G2's as defined under each. A masked word no one could read (G5) aligns to any one output
  token at a cost of 1.5: less than leaving both unaligned, more than any pair close enough to
  count as present.
- **Leakage** = struck sub-tokens scored live ÷ struck sub-tokens. **Over-deletion,** twice: live
  sub-tokens scored marked ÷ live sub-tokens, and live sub-tokens scored absent ÷ live sub-tokens.
  The **unmarked share** = struck sub-tokens scored live or absent ÷ struck sub-tokens, and **read
  leakage** (exploratory) = struck sub-tokens scored live ÷ those scored live or marked: on
  handwriting a system that cannot read a struck word leaves it absent, which leakage alone
  rewards.
- **F1,** at sub-token level: a struck sub-token scored marked is a hit; struck and scored live or
  absent is a miss; live and scored marked is a false alarm. Live sub-tokens scored absent count in
  over-deletion, not in F1.
- **Hallucinated tokens:** inserted output tokens, reported per 1,000 ground-truth tokens, with
  those holding a digit counted separately.
- **Live-text error rates:** CER and WER of the output with marked text removed, against the
  ground truth's live text.
- **Calibration** (exploratory): an R4–R6 answer's probability that a word is struck (0.5 under a
  stray `unsure`, whatever its confidence; its `confidence` under a `struck` verdict and one minus
  it under `clean`, none for a confidence missing or outside 0 to 1 or neither a number nor a string
  holding one; a struck item's goes to its struck part alone), and P-scan's `cnn_prob`, which only
  the words its geometry proposes get. Each is the highest over the tokens aligned to the word,
  which is struck when any of it is. Per arm and set, three views: each side over every word it
  gives a probability, with that share of the scored and of the struck words; P-scan over every
  word, a word it gives none counted at 0; and the model and P-scan on the words both give one, a
  view conditional on P-scan's geometry, with the model's Brier score minus P-scan's. Reliability
  diagrams in tenths, each bin with its count and Wilson interval (a bin under 10 words is not
  read), and the expected calibration error and Brier score with cluster bootstrap intervals (none
  under six clusters), on the ladder's dev pages and, where a provider's R-best is R4–R6, on test.
- **The cascade** (exploratory): each model's outcomes on the words P-scan leaves uncertain (a
  P-scan token aligned to the word with `cnn_prob` strictly inside a band, where the package has
  decided neither way), against P-scan's own on the same words; and the whole pages' leakage,
  marked over-deletion and F1 with the model answering those words and P-scan the rest, against
  P-scan's alone. Each is read in the model meta's band (`p_lo`, `p_hi`) and in three fixed ones,
  (0.05, 0.95), (0.3, 0.7) and (0.4, 0.6), each with the share of sub-tokens and of pages handed
  off and its $ per 1,000 pages: P-scan on every page, and the model on the pages handed off. From
  the same outputs; a page-level stand-in for the app's second look, which works on crops and is
  measured on G3 and G4.
- **Failures.** A page with no usable output (a refusal, an output-limit stop, an empty answer or
  one that fails to parse, a request error after three resubmissions) scores worst case: every
  struck sub-token live and every live sub-token absent. Every claim is also tested without such
  pages, dropping each from both sides (see the decision rule), and failure, refusal and
  parse-failure rates are reported per arm.

## Statistics

- **Clusters:** a G0 page, a G1 document, a G2 bill, a G5 page.
- **Tests.** A leakage or over-deletion claim about sides A and B (model and package, or C1 and C2)
  with bound b, a proportion (1 point is 0.01), uses a sign-flip test on per-cluster counts. The
  statistic is the sum over clusters of ±(E_A − E_B − b·S), where S is the cluster's struck
  sub-tokens (its live ones, for over-deletion) and E_A, E_B count those each side gets wrong
  (struck scored live; live scored marked). Clusters with S = 0 are left out of n. For a bound
  other than 0 (endpoint 2) the test assumes each cluster's difference is symmetric about the
  bound, and the bootstrap interval is reported beside it. An F1 claim instead swaps the two sides'
  outputs in the flipped clusters and recomputes pooled F1, keeping every cluster. The p-value is
  the share of sign patterns at least as favourable to the claim as the observed one, ties
  included: all 2^n of them when n is 16 or fewer, otherwise 10,000 random ones from NumPy's PCG64
  seeded with `20261012`, as (1 + count) ÷ 10,001, with clusters taken in their IDs' order.
- **Intervals:** a paired cluster bootstrap (clusters resampled with replacement, each with all its
  pages), 10,000 replicates from PCG64 seeded with `20261017`, both sides scored on the same draw
  and each metric pooled over the draw's sub-tokens; 95% percentile intervals.
- Per-cluster results sit beside the pooled ones, and per-stratum results for G0. G1 and
  G2-strict, too few clusters to read an interval from, are reported per cluster with none.

## The audit

- **Items.** A disagreement between an arm's output and the ground truth, of one of four kinds: a
  leak (a struck sub-token scored live), a missed strike (one scored absent), an over-deletion (a
  live sub-token scored marked or absent), a hallucinated token. The pilot audit samples up to 20
  per pilot arm (5 per kind), up to 160 items, with seed `20261011`. The test audit samples 160
  items, with seed `20261014`, across the arms that enter a primary endpoint (the package's arms and
  each flagship's R0 C1, R0 C2, R1 C2 and R-best cells): each arm and kind gets an equal share,
  rounded by largest remainder, and a cell with too few disagreements passes its unused share to the
  other kinds of the same arm, then to the other arms, in a seeded order.
- **The page** `audit.py` writes holds, per item, the page crop with the word boxed, then three
  questions in order: (1) on the crop alone, is the word struck through: *struck* (which
  characters, if only some), *not struck*, or *can't tell*; (2) shown the ground-truth label, is it
  right; (3) shown the arm's output in one neutral rendering and the scorer's verdict, did the
  scorer read the output correctly. The arm is never shown. Labels come back as a CSV.
- **Consequences.** A *no* on question 2 is a ground-truth error and on question 3 a scorer bug;
  each is fixed at its source and every affected score recomputed. No number is overridden by
  hand.
- **Who labels.** The author, alone, on one page per task with no practice items, publishing the
  labels: before `prereg-2`, the pilot audit, a triage that hunts scorer and ground-truth bugs
  before both are frozen, and G2's span check; after the test run, the test audit, G1's
  error-rate sample and its relabel. An item is decided by the author's answer. With one labeller
  there is no agreement statistic, so the labels are published for anyone to label again; an
  addendum may add independent annotators before the test audit is labelled.
- **G1's error rate:** 400 G1-test words drawn with seed `20261015`, 200 of them struck in the
  ground truth and 200 live words on lines carrying a strike, labelled from the page crop with
  question 1 only, without the ground truth or any arm's output. Its sha256 is fixed in
  `prereg-2`, and it is labelled once the test run's answers are in, before any of them is scored.
  The error rate is reported separately for the struck words and for the live words on struck
  lines, with Wilson intervals and the date its labels came in, *can't tell* left out and counted.
  After the test run, words on which at least three of the four flagships' R-best arms disagree
  with the ground truth (a leak or an over-deletion; a failed page disputes nothing) are labelled
  the same way, the struck characters part of the answer. Once every one is decided, the
  corrections go to `data/gt/G1-corrections.jsonl` and apply to every arm: they change labels on
  the pages drawn, never the draw or the error-rate sample.
- **G2's span check:** before `prereg-2`, the author's check of 100 aligned spans of G2-test (see
  G2), each asking whether every word boxed as struck is struck and no live neighbour is: TBD (B2).

## Operations

- **Batches.** Each provider's batch API where the model has one; otherwise sequential requests at
  the provider's rate limits. Azure offers no batch for GPT-6, so GPT-6 is sent request by request
  at Azure's Flex tier (`service_tier: flex`, half price), on Global Standard deployments. A
  request Flex turns away waits and goes again at Flex; once five days of its model's test window
  have passed, a test request not yet final may go at the standard tier instead, billed as served,
  and its record says so.
- **Retries.** A request error is resubmitted up to three times, then scored as a failure; a
  rate-limit, authentication or quota error is not an attempt, and the request waits for the
  provider. Five request or account errors in a row (an account error is one waiting won't fix: a
  key, a quota, a spend limit) stop sending to that model until the next run; a capacity limit (a
  429 asking the client to wait, as Flex gives when busy) waits longer instead, and never counts
  toward that stop.
- **The test window.** Each model's test requests run within 7 days. A 10-page dev canary at R0 on
  C2, sent the way the model's test requests are, runs three times at the start of the window and
  once at its end. Two counts are tested: struck sub-tokens left live, on the pages holding a
  strike, and live sub-tokens marked, on the pages holding live text. With no drift a page's four
  runs are exchangeable, so each count's p is exact over the 4^n ways of taking one of each
  page's runs as the end run: twice the smaller share whose total over the pages is at least, or
  at most, the end run's, ties included, capped at 1. A p below 0.025 for either count is
  reported beside the model's results as drift; a model missing any of the four runs is reported
  as untested. A window longer than 7 days, or a test request (the latency sample's included) sent
  outside the canaries, is reported beside the model's results.
- **Model IDs.** Every response's reported model ID is checked against the pinned one, and a
  mismatch stops the run. For GPT-6, whose response names the deployment, the check reads Azure's
  `x-ms-served-model` header. Provider-side fallbacks to other models are off where the API offers
  them. If a version changes mid-run, the pin follows it, and a stage holding answers from two
  versions of a model is refused at scoring: the model's test requests the earlier version
  answered are sent again if the cap allows; otherwise its answers are scored by version, and its
  claims are reported on the pinned version's pages, flagged, outside the Holm families. The
  canaries are never sent again, since the drift test compares across their runs.
- **Cost:** each response's `usage` at the price list in force on the run date
  (`config/prices.toml`, each price with its date and source; Azure's for GPT-6, Vertex AI's for
  Gemini), thinking included, at batch prices where the request was batched and at Flex prices where
  Azure reports the Flex tier served it, GPT-6's input written to Azure's prompt cache at the
  cache-write price Azure bills; a hosted OCR arm's at its price per page. A promotional price
  stops sending once past its last day, until the price list then in force is recorded; each
  model's $ per 1,000 pages is also reported at its list price, which the app pays once promotions
  end. A free arm's cost is its CPU-seconds per page, OCR included, at a named on-demand price per
  vCPU-hour, measured on a named, otherwise idle machine, with the operating system's power
  throttling off for the timing process and every onnxruntime session pinned to one thread. Each arm
  runs alone over its pages, the package's first, after one untimed call that builds the engines
  later calls reuse, and each timed call starts once the process is idle (waiting at most 10 s).
- **Latency:** an interactive sample of 30 test pages of G0–G2 and G5's core, stratified by set and
  drawn with seed `20261019`, per model (Fable 5.1 apart) at R-best on C2, sent one at a time (GPT-6
  at the standard tier) from a process with the operating system's power throttling off, timed
  from request to last byte, retries and back-off excluded (the SDKs' own retries are off); a
  model's seconds per page come from this sample alone. The free arms are timed on the same pages
  in a second pass, run the same way at the libraries' default threads, which stops if an output
  differs from the cost pass's. Q5's seconds per page are reported for these arms.
- **Repeats:** three runs of a 20-page dev subset at R0 and R-best on C2 per ladder model measure
  nondeterminism.
- **Shared answers.** The repeats and the effort ablation are sent together, page by page, and
  an identical request (the same model, settings, prompt, input and repeat number) once: the
  ablation's runs at `high` are the repeats'. In both, an answer counts only while the model that
  gave it is still the one pinned, so after a re-pin both are sent again. A request's attempts in
  either count toward its resubmissions. Every other stage sends its own, the ladder's R2
  included, as R-best compares it with the rungs sent beside it.
- **The dry run** prices every stage before anything is spent, from the `usage` seen so far (the
  smoke tests, then dev), sending nothing: input tokens at the median for the same model, route and
  image size; output and thinking tokens at the 95th percentile per model and rung; xAI's document
  searches at the 95th percentile per input. The first, for stage B2, prices from the smoke tests,
  output and thinking at the 95th percentile per model across them, doubled, and takes every choice
  not yet made at its costliest (R6 as R-best and as the flagships' rung, R5 as the runner-up,
  C2-tiled as the image condition).
- **The caps,** each on everything spent so far, the smoke tests included, plus what is in flight:
  stages B1–B2's, set by the author from the stage B2 dry run once the smoke tests have run,
  $250; and stages B1–B3's, set from the stage B3 dry run, no lower than the price of the cells
  never dropped, TBD (B2). A reserve of $150 above the second is kept for G3's and G4's addenda and
  stage B4: no request of stages B1–B3 is sent into it, though answers costing more than their
  price can run into it. A stage whose price, with what is spent and in flight, passes its cap is
  not sent, and a model sent request by request stops once its next request's price no longer
  fits in what the cap leaves after what is spent, what is in flight and the price of the models
  still to go in the same submission; its other requests wait. Requests already out finish, and
  what they cost over their price passes the cap rather than come out of the models still to go.
  If the stage B3 dry run passes the cap, cells are dropped in this order until it fits: the
  flagships' C1 at R1 and R-best, their C1′, the Fable 5.1 subset, the QA probe. No other cell is
  ever dropped.
- **Deviations** go in `DEVIATIONS.md`: the date, what changed, why, and the results both ways.
  Every score and report records the commit it ran at, the tags in force, and which files of
  `bench/`, `prompts/` and `config/` differ from `prereg-2` (the price list and the budget's
  `dropped` list apart, which change legitimately; a cap or the drop order differing from what the
  tag froze is listed), so no change after the tag is silent.

## What each tag fixes

- **`prereg-1`** (end of stage B1): this file with every `TBD (B1)` filled, and its sha256: the
  pinned model IDs, the IDs their responses must report (each confirmed by a smoke-test answer),
  their knowledge cutoffs, their C1 routes and the G2 cutoff date; the G0 generator commit, the
  held-back seed's sha256 and every G0-test PDF's sha256; the G1 dev documents, page list and
  documents' sha256s; the dev pages for the pilot, the ladder, the repeats (the effort ablation's
  too) and the canary; the G2 bill lists, G2-dev's included, with their PDF and XML sha256s; G5's
  page list, its core and their sha256s; the prompts' sha256s as drafted; stages B1–B2's cap and
  the drop order. `plans freeze` writes the lists to `config/prereg1.json`, which is tagged with
  this file; it refuses while a `TBD (B1)` is left here, while the code differs from its commit,
  while the held-back seed's sha256 is not the one published under G0, and while the drop order
  is not this file's.
- **`prereg-2`** (end of stage B2): every `TBD (B2)` filled, and the sha256 of every file the rules
  live in (`bench/`, `prompts/`, `config/`, this file, the environment's requirements and the
  corpus manifest), of every test page's ground-truth record, of G1's error-rate sample and of
  G0's QA items; the per-model settings table, R-best, the runner-up, the image condition and each
  flagship's rung per provider, the prompts as tuned (paraphrases and few-shot images included),
  the M+P prompt, the G2 alignment rule and its coverage, the G2 span-check result, and stages
  B1–B3's cap. `plans freeze --stage 2` writes them to `config/prereg2.json`, which is tagged with
  this file; it refuses while any `TBD` is left here, while the code differs from its commit,
  before `config/prereg1.json` exists, and while a cap is not a number or the drop order is not
  this file's.

No arm, free or paid, touches a test page before `prereg-2`.

# Part B: build guide

For whoever builds and runs the harness (stages B1 to B3): the rules are Part A's, and this is how
to meet them. Everything runs in the pinned environment (README.md, Setup, in the private
repository), which also lists the commands and the code's layout.

## Records

JSONL throughout, one record a line, every file listed by sha256 in a manifest; what the tags fix
is in `config/prereg1.json` and `config/prereg2.json`.

- **Ground-truth page:** `page_id` (set, document, page index), `set`, `split`, `doc`, `cluster`,
  the source file and its sha256 (G2's XML and G5's transcription too), and `words`: each with
  `text`, `bbox` (page fractions; none on G5), `struck` (character spans), `masked`,
  `furniture`, `line` and, on G0, `strata`.
- **Inputs and staleness:** each input's sha256 is in `data/inputs/manifest.jsonl`, with the
  sha256 and page of the source it was built from, and a free arm's output carries its input's
  sha256. Each source PDF is hashed again before anything is built from it or sent; an input or
  output built from another source is stale, and nothing sends or scores it. Once `plans freeze`
  has written what `prereg-1` tags (`config/prereg1.json`), a test page it doesn't list, or whose
  sources are not the ones it froze, is refused too; once `plans freeze --stage 2` has written
  `config/prereg2.json`, so is a test page whose ground-truth record differs from the one it froze
  (G1's relabelled pages apart). A tag without its lists, or on a commit that doesn't hold them, is
  refused, never passed over: the lists are committed first, then tagged.
- **Response:** `run_id`, `arm`, `page_id`, `input`, `rung`, `para`, the pinned and the
  reported model IDs, `settings`, the prompt's and the input's sha256, timestamps, `batch_id`,
  `attempt`, `status` (`ok`, `refusal`, `length`, `parse_error`, `request_error`; `transient` for
  a limit that is not an attempt, `model_mismatch` for another model's answer), the raw output
  text, the raw `usage`, `cost_usd`, the harness commit, the package fingerprint (version, the ONNX
  sha256, `get_model_meta()`), the pre-registration tags in force, and `sent_tier` where a Flex
  request went at the standard tier. The prompt's sha256 covers every text part sent, in order;
  the registered hash (`<rung>.p<para>` in `prompts hash`) covers the rung's system line and
  instruction, so the two agree where a request sends no other text (C2-tiled adds its fixed
  line, C1′ the page's text, R6 its examples' framing).
- **Score:** per arm and page, the counts behind every metric (sub-tokens by truth and outcome,
  insertions, characters for CER), so tables are sums and nothing is re-derived from text; the
  same counts on the words P-scan leaves uncertain, in each band; and, from an arm that gives
  them, each scored word's probability of being struck, or none. An answer another stage sent
  names it (`reused_from`). Each scored stage records the commit, the tags and the files changed
  since `prereg-2` it was scored at (`provenance.json`).

Raw responses go to the Hugging Face dataset as `frontier/<run-id>/responses.jsonl`; scores and
the audit labels go beside them.

## Stage B1: build (only the smoke tests call an API)

In order; each step ends in a check.

1. **Pin the models.** Read each provider's live model list, fill `config/models.toml` and the
   tables above, and record each model's knowledge cutoff. Smoke-test each model on one dev page:
   the response reports the pinned ID (GPT-6's in `x-ms-served-model`); every setting in its row
   is accepted (`detail: original` on GPT-6 Sol, for one), and so is each ladder model's `low`
   effort; for every batched model, one batch
   holding an image request and a PDF request completes and reports the served model (Vertex's
   through a Cloud Storage bucket), and for GPT-6 the same two requests at Flex report the Flex
   tier; and xAI's document input gets an image-only PDF and GPT-6's a born-digital page, to learn
   what each passes to the model. Pin each `expect` to the most specific string the responses
   report, record GPT-6's C1 route (`c1_route`), and record as `max_output` any model whose own
   output maximum is below 64,000 tokens. A failure here changes the model list or a setting
   before `prereg-1`, never after.
2. **G0.** The same seed gives byte-identical files; every quota is met; P-native finds every
   E1–E4 strike on G0-dev and reports nothing in N1–N3, both apart from the declared limitations.
   A miss or a false report that no limitation declares is either a generator bug or a limitation
   to declare before `prereg-1`.
3. **G1.** Over the whole corpus, the vector detector's count and the vector-flag agreement match
   the package's confirmation check (55,171 vector detections, 99.85% confirmed); the page sample
   is reproducible from its seed; and B-naive's output aligns to at least 99% of the ground-truth
   words on every G1-dev page. Then draw the G1 error-rate sample.
4. **G2 and G5.** Fetch the reported bill versions after the cutoff date, XML and PDF, with a
   sha256 manifest; check the furniture rule on two bills; align; apply the 98% rule. Check: 45
   bills in G2-test and 3 others in G2-dev, with the strict subset counted; every kept page agrees
   with the strikes drawn on it; the coverage report is saved. Fetch BRESSAY with a sha256
   manifest, build G5 from its transcriptions and draw its pages. Check: every transcription in
   the test split parses, every struck page is in, and the core's struck pages, at least 40, hold
   at least 200 struck sub-tokens. Then draw the dev pages (Part A, Prompts).
5. **Inputs.** Identical hashes on a rerun; C2 is 200 dpi; the bands of C2-tiled cover every pixel
   row, and the last ends at the bottom edge.
6. **Free arms.** Run all four on dev. Check: on every dev page the unmarked words equal
   `clean_text` as a multiset of tokens (Part A, Arms and inputs).
7. **Scorer.** Golden tests, at least: a struck word output live (a leak), output marked, and
   absent; `~~semi-~~monthly`; `~~December~~May`; a JSON answer with a partial `chars`, and one
   whose `chars` occurs twice; U+0336; a misread struck word; a hyphen split across a line end; a
   markdown table and heading; a sub-token that normalizes to nothing; a furniture token; a masked
   word; each failure kind.
8. **Statistics.** The bootstrap reproduces by seed; the sign-flip test matches hand-worked
   examples, exact enumeration included, and the relabeling test brute-force enumeration; Holm
   and the largest-p rule match hand-worked examples.
9. **Harness.** Against the fake provider: caching (a rerun never calls an API), the model-ID
   check, resubmission, the error and capacity rules, the cap's guard and dry-run pricing. Then
   the dry run of stage B2, priced from the smoke tests with every choice not yet made at its
   costliest; the author sets stages B1–B2's cap from it.
10. **The audit tool.** Items to an HTML page and a CSV, labels back in, κ; an adjudication naming
    an item already decided is refused.
11. **Tag.** Fill every `TBD (B1)`, commit, run `plans freeze`, commit `config/prereg1.json`, tag
    that commit `prereg-1`, and publish it (PLAN.md's steps). `harness submit` refuses stage B2
    before the `prereg-1` tag, and stage B3 before `prereg-2`.

## Stage B2: dev and the pilot

1. **The pilot:** 34 dev pages (30 of G0-dev and G1-dev, 4 of G2-dev), the four ladder models at
   R2 on C2, plus the free arms. Score; the author triages the pilot audit; fix what it finds,
   rescore.
2. **The ladder:** R0–R6 on the ladder's 24 dev pages, on C2, on each ladder model; choose
   R-best, run it on C2-tiled, and choose the image condition. Then the paraphrases, the repeats,
   the effort ablation, and R-best and the runner-up on each flagship, which fixes its rung.
3. **Notations and labels:** list the struck words the dev outputs at R0 and R1 leave live, and
   add any notation that qualifies to the scorer (Part A, Scoring); the author's G2 span check;
   then fix the G2 alignment rule.
4. **Stage B4's prompt:** write the M+P prompt and hash it.
5. **Freeze:** the dry run of stage B3 (its test pages priced from local files, nothing sent)
   sets stages B1–B3's cap; fill every `TBD (B2)`, commit, run `plans freeze --stage 2`, commit
   `config/prereg2.json`, tag that commit `prereg-2`, and publish it.

## Stage B3: the test run

1. If the dry run of stage B3, with what is spent and in flight, passes the cap, apply the drop
   order.
2. Submit each model's cells, canary first and last, within its 7-day window. Run the free arms
   on the test pages, then the latency sample and the free arms' latency pass on its pages.
3. Collect; label the G1 error-rate sample before scoring; then score and run the test audit,
   then the G1 relabel. Fix at the source and rescore; never edit a number.
4. Report: the results table, the leakage-against-cost scatter, the failure gallery, and the
   per-stratum and per-cluster tables. Publish the responses, scores and labels.

What the author still provides is listed in the private repository's README.md, under Before the
pilot.
