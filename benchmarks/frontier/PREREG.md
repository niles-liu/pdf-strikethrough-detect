# Frontier-model benchmark: pre-registration

**Status: draft (stage B0), not yet pre-registered.** This file binds once it is tagged `prereg-1`
(end of stage B1), and again at `prereg-2` (end of stage B2), as
[PLAN.md](PLAN.md#pre-registration) describes. Until `prereg-1` any value here can change.
`TBD (B1)` and `TBD (B2)` mark what is filled in during that stage, before its tag. A change after
a tag is a deviation: it is logged in `DEVIATIONS.md`, with the results shown both ways.

**How to read it.** [PLAN.md](PLAN.md) gives the reasons and this file gives the rules; where the
two differ, this file governs. **Part A** is the protocol. **Part B** is the build guide for stages
B1 to B3: what to build, in what order, and how to tell it works. Part B is guidance, so changing
it is not a deviation unless the change alters a rule in Part A. The names (Q1–Q6, G0–G4, the
arms, C1–C4, R0–R6, stages B0–B6) are PLAN.md's.

# Part A: protocol

## Models

Exact IDs are pinned at stage B1 against each provider's live model list, and the per-model
settings at stage B2. A model released after `prereg-1` is an extra arm, never a replacement. A
pinned model retired before its test requests run is dropped and reported as dropped; nothing takes
its place in a primary endpoint.

- **Anthropic** (Messages API, Message Batches): Claude Opus 5.5 `claude-opus-5-5`, Claude
  Sonnet 5.5 `claude-sonnet-5-5` and Claude Haiku 4.5 `claude-haiku-4-5-20251001`, plus Claude
  Fable 5.1 `claude-fable-5-1` on a ceiling subset. Haiku 4.5 is listed as legacy, retired no
  earlier than 2026-10-15, so it may be dropped under the rule above.
- **Google** (Gemini API, its Batch API): Gemini 3.1 Pro `gemini-3.1-pro-preview` and Gemini 3.8
  Flash `gemini-3.8-flash`. Google has no stable 3.x Pro, so the flagship is a preview, which Google
  may change or close with two weeks' notice; a stable Pro released before `prereg-1` replaces it.
- **OpenAI** (Responses API, Batch API): GPT-6 Astra `gpt-6-astra` and GPT-6 Luna `gpt-6-luna`,
  the family's smallest tier (GPT-6 has no mini).
- **xAI** (Chat Completions API, which documents the image `detail` setting; the Batch API for
  Grok 4.3 only): Grok 4.7 `grok-4.7` and Grok 4.3 `grok-4.3`. If Grok 4.3 fails a stage B1 smoke
  test (images in a batch, or a PDF attachment), `grok-4.20-0309-reasoning` replaces it before
  `prereg-1`: the same price, images and a 20% batch discount, and a dated ID.
- **Open weights,** each on a host that serves its released weights at their released precision,
  with the provider pinned and routing fallbacks off: Qwen3.8-27B (`Qwen/Qwen3.8-27B`, Apache-2.0)
  at BF16 on DeepInfra, and Kimi K3 (`moonshotai/Kimi-K3`) at its native MXFP4 on Moonshot's own
  platform (`kimi-k3`). They may be reached directly or through OpenRouter with `provider.only`
  and `allow_fallbacks: false`. If Kimi K3 fails a stage B1 smoke test, Qwen3.5-397B-A17B at its
  official FP8 replaces it. DeepSeek is left out: its API resizes an image to about 1,300 × 1,300
  px, well below a 200 dpi page, and its earlier vision model's name now points at another model.
- **OCR-to-markdown:** Chandra OCR 2 (`datalab-to/chandra-ocr-2`), the one such model whose
  documented prompt allows a strikethrough tag (`<del>`), run as documented through Datalab's API
  or from its released checkpoint. It joins only if it emits strike markup on dev at all (stage
  B2), and then runs one cell, its own prompt on C2, reported beside the free arms.

The GPT-6 and Grok IDs, and Gemini 3.1 Pro's preview ID, name a model that its provider can update
in place: none has a dated snapshot, and OpenAI has already changed GPT-6 models' image handling in
place once. So every response also records what the provider reports about the version
(`system_fingerprint` where it exists, and xAI's model `version`), and the canary (see
[Operations](#operations)) checks for drift.

- **Flagships,** the models the primary endpoints test: Claude Opus 5.5, Gemini 3.1 Pro, GPT-6 Astra
  and Grok 4.7. The other models are reported in the same tables, as exploratory results.
- **Ladder models** choose R-best and the image condition for their provider (see
  [Prompts](#prompts)): Claude Sonnet 5.5, Gemini 3.8 Flash, GPT-6 Luna and Grok 4.3. Each open
  vision model chooses for itself.
- **Fable 5.1** runs on a ceiling subset of 50 test pages, stratified by set, at R0, R1 and
  Anthropic's R-best, on C2.

**Settings.** One row per model, fixed in `prereg-2` and recorded with every response:

| Model | Effort | Image | Batch |
|---|---|---|---|
| Opus 5.5, Sonnet 5.5, Fable 5.1 | `effort: high` | as sent (up to 2,576 px) | Message Batches |
| Haiku 4.5 | thinking on, 16,000-token budget | as sent (scaled to 1,568 px) | Message Batches |
| Gemini 3.1 Pro, Gemini 3.8 Flash | `thinking_level: high` | `media_resolution: high` | Batch API |
| GPT-6 Astra, GPT-6 Luna | `reasoning.effort: high` | `detail: original` | Batch API |
| Grok 4.7 | `reasoning_effort: high` | `detail: high` | none: sequential |
| Grok 4.3 | `reasoning_effort: high` | `detail: high` | Batch API |
| Qwen3.8-27B, Kimi K3 | thinking on, if the host can | as sent; host limit recorded | sequential |

- Effort is set explicitly at each model's documented `high` level, or its nearest equivalent,
  because the models get their best shot and provider defaults differ. The effort ablation runs
  the low level on a 20-page dev subset.
- Sampling parameters (temperature, top-p) are never sent, so each model runs at its provider's
  default; OpenAI asks for them to be left out of reasoning requests anyway.
- The output limit is 32,000 tokens, so that an output-limit stop is rare; it includes thinking
  everywhere except xAI, whose limit excludes reasoning tokens. A model that stops at the limit
  scores the page as a failure.
- Region and data retention stay at the provider's default for the account, and are recorded.

## Primary endpoints and decisions

Four primary endpoints; everything else is exploratory. All are paired comparisons on the same
pages. "Points" are percentage points.

1. **Default leakage (Q1).** Leakage at R0, per flagship, on G0-test, G1-test and G2-test, on C1
   and on C2, against P-native. On G0 and on G2, the package wins against a flagship when model
   minus package leakage lies above 0 on both C1 and C2. G1's numbers are reported with their CIs
   and are not a contest.
2. **The one-line prompt (Q6).** Leakage at R1, per flagship, on the C2 pages of G0-test and
   G2-test, against P-native: the model matches the package when model minus package leakage lies
   within ±1 point and model minus package marked over-deletion lies below 1 point.
3. **Detection at its best (Q2).** Struck-word F1 at R-best, on each provider's chosen image
   condition, per flagship, on G0-test and G2-test, against P-scan, which reads the same 200 dpi
   page. Either side wins when the difference lies on its side of 0. P-native's F1 is reported
   beside it.
4. **Input route (Q3).** Leakage at R0, C1 minus C2, per flagship, on G0-test, G1-test and
   G2-test: C1 leaks more when the difference lies above 0.

**The decision rule.** Each claim above is one or more one-sided hypotheses about a paired
difference. Each has a bootstrap p-value (see [Statistics](#statistics)); a claim needing several
components to hold (both inputs in endpoint 1, three bounds in endpoint 2) takes the largest of
their p-values. Within each endpoint the p-values are Holm-adjusted across flagships and sets
(endpoint 3 counts each direction as its own hypothesis), and a claim holds when its adjusted
p-value is below 0.025, the one-sided form of a 95% two-sided interval. Reported intervals are
unadjusted 95% percentile intervals.

**Insufficient data.** A cell with fewer than 200 struck sub-tokens, or fewer than 3 documents,
reads "insufficient data" and leaves its family before the Holm adjustment.

## Ground truth

### G0 synthetic redlines

A seeded generator, `g0.py`, grown from `examples/native_quickstart.py::build_redline_pdf`.

- **Text:** contract-style clauses (parties, dates, amounts, rates, obligations) from a seeded
  template grammar with random fillers. No LLM writes it, so the generator is deterministic,
  offline and novel to every model.
- **Pages:** US Letter, 1-inch margins, one column, 9–13 pt body text, a running head and a page
  number (furniture), and some ruled tables. About 350 words a page.
- **Struck strata.** Each struck word has exactly one encoding:
  E1 a stroked vector line through the middle of the x-height, E2 a thin filled rectangle, E3 a
  `/StrikeOut` annotation with an appearance stream, E4 an ink annotation, E5 a line slanted over
  2° that rises more than 1.5 pt end to end (a declared limitation), E6 U+0336 combining strokes in
  the text (a declared limitation). Styles, on E1 unless noted: S1 single (the default), S2
  double, S3 coloured (red or blue), S4 thick (2 pt or more, an expected model win), S5
  partial-word, S6 a passage running over a line break, S7 a struck number beside its written
  correction (`~~5%~~ 3%`). Fonts rotate through Helvetica, Times and Courier across every
  stratum.
- **Negatives:** N1 strike-free pages; N2 underlined words, including Courier with the underline
  at the baseline (a declared limitation); N3 table rules running close to text.
- **Quotas,** in G0-test: at least 220 struck sub-tokens in each of E1–E6 and S2–S7, at least 1,500
  live sub-tokens under N2 or beside N3, and 8 N1 pages. The generator packs strata onto pages
  until every quota is met (about 45 pages); G0-dev holds 40% of each quota.
- **QA items:** for each S7 correction the generator also writes a question whose answer the
  deletion changed, with its live and struck answers, for the QA probe.
- **Seeds:** dev `20261008`, test `20261009`. The held-back set's seed is chosen at stage B1,
  kept private, and only its sha256 is published: TBD (B1).
- **Ground truth** is written by the generator from what it draws: each word's text, box and
  struck character spans. Each test file's sha256 is tagged in `prereg-1` (TBD (B1)), with the
  generator's commit.

### G1 public redline corpus

- **Documents:** the nine in [`../manifest.json`](../manifest.json) left once the manifest's
  `scanned_pages` are removed (the Copyright Office document leaves entirely).
- **Ground truth:** per page, the package's `vector` and `flag` detectors (0.12.1, as frozen) run
  separately. A word is struck where both report it, with the agreement's character span; a word
  only one of them reports is masked for every arm and counted; every other word is live. Words
  come from PyMuPDF's `page.get_text("words")` in extraction order.
- **Dev documents:** two of the nine, drawn with `random.Random(20261010).sample(sorted(files), 2)`.
  The other seven are G1-test.
- **Pages,** per test document: up to 10 struck pages, sampled with seed `20261010` across six
  cells (the document's strike-density tertiles crossed with whether the page has a partial-word
  strike) in proportion to the cells' sizes, all of them where fewer exist; and up to 3
  strike-free pages where the document has any. The page list is tagged in `prereg-1`: TBD (B1).
- **Its error rate** is measured on a sample of G1-test words (see [The audit](#the-audit)).

### G2 US Congress bills

- **Source:** GovInfo's `BILLS` collection, which publishes each bill version's XML and PDF.
  Reported versions mark text a committee struck with `<deleted-text>`, and the PDF draws it
  struck through.
- **Selection:** bill versions carrying `<deleted-text>` and published after the latest training
  cutoff any model in the run states (its release date where it states none), so that no model can
  have seen them: the cutoff date is TBD (B1),
  once the IDs are pinned, and is no earlier than 2026-09-01, since Grok 4.7's model card reports
  training data into August 2026. At least 4 bills and about 30 pages: per bill, up to 10 pages
  carrying deleted text and up to 2 without, in page order. The bill list, its sha256s and the
  cutoff date are tagged in `prereg-1`: TBD (B1).
- **Alignment:** the XML's word sequence, each word flagged deleted or not, is aligned to the PDF's
  words (PyMuPDF, extraction order) with the edit-distance alignment the scorer uses. A PDF word
  aligned to a deleted XML word is struck. Coverage per page is the share of body words aligned;
  pages under 98% are dropped. The annotators check 100 aligned spans by eye before `prereg-2`; a
  bad span fixes the rule, and every G2 page is rebuilt.
- **Furniture:** margin line numbers, running heads and page numbers.

### G3 and G4

Exploratory, in stages B5 and B6. G3 is crop-level ("is this word struck?") on the HWG dataset's
`written` and `synthetic` parts and the ASAP and GoBo crops of `collected`, by strike type:
StrikeNet as shipped against the models. G4 is hand-verified logbook pages from a related project,
with its consent. Each is specified in an addendum before it runs, logged like a deviation.

## Arms and inputs

**Free arms**, run by the harness on every page:
- **B-naive:** PyMuPDF `page.get_text()` on the born-digital page. Every word is output live.
- **B-pm4llm:** `pymupdf4llm.to_markdown(doc, pages=[p])` at the pinned version.
- **P-native:** `detect_pdf(path, method="vector")` on the single-page born-digital PDF.
- **P-scan:** `detect_pdf(scan, ocr=rapidocr_backend(), scan_config=ScanConfig.confidence_free(),
  dpi=200)`, where `scan` is the page rasterized at 200 dpi by `_scanned.build_scanned_pdf`.

The package is scored from its word records, never its `~~` markdown. P-native's output is the
page's PyMuPDF words with the character spans of every `final` record marked; P-scan's is the
words the same RapidOCR backend reads from the same 200 dpi render, marked the same way. On every
dev page the reconstructed live text must equal the call's own `clean_text` (whitespace aside), or
the harness stops.

**Model inputs**, all built by `render.py` and hashed:
- **C1 PDF upload:** the page as a single-page PDF (PyMuPDF `insert_pdf`, annotations kept), sent
  through the provider's own document input. Claude receives each page as an image plus its
  extracted text, and OpenAI's models the extracted text plus an image of the page (`detail: high`,
  the most a PDF allows). Gemini reads each PDF page as an image plus its embedded text
  (`media_resolution: high`, again the most a PDF allows). xAI's document input, by its docs, gives
  the model a search tool over the file rather than the page; stage B1 checks this with an
  image-only PDF. It is Grok's C1 all the same, since it is the route a developer gets. The open
  vision models have no document input, so no C1.
- **C1′ image plus text:** the C2 image, then a text block holding the page's `page.get_text()`.
- **C2 page image:** `page.get_pixmap(dpi=200)`, RGB PNG, annotations rendered: about
  1700 × 2200 px for US Letter. The same bytes for every model; each provider downscales by its own
  rules, and the effective pixel size is recorded per response.
- **C2-tiled:** the C2 image cut into horizontal bands of the full page width, each at most 1,000 px
  tall with 150 px of overlap, sent in order in one request, with one added prompt line: "The page
  is given as N overlapping horizontal bands, top to bottom. Transcribe the page once."
  Anthropic's high-resolution models and OpenAI's at `detail: original` take the bands unscaled,
  and Haiku 4.5 scales them to 92%. Gemini's `high` budgets 1,120 tokens per image, each band
  included. xAI documents no resolution limit or image-token formula, so its image tokens are
  recorded from `usage` in place of an effective size.
- **C3 degradation** (stage B5): five levels of one fixed recipe (blur, noise, JPEG and a small
  rotation, each stepped together), on G0-dev pages, up to the level where the author can no longer
  read 95% of a 100-word sample; plus 20 G0-test pages printed and scanned at 300 dpi, then
  resampled to 200 dpi. The recipe is fixed before the first C3 call.
- **C4 handwritten crops:** G3's crops as distributed.

**Cells.** Every model runs R0, R1 and R-best on C1 and on C2; C1′ at R0 and R1; C2-tiled at
R-best; and the three R1 paraphrases on C2. A model without C1 skips the C1 cells. Endpoint 3 reads
R-best on the provider's chosen image condition, C2 or C2-tiled.

## Prompts

The texts live in `prompts/`, one file per rung and paraphrase, each with its sha256 in
`prereg-2`. Each rung adds one thing to the rung before. Drafts, tuned on dev until `prereg-2`:

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
- **R4** = R3 with JSON in place of markdown: "Output one JSON object, `{"words": [...]}`, listing
  every word on the page in reading order as `{"text": ..., "chars": ..., "verdict": ...,
  "confidence": ...}`: `chars` is the struck part of the word (empty unless struck), `verdict` is
  `struck`, `clean` or `unsure`, and `confidence` is a number from 0 to 1."
- **R5** = R4 plus a procedure: "First transcribe the page. Then re-inspect it line by line for
  strokes through glyphs, and only then write the JSON."
- **R6** = R5 plus three worked examples: a struck word, an underlined word and a table rule, each
  as an image crop with its JSON answer, taken from dev documents only.

**Placement.** From R1 on, the system line goes in each API's system field. The page (image, bands
or PDF) comes before the instruction text, except in Gemini's image requests, where Google's
guidance puts the text first.

**R-best** is chosen per provider on dev, on its ladder model: the rung with the lowest leakage
whose marked over-deletion stays within 1 point of R0's, ties going to the cheaper rung. It is
confirmed on the provider's flagship before `prereg-2`; if the flagship's best rung differs, the
ladder model's choice stands and the gap is reported. The image condition (C2 or C2-tiled) is
chosen the same way. The choices are tagged in `prereg-2`: TBD (B2).

**Paraphrases:** three of every rung, written and hashed before the first dev call, since R-best
is not known yet. Those of R0, R1 and R-best run on dev, and the R1 paraphrases also on test.

**The QA probe** (exploratory): 40 questions from G0-test's S7 items, each with a live answer and a
struck one ("what is the rate?" where 5% is struck and 3% written beside it), asked with R0's
framing and again with R1's system line, on C2, to the flagships. An answer is scored as giving the
live value, the struck value, both, or neither.

## Scoring

The scorer, `score.py`, is built and golden-tested before any paid call, and its code's hash is
tagged in `prereg-2`.

- **Sub-tokens.** Each ground-truth word is split at its struck/live character boundaries, so
  `~~semi-~~monthly` is the struck sub-token `semi-` and the live sub-token `monthly`.
- **Parsing an output.** Characters inside a marker on the fixed list (`~~…~~`, `<del>`, `<s>`,
  `<strike>`, `[deleted: …]`) are marked, as is any character carrying U+0334–U+0338; the markers
  are then removed. A JSON answer (R4–R6) is flattened word by word: a `struck` verdict marks the
  word's `chars` where they occur in its `text`, else the whole word; `clean` and `unsure` count as
  live. The last JSON object in the output is the answer.
- **Normalization,** for alignment only: NFKC, case-folding, typographic quotes and dashes to
  ASCII, U+0334–U+0338 removed, and leading and trailing punctuation stripped from each token.
- **Alignment.** Output tokens are aligned to ground-truth words by a global edit-distance
  alignment over normalized tokens: gaps cost 1 and a substitution costs twice the pair's
  normalized character edit distance. An output token that equals two adjacent ground-truth words
  joined (a hyphen split at a line end) may align to both. Within each aligned pair, characters are
  aligned the same way.
- **Outcomes.** A ground-truth sub-token is *marked* when at least half of its characters align to
  marked output characters, *absent* when fewer than half align to any output character, and
  *live* otherwise. A pair whose normalized edit distance exceeds 0.5 counts as absent plus an
  inserted token. A misread word (distance up to 0.5) is present.
- **Masked and furniture words** are removed from the ground truth before scoring, and output
  tokens aligned to them are ignored. Furniture is G0's running heads and page numbers, G2's margin
  line numbers, heads and page numbers, and on G1 any word whose box centre lies within 6% of the
  top or bottom edge.
- **Leakage** = struck sub-tokens scored live ÷ struck sub-tokens. **Over-deletion,** twice: live
  sub-tokens scored marked ÷ live sub-tokens, and live sub-tokens scored absent ÷ live sub-tokens.
- **F1,** at sub-token level: a struck sub-token scored marked is a hit; struck and scored live or
  absent is a miss; live and scored marked is a false alarm. Live sub-tokens scored absent count in
  over-deletion, not in F1.
- **Hallucinated tokens:** inserted output tokens, reported per 1,000 ground-truth tokens, with
  those holding a digit counted separately.
- **Live-text error rates:** CER and WER of the output with marked text removed, against the
  ground truth's live text.
- **Calibration** (exploratory): the `confidence` of R4–R6 answers against `cnn_prob`, as
  reliability diagrams, over the same words: those on scanned pages the package scored.
- **Failures.** A page with no usable output (a refusal, an output-limit stop, an answer that
  fails to parse, a request error after three resubmissions) scores worst case: every struck
  sub-token live and every live sub-token absent. A sensitivity table repeats the results without
  such pages, and refusal and parse-failure rates are reported per arm.

## Statistics

- **Bootstrap:** 10,000 paired replicates with seed `20261012`, two-stage: documents resampled
  with replacement, then pages within each drawn document. Both sides of a comparison are scored
  on the same draw; each metric is the pooled ratio over the draw's sub-tokens.
- **G0 documents** are the generator's files: each holds 5 pages, and the file is the cluster.
- **p-values:** one-sided, (1 + the number of replicates on the wrong side of the bound) ÷
  10,001.
- Per-document results sit beside the pooled ones, and per-stratum results for G0.

## The audit

- **Items.** A disagreement between an arm's output and the ground truth, of one of four kinds:
  a leak, an over-deletion (marked or absent), a wrongly marked or missed strike, a hallucinated
  token. The pilot audit samples up to 20 per pilot arm (5 per kind), up to 160 items. The test
  audit samples 160 items across the arms that enter a primary endpoint (the package's arms and
  each flagship's R0 C1, R0 C2, R1 C2 and R-best cells), equally per arm and kind. Seed
  `20261011`.
- **The page** `audit.py` writes holds, per item, the page crop with the word boxed, then three
  questions in order: (1) on the crop alone, is the word struck through: *struck* (which
  characters, if only some), *not struck*, or *can't tell*; (2) shown the ground-truth label, is it
  right; (3) shown the arm's output in one neutral rendering and the scorer's verdict, did the
  scorer read the output correctly. The arm is never shown. Labels come back as a CSV.
- **Consequences.** A *no* on question 2 is a ground-truth error and on question 3 a scorer bug;
  each is fixed at its source and every affected score recomputed. No number is overridden by
  hand.
- **Annotators:** two people who had no part in the package's development, named in the write-up
  with their consent. Each first labels 10 practice items with known answers. They split the items,
  with a random 25% (seed `20261011`) labelled by both, and each checks every *no* the other gives
  on questions 2 and 3. Agreement on question 1 is Cohen's κ over the shared items, with a
  bootstrap CI; below 0.7, the definitions are revised and the round repeated. The author
  adjudicates disagreements, and the pre-adjudication labels are published.
- **G1's error rate,** before any test call: 400 G1-test words drawn with seed `20261011`, 200 of
  them struck in the ground truth and 200 live words on lines carrying a strike, labelled from the
  page crop with question 1 only, without the ground truth. The error rate is reported separately
  for the struck words and for the live words on struck lines, where a missed strike would hide,
  with Wilson intervals. After the test run, words on which at least three of the four flagships'
  R-best arms disagree with the ground truth are labelled by both annotators the same way;
  corrections apply to every arm.
- **G2's span check:** 100 aligned spans, split between the annotators (see G2).

## Operations

- **Batches.** Each provider's batch API where the model has one; otherwise sequential requests at
  the provider's rate limits. A request error is resubmitted up to three times, then scored as a
  failure. Each model's test requests run within 7 days. A 10-page dev canary at R0 on C2 runs
  three times at the start of each model's test window and once at its end; an end-of-window
  leakage outside the range of the three start runs is reported as drift.
- **Model IDs.** Every response's reported model ID is checked against the pinned one, and a
  mismatch stops the run. Provider-side fallbacks to other models are turned off where the API
  offers them.
- **Cost:** each response's `usage` at the price list in force on the run date
  (`config/prices.toml`, each price with its date and source), thinking included, at batch prices
  where the request was batched. The package's cost is measured CPU-seconds per page on a named
  machine, OCR included, at a named on-demand price per vCPU-hour.
- **Latency:** an interactive sample of 30 test pages per model at R0 and at R-best on C2, sent one
  at a time, timed from request to last byte, retries and back-off excluded; the free arms are
  timed on the same pages. Q5's seconds per page are reported for these arms.
- **Repeats:** three runs of a 20-page dev subset at R0 and R-best on C2 per ladder model measure
  nondeterminism.
- **The dry run** prices every stage before anything is spent. Input tokens come from Anthropic's
  and OpenAI's exact token counting; for Gemini, whose image counts are estimates, for xAI, which
  counts text only, and for the open models' host, they come from the `usage` seen on dev for the
  same route and image size. Output and thinking tokens, and xAI's document searches, are taken at
  the 95th percentile seen on dev, per model and rung.
- **The cap** is TBD (B1), set by the author. If the dry run's total passes the cap, cells are
  dropped in this order (*proposed*), and never a cell that enters a primary endpoint, a free arm or
  the audit: the R1 paraphrases on test; C1′ at R1; C2-tiled for the non-flagship models; the Fable
  5.1 subset; the QA probe; the non-flagship models' R-best cells.
- **Deviations** go in `DEVIATIONS.md`: the date, what changed, why, and the results both ways.

## What each tag fixes

- **`prereg-1`** (end of stage B1): this file with every TBD (B1) filled: the pinned model IDs,
  their knowledge cutoffs and the G2 cutoff date; the G0 generator commit, the held-back seed's
  sha256 and every G0-test file's sha256; the G1 dev documents and page list; the G2 bill list and
  sha256s; the cap and the drop order.
- **`prereg-2`** (end of stage B2): every TBD (B2) filled: the scorer's hash, the per-model settings
  table, R-best and the image condition per provider, the prompt files' sha256s (paraphrases and
  few-shot images included), the M+P and text-only prompts, and the G1 error-rate and G2 span-check
  results.

No arm, free or paid, touches a test page before `prereg-2`.

# Part B: build guide

Written for whoever builds the harness (stages B1 to B3). The rules are Part A's; this is how to
meet them. Everything runs in the pinned environment ([PLAN.md](PLAN.md#harness-and-operations)).

## Layout

Suggested, not binding:

```text
benchmarks/frontier/
  PLAN.md  PREREG.md  DEVIATIONS.md      DEVIATIONS.md appears with the first deviation
  requirements.in  requirements.txt      the pinned environment
  config/models.toml                     pinned IDs and the settings table
  config/prices.toml                     prices with their date and source
  config/seeds.toml                      the public seeds
  prompts/                               rungs, paraphrases, few-shot crops and answers
  g0.py  g1.py  g2.py                    ground-truth builders
  render.py                              C1, C1′, C2, C2-tiled (C3 at stage B5)
  free_arms.py                           B-naive, B-pm4llm, P-native, P-scan
  providers/                             one adapter per provider, plus a fake for tests
  harness.py                             the CLI: build, dry-run, submit, collect, score
  score.py  stats.py  audit.py  report.py
  tests/                                 golden tests for the scorer and the statistics
  data/  runs/                           generated, git-ignored, pinned by sha256 manifests
```

## Records

JSONL throughout, one record a line, every file listed by sha256 in a committed manifest.

- **Ground-truth page:** `page_id` (set, document, page index), `set`, `split`, `doc_id`, the
  source file's sha256, and `words`: each with `text`, `bbox` (page fractions), `struck` (character
  spans), `masked`, `furniture` and, on G0, `strata`. Plus each input's path and sha256.
- **Response:** `run_id`, `arm_id`, `page_id`, `input`, `rung`, `paraphrase`, the pinned and the
  reported model IDs, `settings`, the prompt's and the input's sha256, timestamps, `batch_id`,
  `attempt`, `status` (`ok`, `refusal`, `length`, `parse_error`, `request_error`), the raw output
  text,
  the raw `usage`, `cost_usd`, the harness commit, the package fingerprint (version, the ONNX
  sha256, `get_model_meta()`), and the pre-registration tags in force.
- **Score:** per arm and page, the counts behind every metric (sub-tokens by truth and outcome,
  insertions, characters for CER), so tables are sums and nothing is re-derived from text.

Raw responses go to the Hugging Face dataset as `frontier/<run-id>/responses.jsonl`; scores and
the audit labels go beside them.

## Stage B1: build, at no API cost

Do these in order; each ends in a check.

1. **Pin the models.** Read each provider's live model list, fill `config/models.toml` and the
   tables above, and record each model's knowledge cutoff. Smoke-test each model on one dev page:
   the response reports the pinned ID; every setting in its row is accepted (OpenAI's docs do not
   confirm `detail: original` for GPT-6 Luna, for one); for every batched model, one batch holding
   an image request and a PDF request completes; and xAI's document input is given an image-only
   PDF, to learn whether it reads anything beyond a text layer (the answer is reported beside
   Grok's C1). A failure here changes the model list or a setting before `prereg-1`, never after.
2. **G0.** Write `g0.py`. Checks: the same seed gives byte-identical files; every quota is met;
   P-native finds every E1–E4 strike on G0-dev and reports nothing in N1–N3, both apart from the
   declared limitations. A miss or a false report that no limitation declares is either a
   generator bug or a limitation to declare before `prereg-1`.
3. **G1.** Write `g1.py`. Checks: over the whole corpus, the vector detector's count and the
   vector-flag agreement match `benchmarks/confirmation_rate.py` (55,171 vector detections, 99.85%
   confirmed); the page sample is reproducible from its seed; and B-naive's output aligns to at
   least 99% of the ground-truth words on every G1-dev page (it is the same text layer), so
   extraction order is not breaking the alignment.
4. **G2.** Write `g2.py`: list candidate bill versions after the cutoff date, fetch XML and PDF,
   align, apply the 98% rule. Check: at least 4 bills survive; the coverage report is saved.
5. **Inputs.** Write `render.py`. Checks: identical hashes on a rerun; C2 is 200 dpi; the bands of
   C2-tiled cover every pixel row.
6. **Free arms.** Write `free_arms.py` and run all four on dev. Check: the reconstructed live text
   equals `clean_text` on every dev page (Part A, Arms).
7. **Scorer.** Write `score.py` with golden tests first, at least: a struck word output live (a
   leak), output marked, and absent; `~~semi-~~monthly`; `~~December~~May`; a JSON answer with a
   partial `chars`; U+0336; a misread struck word; a hyphen split across a line end; a furniture
   token; a masked word; each failure kind.
8. **Statistics.** Write `stats.py`. Checks: the bootstrap reproduces by seed; Holm and the
   largest-p rule match hand-worked examples.
9. **Harness.** Write `harness.py` and the provider adapters, tested against the fake provider:
   caching (a rerun never calls an API), the model-ID check, resubmission, and `--dry-run` pricing.
   Then a real dry run of stage B2.
10. **The audit tool.** Write `audit.py`: items to an HTML page and a CSV, labels back in, κ.
11. **Tag.** Fill every TBD (B1), merge, and publish `prereg-1` (PLAN.md's steps).

## Stage B2: dev and the pilot

1. **The pilot:** 30 dev pages, the four ladder models at R2 on C2, plus the free arms. Score,
   run the pilot audit, fix what it finds, rescore.
2. **The ladder:** R0–R6 on about 20 dev pages, on C2 and C2-tiled, on each ladder model and each
   open vision model; the paraphrases; the repeats; the effort ablation. Choose R-best and the image
   condition, and confirm them on the flagships.
3. **Labels:** the G1 error-rate sample and the G2 span check.
4. **Freeze:** the scorer's hash, the prompts, the settings, then fill every TBD (B2) and publish
   `prereg-2`.

## Stage B3: the test run

1. A dry run of every test cell against the cap; apply the drop order if needed.
2. Submit each model's cells, canary first and last, within its 7-day window. Run the latency
   sample.
3. Collect, score and run the test audit. Fix at the source and rescore; never edit a number.
4. Report: the results table, the leakage-against-cost scatter, the failure gallery, and the
   per-stratum and per-document tables. Publish the responses, scores and labels.

## What the author provides

- API access for each provider in [Models](#models), each with a spend limit, the keys in the
  repo's git-ignored `.env`: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`,
  `DEEPINFRA_API_KEY` and `MOONSHOT_API_KEY` (or one `OPENROUTER_API_KEY` for both open models),
  and `DATALAB_API_KEY` unless Chandra runs from its checkpoint.
- The spending cap, and a final word on the drop order, before `prereg-1`.
- Zenodo's GitHub integration switched on before `prereg-1` is published.
- Two annotators, with their consent to be named: about 2 to 3 hours each at the pilot audit,
  about an hour for the G1 sample and the G2 check, and about 2 to 3 hours at the test audit.
- A named machine for the package's CPU timing.
