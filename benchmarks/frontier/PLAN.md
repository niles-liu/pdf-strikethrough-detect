# Frontier-model benchmark: plan

**Status: a working plan.** The binding protocol is [`PREREG.md`](PREREG.md), a draft until it is
tagged in two steps (see [Pre-registration](#pre-registration)) before any arm touches a test page.
This file gives the reasons; where the two differ, `PREREG.md` governs. Where the work stands is in
the README beside this file in the private repository.

**Names.** Questions Q1–Q7; ground-truth sets G0–G5; arms B-* (baselines), P-* (this package), M
(a model alone), M+P (a model with the package) and P→M (the package's text to a model); inputs
C1, C1′, C2, C2-tiled, C3 and C4; prompt rungs R0–R6, with R-best the rung picked on dev; stages
B0–B7 (see [Order](#order)), written "stage B1" in prose.

## The question

Can a frontier model do this package's job: find struck-through (deleted) text in a PDF or a scan,
and keep it out of the live text? No published number answers it. ParseBench
([arXiv 2604.08538](https://arxiv.org/abs/2604.08538)) scores strikethrough markup on real
enterprise pages, but folds it into one semantic-formatting score with bold, superscripts and
headings. In [arXiv 2603.08497](https://arxiv.org/abs/2603.08497) ("Reading ≠ Seeing"),
font-style detection is "universally poor" across 15 VLMs, with strikethrough not broken out.
olmOCR-bench and OmniDocBench do not score it.

## How the results are used

The project's app sends each submitted page to the system that does best on its kind of page:
born-digital, an image of print, handwriting, or a logbook (detected per page, which the user can
override). So beside the pre-registered contests, the benchmark reports every system's accuracy,
cost and speed per set and input (Q7). The default routing rule is fixed in `PREREG.md` as an
engineering default rather than a claim, so its date is checkable: per kind of page, the free
system with the fewest misses, and as the paid option the cheapest system not shown to be more
than a point worse.
- **Free by default.** The app runs without paid services, so paid models are opt-in. The free
  route is the package, plus Chandra OCR 2's open weights if they mark strikes on dev (one GPU).
  The larger open-weights models are not a free route: Qwen3.8-27B at BF16 needs about 54 GB for
  its weights alone, and Kimi K3 runs on a hosted API.
- **Route or cascade.** A cascade runs the package first and sends only what it is unsure of to a
  model. Both are measured, and a cascade replaces a route only when it passes the same test at no
  more than half the cost.

## Ground rules

- Questions, primary endpoints, win rules, data, scorer and prompts are pre-registered before the
  results they decide can be seen.
- The models get their best shot: both input routes, a prompt ladder, explicit per-model settings
  (effort or thinking budget, output limit, image resolution) and a tiled-image condition.
- Ground truth is independent of the package where possible (G0, G2). Where it is not (G1), the
  package is not scored against it as a contest, and a hand-labelled sample measures its error
  rate.
- The package runs [frozen](#package-under-test-frozen), never tuned on the test split.
- Every loss sits in the same table as the wins, and the failure gallery shows every arm's worst
  failures, the package's included.
- Code, prompts, cached responses and annotation labels are public, so anyone can re-score the run,
  or re-run it with their own keys.

## Expected outcomes

Hypotheses, reported whichever way they come out.
- **The package should win** on default leakage on born-digital documents (Q1), on cost, and on
  partial-word character spans. It is also deterministic by construction, a property rather than a
  contest.
- **Models should win** on handwriting (G3) and may come close on, or win, scanned-path detection.
  Heavily degraded scans and the meaning of a strike (which struck value goes with which
  correction) are also likely model strengths, which this benchmark mostly does not score.

## Questions and primary endpoints

- **Q1 Default behaviour.** Asked only to transcribe, with no mention of strikes, how much deleted
  text do models pass through as live? The real-pipeline failure, and the headline.
- **Q2 Capability ceiling.** Told to mark struck text, how well do they detect it? Measured at
  R-best, on the image condition (C2 or C2-tiled) chosen per provider on dev.
- **Q3 Input route.** Does the PDF route leak more than the image? Hypothesis: yes, because the
  text layer still carries every struck word (a strike is a drawing, not a text attribute). C1′,
  the page image plus its extracted text as a text block, separates the text layer from each
  provider's own rendering.
- **Q4 Does the package help?** A model alone, the package alone, and a model given the package's
  output.
- **Q5 Cost and latency:** $ per 1,000 pages and seconds per page, per arm.
- **Q6 The prompting gap.** How much of the distance between default behaviour and the package can
  prompting close, at what cost, and does it trade leakage for over-deletion?
- **Q7 Routing.** For each kind of page, which system does best on accuracy, cost and speed? The
  tables come from the runs above; the choice is made from them after the results.

**Primary endpoints:** four, with their win rules and tests fixed in `PREREG.md`, all tested on G2
alone: R0 leakage against P-native on C1 and on C2 (Q1); R1 no worse than P-native, in each of R1's
four wordings, on leakage, on marked over-deletion and on text left out against the same model at
R0, each within 1 point (Q6); struck-word F1 at R-best against P-scan (Q2); and C1 against C2
leakage at R0 (Q3). A claim must hold both with failed pages scored worst case and with them left
out. Everything else is exploratory.
- **Why G2 alone.** Its ground truth is independent of the package, and its bill versions were
  published after every model's training cutoff. G0's strata are quotas fixed by design, two of them
  limitations the package declares, so a pooled G0 verdict would weigh them by quota: P-native leaks
  17% of G0-dev's struck text, nearly all of it in those two strata, and 0.7% without them. G0 is
  reported per stratum and pooled, untested; G1 per cluster, untested.
- **Why one page per bill.** A bill is a cluster. With 13 bills of up to 3 pages, one struck block
  of 364 words was 8.5 points of leakage, and the bills behaved like 11 independent samples; 45
  bills of one page each behave like about 24.
- **Why three bounds in endpoint 2.** A one-sided "no worse than the package" test, with the
  margin R-best itself uses: 1 point on G2 is about 20 struck sub-tokens, roughly one struck
  phrase. The third bound catches a model that obeys R1 by leaving text out. Every wording must
  pass, since an average could hide one that fails.

## Ground truth

- **G0 Synthetic redlines.** A seeded generator (`bench/g0.py`). Its text is novel, so no model
  has seen it, and its ground truth is exact by construction.
  - **Strata** (counts in `PREREG.md`): the strike encodings the package reads, and two it
    declares it can't (lines slanted over 2°, U+0336 combining strokes); single and double,
    coloured, thick, partial-word and multi-line strikes, and struck numbers beside a written
    correction; and Courier, a known limitation. Thick strikes are an expected model win against
    P-scan, which rejects heavy strokes as bold glyph strokes (see the known limitations); G0's are
    2–3 pt, under P-native's highlighter limit, so P-native reads them.
  - **Negatives:** strike-free pages, and pages with underlines and table rules.
  - **Seeds:** dev, test, and a private held-back set, so a future model trained on the public
    pages can still be checked.
  - Synthetic is not real: results are per stratum, and G0 is never tested.
- **G1 The public redline corpus** ([`corpus_manifest.json`](corpus_manifest.json): 10 documents,
  55.2k struck words, sha256-pinned).
  - **The package is the ground truth here.** It is the agreement set of the package's vector and
    flag detectors, and the flag detector is MuPDF's verdict on the same drawn lines, not
    independent evidence. Words only one detector finds are masked for every arm. On what remains,
    P-native and B-pm4llm agree with the ground truth by construction, so their G1 rows are
    flagged as such, never read as results. G1 scores the models (and P-scan, labelled
    in-sample); package-against-model claims are tested on G2 alone.
  - **G1 is the package's development corpus** (CHANGELOG 0.9.0 to 0.12.0): its native heuristics
    and its scanned path were checked against these documents. The scanned pages are excluded,
    which removes the Copyright Office document, so G1 samples nine documents.
  - **Its error rate is measured:** the author labels a sample of G1-test words from the page
    image, without seeing the ground truth or any output; the sample is fixed at `prereg-2` and
    labelled once the test run's answers are in, before they are scored. Words at least three of
    the four flagships' R-best arms dispute are relabelled the same way after the test run, and
    corrections apply to every arm.
  - G1's documents are public, so a model may have seen them, or their final text, in training. G0
    and G2 are the controls.
- **G2 US Congress bills.** GPO's bill XML marks struck text (`<deleted-phrase>`,
  `changed="deleted"`); aligned to the printed strike, that gives ground truth independent of this
  package; only Senate reported versions are used, since House ones leave the struck original out.
  New bills appear daily, so some postdate the latest training cutoff of every model in the run. A
  slice of 45 such bills, one struck page each, joins the main run (stage B3), and more at stage
  B6; three more bills are G2-dev, so the prompt ladder sees bills before the test run does. Most
  bills strike whole sections, and a wholly struck page says little about where a strike ends, so
  a page mixing struck and live text is preferred, and the bill with the most inline strikes
  joins.
- **G3 Handwritten strike-outs:** the HWG dataset
  ([Zenodo 21560739](https://zenodo.org/records/21560739), CC-BY 4.0): its `written` and
  `synthetic` parts and the ASAP and GoBo crops of `collected` (its IAM samples and the `SOW` part
  ship labels only). Crop level, "is this word struck?", by strike type: the package's CNN
  (StrikeNet) as shipped, against the models. Crop level is not page level: it says nothing about
  which words the geometry stage would have proposed.
- **G4 Logbook cells:** NARA logbook pages from the project, the app's main case, held out of all
  its training. They come from its hand-verified pages: chosen by a person, never by an arm, with a
  quota of pages holding struck cells (strikes are rare), and labelled on the dry-bulb column (each
  cell's live value, struck value, correction and strike type) with no arm's output in view, so no
  system is favoured. Scored by cell, not by page: full-page logbook ground truth is costly, and
  handwriting-recognition errors would swamp the strike signal. The outline, fixed in
  `PREREG.md` and specified by addendum once the labels exist:
  - **Pages:** about 10 G4-dev pages, to tune the prompt on, and about 50 G4-test pages holding
    about 100 struck cells.
  - **Unit:** one dry-bulb cell, sent as a crop of its labelled cell box with the rows above and
    below and the wet-bulb neighbour in view, the target boxed; one JSON prompt asks whether it is
    struck, its live and struck values, the strike type and a confidence.
  - **Measures,** per cell: missed strikes, false flags, value match, strike type, calibration, and
    $ and seconds per cell, on every cell and on the cells the package flags or leaves uncertain,
    which is where the app's second look would act.
  - **Arms:** the flagships, the ladder models and the package; about $35–60. The money is set
    aside before the test run (see [Budget](#budget)), and the target is November 2026.
- **G5 Handwritten pages:** BRESSAY ([Zenodo 11637681](https://zenodo.org/record/11637681), CC BY
  4.0), Brazilian Portuguese essays whose transcriptions already mark crossed-out text (`--text--`,
  or `--xxx--` when unreadable): page-level handwriting with ground truth people made, and no
  labelling needed. Every test page holding a readable strike (133) and 10 strike-free ones, since
  strikes are sparse (a median of 3 a page): R-best runs on all 143 and R0 on a 50-page core. It
  is prose, not tables, and public since 2024, so it carries G1's caveat; read here as a transfer
  test of a prompt tuned on print. P-scan's OCR backend is not built for handwriting, so its G5
  scores show that gap more than its strike detection; read leakage (leakage among the struck words
  a system read at all) and the unmarked share are reported beside leakage for that reason.

## Arms

- **B-naive:** PyMuPDF's plain `page.get_text()`, the default in most pipelines. It leaks every
  drawn native strike by construction, which shows the problem exists.
- **B-pm4llm:** `pymupdf4llm` markdown, whose `~~` comes from MuPDF's strikeout flag. Probably the
  strongest free competitor on born-digital PDFs. Its own OCR is off: it would make the baseline
  depend on which OCR engines the machine has.
- **P-native:** `detect_pdf(path, method="vector")`, the package's default call.
- **P-scan:** the scanned path with RapidOCR, the exact call in `PREREG.md`. Any other package
  configuration (`method="both"`, say) is a separate arm, never a replacement.
- **M, a model alone:** at least a flagship and a cheaper tier per provider, with explicit settings
  (effort or thinking budget, output limit, image detail), since provider defaults differ. IDs,
  settings and roles are in `PREREG.md`.
  - Anthropic: Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5, plus Fable 5.1 on a subset as the
    ceiling; Anthropic's IDs are pinned snapshots.
  - Google, on Vertex AI: Gemini 4 Argon and Gemini 3.8 Flash. Argon (GA 2026-10-13) replaced
    the Gemini 3.1 Pro preview, which Google could retire mid-run on short notice. No Gemini 4
    Flash exists yet, so 3.8 Flash picks Argon's prompt, and the flagship check tests that choice.
  - OpenAI, through Azure: GPT-6 Astra and GPT-6 Sol. Sol, the middle tier, rather than Luna, the
    smallest: the cheaper model picks the prompt its flagship runs, so it should behave like the
    flagship, as Sonnet does for Opus.
  - xAI: Grok 4.7 and Grok 4.3, whose fallback, `grok-4.20-0309-reasoning`, has the same price,
    images and batch discount, with a dated ID.
- **Hosted OCR:** Datalab's hosted Chandra, an OCR model its makers say now keeps strikethrough in
  its output, as two exploratory arms. **D-track** is its track-changes add-on, a "visual redline
  model" that marks deletions in PDFs and scans: the package's job sold as a product, and the
  closest commercial competitor found. **D-plain** is the plain conversion: does a top OCR model
  keep struck text live by default? Cheap (about $10–16 per 1,000 pages, inside the free tier);
  deferred to stage B7, after the frontier models.
- **A later package:** a later package version or a retrained model runs as an extra exploratory
  arm (`P-app:<version>`), in its own pinned environment and cache, and never chooses a model.
- **Deferred to stage B7, after the test run: the open-weights models.** The frontier models come
  first. None of these enters a primary endpoint, and released weights do not change, so running
  them later tests the same model. An addendum to `PREREG.md` fixes them before they run. They are
  hosted models, not a free route for the app (see [How the results are
  used](#how-the-results-are-used)), except Chandra OCR 2. Planned:
  - **Open-weights vision models,** each on a host serving its released weights at their released
    precision, with the provider pinned and routing fallbacks off, so anyone can re-run the same
    weights: Qwen3.8-27B (`Qwen/Qwen3.8-27B`, Apache-2.0) at BF16 on DeepInfra, and Kimi K3
    (`moonshotai/Kimi-K3`) at its native MXFP4 on Moonshot's own platform (`kimi-k3`), directly
    or through OpenRouter with `provider.only` and `allow_fallbacks: false`. If Kimi K3 fails a
    smoke test, Qwen3.5-397B-A17B at its official FP8 replaces it. DeepSeek is left out: its API
    shrinks an image to about 1,300 × 1,300 px, and its earlier vision model's name now points at
    another model.
  - **OCR-to-markdown:** Chandra OCR 2's open weights (`datalab-to/chandra-ocr-2`, 5B, free for
    research), run as documented on one GPU: the self-hostable twin of the hosted arms. It joins
    only if it emits strike markup on dev, and then runs one cell, its own prompt on C2, reported
    beside the free arms.
- **M+P context:** P-scan's reading of the page's scan (its struck words with their struck
  characters, and its live text) in the prompt, with the page image: the app's second look on
  scans. It needs no tool loop, so it is cheap, and it runs before the tool arm, on G0's and G2's
  test pages. A version given P-native's output is not run: on born-digital pages, model alone
  against P-native already answers it.
- **P→M text-only** (the package's `provenance_text` to a model, with no image): parked; it waits
  for an addendum.
- **M+P tool:** a real tool-use loop through an MCP (Model Context Protocol) server for the package,
  which also measures whether the model chooses to call it. A small run first (about 10 dev pages,
  one or two flagships) once that server exists; the full arm later.

## Inputs

- **C1 PDF upload,** each provider's native document input: the route a developer gets, whatever it
  passes the model. By the providers' docs, Claude and GPT-6 receive page images plus the extracted
  text (GPT-6's image at `detail: high`) and Gemini an image plus the embedded text (at
  `media_resolution: high`), both below their C2 settings, while xAI gives the model a search
  tool over the file. The stage B1 smoke test checked all but Claude's:
  - GPT-6 transcribes an image-only PDF and marks a born-digital page's underlines, so it gets
    the page image as well as the text.
  - Gemini's usage counts text and image tokens apart: Gemini 4 Argon gets both, Gemini 3.8
    Flash the page image alone.
  - Grok transcribes an image-only PDF with no search call, and counts no more input tokens for a
    born-digital page than for an image-only one, so it gets the page image alone.

  Each model's route is recorded as `c1_route`. Where a route passes text alone, C1's extra
  leakage is by construction, and endpoint 4 says so; no model's does. The open vision models
  (stage B7) have no document input, so no C1. Every model runs C1 at R0; only the flagships run
  it at R1 and R-best, since C1 matters for born-digital pages, where P-native is free and
  expected to win.
- **C1′ Image plus text:** the C2 image with the page's extracted text as a text block, the same for
  every provider, at R0, for the flagships. It isolates the text layer for Q3.
- **C2 Page image:** 200 dpi, identical bytes for every model.
- **C2-tiled:** the same page as overlapping full-resolution bands, at R-best where the provider
  chose it. A thin strike can vanish when a provider downscales a page: Claude's models take up to
  2576 px on the long edge or about 4.8k image tokens, so a 200 dpi page arrives at about 99%, and
  OpenAI's `detail: high` (GPT-6 Sol's fallback; the run asks for `original`) shrinks one to about
  81%. Effective pixels are recorded.
- **C3 Degradation,** after the test run, by addendum: a grid of degradation levels on G0-dev
  pages, up to the level where the author can no longer read 95% of a 100-word sample, reported per
  level; plus 20 G0-test pages printed and scanned. Until then the image-of-print route rests on
  clean renders.
- **C4 Handwritten crops** (G3).

## Prompts: a ladder

"You prompted it wrong" is the first rebuttal to any "models can't do X" result, and two prompts
cannot answer it. A ladder can, and turns the objection into a result. It targets these behaviours:
notice deletions unprompted; never present struck text as current; mark partial-word strikes; keep a
struck value beside its correction, as `provenance_text` does; say how sure it is rather than hide
doubt; judge from the image, not the PDF's text layer; and follow a fixed output format. Each rung
adds one thing to the rung before, so a change is attributable to it (the texts are in `PREREG.md`):

- **R0 neutral:** transcribe the page to markdown, with no mention of strikes (Q1).
- **R1 one system line:** never present struck text as current. The cheap fix an ingestion
  pipeline would actually add. If it removes most leakage, the package's case rests on cost, exact
  location and calibration, and the write-up says so.
- **R2 explicit:** wrap struck text in `~~`.
- **R3 + definition:** what counts as a strike and what doesn't, plus the warning that a PDF's
  text layer still holds deleted words, so judge from the image.
- **R4 + output format:** JSON per word, modelled on the package's `StruckWord` (`text`, `chars`,
  `verdict` of struck or clean), plus a `confidence` that carries any doubt. There is no `unsure`
  verdict: scoring has to read a word as struck or live, so an abstention would always count as a
  leak, and the R3-to-R4 step would measure the format and the abstentions at once.
- **R5 + procedure:** transcribe first, then re-inspect line by line for strokes through glyphs.
  The prompt-side twin of the effort setting; the analysis keeps the two apart.
- **R6 + few-shot:** three example crops (a struck word, an underline, a table rule) with their
  answers, cut from one G0-dev page that no dev list draws. The added image tokens are counted.

How the ladder is used:
- It is designed on the dev split only, and one ladder serves every provider: provider settings
  are recorded inputs, not prompt edits. The dev lists are drawn toward struck pages, across both
  G1-dev documents, with four G2-dev pages added to the pilot's and the ladder's.
- **R-best** is chosen per provider on its ladder model (Sonnet 5.5, Gemini 3.8 Flash, GPT-6 Sol,
  Grok 4.3) by the rule in `PREREG.md`, then the image condition at that rung (C2-tiled runs at that
  rung only). The rule takes the cheapest rung that keeps over-deletion within a point of R0's and
  leaves at most 2 points more struck text unmarked than the best such rung: on about 1,100 struck
  sub-tokens, rungs that close are tied, and the app inherits the chosen rung's cost and latency
  (a JSON answer runs several times longer than markdown). A rung failing more pages than R0, plus
  one, is out, and the rest are compared on the pages none failed, so one truncated answer cannot
  disqualify a rung. Each flagship then runs R-best and the runner-up and keeps whichever the same
  rule picks, so the check it pays for decides something.
- The test run carries R0, R1 and R-best for every model, and they answer Q6; the per-rung curve
  is exploratory.
- Three paraphrases each of R0, R1 and R-best give a paraphrase spread on dev, and the flagships'
  R1 paraphrases run on G2-test's image pages, for endpoint 2, where every wording must pass: a
  finding that holds for one wording only is not a finding.
- Sampling stays at each provider's default. Repeats on a dev subset measure nondeterminism, and
  low effort runs on the same pages, against the repeats' runs at high effort.
- **The QA probe:** questions whose answer a deletion changed ("what is the rate?" where 5% is
  struck and 3% inserted), asked with no mention of strikes and again with R1's system line: does
  the answer give the struck value?
- Automated prompt optimization (DSPy, GEPA, a model as optimizer) is not used: it overfits a small
  dev split and is hard to explain. It may be tried later, bounded to about 10 dev-only rounds, if
  the hand ladder leaves a gap that looks promptable.

Deliverables: the prompting curve on dev (leakage against over-deletion per rung, with cost), and
the minimal prompt line to use where the package can't run.

## Scoring

The scorer is the part most likely to be wrong, so it is built and tested before any paid run, and
its code is hashed into `prereg-2`. The rules are in `PREREG.md`; the design:
- **Units.** Ground-truth words are split at struck/live character boundaries into sub-tokens, so
  `~~semi-~~monthly` and `~~December~~May` have defined outcomes. Every arm, the package included,
  goes through the same edit-distance alignment on normalized tokens, not bag-of-words. The
  package is rendered from its word records, never its `~~` markdown.
- **Outcomes.** Each sub-token is output *live*, *marked* (by any marker on a fixed list) or
  *absent*. A misread word counts as present, live or marked by its markup, never absent. A
  notation the list misses would count as a leak, a bias toward the package, so before `prereg-2`
  every struck word the dev outputs leave live at R0 and R1 is read in context, and any notation
  that marks struck text and nothing else joins the list.
- **Leakage,** struck sub-tokens output live, is the headline and the dangerous error.
  **Over-deletion** is reported twice: live sub-tokens output marked, and output absent. Page
  furniture (running heads, page and line numbers) is excluded by a fixed rule.
- **Failures.** A page with no usable output scores worst case on both errors, so refusing a hard
  page never helps a model; and every claim must also hold with those pages left out of both sides,
  so a model's failures never make a package win. Refusal and parse-failure rates are reported,
  never dropped. The output limit is 64,000 tokens, since a JSON answer on the densest page runs
  21–29k tokens before any thinking.
- **Other metrics:** struck-word F1 at sub-token level, which scores partial-word strikes (G0's S5
  stratum); live-text character and word error rates (CER, WER), so the package's OCR is not
  flattered; hallucinated tokens, digits above all; read leakage, leakage among the struck words a
  system read at all, for handwriting; verbalized confidence against the package's `cnn_prob`, as
  reliability diagrams.
- **Cost:** API spend from each response's `usage` (thinking included, at batch or Flex prices
  where the request uses them), and beside it at list prices, which the app pays once the
  promotions end (Gemini 4 Argon's list price is twice its promotional one, before the batch
  discount). The free arms: CPU-seconds per page at a named on-demand rate per
  vCPU-hour, timed with onnxruntime pinned to one thread, as pages run when a batch fills every
  core; their latency comes from a second pass at default threads, as one page alone runs. That
  mirrors the models: discounted prices for cost, one call at a time for latency. The cost
  frontier's accuracy axis is both errors: an arm dominates only if it is better on leakage and on
  marked over-deletion.
- **Latency:** from a separate interactive sample, 30 pages per model across the test sets, G5's
  handwriting included, at R-best, the prompt the app would use, sequential, back-off excluded;
  batch turnaround and Flex's queueing are not latency.
- **The cascade:** each model's outcomes on the words P-scan is unsure of, from the same outputs,
  against P-scan's own: how much a model adds where the package would hand off. Read in the
  shipped confidence band and in three fixed ones, each with the share of words and of pages
  handed off and the $ per 1,000 pages: at page level the shipped band hands off 81% of dev pages,
  though only 7% of the words carrying a probability. It is a page-level stand-in for the app's
  second look, which works on crops and is measured on G3 and G4.
- **Properties, not contests:** determinism and native CER hold by construction.

## Checks and the audit

- **Golden unit tests:** hand-made outputs with known scores.
- **Self-check:** on G0-dev, P-native finds every strike it can read and reports nothing on the
  negatives, the declared limitations apart and reported separately.
- **G1's ground-truth error rate,** as above.
- **The hand audit** runs after the pilot (dev) and again on the test run, stratified by error kind
  (leak, missed strike, over-deletion, hallucinated token). Its three questions, asked without
  showing the arm, tell an arm's error from a ground-truth error or a scorer bug; each bug is fixed
  at its source and no number is overridden by hand (`PREREG.md`, The audit).
- **Who labels.** The author, alone, publishing the labels.
  - **Before `prereg-2`:** the pilot audit, a dev-only hunt for scorer bugs whose fixes `prereg-2`
    then freezes, and the G2 span check, which can change G2's labels.
  - **After the test run:** the G1 sample (about an hour, before the run is scored), then the test
    audit (about 3 to 5 hours) and the G1 relabel.

  So no labelling stands between the pilot and the test run. Independent annotators, people with
  no part in the package's development, would be stronger, with a double-labelled share giving
  Cohen's κ; the audit tool supports them, and an addendum can add them before the test audit.

## Package under test: frozen

- **Version:** `pdf-strikethrough-detect` 0.12.1 from PyPI, installed from `requirements.txt`, which
  pins every package; never a checkout. Every results file records `pdf_strikethrough.__version__`,
  the sha256 of the `strike_verdict_cnn.onnx` beside `pdf_strikethrough/cnn.py` (expected
  `fac2c51b…`, with `PDF_STRIKETHROUGH_MODEL_DIR` asserted unset), `get_model_meta()`, and the
  commit the harness ran at.
- **The pin moves only before `prereg-1`.** A known limitation is fixed, released and pinned
  before then, or not until the test run is scored.
- **The calls are frozen:** P-native and P-scan as given in `PREREG.md`.
- **Known limitations, declared in advance** (0.12.0's "Documented (not yet fixed)" list, the
  native and scanned paths' design, and the model card):
  - The default `method="vector"` reads near-horizontal drawn strokes, annotation appearances
    included. Strikes on content drawn rotated in text space need `method="flag"` or `"both"`;
    annotation forensics (author, date, colour) need `"annot"` or `"both"`.
  - U+0336 combining strokes are characters, not drawings: no method reports them.
  - On the native path, a line wider than 30% of the word's height reads as a highlighter, and a
    filled bar taller than 3.5 pt, or a line that leans more than 2° and rises more than 1.5 pt end
    to end, is not read as a strike.
  - A struck word narrower than 3 pt (a lone `I` at 10 pt) is not reported, and a strike stroke
    shorter than 6 pt is not read, so a struck word narrower than that (a single digit at 10 pt)
    is missed unless the stroke runs past it.
  - The underline exclusion band comes from the font's box, so for some fonts (Courier, Symbol) a
    line at the baseline counts as a strike.
  - On a scan, a strike over the first letters of a long word can read as a full-word strike, and
    a strike heavier than about 1.5 pt can be rejected as a bold glyph stroke
    (`lines.MAX_STROKE_RUN_PX`), partly on 12 pt text and entirely on large text.
  - In `provenance_text` and the `~~` markdown, a struck word that ends in `~` confuses the markup;
    the scorer reads word records instead, but the parked text-only arm would see
    `provenance_text`.
  - The flag detector can include the character a strike line ends against (`DecemberM`);
    character spans come from `vector`.
  - StrikeNet's training data is not recorded in this repository (see the model card).

## Harness and operations

The code, its environment and its commands are in the README beside this file in the private
repository, and the rules (the model-ID check, resubmission, the test window, the dry run, the
caps) in `PREREG.md`, Operations.
- Raw responses are cached with everything sent and `usage`, so re-scoring never calls an API.
- Requests go to each provider's API (Azure OpenAI for GPT-6, Vertex AI for Gemini), batched where
  the provider allows: about 50% off at Anthropic and Google, 20% off Grok 4.3, and nothing for
  Grok 4.7, which has no batch API. Azure offers no batch for GPT-6; its Flex tier gives the same
  50% off, and a request it turns away waits for the next run rather than go at full price, until
  five days of the model's window have passed, when what is left may go at the standard tier. A
  busy signal waits longer and never stops a run; five request or account errors in a row do. API
  keys come from the environment or the repo's git-ignored `.env`, never a committed file.
- The cap is enforced in code: `submit` prices what it is about to send, adds what is spent and
  in flight, and refuses past the cap in force; a model sent request by request is also stopped
  mid-stage once the cap leaves no room for its next request. Azure and Google alert on spend
  rather than stop it, so the cap is the stop; the account limits are the second line.
- A model version that changes mid-run is caught by the model-ID check; its test requests are sent
  again under the new version if the cap allows, else it is reported by version, outside the
  tested families.
- RapidOCR downloads its OCR models on first use; their sha256s go with every free-arm record, and
  the files are mirrored.
- Cached responses go to the Hugging Face dataset
  [`niles-liu/strikethrough-benchmark`](https://huggingface.co/datasets/niles-liu/strikethrough-benchmark)
  as `frontier/<run-id>/responses.jsonl`, pinned by sha256 in a manifest.
- `requirements.txt` is regenerated (the command is in its header) only between runs: a lock
  change is a new snapshot.

## Pre-registration

A pre-registration counts only if its date can be checked, and a commit's date is whatever its
author sets. So `PREREG.md` is tagged and archived twice: `prereg-1` before the pilot (end of
stage B1), and `prereg-2` before any arm touches a test page (end of stage B2). What each fixes is
in `PREREG.md`. The tags go in the package's public repository, which gets `PREREG.md` and what
`plans freeze` wrote at each tag (`config/prereg1.json`, then `config/prereg2.json`), and at
`prereg-1` also this file and `corpus_manifest.json`; everything else stays in the private
repository until it moves. Each tag:

1. Commit those files to the package repository's `main` through a PR, and merge it.
2. Publish a GitHub Release on that commit:
   `gh release create prereg-1 --target <full commit sha> --title "Frontier benchmark pre-registration 1" --notes "PREREG.md sha256 <digest>" --latest=false`.
   With immutable releases on, the tag cannot move to another commit, and GitHub records when the
   release was published. A tag that does not start with `v` runs none of the package's publish
   jobs, and `--latest=false` keeps the package's release marked latest.
3. The same files go to Zenodo by hand, as a new version of one record at each tag: Zenodo mints a
   DOI and keeps a dated copy outside the author's GitHub account, and the record cites the
   release.
4. Tag this repository's commit the same, the commit holding the lists (`git tag prereg-1`, then
   push the tag). The harness reads the tags in force from here, records them with every
   response, refuses a tag on a commit without its lists, sends stage B2 only once `prereg-1`
   exists, and stage B3 and the test pages only once `prereg-2` does.

`plans freeze` refuses while the code differs from its commit, so the commit it names is the code
that ran; `prereg-2` adds the sha256 of every code, prompt and config file and of every test
page's ground truth, so the public files pin the private code. Every response records the tags,
and every score and report also records the commit and which files have changed since
`prereg-2`. A change after a tag is a logged deviation, with the results shown both ways.

## Data and licences

The responses dataset's card states each source's licence: the corpus documents (the federal
documents are US government works, while the California documents and the municipal code carry
their own terms), the HWG crops and BRESSAY's pages (CC BY 4.0, attributed; BRESSAY's README also
asks for non-commercial research or teaching use and a citation of its ICDAR 2024 paper), and the
logbook pages (the project's). The G0 generator and its dev and test seeds are public; the
held-back seed set is not.

## Budget

Rough, from a token model of every request the plan sends, at batch and Flex prices; the dry run
prices each stage from the usage actually seen before it runs. Every model runs at high effort, so
thinking tokens, billed as output, swing the total most; so does R-best, since a JSON answer is
several times longer than markdown.

| Stages | No thinking, R-best markdown on C2 | 1.5k thinking tokens a request | 1.5k, JSON R-best on C2-tiled | 4k, R6 on C2-tiled |
|---|---|---|---|---|
| B1, the smoke tests | $5 | $7 | $7 | $10 |
| B2, the pilot, the ladder and the checks | $25 | $40 | $79 | $106 |
| B3, the test run | $132 | $273 | $777 | $1,078 |
| B1–B3 | $161 | $320 | $863 | $1,193 |
| B1–B3's cells never dropped | $130 | $248 | $725 | $984 |
| B4, M+P on G0 and G2 | $7 | $14 | $14 | $26 |

- **The test run** is about 10,200 requests: 177 test pages of G0–G2 (53 G0, 73 G1, 51 G2) and
  G5's 143, with the flagships' R1 paraphrases on G2 alone. GPT-6 Astra is about two fifths of
  the total.
- **Two caps,** so the money is fixed when it can be known: stages B1–B2's at `prereg-1`, from a
  dry run priced off the smoke tests (which overstates, pricing every choice not yet made at its
  costliest), and stages B1–B3's at `prereg-2`, from a dry run of the test run once R-best and the
  image condition are chosen, no lower than the cells never dropped. If it doesn't fit, the
  flagships' C1 at R1 and R-best go first, then their C1′, the Fable 5.1 subset and the QA probe.
- **A reserve of $150** above both caps is kept for G3 (about $5), G4 (about $35–60) and stage B4
  (about $14), so the test run can't send into the app's own case. Those figures assume 1.5k
  thinking tokens a request; the smoke tests saw several arms think well past 4k, where the three
  come to about $150.
- Stage B7 is priced when it comes: the hosted OCR arms, about $13 on the test pages, fit inside
  Datalab's free tier.

## Order

- **B0** Draft `PREREG.md`.
- **B1** Pin the models and smoke-test each one. Build and check the ground truth (G5 included), the
  inputs, the free arms (on dev only), the scorer, the statistics, the harness with its dry run,
  and the audit tool. Set stages B1–B2's cap from the smoke-priced dry run, freeze the data, and
  tag `prereg-1`.
- **B2** The pilot on dev: the author's audit triage, then scorer fixes. The prompt ladder on dev,
  then R-best on C2-tiled: choose R-best and the image condition per provider, and each
  flagship's rung. The notation check and the G2 span check. Set stages B1–B3's cap from the test
  run's dry run, and tag `prereg-2`.
- **B3** The test run, batched where the provider allows (GPT-6 at Flex), with the G2 slice and
  G5: every arm, the flagships' R1 paraphrases, the Fable 5.1 subset, the QA probe, the
  interactive latency sample, the G1 error-rate sample's labels, the test-run audit, confidence
  intervals, the results table, a leakage-against-cost scatter ($ per 1,000 pages on a log axis),
  and a gallery of each arm's worst failures, such as a model quoting a deleted clause as in force.
- **B4** The M+P context arm, on P-scan's reading of each scan.
- **B5** G3, the handwritten crops, once the crop input exists, and G4's logbook cells once their
  labels exist (targeted for November 2026), each by addendum.
- **B6** More of G2, and the M+P tool arm, a small run first.
- **B7** After the test run: the open-weights models, the hosted OCR arms, C3's degradation grid
  and the printed-and-scanned pages, and P→M, each by addendum.

## Threats to validity, stated in the write-up

- G1's ground truth is the package's own detectors, and G1 is its development corpus. Mitigated by
  masking, the independent error-rate sample, testing package-against-model contests on G2 only,
  and labelling P-scan's G1 results in-sample.
- The package's author designs the benchmark and labels its audits and the G1 sample. Mitigated by
  the pre-registration tags and archive, audit pages that never show the arm, a G1 sample labelled
  from crops before the test run is scored, the published labels, G0's fixed strata, and public
  responses.
- Synthetic pages and renders are cleaner than real scans; G2's printed bills and G5's scans are
  the checks until the printed-and-scanned pages run.
- Contamination: G0 and G2 are the controls, and a held-back G0 seed set covers future models.
  G5 is public, like G1; G4's labels exist nowhere public.
- Provider downscaling: effective pixels are recorded, and C2-tiled is the countermeasure.
- Model drift: model IDs are pinned, dated where the provider offers it (OpenAI has changed
  GPT-6's image handling in place once; Azure's versions are dated, and every GPT-6 response
  names the version that served it), and the model ID and version each response reports are
  recorded; each model's test requests run within a week, between canaries.
- Prompt sensitivity: a seven-rung ladder chosen on dev, plus the paraphrase spread.
- Scorer bugs: golden tests, the hash in `prereg-2`, and the audits.
- Annotation: one labeller, the author, so no agreement statistic; the labels are published for
  anyone to label again, and an addendum can add independent annotators.
- Small samples: exact sign-flip tests over 45 G2 bills of one page each, bootstrap intervals over
  clusters, per-cluster results, no intervals where there are too few clusters to read one (G1,
  G2-strict), and "insufficient data" cells.
