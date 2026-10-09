# Frontier-model benchmark: plan

**Status: a working plan. Nothing here has run, and nothing is pre-registered yet.** The frozen
protocol is [`PREREG.md`](PREREG.md), a draft until it is committed and timestamped in two steps
(see [Pre-registration](#pre-registration)) before any arm touches a test page. Where the two
differ, `PREREG.md` governs. Values marked *proposed* are this plan's defaults until `PREREG.md`
fixes them.

**Names used below.** Questions Q1–Q6; ground-truth sets G0–G4; arms B-* (baselines), P-* (this
package), M (a model alone), M+P (a model with the package) and P→M (the package's text to a
model); inputs C1, C1′, C2, C2-tiled, C3 and C4; prompt rungs R0–R6, with R-best the rung picked
on dev; stages B0–B6 (see [Order](#order)), always written "stage B1" outside that list.

## The question

Can a frontier model do this package's job: find struck-through (deleted) text in a PDF or a scan,
and keep it out of the live text? No published number answers it. ParseBench
([arXiv 2604.08538](https://arxiv.org/abs/2604.08538)) scores strikethrough markup on real
enterprise pages, but folds it into one semantic-formatting score with bold, superscripts and
headings. In [arXiv 2603.08497](https://arxiv.org/abs/2603.08497) ("Reading ≠ Seeing"),
font-style detection is "universally poor" across 15 VLMs, with strikethrough not broken out.
olmOCR-bench and OmniDocBench do not score it.

## Ground rules

- Questions, primary endpoints, win rules, data, scorer and prompts are pre-registered before the
  results they decide can be seen.
- The models get their best shot: both input routes, a prompt ladder, explicit per-model settings
  (effort or thinking budget, output limit, image resolution) and a tiled-image condition.
- Ground truth is independent of the package where possible (G0, G2). Where it is not (G1), the
  package is not scored against it as a contest, and independent annotators measure its error
  rate.
- The package runs [frozen](#package-under-test-frozen), never tuned on the test split.
- Every loss sits in the same table as the wins, and the failure gallery shows every arm's worst
  failures, the package's included.
- Code, prompts, cached responses and annotation labels are public, so anyone can re-score the run,
  or re-run it with their own keys.
- The package's author designs and runs the benchmark. The pre-registration, the independent
  annotators and the public responses are the checks on that.

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

**Primary endpoints:** at most four, each with a win rule; everything else is exploratory.
*Proposed:*
1. Leakage at R0 (Q1), per flagship model, on G0-test, G1-test and G2-test, with its confidence
   interval. On G0 and G2 the package (P-native) wins if model minus package leakage lies above 0
   on both C1 and C2.
2. Leakage at R1 (Q6), per flagship, on the C2 pages of G0-test and G2-test: a model matches the
   package (P-native) if model minus package leakage lies within ±1 point and model minus
   package marked over-deletion lies below 1 point.
3. Struck-word F1 at R-best (Q2), on the image condition chosen per provider on dev and fixed in
   `prereg-2`, model against P-scan on G0-test and G2-test: either side wins when the difference
   lies on its side of 0.
4. C1 against C2 leakage at R0 (Q3), per flagship, on G0-test and G2-test (G1 is reported, not
   tested): C1 leaks more if C1 minus C2 lies above 0.

`PREREG.md` decides each claim by an exact sign-flip test over clusters (documents, or G0 pages),
Holm-adjusted within the endpoint, with the intervals reported beside it. Cells holding fewer
than 200 struck sub-tokens or fewer than 6 clusters (*proposed*) read "insufficient data".
Per-cluster results sit beside the pooled ones.

## Ground truth

- **G0 Synthetic redlines.** A seeded generator grown from
  `examples/native_quickstart.py::build_redline_pdf`. Its text is novel, so no model has seen it,
  and its ground truth is exact by construction.
  - **Strata**, with per-stratum counts fixed in `PREREG.md`. Encodings: vector lines, filled
    rectangles, `/StrikeOut` and ink annotations, lines slanted over 2° and rising more than 1.5 pt,
    and U+0336 combining strokes. Styles: single and double, colour, thickness, partial-word,
    multi-line passages, and struck numbers beside a written correction. Fonts, including Courier
    (a known limitation below). Thick strikes, an expected model win: on the native path a line
    wider than about 30% of the word's height reads as a highlighter, and the scanned path rejects
    heavy strokes as bold glyph strokes (in a quick sweep a strike of about 1.5–2 pt or more on
    24 pt text gave no line at all, while 12 pt text kept strikes of about 2 pt and 3 pt only in
    fragments).
  - **Negatives:** strike-free pages, and pages with underlines and table rules.
  - **Seeds:** dev, test, and a private held-back set for later re-runs, hashed in `PREREG.md`. The
    generator's commit and every G0-test file's sha256 are tagged in `prereg-1`; the harness
    refuses a mismatch.
  - Rasterized with `_scanned.build_scanned_pdf` for the image route. Synthetic is not real: results
    are per stratum, and G0 never carries a headline alone.
- **G1 The public redline corpus** ([`../manifest.json`](../manifest.json): 10 documents, 55.2k
  struck words, sha256-pinned).
  - **The package is the ground truth here.** It is the agreement set of the package's vector and
    flag detectors, and the flag detector is MuPDF's verdict on the same drawn lines, not
    independent evidence. Words only one detector finds are masked for every arm and counted. On
    what remains, P-native and B-pm4llm agree with the ground truth by construction and M+P is
    handed it, so their G1 scores are reported as such, never as results. G1 scores the models (and
    P-scan, labelled in-sample, see the next point); package-against-model detection claims come
    from G0 and G2.
  - **G1 is the package's development corpus** (CHANGELOG 0.9.0 to 0.12.0): its native heuristics
    and its scanned path were checked against these documents. The manifest's `scanned_pages` are
    excluded from the G1 sample, which removes the Copyright Office document (all four of its
    pages), so G1 samples nine documents.
  - **Its error rate is measured.** Before any test call, the two annotators (see the
    [audit](#checks-and-the-audit)) label a stratified sample of G1-test words from the page image,
    without seeing the ground truth; the error rate is reported with a CI. Words on which most
    R-best model arms disagree with the ground truth are adjudicated the same way, and corrections
    apply to every arm.
  - **Sample:** per document, up to 10 struck pages stratified by strike density and partial-word
    strikes (all of them where fewer exist), and about 3 strike-free pages from each document that
    has any (five of the nine do). Two documents are dev; the page list and its seed are tagged.
  - G1's documents are public, so a model may have seen them, or their final text, in training. G0
    and G2 are the controls.
- **G2 US Congress bills.** GPO bill XML marks struck phrases with `<deleted-phrase>` and struck
  sections with `changed="deleted"`; aligned to the rendered strike, that gives ground truth
  independent of this package, and new bills appear daily, so some postdate the latest training
  cutoff of every model in the run. A slice of 12 or more such bills (about 40 pages) joins the
  main run (stage B3). Alignment coverage is reported per page and pages under 98% are dropped;
  the annotators hand-check 100 aligned spans; GPO's typesetting slug, margin lines, running foot
  and line numbers are page furniture.
- **G3 Handwritten strike-outs:** the HWG dataset
  ([Zenodo 21560739](https://zenodo.org/records/21560739), CC-BY 4.0): its `written` and
  `synthetic` parts and the ASAP and GoBo crops of `collected` (its IAM samples and the `SOW` part
  ship labels only). Crop level, "is this word struck?", by strike type: the package's CNN
  (StrikeNet) as shipped, against the models. Crop level is not page level: it says nothing about
  which words the geometry stage would have proposed.
- **G4 Logbook cells:** hand-verified pages from a related historical-logbook project, with its
  consent. The target domain for handwriting; lands with that project.

## Arms

- **B-naive:** PyMuPDF's plain `page.get_text()`, the default in most pipelines. It leaks every
  drawn native strike by construction, which shows the problem exists.
- **B-pm4llm:** `pymupdf4llm` markdown, whose `~~` comes from MuPDF's strikeout flag. Probably the
  strongest free competitor on born-digital PDFs.
- **P-native:** `detect_pdf(path, method="vector")`, the package's default call.
- **P-scan:** `detect_pdf(scan, ocr=rapidocr_backend(), scan_config=ScanConfig.confidence_free(),
  dpi=200)`. Any other package configuration (`method="both"`, say) is a separate arm, never a
  replacement.
- **M, a model alone.** At least a flagship and a small tier per provider:
  - Anthropic: Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5, plus Fable 5.1 on a subset as the
    ceiling.
  - Google: Gemini 3.1 Pro, a preview (Google has no stable 3.x Pro), and Gemini 3.8 Flash.
  - OpenAI: GPT-6 Astra and GPT-6 Luna, the family's smallest tier (GPT-6 has no mini).
  - xAI: Grok 4.7 and Grok 4.3.
  - Two open-weights vision models, Qwen3.8-27B and Kimi K3, each served through a hosted API at
    its released precision with the provider pinned: anyone can re-run them with the same weights.
  - One OCR-to-markdown model, Chandra OCR 2, whose documented prompt allows a strikethrough tag,
    run as documented once it is checked to emit strike markup at all.

  `PREREG.md` pins exact model IDs and a per-model settings table: effort or thinking budget (set
  explicitly; provider defaults differ), output token limit, maximum image resolution or detail, and
  region. Each result records the model ID the provider reports and the run date.
- **M+P context:** the package's struck spans and clean text in the prompt, with the page. It needs
  no tool loop, so it is cheap, and it runs before the tool arm.
- **P→M text-only:** `clean_text` / `provenance_text` to a text-only model, with no image. Does the
  image add anything?
- **M+P tool:** a real tool-use loop through an MCP (Model Context Protocol) server for the package,
  which also measures whether the model chooses to call it. Runs once that server exists.

## Inputs

- **C1 PDF upload,** each provider's native document input (Claude receives page images plus the
  extracted text). By xAI's docs, its document input treats a PDF as a text format and gives the
  model a search tool over it, not the page (stage B1 checks this with an image-only PDF); it is
  Grok's C1 all the same, since it is the route a developer gets. The open vision models have no
  document input, so no C1.
- **C1′ Image plus text:** the C2 image with the page's extracted text as a text block, the same for
  every provider. It isolates the text layer for Q3.
- **C2 Page image:** 200 dpi, identical bytes for every model.
- **C2-tiled:** the same page as overlapping full-resolution bands, at R-best for every model. A
  thin strike can vanish when a provider downscales a page: Claude's models take up to 2576 px on
  the long edge or about 4.8k image tokens, so a 200 dpi page arrives at about 99%, and OpenAI's
  `detail: high` shrinks one to about 81%. Effective pixels are recorded.
- **C3 Degradation:** a grid of degradation levels on G0 dev pages, up to the level where a person
  can no longer read 95% of a page, reported per level; plus 20 G0-test pages printed and scanned.
- **C4 Handwritten crops** (G3).

## Prompts: a ladder

"You prompted it wrong" is the first rebuttal to any "models can't do X" result, and two prompts
cannot answer it. A ladder can, and turns the objection into a result. It targets these behaviours:
notice deletions unprompted; never present struck text as current; mark partial-word strikes; keep
a struck value beside its correction, as `provenance_text` does; abstain with "unsure" rather than
guess; judge from the image, not the PDF's text layer; and follow a fixed output format. Each rung
adds one thing to the rung before, so a change is attributable to it:

- **R0 neutral:** "transcribe this page to markdown", with no mention of strikes (Q1).
- **R1 one system line:** "Documents may contain struck-through (deleted) text; never present it as
  current." The cheap fix an ingestion pipeline would actually add. If it removes most leakage, the
  package's case rests on cost, exact location and calibration, and the write-up says so.
- **R2 explicit:** wrap struck text in `~~`.
- **R3 + definition:** what counts as a strike (single, double, wavy, handwritten, partial-word) and
  what doesn't (underlines, table rules, borders, redaction bars), plus the warning that a PDF's
  text layer still holds deleted words, so judge from the image.
- **R4 + output format:** structured JSON per word, modelled on the package's record: `text`,
  `chars` and `verdict` (struck / clean / unsure) as in `StruckWord`, plus a `confidence`.
- **R5 + procedure:** transcribe first, then re-inspect line by line for strokes through glyphs. The
  prompt-side twin of the effort setting; the analysis keeps the two apart.
- **R6 + few-shot:** three example images (a struck word, an underline, a table rule) with their
  answers, taken from dev documents only. The added image tokens are counted.

Protocol:
- The prompt texts live in this directory. The ladder is designed on the dev split only: 2 G1
  documents and a G0 dev seed. The pilot draws from dev only.
- One ladder serves every provider. Provider settings are recorded inputs, not prompt edits.
- **R-best is chosen per provider on dev** (*proposed*): of the rungs that ask for marks (R2–R6),
  the one leaving the fewest struck sub-tokens unmarked (live or absent) whose over-deletion, marked
  and absent, stays no more than 1 point above R0's, ties going to the cheaper rung. It is chosen on
  one non-flagship model per provider (Sonnet 5.5, Gemini 3.8 Flash, GPT-6 Luna, Grok 4.3) and
  checked on that provider's flagship, which runs R-best and the runner-up, before the second
  pre-registration tag. If the runner-up does better there, the non-flagship's choice stands and the
  gap is reported. Each open vision model chooses for itself. The image condition, C2 or C2-tiled,
  is then chosen per provider at that rung the same way.
- The full test run carries R0, R1 and R-best for every model. The per-rung curve is exploratory;
  Q6 is answered on test by R0, R1 and R-best.
- Three paraphrases each of R0, R1 and R-best give a paraphrase spread on dev; the R1 paraphrases
  also run on test. Paraphrases and few-shot images are hashed before the first dev call. A finding
  that holds for one wording only is not a finding.
- Sampling stays at each provider's default. Three repeats on a 20-page dev subset measure
  nondeterminism; low against high effort runs on a subset.
- **The QA probe,** on a subset: questions whose answer a deletion changed ("what is the rate?"
  where 5% is struck and 3% inserted), asked with no mention of strikes and again with R1's system
  line, scored on whether the answer gives the struck value.
- Automated prompt optimization (DSPy, GEPA, a model as optimizer) is not used: it overfits a small
  dev split and is hard to explain. It may be tried later, bounded to about 10 dev-only rounds, if
  the hand ladder leaves a gap that looks promptable.

Deliverables: the prompting curve on dev (leakage against over-deletion per rung, with cost), and
the minimal prompt line to use where the package can't run.

## Scoring

The scorer is the part most likely to be wrong, so it is built and tested before any paid run, and
its code is hashed into the second pre-registration tag.
- **Units.** Ground-truth words are split at struck/live character boundaries into sub-tokens, so
  `~~semi-~~monthly` and `~~December~~May` have defined outcomes. Every arm, the package included,
  goes through the same text alignment: an edit-distance alignment on normalized tokens (case,
  punctuation, hyphenation, ligatures), not bag-of-words. The package is rendered from its word
  records (character spans, `final`), never its `~~` markdown.
- **Outcomes.** Each sub-token is output *live*, *marked* (by any marker on a list fixed in
  `PREREG.md`: `~~`, `<del>`, `<s>`, `[deleted: …]`), or *absent*. A sub-token aligned by
  substitution (a misread word) counts as present, live or marked by its markup, never absent.
  Normalization strips U+0334–U+0338, and a token that carried them counts as marked. A JSON
  answer is flattened word by word: a `struck` verdict marks the word's `chars` (the whole word if
  they are empty); `clean` and `unsure` count as live.
- **Leakage** = struck sub-tokens output live ÷ struck sub-tokens, the headline and the dangerous
  error. **Over-deletion** is reported twice: the share of live sub-tokens output marked, and the
  share output absent. Page furniture (running heads, page and line numbers) is excluded by a
  fixed rule.
- **Failures.** A page with no usable output (a refusal, an output-limit stop, an unparseable
  answer, a request error after three resubmissions) scores worst case on both errors, with a
  sensitivity table without such pages. Refusal and parse-failure rates are reported, never
  dropped.
- **Other metrics:** struck-word precision, recall and F1 at sub-token level; character spans (G0,
  G2); live-text character and word error rates (CER, WER), so the package's OCR is not flattered;
  hallucinated tokens, digits above all; verbalized confidence against the package's `cnn_prob` as
  reliability diagrams, both over the same words: those on scanned pages the package scored.
- **Cost:** API spend from each response's `usage` (thinking included, at batch prices where the
  request can be batched). The package: measured CPU-seconds per page on a named machine, OCR
  included, at a named on-demand rate per vCPU-hour. The cost frontier's accuracy axis is both
  errors: an arm dominates only if it is better on leakage and on marked over-deletion.
- **Latency:** from a separate interactive sample, 30 pages per arm, sequential, back-off excluded;
  batch turnaround is not latency.
- **Properties, not contests:** determinism and native CER hold by construction. Word location
  counts only against an R4 variant that returns boxes (IoU ≥ 0.5).
- **Statistics:** decisions by an exact paired sign-flip test over clusters (documents, or G0
  pages); intervals from a paired cluster bootstrap, 10,000 replicates with a fixed seed.

## Checks and the audit

- **Golden unit tests:** hand-made outputs with known scores.
- **Self-check:** on a separate G0 vector-line seed, the package scores 100% on its native path
  apart from the declared known limitations, reported separately.
- **G1's ground-truth error rate,** as above.
- **The hand audit** runs after the pilot (dev) and again on the test run, stratified by error kind
  (leak, over-deletion, missed or wrongly marked strike, hallucinated token). Its unit is a
  disagreement between an arm's output and the ground truth; an arm here is one system as scored (a
  model at one prompt and input, or a free arm). The harness writes the items as a self-contained
  HTML page and a CSV.
  1. First, on the page crop alone, with the word boxed: is it struck through? *struck* (which
     characters, if only some), *not struck*, or *can't tell*. Underlines, table rules, borders and
     redaction bars are not strikes.
  2. Then the ground-truth label is shown: is it right?
  3. Then the arm's output, in one neutral rendering, and the scorer's verdict: did the scorer
     read the output correctly?

  The arm is never shown. A *no* on question 2 is a ground-truth error, and on question 3 a scorer
  bug; each is fixed at its source and every affected score recomputed. No number is overridden by
  hand.
- **Two annotators** do the labelling: people who had no part in the package's development, named
  in the write-up with their consent. Each reads the definitions above and labels 10 practice
  items with known answers. They then split the audit items, the G1 sample and the G2 span check,
  with 25% of the audit and G1 items labelled by both, independently, and each checks every *no*
  the other gives on questions 2 and 3. Agreement on question 1 is reported as Cohen's κ with a
  CI; below 0.7 (*proposed*), the definitions are revised and the round repeated. The author
  adjudicates disagreements, and the pre-adjudication labels are published. A pilot audit of up
  to 160 items (up to 20 per arm) takes each annotator about 2 to 3 hours, and the test-run audit,
  160 items across the arms that enter a primary endpoint, about as long again. The G1 sample
  (sized in `PREREG.md`) and the G2 spans add about an hour per annotator.

## Package under test: frozen

- **Version:** `pdf-strikethrough-detect` 0.12.1 from PyPI, installed from
  [`requirements.txt`](requirements.txt), which pins every package; never the checkout. The repo's
  `src/` layout keeps the checkout off the import path, so a script run from the repo root still
  imports the installed wheel. Every results file records `pdf_strikethrough.__version__`, the
  sha256 of the `strike_verdict_cnn.onnx` beside `pdf_strikethrough/cnn.py` (expected
  `fac2c51b…`, with `PDF_STRIKETHROUGH_MODEL_DIR` asserted unset), `get_model_meta()`, and the
  commit the harness ran at.
- **The pin moves only before the first pre-registration tag.** A known limitation is fixed,
  released and pinned before then, or not until the test run is scored.
- **The calls are frozen:** P-native and P-scan as given under [Arms](#arms).
- **Known limitations, declared in advance** (0.12.0's "Documented (not yet fixed)" list, the
  native and scanned paths' design, and the model card):
  - The default `method="vector"` reads near-horizontal drawn strokes, annotation appearances
    included. Strikes on content drawn rotated in text space need `method="flag"` or `"both"`;
    annotation forensics (author, date, colour) need `"annot"` or `"both"`.
  - U+0336 combining strokes are characters, not drawings: no method reports them.
  - On the native path, a line wider than 30% of the word's height reads as a highlighter, and a
    filled bar taller than 3.5 pt, or a line that leans more than 2° and rises more than 1.5 pt end
    to end, is not read as a strike.
  - A struck word narrower than 3 pt (a lone `I` at 10 pt) is not reported.
  - The underline exclusion band comes from the font's box, so for some fonts (Courier, Symbol) a
    line at the baseline counts as a strike.
  - On a scan, a strike over the first letters of a long word can read as a full-word strike, and
    a strike heavier than about 1.5 pt can be rejected as a bold glyph stroke
    (`lines.MAX_STROKE_RUN_PX`), partly on 12 pt text and entirely on large text.
  - In `provenance_text` and the `~~` markdown, a struck word that ends in `~` confuses the markup;
    the scorer reads word records instead, but the text-only arm sees `provenance_text`.
  - The flag detector can include the character a strike line ends against (`DecemberM`);
    character spans come from `vector`.
  - StrikeNet's training data is not recorded in this repository (see the model card).

## Harness and operations

`benchmarks/frontier/`, reusing `_corpus.py`, `manifest.json` and `_scanned.py`.
- Raw responses are cached as JSONL with the request parameters, the model ID the provider reports,
  timestamps and `usage`, so re-scoring never calls an API. The harness fails on a model-ID
  mismatch, and provider fallbacks to another model are off.
- Each model's test requests run within 7 days, with a 10-page canary at the end; a model released
  meanwhile is an extra arm, never a replacement. Request errors are resubmitted three times, then
  scored as failures.
- `--dry-run` prices a run before anything is spent: input tokens exactly by the provider's token
  counting where it counts images, otherwise from the `usage` seen on dev for the same route and
  image size; output and thinking tokens, and xAI's billed document searches with the input they
  add, at the 95th percentile seen on dev (the pilot and the ladder) per model and rung.
- Requests go to each provider's own API, batched where the provider allows: about 50% off at
  Anthropic, Google and OpenAI, 20% off Grok 4.3, and nothing for Grok 4.7, which has no batch
  API. The open models go through a hosted API whose provider and precision are pinned and
  recorded. API keys come from the environment or the repo's `.env` (git-ignored), never a
  committed file.
- RapidOCR downloads its OCR models on first use: their sha256s are recorded and the files mirrored.
- Cached responses go to the Hugging Face dataset
  [`niles-liu/strikethrough-benchmark`](https://huggingface.co/datasets/niles-liu/strikethrough-benchmark)
  as `frontier/<run-id>/responses.jsonl`, pinned by sha256 in a manifest like the DI results.

The environment, from the repo root:

```bash
uv venv benchmarks/frontier/.venv --python 3.12     # git-ignored
uv pip sync --python benchmarks/frontier/.venv benchmarks/frontier/requirements.txt
```

`requirements.in` lists what the harness needs. Regenerate `requirements.txt` with the command in
its header only between runs: a lock change is a new snapshot.

## Pre-registration

A pre-registration counts only if its date can be checked, and a commit's date is whatever its
author sets. So each stage of `PREREG.md` is tagged and archived:

- **`prereg-1`, before the pilot** (end of stage B1): the questions, primary endpoints and win
  rules, the arms, the model list and the exact package calls, the G1 page list, seed and dev
  documents, the G0 generator commit, seeds and test-file sha256s, the G2 bill list with its
  sha256s and cutoff date, the failure rule, the spending cap and the order in which arms are
  dropped if it is reached.
- **`prereg-2`, before any arm touches a test page** (end of stage B2): the scorer's and the
  statistics' code hashes, the R-best and image-condition choices, the prompt texts with their
  paraphrases and few-shot images, the M+P and text-only prompts, the per-model settings table, and
  the G2 alignment rule, which the span check may change.

No arm, free or paid, touches a test page before `prereg-2`.

Each tag:
1. Commit `PREREG.md` to `main` through a PR, and merge it.
2. Publish a GitHub Release on that commit:
   `gh release create prereg-1 --target <full commit sha> --title "Frontier benchmark pre-registration 1" --notes "PREREG.md sha256 <digest>" --latest=false`.
   The repository's releases are immutable, so the tag cannot move to another commit, and GitHub
   records when the release was published. A tag that does not start with `v` runs no publish job,
   and `--latest=false` keeps the package's release marked latest.
3. Zenodo archives the release, with a DOI. Its GitHub integration archives every release of the
   repository once switched on, so it is switched on before `prereg-1` is published.

Every results file records both tags. A change after a tag is a logged deviation, with the results
shown both ways.

## Data and licences

The responses dataset's card states each source's licence: the corpus documents (the federal
documents are US government works, while the California documents and the municipal code carry
their own terms), the HWG crops (CC-BY 4.0, attributed), and the logbook pages (with the project's
consent). The G0 generator and its dev and test seeds are public; the held-back seed set is not, so
a future model trained on the public pages can still be checked.

## Budget

- The pilot: 30 dev pages × 4 models × R2, about $6.
- The prompt ladder on dev: about 20 pages × 7 rungs × 4 non-flagship models, plus paraphrases and
  C2-tiled, $20–30; the open vision models' own ladders add a few dollars.
- The main run: about 165 test pages (G0, G1 and the G2 slice) at roughly $0.15 per page for the
  seven Anthropic, Google and OpenAI models together, with markdown-length answers. That is about
  $25 per prompt × input cell, so R0, R1 and R-best on C1 and C2 come to about $150, or $75
  batched. C1′ at R0 and R1, C2-tiled at R-best and the R1 paraphrases add about $90–115 batched.
  xAI's two models add about $60–80, nearly all unbatched, and the open vision models a few
  dollars.
- The Fable 5.1 subset, the QA probe, the latency sample (unbatched) and C3: about $35–55.
- The handwritten crops about $5; repeats and effort ablations $10–20.
- All in for stages B2, B3 and B5: about $300–390, against a cap set in `PREREG.md`; the dry run
  prices it first, and the drop order decides if it is reached. If R-best is a JSON rung (R4–R6),
  output tokens rise several-fold. Stage B4 is priced by the dry run when it comes.

## Order

- **B0** Write `PREREG.md` (drafted): the questions, endpoints, win rules, arms, package calls and
  spending cap, and the model list.
- **B1** Pin the models and smoke-test each one, xAI's document input with an image-only PDF
  among the checks. Build the G0 generator, the G1 page sampler, the scorer and its golden tests,
  and the harness with its dry run; run the free arms (B-naive, B-pm4llm, P-native, P-scan) on
  dev only. Freeze the data, and tag `prereg-1`.
- **B2** The pilot on dev (about $6): hand-audit, fix the scorer. The prompt ladder and C2-tiled
  on dev ($20–30, and a few dollars for the open models): choose R-best and the image condition
  per provider and confirm them on the flagships. The G1 error-rate sample. Tag `prereg-2`.
- **B3** The test run, batched where the provider allows, with the G2 slice: every arm, the R1
  paraphrases, the Fable 5.1 subset, the QA probe, the interactive latency sample, the test-run
  audit, confidence intervals, the results table, a leakage-against-cost scatter ($ per 1,000
  pages on a log axis), and a gallery of each arm's worst failures, such as a model quoting a
  deleted clause as in force.
- **B4** The M+P context and text-only arms.
- **B5** G3, the handwritten crops, and C3.
- **B6** More of G2, G4's logbooks, and the M+P tool arm.

## Threats to validity, stated in the write-up

- G1's ground truth is the package's own detectors, and G1 is its development corpus. Mitigated by
  masking, the independent error-rate sample, scoring package-against-model contests on G0 and G2
  only, and labelling P-scan's G1 results in-sample.
- The package's author designs the benchmark. Mitigated by the pre-registration tags and archive,
  the independent annotators, G0's fixed strata, and public responses and labels.
- Synthetic pages and scans are cleaner than real ones; the printed-and-scanned G0 pages and G2 are
  the checks.
- Contamination: G0 and G2 are the controls, and a held-back G0 seed set covers future models.
- Provider downscaling: effective pixels are recorded, and C2-tiled is the countermeasure.
- Model drift: model IDs are pinned, dated where the provider offers it, and the model ID and
  version each response reports are recorded; each model's test requests run within a week, and
  a canary closes them.
- Prompt sensitivity: a seven-rung ladder chosen on dev, plus the paraphrase spread.
- Scorer bugs: golden tests, the hash in `prereg-2`, and the audits.
- Annotation: two annotators from outside the package's development do the labelling, with 25% of
  the audit and G1 items double-labelled for κ; the author only adjudicates.
- Small samples: exact sign-flip tests and bootstrap intervals over clusters, per-cluster
  results, and "insufficient data" cells.
