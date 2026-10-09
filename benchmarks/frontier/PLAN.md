# Frontier-model benchmark: plan

**Status: a working plan. Nothing here has run, and nothing is pre-registered yet.** The frozen
protocol will be `PREREG.md` in this directory, committed and timestamped in two steps (see
[Pre-registration](#pre-registration)) before any arm touches a test page. Where the two differ,
`PREREG.md` governs. Values marked *proposed* are this plan's defaults until `PREREG.md` fixes them.

**Names used below.** Questions Q1–Q6; ground-truth sets G0–G4; arms B-* (baselines), P-* (this
package), M (a model alone), M+P (a model with the package) and P→M (the package's text to a
model); inputs C1, C1′, C2, C2-tiled, C3 and C4; prompt rungs R0–R6, with R-best the rung picked
on dev; stages B0–B6 (see [Order](#order)), always written "stage B1" outside that list.

## The question

Can a frontier model do this package's job: find struck-through (deleted) text in a PDF or a scan,
and keep it out of the live text? No published number answers it. No strikethrough evaluation of
LLMs or VLMs turned up, and olmOCR-bench and OmniDocBench do not score it. The closest evidence is
[arXiv 2603.08497](https://arxiv.org/abs/2603.08497) ("Reading ≠ Seeing"), where font-style
detection is "universally poor" across 15 VLMs, with strikethrough not broken out.

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
   interval. On G0 and G2 the package wins if the paired 95% CI of model minus package leakage
   lies above 0.
2. Leakage at R1 (Q6), per flagship, on the C2 pages of G0-test and G2-test: a model matches the
   package if the paired CI of model minus package leakage lies within ±1 point and the upper
   bound of the difference in marked over-deletion is under 1 point.
3. Struck-word F1 at R-best (Q2), on the image condition chosen per provider on dev and fixed in
   `prereg-2`, model against package on G0-test and G2-test: the side whose paired CI excludes 0
   wins.
4. C1 against C2 leakage at R0 (Q3), per flagship: C1 leaks more if the paired CI of C1 minus C2
   lies above 0.

Comparisons across flagships are Holm-adjusted. Cells holding fewer than 200 struck sub-tokens or
fewer than 3 documents (*proposed*) read "insufficient data". Per-document results sit beside the
pooled ones.

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
- **G2 US Congress bills.** GPO bill XML marks deleted text with `<deleted-text>`; aligned to the
  rendered strike, that gives ground truth independent of this package, and new bills appear daily,
  so some postdate the latest training cutoff of every model in the run. A slice of 4 or more such
  bills (about 30 pages) joins the main run (stage B3). Alignment coverage is reported per page and
  pages under 98% are dropped; 100 aligned spans are hand-checked; margin line numbers are page
  furniture.
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
  - Anthropic: Claude Opus 5.5, Sonnet 5.5 and Haiku 4.5, plus Fable 5.1 on a subset as the
    ceiling.
  - Google: Gemini Pro and Flash.
  - OpenAI: the flagship and its mini.
  - One or two open-weights vision models (for example Qwen's VL line or Kimi-VL), served through a
    hosted API with the provider and the precision pinned: anyone can re-run them with the same
    weights.
  - One OCR-to-markdown model (olmOCR, Mistral OCR or PaddleOCR-VL), with its documented prompt,
    once it is checked to emit `~~` at all.

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
  extracted text).
- **C1′ Image plus text:** the C2 image with the page's extracted text as a text block, the same for
  every provider. It isolates the text layer for Q3.
- **C2 Page image:** 200 dpi, identical bytes for every model.
- **C2-tiled:** the same page as overlapping full-resolution bands, at R-best for every model. A
  thin strike can vanish when a provider downscales a page: Claude's high-resolution models take up
  to 2576 px on the long edge (about 4.8k image tokens), and Haiku 4.5 about 1.15 MP, so a 200 dpi
  page is downscaled for it. Effective pixels are recorded.
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
- **R-best is chosen per provider on dev** (*proposed*): the rung with the lowest leakage whose
  marked over-deletion stays within 1 point of R0's, ties going to the cheaper rung. It is chosen on
  one non-flagship model per provider (Sonnet 5.5, Gemini Flash, the OpenAI mini) and confirmed on
  that provider's flagship before the second pre-registration tag. If the flagship's best rung
  differs, it keeps the non-flagship's choice and the gap is reported. The image condition, C2 or
  C2-tiled, is chosen per provider on dev the same way.
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
  answer, a batch error after three resubmissions) scores worst case on both errors, with a
  sensitivity table without such pages. Refusal and parse-failure rates are reported, never
  dropped.
- **Other metrics:** struck-word precision, recall and F1 at sub-token level; character spans (G0,
  G2); live-text character and word error rates (CER, WER), so the package's OCR is not flattered;
  hallucinated tokens, digits above all; verbalized confidence against the package's `cnn_prob` as
  reliability diagrams, both over the same words: those on scanned pages the package scored.
- **Cost:** API spend from each response's `usage` (thinking included, at batch prices). The
  package: measured CPU-seconds per page on a named machine, OCR included, at a named on-demand
  rate per vCPU-hour (the local VLM per GPU-hour). The cost frontier's accuracy axis is both errors:
  an arm dominates only if it is better on leakage and on marked over-deletion.
- **Latency:** from a separate interactive sample, 30 pages per arm, sequential, back-off excluded;
  batch turnaround is not latency.
- **Properties, not contests:** determinism and native CER hold by construction. Word location
  counts only against an R4 variant that returns boxes (IoU ≥ 0.5).
- **Statistics:** paired bootstrap, 10,000 replicates with a fixed seed, pages nested in documents.

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
- **Two annotators** do the labelling: members of a related course project who had no part in the
  package's development, named in the write-up with their consent. Each reads the definitions
  above and labels 10 practice items with known answers. They then split the audit items and the
  G1 sample, with 25% of the items labelled by both, independently, and each checks every *no* the
  other gives on questions 2 and 3. Agreement on question 1 is reported as Cohen's κ with a CI;
  below 0.7, the definitions are revised and the round repeated. The author adjudicates
  disagreements, and the pre-adjudication labels are published. For a pilot audit of up to 140
  items (up to 20 per arm), that is about an hour each.

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
- Each model's test batch runs within 7 days, with a 10-page canary at the end; a model released
  meanwhile is an extra arm, never a replacement. Batch errors are resubmitted three times, then
  scored as failures.
- `--dry-run` prices a run before anything is spent: input tokens exactly by each provider's token
  counting, output and thinking tokens at the 95th percentile seen on dev (the pilot and the
  ladder) per model and rung.
- Requests go to each provider's own API, where batching (about 50% off at all three) and token
  counting are native; the open models go through a hosted API whose provider and precision are
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
  sha256s, cutoff date and alignment rule, the failure rule, the spending cap and the order in
  which arms are dropped if it is reached.
- **`prereg-2`, before any arm touches a test page** (end of stage B2): the scorer's code hash,
  the R-best and image-condition choices, the prompt texts with their paraphrases and few-shot
  images, the M+P and text-only prompts, and the per-model settings table.

No arm, free or paid, touches a test page before `prereg-2`.

Each tag:
1. Commit `PREREG.md` to `main` through a PR, and merge it.
2. Publish a GitHub Release on that commit:
   `gh release create prereg-1 --target <full commit sha> --title "Frontier benchmark pre-registration 1" --notes "PREREG.md sha256 <digest>" --latest=false`.
   The repository's releases are immutable, so the tag cannot move to another commit, and GitHub
   records when the release was published. A tag that does not start with `v` runs no publish
   job, and `--latest=false` keeps the package's release marked latest.
3. Zenodo archives the release, with a DOI: its GitHub integration does this for every release of
   the repository once switched on.

Every results file records both tags. A change after a tag is a logged deviation, with the results
shown both ways.

## Data and licences

The responses dataset's card states each source's licence: the corpus documents (the federal
documents are US government works, while the California documents and the municipal code carry
their own terms), the HWG crops (CC-BY 4.0, attributed), and the logbook pages (with the project's
consent). The G0 generator and its dev and test seeds are public; the held-back seed set is not, so
a future model trained on the public pages can still be checked.

## Budget

- The pilot: 30 dev pages × 3 models × R2, about $5.
- The prompt ladder on dev: about 20 pages × 7 rungs × 3 non-flagship models, plus paraphrases and
  C2-tiled, $15–25.
- The main run: about 150 test pages (G0, G1 and the G2 slice) at roughly $0.15 per page for all
  seven paid models together, with markdown-length answers (the open vision models, through a
  hosted API, add a few dollars). That is about $25 per prompt × input cell,
  so R0, R1 and R-best on C1 and C2 come to about $150, or $75 batched. C1′ at R0 and R1, C2-tiled
  at R-best and the R1 paraphrases add about $90–115 batched.
- The Fable 5.1 subset, the QA probe, the latency sample (unbatched) and C3: about $30–50.
- The handwritten crops about $5; repeats and effort ablations $10–20.
- All in for stages B2, B3 and B5: about $230–300, against a cap of $300 (*proposed*); the dry run
  prices it first, and the drop order decides if it is reached. If R-best is a JSON rung (R4–R6),
  output tokens rise several-fold. Stage B4 is priced by the dry run when it comes.

## Order

- **B0** Write `PREREG.md`: the questions, endpoints, win rules, arms, package calls and spending
  cap, and the model list.
- **B1** Build the G0 generator, the G1 page sampler, the scorer and its golden tests, and the
  harness with its dry run; run the free arms (B-naive, B-pm4llm, P-native, P-scan) on dev only.
  Freeze the data, and tag `prereg-1`.
- **B2** The pilot on dev (about $5): hand-audit, fix the scorer. The prompt ladder and C2-tiled
  on dev ($15–25): choose R-best and the image condition per provider and confirm them on the
  flagships. The G1 error-rate sample. Tag `prereg-2`.
- **B3** The test run, batched, with the G2 slice: every arm, the R1 paraphrases, the Fable 5.1
  subset, the QA probe, the interactive latency sample, the test-run audit, confidence
  intervals, the results table, a leakage-against-cost scatter ($ per 1,000 pages on a log axis),
  and a gallery of each arm's worst failures, such as a model quoting a deleted clause as in force.
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
- Model drift: model IDs are dated, each test batch runs within a week, and a canary closes it.
- Prompt sensitivity: a seven-rung ladder chosen on dev, plus the paraphrase spread.
- Scorer bugs: golden tests, the hash in `prereg-2`, and the audits.
- Annotation: two annotators from outside the package's development do the labelling, 25% of it
  double-labelled for κ, and the author only adjudicates.
- Small samples: the bootstrap over documents, per-document results, and "insufficient data" cells.
