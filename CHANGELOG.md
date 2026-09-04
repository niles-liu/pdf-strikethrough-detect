# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.0] — 2026-09-04

Minor, not patch: adds public API (`ScanConfig.rescue_clean_chains`). The headline is the shaded-
and highlighted-block fix (issue #15), which changes scanned-path output on any page carrying a
coloured or grey ground. Pages without one are unaffected, and verified so: per-document predictions
on the 2170-strike recovery benchmark are identical to 0.10.0.

### Added
- **`benchmarks/confidence_veto.py --switches`** — scores all four combinations of the two
  provisional precision switches against the label set, reporting what each buys alone and how much
  they overlap. The `--ab` mode only ever compared the default against `ruled_forms()`.
- **`tests/test_corpus_flag_detector.py`** — regression guard for the crash below, in two halves.
  Corpus-gated tests run the flag detector over whatever PDFs are in `benchmarks/corpus/` and skip
  when it is unpopulated; version-guard tests pin `native.FLAG_MIN_PYMUPDF`, hold it equal to the
  `pymupdf>=` floor in `pyproject.toml`, and run everywhere, since CI has no corpus. Closes the gap
  that hid the crash: every other flag-detector test builds its PDF with fitz in-test, and those
  synthetic pages did not trigger it.

### Performance
- **`lines.strike_lines` ~1.5x faster**, which is ~75% of the per-page cost on the scanned path.
  `_stitch_fragments` (the hottest loop — thousands of fragments on a dense scan) rebuilt its
  candidate list as a fresh Python slice per fragment and tested every pair scalar-wise; the pair
  tests are now vectorized behind a sorted-order `searchsorted` bound, taking it from ~3.8s to ~0.46s
  on a dense page. The near-duplicate dedup made a scalar IoU call per candidate/kept pair (~257k on
  that page) and now runs one vectorized IoU pass per candidate. `_spine_fill` samples all
  perpendicular offsets in one indexing pass instead of looping.

  **Output is unchanged, and verified so:** every line field and every stitch-group partition was
  compared against the previous implementation across 41 corpus pages — 0 differences — and the
  corpus false-positive counts are identical (113 / 42 / 95 / 28).
- **`ScanConfig.rescue_clean_chains` — an independent second precision switch for degraded ruled
  scans** (issue #7 follow-up). **PROVISIONAL — see Stability.** `classify_lines` rejects a
  marginal-fill line whose every word OCRs *cleanly* as a glyph chain, on the principle that a real
  strike corrupts what it crosses. An escape spared such a line anyway when the page looked
  pen-edited (`edit_prior >= page_edited_min`) and the line was substantial, handing the decision to
  the CNN — which, saturated on faint scans, confirmed nearly all of them. That escape is now a flag.

  **`True` by default, and NOT bundled into `ruled_forms()`** — opt in per switch. The two switches
  are separate code paths and each costs recall on its own terms, so a caller who has validated one
  should not acquire the other by upgrading:

  | configuration | struck-final | false positives | recall |
  |---|---|---|---|
  | `ScanConfig()` | 115 | 113 | 2/3 |
  | `ruled_forms()` — printed-rule veto only | 44 | 42 (−63%) | 2/3 |
  | `ScanConfig(rescue_clean_chains=False)` — this switch only | 97 | 95 (−16%) | 2/3 |
  | `ruled_forms(rescue_clean_chains=False)` — **both** | 30 | **28 (−75%)** | 2/3 |

  Private 8-document ruled-table corpus, scored against its verified label set; **recall is unchanged
  at 2/3 in every configuration.** The two overlap on only 4 false positives (89 removed separately,
  85 together), so they are substantially complementary rather than redundant. Reproduce:
  `python benchmarks/confidence_veto.py --switches`.

  **Why the escape misfires on ruled forms** (measured 2026-08-05): `edit_prior` is the fraction of a
  page's words OCRing below 0.90, which on a degraded multilingual scan tracks *scan quality*, not pen
  edits. Across the corpus it is **anti-correlated with real edits** — the one page carrying genuine
  strikes scores `0.027`, below the `0.03` floor, while edit-free degraded scans score `0.09`–`0.32`.
  The escape is not needed for those strikes either: all three damage their word's OCR
  (`0.71` / `0.84` / `0.94` against neighbours at `0.98`–`1.0`), so the gate's primary condition
  already spares them. A row-local variant of the edit test was **rejected on measurement** — a
  damaged word in the line's own row band is present for 41 escape-taking lines and absent for 46.

  **Left ON by default** because that evidence is one document wide, and a real strike that leaves
  OCR undamaged is exactly the case the escape exists for. ⚠️ **Limitation:** the whole chain gate
  sits behind `confidence_gating`, so unlike the printed-rule veto this switch is **inert on
  confidence-free engines** (RapidOCR) and when words carry no confidence.

### Fixed
- **A word on a grey highlight block read as struck (issue #15).** Three independent causes, all in
  how colour reaches the classifier. A reporter's shaded contract page returned 41 strikes under the
  default `ScanConfig()` and 4 under a precision-tuned one; the page holds **none**.

  1. **`cnn.std_crop` inverted the shaded ground into an ink pedestal.** `1 - gray/255` maps a grey
     211 ground to a uniform 0.17 across the whole 32x160 net input, which StrikeNet reads as a wash
     over the word and scores at **p=1.00** — off-distribution, not borderline, which is why raising
     `cnn_p_hi` did nothing. The crop's paper ground is now taken as the **mode** of its light pixels
     (a percentile reads the white margin `PAD_Y` pulls in, and misjudged 18 of 41) and clamped back
     to white when it is shaded. Ordinary paper is untouched.
  2. **`lines.to_gray_u8` collapsed RGB with a channel mean.** PyMuPDF's `csGRAY`, which the PDF path
     renders through, uses Rec.709 luminance in linear light. A yellow highlighter lands at 170 under
     the mean and 248 under `csGRAY`, so the same page was readable through `detect_pdf` and solid
     ink through an RGB array. Converts properly now.
  3. **`lines.ink_mask` used one global Otsu for the whole page.** A single threshold cannot serve a
     page holding both white and shaded regions: once a block's ground is dark enough Otsu splits
     page-from-block instead of ink-from-paper and the block returns entirely as ink (ink fraction
     0.07 at ground 211, **1.00** at 195, and every downstream stage then finds nothing). Falls back
     to a block-wise background flatten, **gated** on the mask coming back implausibly inky rather
     than run unconditionally, because the geometry filters downstream are calibrated against the
     plain global mask.

  Fixes 2 and 3 are interdependent on strongly-coloured grounds — the fallback makes the words
  detectable at all, the crop flatten makes the CNN judge them correctly.

  **Measured.** Issue #15's page: 41 -> 2 under the default config, 160 -> 5 under
  `confidence_free()`, and **4 -> 0** under the reporter's own configuration. Manufactured colour
  grounds (image path), recall before -> after: magenta 0% -> 87%, pink 0% -> 92%, yellow 0% -> 83%,
  green 0% -> 87%, cyan 0% -> 82%. Manufactured grey shading over 610 known strikes: 35-58 false
  positives -> 0, at a cost of one recall point.

  **No regression.** The 2170-strike recovery benchmark is unchanged at 96.1% with **per-document
  predictions identical** to before, and the ruled-forms corpus improves (113/42/95/28 -> 109/38/92/25
  false positives) at unchanged recall.

  **Known limits.** `SHADE_WHITE_FRAC = 0.85` and `BG_INK_MAX = 0.35` each rest on a single measured
  document and are marked provisional in the source. A uniformly **dim** scan (ground ~110, no
  colour) still recovers ~4% and is a different failure, untouched here. The `ink_mask` fallback
  costs ~550ms when it fires; it fires on neither benchmark corpus.
- **`pymupdf` floor raised to `>=1.26.6` — 1.26.3/1.26.4/1.26.5 segfault in the flag detector.**
  `native_flag_strikes` extracts with `TEXT_COLLECT_VECTORS` (which is what populates
  `FZ_STEXT_STRIKEOUT`), and on those versions `get_text("dict", flags=…)` raises a native access
  violation inside `JM_make_textpage_dict` — on **page 0 of every document in the public benchmark
  corpus**, i.e. ordinary real-world PDFs, not exotic ones. It is a process-level crash, so
  `method='flag'` and `method='both'` take the host process down and leave the caller nothing to
  catch. Bisected rather than guessed: 1.26.5 crashes, 1.26.6 does not. `method='vector'` (the
  default) and `method='annot'` never collect vectors and were never affected.

  **The full test suite passed throughout**, because every existing flag-detector test builds its
  PDF with fitz in-test and those synthetic pages do not trigger it — so this shipped undetected.
  `tests/test_corpus_flag_detector.py` closes that gap. ⚠️ 1.26.3 was also **numerically different**
  on the scanned path where it did not crash (one corpus document scored 8 false positives instead
  of 10), so it was quietly wrong as well as crash-prone; the figures in this file are reproduced on
  1.28.2.
- **`native_flag_strikes` now raises instead of crashing on a bad PyMuPDF.** The floor above only
  binds a fresh resolve; an environment that already had 1.26.3–1.26.5 installed kept segfaulting
  with no traceback and nothing pointing at the cause. The detector checks
  `native.FLAG_MIN_PYMUPDF` first and raises `RuntimeError` naming the installed version, the
  upgrade command, and the two methods that still work. An unparseable version string fails
  **open** — unknown is not evidence of a bad build.

### Notes
- **Issue #7 is closed — the residual is *accepted*, not pending.** This supersedes the 0.10.0 note
  below, which said it stays open pending the hard-negative retrain. Nothing regressed and no further
  suppression shipped; what changed is the expected-value call. The acceptance bar on this document
  class is **zero** false positives, strikethroughs are near-absent on it to begin with (3 real
  strikes across 8 documents; a 315-document sweep turned up nothing the native path could not
  handle), and so the best available configuration — 28 false positives against 2 recovered strikes,
  ~14:1 against **after** a 75% improvement — is EV-negative. Precision tuning of this shape cannot
  fix that ratio, and zero-FP is not reachable by moving a CNN operating point at all, because the
  genuine and spurious distributions overlap fully at the top of the range.

  **Guidance for degraded ruled forms: use the native vector path and keep the scanned path off**, or
  route it to human review rather than an automatic scrub. If you do run it on this class, enable
  both switches — `ScanConfig.ruled_forms(rescue_clean_chains=False)` — and treat the output as a
  review queue. The retrain that would have made it unattended is **descoped**: the mechanism that
  could meet the bar is abstention with a conformal guarantee, which needs mass negatives in the
  failing regime, and no supply of them has been found. Full reasoning in the issue thread.

## [0.10.0] — 2026-07-29

Minor, not patch: adds public API (`ScanConfig.ruled_forms()`, `veto_printed_rules`, the
`straightness` line field). Nothing changes by default.

### Added
- **`ScanConfig.ruled_forms()` — printed-rule veto for degraded, heavily-ruled scans** (issue #7,
  follow-up to #4). **PROVISIONAL — see Stability.** On faint Statement-of-Facts–style forms a printed
  rule crossing text is indistinguishable from a pen strike at the point of attribution, so 0.9.1 cut
  *raw* detections but left the *post-gate* over-flagging. This opt-in drops any detected line that
  looks like printed furniture — **solid** (`fill > 0.88`) or **dead-straight** (perpendicular wobble
  `< 1.80` px @200 dpi) — in practice mostly short fragments (glyph strokes, pieces of rules) rather
  than full-width rules. Geometry-only, so it works on any OCR engine.

  **Off by default, and must stay so for general input:** on a clean scan a real strike is also solid
  and straight, so this trades pristine-strike recall for precision. On the private 8-document
  ruled-table corpus, scored against a verified label set, it cuts false positives **113 → 42
  (−63%) with recall unchanged**. Reproduce: `python benchmarks/confidence_veto.py --ab`.
- **`straightness` line field** on `lines.strike_lines` — perpendicular ink wobble (RMS px, normalized
  to `RENDER_DPI`), the metric behind the veto. **`None` when not measurable**; treat None as
  "unknown", never as "straight".
- **`benchmarks/confidence_veto.py --ab`** — scores TP/FP/FN against a `ground-truth.json` label set,
  flagging recall loss and unlabeled documents. A struck-final count is *not* a false-positive count:
  it includes the real strikes.

### Fixed
- **The veto could silently discard every detection.** A missing `fill` defaulted to `1.0`, above the
  "solid" bar, so callers passing hand-built line dicts to `classify_lines` lost all output with no
  error. Missing and `None` metrics now read as "not a rule".
- **An unmeasurable wobble read as dead-straight** (`0.0`, below the bar) and dropped the line; it is
  now reported as `None`.
- **`strike_lines` is 1.7–2.3x faster.** `straightness` ran a per-sample Python loop costing ~37% of
  runtime on every scanned page, including the default path where the veto is off. Now vectorized,
  verified value-identical on the corpus.

### Stability
`ruled_forms()`, `veto_printed_rules` and `straightness` are **provisional** and **outside the v1.0
stability contract**: a stopgap for one input class, calibrated on a single positive document, and
expected to be removed once the model handles ruled forms natively (issue #4·C). Pin exactly if you
depend on it.

### Notes
- **Corrects the unreleased 0.9.2 notes**, which claimed `≈111 → 40 (−64%)` with "both validated
  real-strike morphologies kept intact". The count mixed real strikes in with false positives, and the
  second "morphology" is itself a false positive — a cursive `ft` crossbar on a page that deletes
  nothing. The corpus holds **3 real strikes on 1 document**, so this veto rests on one morphology and
  its thresholds were partly tuned to preserve a false positive.
- **Issue #7 stays open.** The residual 42 false positives are printed form captions, faint
  letterhead, and handwritten entries where the classifier fires on the writer's own ink; only the last
  is unreachable by geometry, and all three are hard-negative material for the CNN retrain (issue
  #4·C). Directions rejected on measurement — DI `tables[]` cell geometry, detrended `straightness`, a
  line-length floor, a higher wobble bar — are recorded with their numbers at each site in the source.

## [0.9.1] — 2026-07-26

Correctness patch for scanned, tightly-ruled tables and forms analyzed via Azure Document
Intelligence (issue #4): the detector over-flagged clean printed text as struck. On a private
8-document ruled-table benchmark the struck-final count dropped **188 → 115** overall, and the
dangerous high-OCR-confidence over-flags (clean body text / header fields / footer boilerplate that
would be deleted downstream) dropped **95 → 22 (−77%)**, with **no real strike lost** and the
confidence-free (RapidOCR) path unchanged. Two root causes were fixed; the residual faint-scan false
positives are the known CNN-saturation limit tracked for a post-1.0 retrain (issue #4·C/D). No API
changes; pure precision.

### Fixed
- **Table-rule / underline ink test on in-band lines** (issue #4·B) — the strike-vs-underline
  through-glyph ink test (`_ink_above_below`) was skipped for in-band long lines, so a solid
  full-width table rule rode its high spine-fill straight to tier `auto`. `scanned.classify_lines`
  now runs the ink test for **every** hit and rejects an in-band, long (`len ≥ TABLE_RULE_MIN_LEN_IN`),
  high-fill (`fill ≥ FILL_STRONG`) line whose ink is **one-sided** (a rule/underline, not a strike).
  Real strikes shatter on the glyphs (fill below the solid-rule threshold) and keep ink on both
  sides, so they are unaffected — covered by the synthetic and full-height-glyph fixtures.
- **DI-confidence veto at the verdict stage** (issue #4·A) — StrikeNet saturates (prob → 1.0) on
  faint ~200-DPI scans and confirmed nearly every candidate; the confidence gate lived only inside
  the marginal-fill branch, so a high-fill line bypassed it and a word OCR'd at 0.99 could not
  defend itself. `detect.apply_cnn_verdict` now downgrades a struck candidate to `final=False` /
  `verdict="unsure"` (flagged `conf_veto`) when its OCR confidence is **> `ScanConfig.max_clean_conf`
  AND** it lacks corroborating strike geometry (a new per-record `geom_corroborated` flag: an
  in-band, both-sided-ink, shattered-fill strike). The veto is **conf AND missing-geometry, never
  confidence alone** (a genuinely struck high-confidence word keeps its geometry and is spared) and
  is **scoped to the calibrated-confidence DI path** — `ScanConfig.confidence_free()` / RapidOCR
  never trigger it, so weak-confidence recall cannot regress.

### Added
- **Struck-record fields** — scanned records now carry `geom_corroborated` (bool) and, when the
  DI-confidence veto fires, `conf_veto: True`.
- **Regression tests** — `tests/test_correctness_0_9_1.py` reproduces the issue-#4 geometry on
  synthetic rasters (one-sided table rule rejected; two-sided strike kept; the veto drops a
  high-confidence uncorroborated word, spares a corroborated one, and stays off under
  `confidence_free`).
- **Precision benchmark** — `benchmarks/confidence_veto.py` reports the struck-final count per
  document split by OCR confidence, over a local directory of scanned ruled-table PDFs + their
  cached DI results (private data, not committed; see `.gitignore`).

### Changed
- **CI / lint hygiene** (unrelated to the fix) — pinned ruff and made `[tool.ruff.lint]` `select`
  explicit (`E4/E7/E9/F`, the documented intent) so a floating newer ruff no longer fails `lint`
  with rules the project never opted into. `dev` extra and the `ruff-action` version now match.

## [0.9.0] — 2026-07-06

Prove it: the evidence & model program. This release ships the *machinery* — operating points,
calibration, an active-learning export, a digest-verified model loader, a reproducible training
script, and a demo — plus the **populated public benchmark corpus** those numbers stand on. All
pure-code, no new runtime dependencies. The StrikeNet weights + model card are now hosted on
Hugging Face ([`niles-liu/strikenet`](https://huggingface.co/niles-liu/strikenet)) and the demo is
live as a Space ([`niles-liu/strikethrough-demo`](https://huggingface.co/spaces/niles-liu/strikethrough-demo)).
Still deferred (needs a hand-labeled corpus + a torch run): retraining StrikeNet from data and the
R-cal precision/recall figure.

### Added
- **Selectable operating points** (R-cal) — `ScanConfig.recall_first()` (legal / audit: never miss
  a deletion) and `ScanConfig.precision_first()` (RAG / indexing: never drop live text) set the
  CNN's struck/clean decision threshold, overridable via the new `cnn_p_hi` / `cnn_p_lo` fields.
- **Threshold calibration** (R-cal) — new `pdf_strikethrough.calibration`:
  `threshold_for_recall` / `threshold_for_precision` (extreme threshold meeting a target on a
  labeled set), `conformal_threshold` (split-conformal, distribution-free recall floor of
  `1 - alpha`), and `pr_curve`. Picks an operating point *from data* instead of a magic number.
- **Active-learning crop export** (R-active) — `dump_crops` / CLI `--dump-crops DIR` writes every
  crop the scanned pipeline scored (as a PNG) plus its verdict and geometry evidence to a
  `crops.jsonl` manifest for labeling; the labeled set feeds retraining and calibration. A
  "contribute a failing page" issue template routes bug reports into the same format.
- **Digest-verified model download** (R-hf) — `cnn.ensure_model(url, sha256, …)` downloads a
  StrikeNet model to a local cache, verifies its sha256 before anything is loaded (a tampered host
  can't swap the graph), and points the loader at it. The shipped model is published at
  [`niles-liu/strikenet`](https://huggingface.co/niles-liu/strikenet) (`.onnx` + `.meta.json` +
  card); `st.ensure_model` is verified end-to-end against that URL. README "Hosted weights" block
  shows the one-call load.
- **Reproducible training script** (R-hf) — `training/train_strikenet.py` trains StrikeNet from a
  `dump_crops` labeling set, calibrates `p_hi` / `p_lo` conformally, and exports ONNX + meta with
  the crop geometry recorded (so preprocessing can't drift). Closes the model reproducibility hole.
- **Gradio demo** (R-space) — `demo/app.py` (+ requirements/README): drag in a PDF or scan, see the
  strikes boxed plus the struck-aware markdown and surviving text. Deployed live at
  [`niles-liu/strikethrough-demo`](https://huggingface.co/spaces/niles-liu/strikethrough-demo); the
  app best-effort pulls the hosted model via `ensure_model` (packaged weights as fallback, so local
  runs still work).
- **Corpus fetcher + populated benchmark corpus** (R-bench) — `benchmarks/fetch_corpus.py`
  downloads the manifest's PDFs into `corpus/` and verifies each sha256. `benchmarks/manifest.json`
  now lists **10 public regulatory redline PDFs** (US Copyright Office, FDIC, CEQ ×2, EPA ×3,
  California CCPA/CPPA, Gretna LA development code — 54.7k struck words, each sha256-pinned). The
  benchmark is one command to reproduce: `confirmation_rate.py` prints **99.8% of vector detections
  independently confirmed by the flag signal** (92.5–100% per document, 8 of 10 ≥99.9%).
- **Reproducible scanned-path benchmark** (R-bench) — `benchmarks/scanned_recovery.py` rasterizes
  the born-digital redline pages into image-only "scans" and scores the scanned path (OCR words +
  geometry + CNN) against the native detector's *known* strikes on the originals. On 3 documents /
  24 pages / 2,170 known strikes the scanned path recovers **95% (Azure DI) / 97% (RapidOCR)** of the
  strike set. `prep_scanned_di.py` is the one-time asset generator (rasterize + one Azure DI call,
  cached). This supersedes the DI-vs-original-pipeline `di_parity.py` / `ocr_backend_table.py`, which
  needed a vanished reference pipeline; both are kept as legacy for a genuinely scanned corpus.

### Fixed
- **RapidOCR ≥ 3.2 version gate** — `rapidocr_backend()` read `rapidocr.__version__`, which the
  `rapidocr` 3.x wheels don't set, so a valid install (3.9.1) was wrongly rejected as "too old". It
  now falls back to `importlib.metadata.version("rapidocr")`.
- **Demo launch crash** — `demo/app.py` referenced an unimported `tempfile` in its `__main__` block,
  raising `NameError` on `python demo/app.py` (and every Space startup). Removed the dead line.
- **`.env` now git-ignored** — the repo pattern for local secrets (`HF_TOKEN`, `CLAUDE_API_KEY`) was
  never in `.gitignore`; added `.env` / `.env.*` so a token file can't be committed.

### Changed
- **README** documents operating points, the calibration surface, the model-improvement loop
  (dump-crops → label → train → load), and expands the handwritten-strike scope note (the main known
  gap) with the failing-page contribution path.
- **Headline confirmation-rate figures reconciled to the reproducible corpus** — the README and
  `native.py` "99.9–100% on 12 public redline PDFs (33k struck words)" (from the original private
  dev corpus) is now the measured "99.8% on 10 public redline PDFs (54.7k struck words; 92.5–100%
  per doc)". The domain description was corrected to the documents actually in the corpus (federal
  & state regulatory redlines + a municipal code; the previous list named court rules / procurement
  clauses / university policy, which the published corpus does not contain).
- **OCR-backend / DI-parity claims reconciled to the reproducible benchmark** — the README's
  "RapidOCR 100% covered / ~99%" table and "reproduces the original Azure-DI pipeline to 99.5%
  (1477 vs 1484)" (both from the original private validation) are now the measured "95% (DI) / 97%
  (RapidOCR) of the native strike set recovered", produced by `scanned_recovery.py`.

## [0.8.0] — 2026-07-06

Scale & armor: robustness on hostile and high-resolution input, dashed/curve-drawn strikes,
column-aware reading order, and a batch CLI. All pure-code — no new dependencies or shipped assets.

### Added
- **Batch / directory CLI mode** — `pdf-strikethrough detect` accepts several files, a directory
  (its top-level PDFs/images/.docx), or a glob (`*.pdf`, expanded internally so it works on shells
  that don't glob). Output is JSONL (`--jsonl PATH`, one result object per line; `--json` behaves as
  JSONL here); `--jobs N` spreads files across worker processes. A single unreadable file yields an
  `error` line instead of aborting the run. Per-file output flags (`--markdown`/`--overlay`/cloud
  results/`--pages`) stay single-file. The picklable worker lives in the new `_batch` module so it
  resolves under both the console script and `python -m`.
- **Dashed & curve-drawn native strikes** (R-dash) — `native.horiz_strokes` now chains collinear
  sub-`MIN_STROKE_LEN` segments (dashes/dots) into runs before the length gate, and recognizes flat
  (near-horizontal) cubic-bezier `"c"` path items as strike segments. A dashed or bezier strike is
  detected like a solid line; an isolated short tick still isn't.
- **Malformed-PDF regression suite** (R-hostile) — truncated / header-only / not-a-PDF / byte-flip
  fuzz inputs are asserted to fail cleanly (catchable error) or recover, never hang or crash.
- **Validated-scripts scope** (R-cjk slice 1) — horizontal CJK strikes are regression-tested
  (native + scanned); the README documents the validated script/layout matrix. Vertical writing
  modes remain out of scope.

### Changed
- **High-DPI normalization + pixel-budget guard** (R-highdpi / R-guard) — scanned rasters above
  ~300 dpi are worked at the 200-dpi calibration point (accuracy-neutral; extra resolution is pure
  cost), and a single page is capped at 128 Mpix — a hostile huge-mediabox page auto-downsamples
  with a warning instead of OOMing. Output boxes are page fractions, so both are invisible to
  results. This resolution normalization keeps CNN crops on-distribution by construction, superseding
  the previously-planned dpi-proportional-pad + ONNX re-export. Documented in `SECURITY.md`.
- **Column-aware reading order** (R-layout) — `markdown` / `clean_text` / `passages` now read
  down each column of a two-column page instead of interleaving across the gutter; narrow-column
  tables are left as rows (a struck table row stays one passage) and strikes across a hyphenated
  line break still group into one passage.

## [0.7.0] — 2026-07-05

Beyond PDFs: the same strike detection now reaches raster images, Word documents, and the AWS/Google
OCR ecosystems, plus an audit-preserving text mode for RAG. All pure-code — no new dependencies.

### Added
- **Image-file input** — `detect_image_file(source, ocr=..., dpi=None)` and CLI support for
  `.png/.jpg/.tiff` (incl. multi-page TIFF), for photos/faxes/scans that never were a PDF. Every
  frame is treated as a scanned page (the pipeline is already image-native); DPI is taken from an
  explicit `dpi=`, else the image metadata, else 200. Returns the same result shape as `detect_pdf`.
- **Cloud-OCR adapters** — `words_from_textract(result)` (AWS Textract) and `words_from_docai(document)`
  (Google Document AI), mirroring `words_from_azure_di`: each converts a pre-fetched result into
  `{0-based page: [Word]}`. Neither cloud flags strikethrough natively — feed the result to
  `detect_pdf(pdf, words_by_page=...)` (new provider-neutral parameter) or `detect_image_file`. CLI
  gains `--textract-result` / `--docai-result`. These confidences aren't calibrated to the scanned
  classifier, so `words_by_page` defaults `scan_config` to `ScanConfig.confidence_free()`.
- **DOCX detection** — `strikethroughs_in_docx(source)` reads strike character formatting
  (`w:strike`/`w:dstrike`) and tracked deletions (`w:del`, carrying `docx_author`/`docx_date`) from
  a Word document. Records use `tier="docx"` with a `para` index instead of `bbox_frac`/`page`.
  Stdlib-only (a .docx is a zip of XML) — no new dependency. The CLI routes `.docx` inputs here.
- **Provenance text for RAG** — `provenance_text(result, template="[deleted: {}]")` and
  `markdown.mark_provenance`: keep struck spans as `[deleted: …]` markers instead of removing them
  (contrast `clean_text`), so struck text entering a vector index is recorded-as-deleted rather than
  silently surfaced. CLI gains `--provenance PATH`; `examples/rag_provenance.py` demonstrates it.

### Changed
- CLI `detect` infers the input kind (PDF / image / .docx) from the file extension; `--dpi` now
  defaults to the image metadata for image files (still 200 for PDFs). `--json` carries the new
  `para`/`docx_*` evidence fields; `--overlay` warns and is skipped for non-PDF input.
- **Fixed a clobbered `## [0.5.0]` heading** in this changelog (the 0.6.0 edit dropped it, orphaning
  the 0.5.0 notes under the 0.6.0 section).

## [0.6.0] — 2026-07-05

Annotations & evidence: complete the *evidence story* for the signals already detected — an
explicit annotation pass with redline forensics, stroke color/width on vector records, a visual
overlay, a library logger, and CLI/API naming unification. Code-only, no new dependencies or
shipped assets.

### Added
- **Explicit `/StrikeOut` annotation pass** (`native_annot_strikes`, `method="annot"`, folded into
  `method="both"`). Reads the redlines Acrobat/Preview/editors write as annotation objects
  (QuadPoints over the struck text) — distinct from the vector drawings the geometry path reads and
  the font-attribute flag path — and snaps them onto the page words (partial spans supported).
  Records carry `tier="annot"` plus **flattened-redline forensics no extractor exposes**:
  `annot_author` (/T), `annot_created`, `annot_modified` (the "who struck this, and when"),
  `annot_color`, and `annot_id`. Hidden annotations paint no ink and are skipped.
- **Stroke color/width on vector records** (`stroke_color`, `stroke_width`). `get_drawings()`
  already carries them; the native geometry path now reports the dominant contributing stroke's
  paint (RGB in [0, 1]) and thickness (pt). Pen-color conventions (red = opposing counsel) are
  evidence in legal review. Both keys are surfaced in the CLI `--json` output alongside the
  annotation forensics.
- **Visual overlay** — `render_overlay(source, result=None, dpi=150, pages=None)` renders each
  struck page to a PIL image with strike boxes drawn (red = full, orange = partial); `save_overlays`
  writes them to disk; and the CLI gains `--overlay PATH` (+ `--overlay-dpi`). Useful for debugging,
  `ScanConfig` tuning, and the before/after documentation figure. No new dependency — PyMuPDF
  renders, Pillow draws (both already core).
- **Library logger.** The package attaches a `NullHandler` to the `pdf_strikethrough` logger (silent
  by default) and logs pipeline diagnostics at DEBUG — page routing, native detector method + record
  counts, OCR and geometry+CNN timings. `warnings` stays reserved for caller-facing hazards.

### Changed
- **`detect_pdf` native-page selector renamed `native_method` → `method`**, matching
  `strikethroughs_in_pdf`, `page_strikes`, and the CLI `--method`. `native_method` remains as a
  deprecated alias (emits a `DeprecationWarning`; passing both with different values raises
  `ValueError`) so 0.5.x callers keep working. The CLI is unchanged (`--method` already matched).
- CLI `--method` gains the `annot` choice; `--json` evidence now includes `stroke_color`/
  `stroke_width`/`annot_*`.
- **CI early-warning legs** (`.github/workflows/ci.yml`): a weekly `deps-latest` job against
  newest + pre-release deps (the `fitz`→`pymupdf` rename bit once); a `lowest-bounds` job that
  installs the exact `>=` floors on Python 3.10 to prove they're real; and a `torch-fallback` job
  that removes onnxruntime and loads a `.pt` through the CNN's torch fallback — a shipped code path
  CI never touched.

## [0.5.0] — 2026-07-05

Reachable & credible: new public API/CLI surface, inline typing, runnable examples, a reproducible
benchmark harness, and release/CI hardening.

### Added
- **`pages=` and `progress=` on `detect_pdf`.** `pages=` restricts work to a subset of 0-based
  page indices (negatives index from the end; out-of-range raises `IndexError`), so a 300-page
  scan no longer has to OCR every page or nothing; the result gains a `pages` key aligned to
  `page_sources`. `progress=` is a `progress(completed, total, page_index)` callback fired after
  each page, so long OCR+CNN runs aren't silent.
- **CLI feature parity with the API.** `--version`; `--pages 1-5,12` (1-based); `--di-result PATH`
  (use a pre-fetched Azure DI result from the CLI); `--scan-config auto|azure-di|confidence-free`;
  `--fail-if-found` (exit 3 for CI gating); `-` for stdin PDF and for stdout on
  `--json`/`--markdown`/`--clean-text`; per-page stderr progress on a TTY;
  `ArgumentDefaultsHelpFormatter` and documented exit codes. JSON output now carries
  `schema_version`, a `warnings` list, evidence fields (`coverage`/`score`/`cnn_prob`/`cnn_agrees`/
  `conf`) alongside each word, and is written with `ensure_ascii=False`.
- **Typed public surface.** New `pdf_strikethrough.types` module with `StruckWord`, `Passage`, and
  `DetectResult` TypedDicts (tier-dependent keys documented); ships `py.typed` so downstream type
  checkers see the annotations. Re-exported from the top-level package.
- **`strikethroughs_in_pdf` warns on scanned input.** A scanned page has no vector strikes, so the
  function still returns `[]` for it — but now emits a `UserWarning` naming the scanned pages
  (previously a silent `[]`, the package's most dangerous confusion).

- **Examples.** `examples/native_quickstart.py` and `examples/scanned_quickstart.py` — each builds
  its own sample PDF, so every quick-start snippet is copy-paste runnable with no assets to fetch.
- **Benchmark harness.** `benchmarks/` with a manifest-driven, sha256-verified corpus loader and
  three reproducible scripts: `confirmation_rate.py` (vector↔flag agreement — the headline native
  claim), `ocr_backend_table.py` (per-backend struck-region coverage vs an Azure DI reference), and
  `di_parity.py` (the "1477 vs 1484" DI-pipeline parity). The corpus (public PDFs) is downloaded
  locally, not committed.
- **Project meta.** `CONTRIBUTING.md`, `SECURITY.md` (the package parses untrusted PDFs),
  `CITATION.cff`, an API-surface table in the README, and a `[torch]` install extra documented.

### Changed
- **Version is single-sourced** from `pdf_strikethrough.__version__` via `dynamic = ["version"]`
  (no more lockstep bump of `pyproject.toml` + `__init__.py` at release time).
- **Scanned auto-tier verdict is now consistent with the ship decision.** A geometrically-"auto"
  word the CNN votes down (`final=False`) no longer keeps a misleading `verdict="struck"`; it
  reports the CNN's read (`clean`/`unsure`). Kept words are unchanged. Invariant: `verdict=="struck"`
  ⇒ `final is True`.
- **CI hardening.** Added a lint job (ruff), a build job that installs the built wheel and runs the
  tests against it, and an extras import-smoke job; the test matrix now spans Linux/macOS/Windows
  and Python 3.10–3.14 (3.14 + `Development Status`/`OS Independent`/`3 :: Only` classifiers added).
  `publish.yml` now tests the built wheel before publishing and adds a `workflow_dispatch` →
  TestPyPI rehearsal with explicit attestations. Added Dependabot for actions + pip.
- **Docs accuracy.** README record schema now lists `tier` and the scanned-only evidence fields;
  the CLI section documents every flag and exit code; the "RGB/float coerced" note moved to the
  function that actually coerces; `render_page_gray` documented as always-uint8.
- `MANIFEST.in`: dropped the dead `recursive-exclude test_docs` line; added `py.typed`.
- The `98%+` confirmation figure in `native.py` reconciled to the README's `99.9-100%` (12 PDFs,
  33k words) and pointed at `benchmarks/confirmation_rate.py`.
- **Native detection extracts each page's word list once.** Under `native_method="both"` a native
  page ran `get_text("words")` up to four times (page-source classification + vector detector +
  flag detector + the markdown word match). The detectors now take an optional `words=` argument
  and `detect_pdf` threads a single per-page extraction through all of them (down to two: the
  classifier's own probe plus one shared pass). `native_page_strikes`/`native_flag_strikes`/
  `page_strikes` gain the `words=` parameter; behavior is unchanged when it's omitted.

### Documented (not yet fixed)
- CNN crop geometry (`PAD_X`/`PAD_Y`) is fixed-pixel and calibrated at 200 dpi; it drifts
  off-distribution at other resolutions. The dpi-proportional fix needs a model re-export and is
  deferred to be done with the high-DPI normalization work (roadmap R-highdpi).
- Native detection is **horizontal (left-to-right) text only** — vertical writing modes and
  non-Latin scripts whose strikes run along a different axis are out of scope (`native.py` module
  docstring). Full support is roadmap R-cjk.

## [0.4.1] — 2026-07-05

### Fixed
- **Encryption gate now recognizes authenticated documents.** Detection gated on
  `needs_pass`, which stays `True` even after a successful `doc.authenticate(password)`, so the
  exact workflow the error message recommends still failed. Both entry points now gate on
  `is_encrypted` through a single shared helper, and `open_pdf` closes the freshly-opened
  document before raising `EncryptedPdfError` (previously it leaked, locking the file on Windows).
- **CLI no longer crashes writing non-cp1252 output.** All markdown/clean-text/JSON writes open
  with `encoding="utf-8"`, `sys.stdout`/`sys.stderr` are reconfigured with `errors="replace"`,
  and the `RuntimeError` open-failure handler is scoped to the open step only (mid-run
  onnxruntime/PyMuPDF errors are no longer mislabeled "cannot open FILE").
- **Page-source classification.** Clipped image rectangles are now unioned on a coarse boolean
  grid instead of summed (the same image placed three times no longer reads as 3× its coverage);
  a page is routed to "scanned" only when heavy image coverage coincides with invisible text or
  no content-stream drawings. A "scanned"-classified page that still has extractable text and no
  OCR backend falls back to the native detector with a warning instead of raising. A partial
  `di_result` now honors `on_missing_ocr="skip"`.
- **Sloped scanned strikes attribute to the right words.** The stroke-y is interpolated at each
  word's x-midpoint from the line endpoints instead of using one global bbox-center for the whole
  line, so a multi-word pen strike at a few degrees of slope no longer misses every word.
- **Native detection.** The flag path now applies the vector path's grazing guard after merging
  spans (no more spurious partials from overshoot into a neighbor word); invisible strokes
  (white / background-colored / zero-opacity lines) are skipped instead of confirmed as strikes.
- **OCR input safety.** Pixel-coordinate `Word` boxes raise instead of silently reporting
  all-clean; the RapidOCR adapter checks for the `>=3.2` result shape and raises a clear error on
  older installs; `words_from_azure_di` raises on missing/zero page dimensions.
- **Model loading.** `PDF_STRIKETHROUGH_MODEL_DIR` is read at load time (the documented
  "set it then import" override no longer no-ops); `torch.load` uses `weights_only=True`; the
  shipped `meta.json` crop/pad geometry is validated against the code constants.
- **Low-DPI and wide-dtype scans.** Scaled stroke-run/stitch thresholds are floored so 72–100 dpi
  scans keep thin strikes; 16-bit and other wide integer rasters are rescaled to 8-bit instead of
  saturating to all-white (`to_gray_u8`) or wrapping mod-256 (`std_crop`).

### Changed
- `import fitz` replaced with `import pymupdf` throughout (the `fitz` alias is deprecated upstream
  and collides with the abandoned `fitz` PyPI package).

## [0.4.0] — first public release

- Detect struck-through (deleted) text in born-digital PDFs (exact vector/flag detection) and scanned pages (stroke geometry + OCR + ONNX CNN), with `~~struck~~` markdown, clean text, and grouped passages via a Python API and CLI.

> Versions 0.1–0.3 were internal pre-release iterations and were never published to PyPI; 0.4.0 is
> the first public release.
