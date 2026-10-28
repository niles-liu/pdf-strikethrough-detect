# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`benchmarks/frontier/` holds a pre-registered benchmark against frontier models.** Can they do
  this package's job, finding struck text and keeping it out of the live text? `PLAN.md` gives the
  reasons and `PREREG.md` the binding protocol, tagged `prereg-1` (2026-10-16) and `prereg-2`
  (2026-10-17), each a release here and a version on Zenodo, with what each tag fixed in
  `config/prereg1.json` and `config/prereg2.json`. The benchmark ran in an environment pinning the
  package to the 0.12.1 release and every dependency (the lock's sha256 is in
  `config/prereg2.json`), so no later change on `main` reaches its results.
- **`benchmarks/frontier/RESULTS.md` reports that benchmark's test run,** pre-registered as
  `prereg-1` and `prereg-2`: the four primary endpoints, every arm on every set, the routes, the
  audits and the limitations, with its tables in `benchmarks/frontier/results/`, every prompt
  template it used in `benchmarks/frontier/prompts/`, and each change to a rule since `prereg-1` in
  `benchmarks/frontier/DEVIATIONS.md`. A new `.gitattributes` checks `benchmarks/frontier/` out
  byte for byte, so its files match the sha256s the pre-registration pins on every platform. The README gains the run's
  summary, in the form the pre-registration fixed.
- **The benchmark's harness and scorer are in `benchmarks/frontier/`:** the code, its tests, its
  config and its pinned environment, committed as their history from `prereg-2` on, with a README
  on checking the code against the pre-registration's pins and re-scoring the run from the Hugging
  Face dataset.
- **`lines.rule_mask(ink, min_len_px, tolerance_px=0)` finds horizontal ruling in an ink mask:**
  the pixels on horizontal runs at least `min_len_px` long (a 1 x `min_len_px` opening), with
  `tolerance_px` rows of vertical slack for a thin rule a skewed scan has broken into steps. The
  scanned path uses it for the underline fix below.
- **`ScanConfig.line_support`, on by default, and a `line_support: True` key on the scanned
  records it keeps,** also in the CLI's JSON output and the `dump_crops` manifest. See Changed.
- **Words a broken character map garbled are flagged, and their page gets a warning.** Some PDFs
  map a font's glyphs to the wrong characters, so the text layer, and `clean_text` with it, says
  something other than the page. Equations Word exports in Cambria Math are the common case:
  `·(0.955) −` reads `⋅ሺ0.955ሻെ`. `detect_pdf` lists such words in a new `garbled_words` key
  (`page`, `text`, `bbox_frac`; type `GarbledWord`), also in the CLI's `--json` and `--jsonl`. A
  struck record of such a word carries `garbled: True`, whichever native detector made it. Each
  page holding any gets an entry in `warnings`, and one `GarbledTextWarning` (a `UserWarning`)
  names those pages. The text is unchanged by default; the new `on_garbled="drop"` leaves those
  words out of `markdown`, `clean_text` and `passages`. A word is flagged for a noncharacter or a
  code point unassigned inside the blocks of Ethiopic, the Indic scripts and similar scripts,
  letters, marks or digits of two of those scripts (Ethiopic beside Malayalam), one of their
  characters beside a mathematical letter or a private-use character, a vowel sign after a
  character of another script with no letter of its own following, or one of their characters
  drawn in a math font. Real Ethiopic, Indic, Arabic, Hebrew and Thai text, digits and Latin
  abbreviations included, is not flagged, nor are letters assigned in Unicode 14 to 18 that an
  older Python calls unassigned. On the dev pages of the frontier benchmark's public redline corpus
  it flags the same 957 words the benchmark's own check flags, and none on newly generated
  synthetic pages or on Senate bills published before the benchmark's cutoff.
- **The CLI's `--on-garbled keep|drop` sets `detect_pdf()`'s `on_garbled`.** With `drop`, words a
  broken character map garbled are left out of `--markdown`, `--clean-text`, `--provenance` and the
  `passages` in `--json`, whose `garbled_words` still lists them. The default, `keep`, gives the
  same output as before. Batch output has no markdown, clean text or passages, so there the option
  is ignored with a warning, as `--markdown` is.
- **`get_model_meta()` reports the loaded model's sha256,** so a log or a report can tell
  retrained models apart where `version` cannot: the training script stamps its own version
  string. It is the digest of `strike_verdict_cnn.onnx` (or the `.pt`), the same one
  `ensure_model` verifies. An ONNX model that keeps its weights in external data files, as torch's
  newer exporter writes by default, also gets `external_data`, the location and sha256 of each
  file, and only both together identify its weights. The model still loads by path, so ONNX
  Runtime finds those files beside it and never in the working directory. `get_model_meta()` now
  returns a deep copy.
- **Run telemetry: `detect_pdf(telemetry=True)` and `detect_image_file(telemetry=True)`.** The
  result gains `telemetry`, a JSON-ready report for monitoring: the package version, the scoring
  model's `version` and `sha256`, the seconds taken, a ten-bin histogram of the scanned records'
  `cnn_prob`, and per page its source, its render, OCR and detection seconds, its candidates per
  tier and its struck count. A scanned page also gives its dpi and OCR word count, and its tiers
  count the geometry stage's candidates, including the `weak` ones the CNN pass drops. The rest of
  the result is unchanged. The CLI's `--telemetry FILE` appends one report per document to FILE as
  a JSON line, batch mode included, where a file that fails gets a line with its error.
  `pdf_strikethrough.types.Telemetry` documents the shape, versioned by `telemetry_version`.

### Changed
- **A vector record's `text` leaves out the U+0336 overlays on the characters it reports.** On a
  word struck with overlay characters, `text` is `of`, not `o̶f̶`; `char_span` and `chars` index
  that text, and `markdown`, `clean_text` and `passages` follow it. A word struck by overlays
  alone has no stroke, so its record carries no `stroke_color` or `stroke_width`. On any word with
  overlays, whether alone or with a drawn line, `coverage` is the share of the word's width its
  struck characters span. `SCHEMA_VERSION` stays 1: only words carrying U+0336 change, and 0.12.1
  reported those only under a drawn line.
- **On scanned pages, a stroke over most but not all of a word no longer strikes the whole word.**
  A word was a full strike whenever its strokes covered 70% of its characters, as estimated from
  the OCR box, so the live end of a partly struck long word was dropped with the struck part. A
  word is now a full strike only when its strokes also reach its first and last character, and a
  character at a stroke's end counts as struck when the stroke covers at least half of it. Both are
  measured against the word's glyph ink near the stroke, not its OCR box, which runs loose and on
  RapidOCR can hold several words, and against the stroke's own ink, not the detected line's ends.
  Pieces of a stroke across short breaks count as one stroke: a pen skip, typed hyphens, or dashes
  at least half a character long with breaks of at most a quarter of one. A strike with shorter
  dashes or wider breaks can still leave a long word partly struck. Partial spans move onto the
  struck characters the same way, so a stroke over the letters alone now leaves the punctuation
  around them live. Which words are reported struck does not change, only their spans and whether
  they are full or partial. A box with no glyph ink to measure keeps the old rule. Measured on its
  own against 0.12.1 with the frontier benchmark's scorer, on pages outside its test sets and not
  used to tune it: the live part of a partly struck word is marked 19 times in 318 where it was 71
  (synthetic pages) and 27 times in 149 where it was 42 (real redlined documents). The character
  error rate of the live text, which also counts a whole struck word now left with a character live,
  falls from 6.36% to 6.06% on the synthetic pages, from 11.29% to 10.96% on the redlined documents
  and from 3.18% to 3.17% on Senate bills.
- **The scanned path keeps short struck words that StrikeNet leaves unsure.** A word of three
  characters or fewer, not counting surrounding punctuation, is mostly padding at StrikeNet's input
  width, so a struck "of" or "a" often scored below the 0.85 struck threshold inside a run of
  confirmed struck words and stayed live. Such a word is now kept, struck whole, when the line
  through it strikes confirmed words on both sides of it, it shows strike geometry on that line,
  and it has a stroke of its own: one unbroken stretch of ink at the line's height across at least
  65% of the word (`detect.LINE_SUPPORT_MIN_RUN`), with more ink there than in the glyph ink just
  above and below it. Two words struck one by one can be joined into one line across the live word
  between them, whose glyph ink at that height ("a", "&", a crossbar) is not one such stroke. On
  the calibrated-confidence path a word read above `max_clean_conf` is never kept.
  `precision_first()` turns the rule off, and `line_support=False` gives the previous output.
  Measured on its own against 0.12.1: on scanned pages of Senate bills published before the
  benchmark's cutoff (the half not used to pick the rule), struck words found rose from 89.2% to
  96.7% of those read, with no new live words marked; on synthetic redlines that strike each word
  on its own, live words marked stayed at 88 (112 without the own-stroke test). A lower `cnn_p_hi`
  is not a substitute: it marks far more live words.
- **Only a release whose tag starts with `v` (or `V`) runs `publish.yml`'s jobs.** A GitHub
  Release with another tag, such as the ones that timestamp the benchmark's pre-registration,
  builds and publishes nothing and asks for no approval. A version tag still has to match
  `__version__`. Releases are now immutable (a repository setting): a published release's tag
  cannot be moved.
- **CI's copies of the dependency pins can no longer drift from `pyproject.toml`.** The lint job
  reads ruff's version from the `dev` extra instead of keeping its own copy, and moves to
  ruff-action v4.1.0 (Node 24; v4 has no floating major tag, which is why Dependabot never offered
  it). The `lowest-bounds` job still copies the dependency floors, and
  `tests/test_dependency_floors.py` now fails when its copy and `pyproject.toml` disagree. The
  `pymupdf` floor may rise above `native.FLAG_MIN_PYMUPDF` (1.26.6, where the flag detector's
  1.26.3–1.26.5 crash ends), but still not fall below it.
- **The `onnxruntime` floor is now 1.18.** PyMuPDF 2.0 loads ONNX layout models when it is
  imported, and they need ONNX IR version 10, which onnxruntime reads from 1.18. Under 2.0 an older
  onnxruntime made `import pdf_strikethrough` fail; on Python 3.10 the layout models that
  pymupdf4llm 1.28 loads broke `clean_markdown()`. 2.0 also makes `import pymupdf` about ten times
  slower, which adds seconds to the start of each CLI `--jobs` worker that starts a fresh
  interpreter.
- **The `pillow` floor is now 10.1.** Pillow 9.2 through 10.0 turn a TIFF with an orientation tag
  upright on load but still report the tag, so `detect_image_file()` rotated it a second time: an
  OCR backend got the page sideways, and supplied word boxes drew a spurious EXIF warning. CI's
  `lowest-bounds` job installed 9.0, which is not affected, so it never showed.
- **The Gradio demo no longer installs the `[markdown]` extra,** which it never used.
- **`rapidocr_backend()` builds its engine with RapidOCR's text-line orientation classifier
  off.** RapidOCR turns it on by default, and it turned struck, underlined and some plain lines
  upside down, which then read as stray letters or not at all: struck text could reach
  `clean_text`, and live text could go missing from it. The default goes in as
  `params={"Global.use_cls": False}` under any `params` you pass, so
  `params={"Global.use_cls": True}` turns the classifier back on; the default also overrides a
  `use_cls` set in a `config_path` file. A page scanned upside down now reads as stray letters:
  turn it upright first, or turn the classifier back on. An engine passed as `engine=` is used as
  is, so build it with `RapidOCR(params={"Global.use_cls": False})`; the Gradio demo now does.
- **`rapidocr_backend()` removes lone CJK characters from a page without CJK text.** RapidOCR's
  default recognizer reads Chinese and Japanese as well as English, and on English redlines it read
  small marks as CJK characters, which reached `clean_text` as inserted text: a table's footnote
  marker `(a)` as `(斗)`, an `=` in an equation as `二`. On a page where no two CJK characters (Han,
  kana, Hangul, bopomofo) come one right after the other in reading order (inside a word, or the
  last of one word and the first of the next, across a line break too; a digit in between breaks
  the run), the backend now removes each one: a word left empty is dropped, and one inside a word
  becomes a space. Real CJK goes too when it stands alone like that: `100 円`, `5 個` and `7 元` lose
  the unit, and `Article 1 (第1条)` reads `(1)`. A page with CJK text, where the characters come in
  runs, is returned as read, and CJK punctuation and fullwidth forms are kept.
  `drop_lone_cjk=False` keeps every character; it applies to an engine passed as `engine=` too. The
  CLI's `--ocr rapidocr` always removes them; to keep them, call `detect_pdf()` from Python with
  `ocr=rapidocr_backend(drop_lone_cjk=False)`.
- **CI's extras job now runs the RapidOCR adapter's tests** against the real engine. The test jobs
  don't install RapidOCR, so its one real-engine test was always skipped there.
- **`detect_scanned_image`, `analyze_scanned_page` and `scanned.classify_lines` default to
  `ScanConfig.confidence_free()`,** as `detect_pdf` without a `di_result`, `detect_image_file`,
  `words_by_page=` and the CLI have since 0.12.0. They defaulted to the Azure DI calibration,
  which reads the near-1.0 confidences other engines give struck words as clean text: on a
  synthetic page, a struck word read at 0.99 scored 0.40 (`review`) where the high-level paths
  score it 0.67 (`auto`). Words without confidences get the same output either way (see Fixed).
  Azure DI words passed with no config also lose that calibration's chain gate and confidence
  veto; pass `config=ScanConfig.azure_di()` to keep the old behavior.

### Removed
- **The TestPyPI rehearsal.** A manual run of `publish.yml` tried to publish to TestPyPI, where this
  repository had no trusted publisher, so it failed at the login each time. A manual run now builds
  the sdist and wheel, checks them and tests the wheel, and publishes nothing; a GitHub Release
  publishes to PyPI as before.
- **`benchmarks/di_parity.py` and `benchmarks/ocr_backend_table.py`.** They read two manifest
  fields, `di_result` and `di_reference_struck`, that no manifest entry has, so they ran on nothing,
  and the README no longer quotes what they measured. `scanned_recovery.py`, added in 0.9.0,
  measures the scanned path's recovery on this repository's own corpus.

### Fixed
- **The vector detector missed text struck with overlay characters.** Online strikethrough
  generators strike text by writing U+0336 COMBINING LONG STROKE OVERLAY after each character
  (`o̶f̶`), with no drawing, so the vector path found nothing and the struck letters stayed in
  `clean_text`. It now strikes each character a U+0336 follows, with its accents, part of a word
  included, so `both` reads them too; `flag` and `annot` don't, and their output is unchanged.
  `clean_markdown()` drops overlaid characters. A record holds one span, so a word with a drawn
  strike reports that span merged with any overlay run it touches. Overlays on characters outside
  the reported span stay in the record's `text` and in `markdown`, and `clean_text` drops those
  characters, as `clean_markdown()` does. Only U+0336 is read: U+0338 is the slash of a
  decomposed ≠ and U+0334 is the IPA mark for velarization. The short overlays U+0335 and
  U+0337, which some generators emit, are left out too, so text struck with them is still missed.
- **The vector detector struck words underlined at the baseline in Courier and other fonts.** The
  strike test used the middle share of the word's box, which in Courier and many other fonts holds
  the baseline. The stroke is now measured in ems from the glyph baseline, read from the
  characters' origins: it counts where its center is more than 0.03 em above the baseline
  (`STRIKE_BAND_LO`) and at most 0.58 em (`STRIKE_BAND_HI_LEVEL`), or 0.65 em for a stroke leaning
  more than 2 degrees (`STRIKE_BAND_HI`). A word with no usable baseline keeps the old box-share
  band. The character pass turns PyMuPDF's quad corrections on for itself, so the output is the
  same after pymupdf4llm turns them off. The edges were picked on newly generated pages from the
  frontier benchmark's synthetic redline generator and on strike-free LaTeX and plotting PDFs. On
  200 such pages, the 687 words with a Courier baseline underline are no longer marked, and nothing
  else changes; 208 pages of Senate bills published before the benchmark's cutoff are unchanged; on
  the dev pages of the benchmark's public redline corpus 16 words change, 14 of them invisible
  text; the strike-free PDFs get 252 marks, down from 270, none new. Reading character data for
  every word a stroke crosses adds a few milliseconds per page, up to 3 times the detector's time
  on a page of underlined text.
- **The scanned path dropped strikes thicker than about 1.4 pt.** A line more than 4 px thick at
  200 dpi was taken for a bold glyph's bar, so a 2 or 3 pt pen or marker strike (5.6 to 8.3 px)
  was lost. The cap is now 9 px, plus 1 px of slack because a thickness is counted in whole pixels
  while the cap scales with the dpi. A line thicker than 4 px is kept only when it is at least 1.4
  times as thick as the stems of the glyph ink just above and below it. A bold glyph's bars, and
  the chains of them stitched along a bold heading, are about as thick as its stems, so they stay
  out: on bold headings in nine fonts at 10 to 24 pt, the cap alone would have sent 2.7 times as
  many words to the CNN, and with the stem test it sends about 10% more. The same test drops a
  real strike less than 1.4 times as thick as the stems it crosses: at 2 pt, that is a strike
  through bold text above about 10 pt or regular text above about 18 pt, which 0.12.1 lost as
  well. On a soft scan a 3 pt strike can merge with the glyphs it crosses into a piece thicker
  than the stitching accepts, and is then still lost. `lines.STEM_RATIO = 0` turns the stem test
  off but keeps the 9 px cap; 0.12.1's behavior also needs `lines.MAX_STROKE_RUN_PX = 4` and
  `lines.RUN_SLACK_PX = 0`.
- **The scanned path marked underlined text sitting on a table rule as struck.** A line below a
  word's middle band is kept only with ink on both sides of it, and under an underline the ink
  below was the table's rule, so underlined numbers in ruled tables became candidates that the CNN
  then confirmed. The test of the ink above and below a line now reads glyph ink: the page's ink
  without its horizontal runs of 0.25 in or more (50 px at 200 dpi, scaled with the dpi). A strike
  low in the glyphs, or a short strike, on ruled or underlined text is now treated as it would be
  on unruled text, since the rule no longer counts as ink. A long solid line through a word's
  middle with no glyph ink on either side is judged on all ink, so one of a double rule is still
  taken for a rule. `ScanConfig(glyph_ink=False)` measures on all ink, as before.
  `scanned.classify_lines` takes the raster's `dpi=` for the scaling, and `glyph_ink=` to supply
  the mask.
- **The flag detector's version guard could not read PyMuPDF 2.0's version.** 2.0 reports `2.0`,
  with no patch number, so the guard read it as unknown. Unknown means unguarded, so 2.0 users were
  never blocked, but the test that the installed PyMuPDF clears the floor failed in every CI job
  that ran it on 2.0. A two-part version now reads as patch 0.
- **The `[markdown]` extra could install without a usable pymupdf4llm.** PyMuPDF 2.0 ships the
  pymupdf4llm module itself, and the pymupdf4llm 2.0 package is an empty placeholder. Beside a
  PyMuPDF below 2.0 (always on Python 3.10, which 2.0 does not support, or under a pin below 2.0)
  the extra resolved that placeholder, and `clean_markdown()` asked for the extra that was already
  installed. The extra now takes `pymupdf4llm>=0.0.27,<2` on Python 3.10 and PyMuPDF 2.0 on 3.11
  and later. On 3.11 and later, `[markdown]` with PyMuPDF pinned below 2.0 now fails to resolve:
  to stay below 2.0 there, install the base package with `pymupdf4llm<2` instead of the extra.
  Dependabot is told not to lift the `<2` cap.
  New tests cover `clean_markdown()`, which no test called before. Detection output on the
  benchmark corpus is unchanged under 2.0; `clean_markdown()`'s layout changes in places, mostly
  around equations.
- **`clean_markdown()` wrote to stdout.** On PyMuPDF 2.0, pymupdf4llm prints a "Document parser
  messages" banner on every call ("RapidOCR is missing" without RapidOCR, "Using RapidOCR" with
  it), and older pymupdf4llm releases print a hint on their first call. Both now go to stderr,
  except where the caller has routed PyMuPDF's messages elsewhere, which is kept. The switch is
  process-wide while a call runs, so calls from several threads now run one at a time. The CLI
  was not affected: its markdown is built from word records.
- **The model card described the CNN thresholds wrongly.** Of the two, only `p_hi` decides what
  is reported: a candidate the CNN does not confirm at `p_hi` is dropped, and `p_lo` only labels
  the rest clean or unsure. The card wrote the unsure band as `[p_lo, p_hi)`, said unsure words
  were deferred to geometry, and gave `ScanConfig.precision_first` a `cnn_p_lo` argument; it
  takes `cnn_p_hi`.
- **`tools/export_model.py` could leave a new model beside an old meta.** It wrote the ONNX into
  the package before its parity check, an `assert` that `python -O` skips. It now writes to a
  temporary file and replaces the model and meta only after the check passes, and no longer
  crashes printing `Δ` on a cp1252 console after writing the files. Both training tools write
  their meta with LF line endings.
- **`training/train_strikenet.py` checked its validation split only after training**, and one
  labeled crop crashed it on an empty training split. It now stops before the first epoch and
  says what the split lacks; the same seed still gives the same split and model.
- **A test pinned the wrong switch.** The chain-gate escape test ran `ScanConfig.ruled_forms()`,
  whose printed-rule veto removes the line before the escape is reached, so nothing tested
  `rescue_clean_chains=False`; now one test does, and another pins the veto's order.
- **One RapidOCR word box off the page no longer loses the whole document.** RapidOCR can place a
  word partly or wholly past the edge of the page (the last word of one equation line ran from
  0.78 to 1.57 of the page width). `Word` refuses a box past 1.5, its guard against pixel
  coordinates, so `rapidocr_backend()` raised `ValueError` inside `detect_pdf()` or
  `detect_image_file()` and the call raised, so nothing came back. The backend now clamps each word
  box to the page, so a word placed wholly off it keeps its text on a zero-width box at the edge; a
  box only slightly past the edge, which `Word` accepted before, is clamped too.
- **The README said RapidOCR reads every CJK script and that CJK scans work end to end.** Its
  default model has no Hangul; the README now gives a runnable Korean route (rapidocr 3.3 or
  later, Python only). On CJK scans the scanned path takes the horizontal stroke inside many
  characters (을, 王, 三) for a strike: on synthetic 200-dpi sans-serif scans it marked 13–28% of
  unstruck Korean, Chinese and Japanese words struck, and no English ones. The README says to
  check a CJK scan's struck words, and its scope table now lists text-layer CJK PDFs and scans
  separately.
- **A strike line leaning more than 2° and rising more than 1.5 pt struck nothing.** The vector
  detector kept a line only when it rose at most 1.5 pt or leaned at most 2°, so a line-tool strike
  drawn by hand across a word was often dropped. Lines leaning up to 12° now count, each word
  tested where the line crosses it, except on a line of text that itself leans more than 2°, where
  a line along the text crosses its upright box. A steeper line or curve is never a strike on its
  own, however short; before, any line rising 1.5 pt or less counted at any angle. The steeper
  pieces of a hand-drawn strike still count when they chain with level ones. On synthetic pages,
  strikes leaning 3° or more went from none found to 96%; the rest lean past 12° across words of
  three letters or fewer.
- **A strike shorter than 6 pt struck nothing,** so a struck digit, comma or period, or the old
  letter of a relettered item (`(ba)`, `(iv)`), stayed in `clean_text`. The floor is now 3 pt.
  A short mark must also be flat: a tick across one character (3 pt across, 1.5 pt up), drawn as
  a line, a curve or a filled shape, is still not a strike, nor is a box or bar under 6 pt wide
  and over 1.5 pt tall. A stroke under 6 pt counts only on a character whose own strike band it
  crosses (the band above the baseline, measured from that character's own origin and em, or the
  middle of its box when it has no usable baseline), so the bar of a small fraction or a root
  beside full-size text is not a strike. A character extracted with no width, such as the 'i' of
  an 'fi' ligature, is struck with the character before it. Segments under 6 pt are still chained
  across the gaps of a dashed strike before the floor applies.
- **Invisible text over an inserted letter could be marked struck.** Some redline tools lay a
  hidden glyph a little low over a letter, so the underline of an inserted letter crossed the
  hidden word's strike band. On an invisible word (render mode 3, or fully transparent) that
  overlaps visible text, a stroke under 6 pt now counts only through the strike band of the
  visible characters beneath it, so the hidden word is struck with a struck letter and not by an
  underline. A strike over a text layer that is all invisible, such as an OCR'd scan with its
  markup flattened, still counts.
- **One word box off the page from Azure Document Intelligence, Textract or Document AI no longer
  loses the whole document.** `Word` refuses a box past 1.5, its guard against pixel coordinates,
  so one such box made `words_from_azure_di()` (which `di_result` goes through),
  `words_from_textract()` or `words_from_docai()` raise `ValueError`, and `detect_pdf()` or
  `detect_image_file()` with it. The adapters now clamp each word box to the page, as
  `rapidocr_backend()` does, and drop a box that is not finite. A page where most boxes lie past
  1.5, as with boxes in pixels or inches, still raises: clamped, its struck words would come back
  as clean text.
- **A blank page in a Textract result refused the whole scan.** `words_from_textract()` listed
  only pages with words, so `detect_pdf()` and the CLI's `--textract-result` raised on a scanned
  page with none. Every page in the result is now a key, with an empty list when it has no usable
  words, as Document AI and Azure DI already report it.
- **`clean_markdown()`'s docstring and the README now say what MuPDF's strikeout flag does.**
  `clean_markdown()` removes what pymupdf4llm marks `~~`, which is MuPDF's strikeout flag, and the
  flag can take in the character a strike ends against when no space separates them: the `M` of a
  `May` printed straight after a struck `December` goes too, as it does with `method="flag"`, more
  often under PyMuPDF 2.0 than earlier versions. `method="vector"` and `"both"` stop at the struck
  characters. The docs also say that `flag` and `annot` find nothing on U+0336 overlay strikes,
  which `clean_markdown()` drops itself, code blocks included, and that it keeps text the flag
  marks where pymupdf4llm sets it as a code block, such as a paragraph set entirely in a monospaced
  font. No code changes.
- **Under the Azure DI calibration, scanned words with no OCR confidence scored at most 0.60.**
  With `ScanConfig()` and words that carry no confidence, the chain gate already ran without
  confidences, but the evidence score kept its confidence term and read every word as clean
  text: on a synthetic page a clear strike scored 0.47 (`review`) where `confidence_free()` gives
  0.78 (`auto`). The score now drops the term whenever the page has no confidences, as the gate
  does, so such a page gets the same output under either calibration. Words with confidences
  score as before.
- **`apply_cnn_verdict` could drop an annotation record.** It leaves native records to their own
  detector, but its list of native tiers predated `annot`, so a `/StrikeOut` record passed to it
  was dropped whenever StrikeNet read the crop as clean. `detect_pdf` never passes it native
  records; only a direct call was affected.
- **A caller-built line without a `fill` could lose its words to the table-rule test.**
  `classify_lines` read a missing `fill` as 1.0 there, so a long line through a word's middle
  with glyph ink on one side only was dropped as a solid rule, the failure 0.10.0 fixed in the
  printed-rule veto. A missing or `None` `fill` now triggers neither fill test (the table-rule test
  and the chain gate, which already skipped it) and still scores as 0.80
  (`scanned.FILL_UNMEASURED`); a `None` used to raise. Lines from `strike_lines` always carry one.
- **A scanned page missing from `words_by_page` raised "no OCR backend was provided".** The
  `OcrRequiredError` now names the pages `words_by_page` covers, as the error for a partial
  `di_result` does; `on_missing_ocr="skip"`'s warning names `words_by_page` too.
- **`markdown.strip_struck` kept characters struck with U+0336 overlays.** It was a copy of
  `strip_struck_markdown` from before overlays were read; it now calls it and trims the result.
- **The vector detector no longer reads fraction bars, radical bars and gridlines behind a legend
  as strikes,** which dropped that text from `clean_text`. A stroke now strikes a character only
  through that character's own band, so a fraction bar no longer strikes a small numerator or
  denominator joined into one word with full-size text, nor a radical's bar its radicand. A radical
  sign is struck only when the strike crosses at least half of it and its radicand is struck too.
  A stroke under a fill at least half opaque, painted between it and the text, is behind the
  text's background, as a plot's gridline is behind its legend's box; where the paint order can't
  be read, the stroke is kept. On 137 strike-free PDFs, mostly LaTeX documents and matplotlib
  figures, false strikes go from 421 to 4. Newly generated synthetic pages and Senate bills
  published before the benchmark's cutoff are unchanged; on the dev pages of the benchmark's
  public redline corpus, 13 marks go, all fraction bars in equations. Limits: a strike over a
  radical sign alone, or drawn under such a fill where it still shows through, is lost; an overline
  or a radical's bar just above x-height letters (`√x`), or a gridline through text with no box,
  is still read as a strike. The detector takes 1–4% longer.
- **The scanned path dropped thick strikes through bold or large text.** A 2 pt strike through bold
  text above about 10 pt, or regular text above about 18 pt, is as thick as the stems it crosses, so
  it read as a bold glyph's bar, and in larger bold text it merged with the bars it touches into one
  piece too thick to stitch. A second pass now adds such a line where none was found: a thick line
  is kept when its ink runs on unbroken for two and a half glyph heights, since a bold glyph's bar
  ends with its glyph, and where a piece is too thick to stitch, the band it fills end to end is
  taken as the line. Every line found before comes out unchanged, and `strike_lines` still needs no
  OCR. On synthetic scanned pages in 42 fonts, struck words found rose from 3,853 to 4,997 of 12,782
  (2 to 3 pt strikes from 944 to 1,717 of 4,247), with none lost and no more live words marked, and
  strike-free bold text on shading, in boxes and under rules marked no more words. Struck bold
  headings drawn with underlines, rules and boxes marked 46 live words where they marked 40. On
  newly generated synthetic pages from the frontier benchmark, the half not used to tune, struck
  sub-tokens found rose from 2,694 to 2,707 of 3,560 with no new over-marks; Senate bills published
  before the benchmark's cutoff and the dev pages of its public redline corpus are unchanged.
  `strike_lines` takes about 10% longer on struck pages and 7% on strike-free bold headings. A
  strike thicker than about 3 pt is still over the 9 px cap, and a thin strike merged into bold
  glyph bars is still lost. `lines.SOLID_GLYPHS = 0` gives the previous behavior.
- **The scanned path dropped struck words whose line ran a few px off the stroke.** Stitching can
  run a line a few px above or below a thin or light stroke, along the glyph pieces it crosses.
  Read from there, the ink on one side of the line is under the glyphs, so one of two tests
  dropped the word: a line outside the word's middle band, or a short one, needs glyph ink on both
  sides, and a long solid line with ink on one side only is taken for a table rule. A word either
  test drops is now tried again at the stroke's own row: the row within 4 px of the line (at 200
  dpi, scaled with the dpi) whose unbroken run of ink covers the most of the columns the word
  shares with the line, and at least 80% of them. The tests themselves are unchanged, and a word
  they pass is judged as before. On the half of each set not used for tuning, struck sub-tokens
  found rose from 2,707 to 2,797 of 3,560 on newly generated synthetic pages from the frontier
  benchmark, from 2,599 to 2,631 of 2,798 on the dev pages of its public redline corpus, and from
  5,548 to 5,561 of 5,752 on Senate bills published before its cutoff, with none lost. Three live
  sub-tokens were newly marked, all in partly struck words: two struck words already found took
  one or two more characters, and one OCR box holding a live word and the struck word after it
  was marked whole. A stroke over less than 80% of those columns, or more than 4 px off, is still
  lost. The reach was tuned at 200 dpi. At 72 and 100 dpi it is 1 or 2 px, and glyph rows of
  serif and bold text can run as far as a stroke: on synthetic pages at those resolutions it
  found 350 and 710 more struck words, and marked 97 and 10 more live ones. A scanned page's
  detection takes 1.5 to 3.2% longer on average, OCR aside, and up to 1.17 times as long on one
  page. `scanned.STROKE_SNAP_PX = 0` gives the previous behavior.
- **`benchmarks/confidence_veto.py` split its counts where the veto does not.** Its default mode
  counted a word read at exactly `max_clean_conf` (0.955) as high-confidence, but the veto acts
  only above it. The split now uses the veto's boundary. One reported word on its ruled-table set
  reads at exactly 0.955 in both 0.9.0 and 0.9.1, so the 0.9.1 figures it reproduces become 94 → 21
  (−78%) high-confidence over-flags, not 95 → 22 (−77%). The struck-final counts (188 → 115) are
  unchanged.

## [0.12.1] — 2026-10-08

Correctness patch for the shipped StrikeNet's documentation and the tooling around it. Nothing
under `src/` changes except the version, so detection output is identical to 0.12.0's. The model
card and training docs no longer say the shipped weights can be reproduced, the training script
exports on current torch and records what a model was trained on, and the scanned-recovery figures
reproduce from a fresh clone.

### Changed
- **The Gradio demo requires 0.12.0** and drops its own guard for RapidOCR's empty-page placeholder,
  which 0.12.0's `rapidocr_backend()` handles. Its deploy steps now say that a new Gradio Space on
  free CPU needs a Hugging Face PRO account and how to push to an existing one, and drop the libGL
  step: RapidOCR's headless OpenCV needs no system packages.

### Added
- **`training/train_strikenet.py` records what a model was trained on.** The exported meta carries
  a `training` block: a sha256 of the labeled set (each labeled crop's sha256 and its label, in
  manifest order), the struck and clean counts, the validation size, the hyperparameters including
  the seed that draws the split, and the package and torch versions. The digest identifies the data
  but cannot restore it, so keep the labeled directory. `--batch`, `--lr` and `--seed` are now
  flags, with the old values as defaults.
- **The scanned-recovery benchmark needs no Azure account.** The cached Azure DI results it reads
  are hosted in the [`niles-liu/strikethrough-benchmark`](https://huggingface.co/datasets/niles-liu/strikethrough-benchmark)
  dataset. `manifest.json` lists each one's URL and sha256, `fetch_corpus.py` downloads and
  verifies them with the PDFs, and `scanned_recovery.py` checks the digest before reading one.

### Fixed
- **The shipped StrikeNet weights were described as reproducible, and they are not.** The model
  card, `training/README.md` and the training script said the shipped weights could be regenerated
  from a labeled crop set with `training/train_strikenet.py`. The weights are byte-identical in
  every release since 0.4.0 and predate that script, which did not produce them; their training
  set, split and settings, and how their thresholds (0.85 / 0.15) were chosen, are not recorded in
  this repository or any release. Those documents now say so, the model card says its scanned-path
  figures measure the whole pipeline, not the model alone, and neither it nor the README says any
  more what the model was trained on.
- **The training script's recall floor was stated without its condition.** The model card called
  the `1 - alpha` recall floor that `p_hi` sets distribution-free; `training/README.md` and the
  script's `--alpha` help called it guaranteed. It holds for crops exchangeable with the
  validation crops; the split is random over crops, not documents, so on a document outside the
  labeled set recall can fall below it. All three now say so.
- **`training/train_strikenet.py` could not export on current torch.** From torch 2.9
  `torch.onnx.export` defaults to the dynamo exporter, which needs `onnxscript`, and either exporter
  imports the `onnx` package, which no setup step listed; so a finished training run ended in an
  import error with nothing written. The script uses the TorchScript exporter, as
  `tools/export_model.py` already did, and checks for `torch` and `onnx` before it trains. The
  export goes to a temporary file and replaces the output directory's model only when its logits
  match the trained net's within 1e-3, so a failed export leaves the previous model and its meta as
  they were. The README, `training/README.md`, `CONTRIBUTING.md` and `tools/export_model.py` now
  list `onnx`.
- **`training/train_strikenet.py` could write a model that calls every candidate struck.** With
  fewer struck validation crops than `--alpha` needs (19 at the default 0.05), `conformal_threshold`
  returns 0.0, so both thresholds came out 0; and a validation split with no struck or no clean crop
  fell back to the shipped 0.85 / 0.15 without saying so. The script now stops and says how many
  crops it needs.
- `CONTRIBUTING.md` gave `tools/export_model.py` a `--checkpoint` flag; the flag is `--ckpt`.
- **A fresh clone could not fetch four of the ten corpus PDFs.** copyright.gov and gretnala.com
  answer urllib's default User-Agent with a 403, and both CEQ documents' URLs now redirect to
  nepa.gov's home page. `fetch_corpus.py` sends its own User-Agent, and the CEQ entries point at
  Internet Archive copies whose sha256 matches the pinned digest.
- **`scanned_recovery.py` did not produce the README's Azure DI figure for `confidence_free()`**: it
  scored the cached DI words under the default calibration only. It now reports both. An entry with
  no DI result reads n/a in the DI columns instead of turning their totals into NaN, and a total
  taken over fewer strikes than the GT column says so.
- **A missing corpus file was reported under `<function corpus_dir at 0x…>/`** instead of its
  directory.
- **Under uv, an editable install kept the version it was installed at.** uv rebuilds a project only
  when its cache keys change, and the defaults miss the version read from `__init__.py`, so after a
  bump `importlib.metadata` still reported the old one. `[tool.uv] cache-keys` now include it.

### Security
- **`.env`, where `benchmarks/prep_scanned_di.py` reads the Azure DI key, was not git-ignored.** It
  is now.

### Notes
- **Corrects 0.9.0's "Closes the model reproducibility hole."** The training script trains a
  replacement and records what it was trained on; it does not reproduce the shipped weights or the
  process that made them (see **Fixed**).

## [0.12.0] — 2026-10-08

A correctness release. An audit of the whole package (native path, scanned path, public API, CLI
and packaging) reproduced every defect it reported before any was fixed; all but the few under
**Documented (not yet fixed)** are fixed here, with regression tests that fail on 0.11.0.
Several fixes change output, so read **Changed** before upgrading.

### Changed
- **Native strike spans are character-exact.** The vector, flag and annotation detectors map a
  strike onto the characters it actually crosses, read from the PDF's own glyph boxes, where they
  used to split the word box evenly by character count; and a word is a **full** strike only when
  every character is struck, where it used to be one whenever the strike covered 70% of its width.
  On the 10-document benchmark corpus (55.2k vector records) this removes 41 records (37 edges of a
  filled watermark shape, 4 of a dashed diagram box), adds 464 (single struck characters: relettered
  labels such as `(e)` → `(d)`, recased words) and corrects 1,140 (502 full strikes become partial
  because trailing punctuation or text inserted after the deletion is not struck, 614 partial spans
  move onto the struck characters, 24 partials become full because a struck sub- or superscript now
  counts). The flag detector changes similarly (287 removed, 491 added, 1,444 corrected). Each kind
  of change was checked against the rendered pages. `confirmation_rate.py` still reports 99.8% of
  vector detections confirmed by the flag signal, and the per-document floor rose from 92.5% to
  99.6%.
- **`detect_pdf(ocr=...)` defaults to `ScanConfig.confidence_free()`**, as `detect_image_file`,
  `words_by_page=` and the CLI already did. The default was the Azure DI calibration, whose
  confidence veto dropped real strikes on RapidOCR's near-1.0 confidences, so the same page gave
  different answers through the API and the CLI. `di_result=` keeps the DI calibration; pass
  `scan_config=ScanConfig()` for the old behaviour with `ocr=`.
- **The Azure DI calibration keeps crisp strikes.** 0.9.1's confidence veto (issue #4) drops a word
  DI reads above `max_clean_conf` unless the line through it is strike geometry, and that required
  the line to break up on the glyphs. A clean scan of a printed strike stays solid and DI reads the
  word cleanly, so the default `di_result=` path kept 28% of the known strikes on the
  scanned-recovery benchmark. Glyph ink on both sides of an in-band line is now enough, and the
  default keeps 91% (`confidence_free()`: 96%). On the private ruled-forms corpus this adds 4 false
  positives at default (109 → 113) and with the chain switch alone (92 → 96), none under
  `ruled_forms()` (38) or with both switches (25); recall is unchanged at 2/3. A one-sided rule
  still earns no reprieve.
- **Arguments that used to be misread now raise at the call.** `pages=` rejects a string (`"12"`
  meant pages 1 and 2), a bool and a float; `words_by_page` rejects keys outside the document
  (1-based keys put page 1's words on page 0), a flat list of words, and use together with
  `di_result`; `method`, `on_missing_ocr` and a non-positive `dpi` are checked even when no page
  would have used them; `ScanConfig` rejects values outside [0, 1] (`cnn_p_hi=85`), NaN and
  strings; `Word` rejects a non-finite box. `provenance_text` raises on a result without
  `markdown` instead of returning `''`, and the calibration helpers raise on a label that is not
  struck/clean, True/False or 1/0 instead of counting it.
- **CLI.** Batch mode follows the form of the arguments (several files, a directory, a glob, or
  `--jsonl`), not the number of files matched. Status lines (`wrote ...`) go to stderr. Usage errors
  exit 1, as documented — argparse's 2 collided with "encrypted / OCR required". `--dpi`, `--jobs`
  and `--overlay-dpi` must be positive. An interrupted run exits 130. Plain output numbers pages
  from 1, like `--pages`.

### Added
- **`schema_version` on every `detect_pdf` / `detect_image_file` result**, the same number the CLI
  `--json` / `--jsonl` payloads already carried. The constant now lives in
  `pdf_strikethrough.types.SCHEMA_VERSION` (still `1`) and the CLI reads it from there, so the
  Python result and the CLI share one number. It versions the documented keys, not the detections.
  The CLI's `--jsonl` error records (a file that failed to process) now carry it too; they were the
  one payload without it.
- **An open binary file is accepted wherever a path is** — `detect_pdf` / `open_pdf` /
  `strikethroughs_in_pdf` (an `io.BytesIO` failed with PyMuPDF's "bad filename") and
  `detect_image_file`.
- **A warning on PDFs MuPDF had to repair to open.** A truncated file used to pass
  `--fail-if-found` as "1 page, 0 struck words" with nothing in `warnings`.

### Fixed
- **`rapidocr_backend()` crashed on a page where OCR found no text.** RapidOCR (3.9 at least) then
  returns the placeholder `(('', 1.0, None),)`: a bare `(text, score, box)` triple where a line of
  them belongs. The adapter unpacked it as a line and raised `ValueError: not enough values to
  unpack`, so `detect_pdf` / `detect_image_file` aborted on any text-free scanned page or image
  instead of returning no words for it. A bare triple is now read as a one-word line, and the
  placeholder's empty text is then skipped like any empty word.
- **The Gradio demo could not read scans.** `demo/requirements.txt` installed the legacy
  `rapidocr-onnxruntime` distribution, which does not provide the `rapidocr` module the OCR adapter
  imports. Because `rapidocr_backend()` imports on first use, the demo's startup guard never fired
  and a scanned upload ended in a bare `ModuleNotFoundError`. The requirements now pull the
  package's own `rapidocr` extra, and the demo builds the OCR engine at startup: if that fails, it
  says so at the top of the page and runs on born-digital PDFs only. A detection error now shows
  its reason in the summary instead of Gradio's bare "Error". Until the demo can require a release
  carrying the fix above, it guards the same empty-page placeholder itself, because a Space
  installs the package from PyPI. It also caps image uploads at `MAX_PAGES` frames, as it already
  did PDF pages.

#### Native path
- **Text inserted straight after a deletion was deleted with it.** A redline that prints the
  replacement with no space (`~~December~~May`) is one word to the extractor; a strike over 70% of
  it counted as a full strike, so `clean_text` lost `May`. See **Changed** for the measured effect.
- **Fill-only shapes read as strikes.** A filled path's outline was judged by its stroke colour,
  which is unset and read as black: a watermark or a white background shape whose edge crossed a
  line of text struck every word on it. Each subpath of a fill-only path now counts as a bar when it
  is as thin as one, and reports its fill colour (a red bar used to report black). A line stroked
  wider than about a third of the word's height (a highlighter swipe) no longer counts either.
- **A line-tool strike drawn at a slight slant was ignored.** The 1.5 pt rise limit applied to the
  whole line, so a 0.5° strike across a line of text struck nothing. Lines leaning up to 2° are now
  tested at the height where they cross each word.
- **Content drawn rotated and turned upright by `/Rotate`** (pdflscape tables, ps2pdf
  auto-rotation): the vector detector read a short mark crossing a vertical word as a strike, and
  the flag detector dropped two of the four text directions. The vector detector now skips words
  that run vertically in text space; the flag and annotation detectors read all four.
- **Scans routed to the native path, which sees no strikes on them**: a full-page scan on a
  `/Rotate 90` legal or tabloid page (image boxes were measured unrotated against the rotated page,
  61% coverage), and a scan covering less than 70% of its page under an invisible OCR layer.
- **Deletion passages joined across live text** (`~~semi-~~monthly ~~rate~~` gave one passage
  "semi- rate"), and **a word straddling the page edge vanished** from `markdown` and `clean_text`
  on a two-column page.
- **Annotation forensics:** a gray or CMYK `/C` colour came through as a 1- or 4-tuple where an RGB
  triple is documented, and a `/NoView` annotation, which paints nothing, was reported.

#### Scanned path and image files
- **A camera's or screenshot tool's 72/96 dpi metadata made strikes vanish.** `detect_image_file`
  scaled every geometry tunable by it, so a 200-dpi photo labelled 72 dpi found nothing. Metadata
  below 100 dpi is now read as absent (200 dpi, with a warning); pass `dpi=` to override.
- **Image files missed strikes the same page found as a PDF or an array.** Frames went through
  PIL's own gray conversion, not the luminance conversion fixed for issue #15 (a green highlighter
  read 150 instead of 219); transparency was dropped, so ink on a transparent PNG read as a black
  page; a phone photo stored rotated was processed sideways. Frames now go through `to_gray_u8`,
  alpha is composited over white, and the EXIF rotation is applied when the package runs `ocr=`
  itself (supplied words are read against the image as stored, with a warning).
- **Wide integer arrays read as black.** `to_gray_u8` divided int32/int64 input by the dtype
  maximum; it now scales by the data range (8-bit values pass through, 16-bit divide by 257).
  `score_word` on a 16-bit page scored the crop as blank and now converts it first.
- **`detect_image_file` treated an open file as a PIL image** (`AttributeError`) and left files it
  opened open, so a file that failed to decode could not be deleted on Windows.
- **`cnn_prob` was missing**, not `None` as documented, on a scanned record whose crop was too small
  to score.
- **A concurrent `set_model_dir` / `ensure_model` could crash scoring** with `TypeError`: the loader
  re-read the model global after releasing its lock.
- **`score_struck` raised `KeyError: 'len_in'`** on caller-built line dicts, which
  `classify_lines` accepts; **`dump_crops` crashed** on numpy float confidences from a custom OCR
  backend (`Word` now stores a plain float).

#### Calibration
- **Manifest labels were read as all struck.** `dump_crops` asks for `"struck"` / `"clean"`, and a
  bare `astype(bool)` turned both into True (and an unfilled `None` into False), so every threshold
  came from a silently relabelled set.
- **`threshold_for_precision` checked precision in the middle of tied probabilities**, which
  rounded manifests and StrikeNet's saturation at 1.0 produce often, and returned a threshold that
  missed its target. **`threshold_for_recall`** lost a positive to floating-point rounding
  (1 - 0.9 = 0.0999...) and returned a lower threshold than the highest that qualifies.

#### Cloud OCR adapters
- **An Azure DI result for a page range put its words on the wrong pages**, keyed by list position
  rather than `pageNumber`; Document AI shards (which start mid-document) did the same. Both are now
  keyed by page number.
- **Words were dropped silently**: Azure DI polygons given as `{x, y}` points (the Python SDK's
  `to_dict()` shape) and Document AI pixel vertices on a page with no `dimension`. Points are read
  now, and a page with words but no usable box raises `ValueError`, as a page with no width
  already did.

#### Word documents
- **Strike formatting from a style was missed**: a character style, a paragraph style (following
  `basedOn`, with the two toggling one another as Word does) or the document defaults. So were
  struck runs inside text boxes (each read once, though Word stores two copies, and reported under
  the body paragraph the box is anchored in, so body numbering is unchanged) and every strike in a
  Strict OOXML file. Malformed XML or a damaged archive now raises the documented `ValueError`,
  naming a `Path` source.

#### CLI
- **Output written to `-` was not UTF-8 on Windows.** A redirected stdout is cp1252, so struck
  non-Latin text became `?` and `--json -` wrote invalid JSON. Output to `-` is UTF-8.
- **`--jsonl` was ignored when a directory or glob matched one file**, leaving the previous output
  in place; **status lines on stdout broke piped JSON and markdown**; **a filename containing `[`**
  was treated as a pattern; **`--pages` past the end** and an unreadable image, malformed `.docx`
  or unwritable output ended in a traceback.
- **A batch held every result until the end**, so Ctrl-C, a crashed worker or a bad output path
  lost the whole run. The output is opened first, after the run's own checks (a missing OCR extra
  no longer empties the previous output), and each line is written as its file finishes. Under
  `--jobs` a file that crashes its worker is re-run alone and becomes the one error line while the
  rest carry on, Ctrl-C stops the workers at once, and a closed output pipe (`| head`) ends the run
  with an error instead of a traceback.
- **The README called a directory "a whole tree"; only its top level is read.** The README now says
  so, and a glob with `**` recurses. A glob now keeps only documents (it handed a `.txt` or a
  subdirectory to PyMuPDF), `-` cannot be mixed with other inputs, and flags that do not apply to a
  `.docx` say so.
- **CI linted with ruff 0.15.20** while the `dev` extra pinned 0.16.x; both now pin 0.16.9.
- `examples/scanned_quickstart.py` crashed printing an OCR'd non-Latin character to a pipe.

#### Benchmarks and published figures
- **`scanned_recovery.py` could score cached Azure DI words against the wrong page images.** It
  re-picked the most-struck pages with the current native detector instead of reading the
  manifest's `scanned_pages`, the pages the DI result was captured on, so any change in per-page
  counts swapped pages (this release's native fix did exactly that, and the DI column fell to 8%).
  It now scores the recorded pages.
- **The README's scanned-recovery figures were stale.** Re-measured with the fixed harness:
  RapidOCR recovers 96% of the 2,170 known strikes (97% was quoted), and Azure DI 96% with
  `confidence_free()` and 91% with its default calibration (95% was quoted; 0.11.0 measures 28%,
  see the veto entry under **Changed**). The other figures are identical on 0.11.0, so nothing
  here moved them. The README, the model card and `benchmarks/README.md` now quote the measured
  values, as do the confirmation-rate figures above.
- **The ruled-forms switch figures were stale.** The README and the `ScanConfig` docstring still
  quoted 0.10.0's false positives on the private corpus, 113 / 42 / 95 / 28 (default /
  `ruled_forms()` / `rescue_clean_chains=False` / both). This release measures 113 / 38 / 96 / 25:
  0.11.0's 109 / 38 / 92 / 25 plus the 4 the DI veto fix adds (see **Changed**).

### Security
- **`ensure_model` could load content it had not verified.** An empty digest skipped the check,
  and `meta_url` without `meta_sha256` — the README's own example — loaded unverified thresholds: a
  tampered `p_hi` of 0 would report every scored word struck. `meta_sha256` is now required with
  `meta_url`, a digest must be 64 hex characters (any case: PowerShell's uppercase digests failed as
  a "mismatch"), and a model whose thresholds are not `0 <= p_lo <= p_hi <= 1` is refused at load.
  The README, the model card and the demo pass the meta digest.

### Documented (not yet fixed)
- `markdown` marks a partial strike in place (`~~semi-~~monthly`); a strict CommonMark renderer
  does not treat a `~~` that touches punctuation this way as a delimiter, and a struck word that
  itself ends in `~` confuses `strip_struck` / `provenance_text`. The word records are exact; a
  different marker is a format change, so it waits.
- The vector detector reads horizontal strokes only, so a strike on content drawn rotated in text
  space is found by `method="flag"` / `"both"`, not by the default.
- The flag detector reports MuPDF's own per-character decision, which can include the character a
  strike line ends against (`DecemberM`); `vector` and `both` stop at `December`.
- A struck word narrower than 3 pt (a lone `I` at 10 pt) is not reported. Lowering that floor added
  22 records on the benchmark corpus, all chart gridlines crossing glyphs, and no real strike.
- The underline exclusion band comes from the font's box, so for some fonts (Courier, Symbol) a line
  at the baseline still counts as a strike.
- On a scan, a strike over the first letters of a long word can read as a full-word strike.
- A tracked move in a Word document (`w:moveFrom`) is not reported.

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
