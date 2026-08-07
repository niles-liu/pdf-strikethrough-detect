# The positives pull — plan, pre-commitment, and what counts as a usable label

Working notes for the data pull that feeds the hard-negatives + abstention program. Scope: **where
struck-through text can be obtained in the regime that actually breaks the detector**, and what to
conclude if it can't be.

Read `benchmarks/private/ruled-tables/CORPUS_README.md` first for why this is needed. The short
version: the scanned CNN path produces **28 false positives against 2 recovered strikes** on real
degraded forms *in its best configuration* (113:2 at default), the acceptance bar is **zero** false
positives (one deleted remark disqualifies the path), and the domain that motivated the work barely
carries strikethroughs at all. So this is not a
"collect more data and retrain" exercise. It is a search for whether the failing regime has a data
supply anywhere, with a pre-committed reading if it doesn't.

**Where this stands:** §3 is closed — the one route that looked cheap does not work, for a reason no
amount of tuning fixes. The open question is §2 rows 1–3, and nothing else here is in flight.

---

## §1 The pre-commitment — written 2026-08-06, BEFORE the sweep ran

Recorded in advance and deliberately committed, so a thin result cannot be talked into a corpus
after the fact. The failure mode being guarded against is real and already happened once in this
repo: `ANT-BULK-HONDURAS` was recorded as a positive document for three releases, and a threshold
was held at 1.80 to preserve what turned out to be a false positive.

**If the sweep finds no domain with routinely *scanned* struck text, then:**

1. **The native vector path is the product.** The scanned CNN stays behind a flag, permanently, and
   the README says so plainly rather than implying parity between the two paths.
2. **That is a finding, and it ships as one.** "No usable data supply exists for this regime" is a
   publishable, useful result — not a failed task to be quietly dropped.
3. **"Not enough positives to converge" gets reported, not worked around.** No relabeling of
   near-misses, no lowering of what counts as a strike, no counting synthetic degradations as though
   they were independent documents.

**What would falsify the pessimistic reading** — any *one* of these is enough to justify the pull
continuing into a retrain:

- ≥100 genuinely scanned pages carrying ≥300 struck words, from ≥10 distinct source documents, in
  the tabular/ruled-form regime;
- or a public clinical-CRF / archival set meeting the same bar without PHI obstacles;
- or the ladder in §3 demonstrating transfer — degraded synthetic positives predicting behaviour on
  the held-out genuine scanned positives.

Anything less is a thin result, and the three clauses above apply. **As of 2026-08-07 all three
falsifiers are unmet**: the third is now settled negative (§3), and the first two are untested.

---

## §2 Domain ranking — status per domain

Ranked by match to the *failing* regime (scanned **and** tabular **and** strike-as-convention).
**The ranking is an informed hypothesis about availability, not a measurement.** Update the status
column as the sweep runs; do not delete a row that turns out empty — an empty row is the finding.

| # | Domain | Why it should carry positives | Status |
|---|---|---|---|
| 1 | **Clinical trial CRFs / regulated source records** | GCP requires a correction be struck through, initialed and dated, never obliterated — so a corrected CRF is a positive *by regulation*, in exactly the failing regime | **not started.** Blocker: PHI. Needs public, redacted or synthetic sets |
| 2 | **Archival tabular records** — ship logs, registers, ledgers, census sheets | Scanned, handwritten, correction-by-strikethrough standard, much of it public domain. Highest expected volume of genuinely scanned handwritten strikes | **not started** |
| 3 | **Chain-of-custody / logbook / QA forms** | Same struck-and-initialed convention as #1 without the regulatory paper trail | **not started** |
| 4 | **Legislative markup + USPTO claim amendments** | Enormous volume, strikethrough is *mandated* convention | **spent.** `manifest.json` holds 10 government redlines, but they are native and exercise the vector path that already works. Fed into §3 as raw material, and §3 failed on them |

Note what row 4 means: the domain the ranking calls "almost certainly native PDF" is the one already
collected, and §3 has now spent it outright. **The pull's whole remaining question is whether rows
1–3 exist in usable form** — the failed gate makes that more load-bearing, not less.

**Cost, per document, do not forget it:** a genuinely scanned document needs its own Azure DI
`prebuilt-layout` result (`di-result.json` beside the PDF) to be usable offline — one live cloud call
each. The §3 route avoided this for born-digital sources, which was most of its appeal.

---

## §3 Degrade-and-relabel — CLOSED, 2026-08-06. It does not reproduce the regime.

> **Verdict.** The gate failed; the confound that made that verdict provisional was cleared (§3.2);
> the collapse was attributed to `blur` and `scale` rather than to the fade model (§3.3). All three
> candidate causes are resolved, and the one left standing — **the redline corpus has none of the
> confusers that make real forms over-flag** — is a property of the document class that no ladder
> can change. **Do not reopen this without new document classes; there is nothing left to tune.**
>
> **The code is archived on branch `explore/positives-pull` and is deliberately not on `main`:**
> [`_degrade.py`](_degrade.py) (the ladder), [`degrade_ladder.py`](degrade_ladder.py) (the gate),
> [`gating_control.py`](gating_control.py) (the §3.2 control), `tests/test_degrade_ladder.py`. It is
> kept so the tables below can be re-derived, not because a re-run is planned.

**The insight, which still holds:** native positives are **free labels**. The vector detector reports
ground truth reliably on a born-digital redline, and degrading the page does not move the ink, so the
labels survive the ladder. That much is verified — geometry preservation and the skew transform are
pinned by tests.

**The part that failed:** the assumption that a degraded redline therefore lands *in the failing
regime*. It does not.

### The gate ran, 2026-08-06 — it FAILED. Wrong failure mode.

RapidOCR under `ScanConfig.confidence_free()`, whole ladder, 3 documents / 24 pages / 2170 known
strikes. Cell = recall of the native strike set | predicted struck count.

| document | GT | L0_clean | L1_light | L2_photocopy | L3_fax | L4_thrice |
|---|---|---|---|---|---|---|
| Copyright 37 CFR 201.10 | 34 | 26.5% \| 9 | 29.4% \| 9 | 35.3% \| 11 | 0.0% \| 0 | 0.0% \| 0 |
| FDIC Fair Lending | 1732 | 97.7% \| 1680 | 94.3% \| 1577 | 60.5% \| 783 | 0.0% \| 0 | 0.2% \| 1 |
| Gretna LA UDO edits | 404 | 95.3% \| 380 | 88.4% \| 349 | 92.8% \| 369 | 2.0% \| 5 | 0.0% \| 0 |
| **TOTAL** (recall-weighted) | **2170** | 96.2% \| 2069 | 92.2% \| 1935 | 66.1% \| 1163 | 0.4% \| 5 | 0.1% \| 1 |

1. **The predicted count never once exceeds GT, on any document, at any rung.** It declines
   monotonically (2069 → 1935 → 1163 → 5 → 1) while recall falls 96.2% → 66.1%. The SOF failure is
   **over**-flagging — 113 false positives against 2 real strikes. This is monotonic
   **under**-detection: the opposite sign.
2. **The ladder is destroying ink, not manufacturing confusion** — the "wrong failure mode" branch of
   the script's own reading guide, not the hoped-for one.
3. **L3/L4 annihilate the page** on all three documents. A real photocopy stays readable — Azure DI
   reads the actual SOFs and returns words with low confidence. A zero-detection rung models a blank
   page, not a bad scan.

⚠ Do not read the L0 column as an accuracy claim. Copyright's 26.5% at L0 is RapidOCR on small type,
not the package's DI-measured 95–97%. The gate compares rungs *within* a row; across-row and
across-backend comparisons are not what it is for.

**Three candidate causes, and where each landed:**

| # | Cause | Outcome |
|---|---|---|
| 1 | The gate is confounded — it ran a config structurally incapable of over-flagging (§3.1) | **dead** (§3.2), and in the opposite direction from the worry |
| 2 | The fade model is wrong — PIL's global `Contrast` lifts ink uniformly until no edges survive | **demoted** (§3.3): contrast is close to innocent; `blur` and `scale` are the cliff |
| 3 | Redlines lack the confusers — the SOFs over-flag on **printed table rules and handwriting**, and typeset prose has neither | **stands, and is now the explanation.** Not fixable by any ladder: the confusers are a property of the document class, not of scan quality |

**What the result does not settle:** it rules out *this* ladder run, not every conceivable ladder.
And it says nothing about transfer, because the holdout that would test transfer is **3 strikes on
one page of one document** — too thin to settle anything. Ladder rungs are not documents; never quote
a rung as a sample.

### §3.1 The confound — found after the run, resolved by §3.2

The 113-FP measurement and the ladder numbers were **not the same experiment.** Three variables moved
between them, only the first of which was intended:

| # | Variable | The 113-FP measurement | The ladder |
|---|---|---|---|
| 1 | Document class | private ruled SOFs | public typeset redlines |
| 2 | OCR backend | Azure DI `prebuilt-layout` | RapidOCR |
| 3 | **Config path** | `ScanConfig()` — gating **ON** | `ScanConfig.confidence_free()` — gating **OFF** |

Row 3 was the one that could invalidate the run. `confidence_gating=False` disables every
confidence-dependent decision, including the chain rejection *and* its escape
([`scanned.py`](../src/pdf_strikethrough/scanned.py) — `rescue = config.confidence_gating and …`).
That escape is a **documented over-flagging mechanism**, and C1 exists precisely because it misfires
on ruled forms — so the ladder may have run a decision path structurally incapable of the failure it
went looking for.

**The reading was pre-committed before the control ran**, and both branches were informative: still
over-flags under `confidence_free()` → the config is exonerated and the fade/confuser hypotheses
stand; under-detects too → the gate tested the wrong path and must be re-run with gating on before
the route can be called dead. ⚠ Recorded at the time because the discipline this file exists to
enforce cuts both ways: a result that flatters the pessimistic reading gets audited on the same terms
as one that flatters the optimistic one.

### §3.2 The config control ran — row 3 is cleared, 2026-08-06

[`gating_control.py`](gating_control.py) holds document class and OCR backend fixed (real private
corpus, cached DI results) and moves **only** the config. Free, as predicted: no cloud call, no OCR
pass.

| | `ScanConfig()` — gating ON | `confidence_free()` — the ladder's config |
|---|---|---|
| struck-final | 115 | 344 |
| **false positives** | **113** | **342** (+203%) |
| recall | 2/3 | 2/3 |

The `ScanConfig()` column reproduces the shipped 113 exactly, which is what makes the other column
trustworthy. ⚠ Measured on **PyMuPDF 1.28.2**. A first pass on 1.26.3 returned 111 — that version is
both crash-prone and numerically wrong here (see `CHANGELOG.md`); numbers from it are void.

**The config is exonerated, and in the opposite direction from the worry.** §3.1 feared that gating
OFF makes the chain gate inert and so *suppresses* the failure. It does the reverse: gating OFF also
removes the vetoes that hold false positives **down**, so FP triples while recall is unchanged. The
ladder therefore ran the single most over-flag-prone configuration available and *still*
under-detected at every rung.

It also sharpens cause 3: the FP-permissive config over-flags freely on ruled forms and not at all on
degraded redlines, which is what a **missing-confuser** gap looks like rather than a
mis-tuned-degradation gap. Degrading a page that has no ruled lines cannot manufacture ruled-line
confusion, at any rung.

### §3.3 The L2→L3 ablation — the cliff is `blur` and `scale`

The L2→L3 step moves **six** parameters at once, so the collapse attributed to nothing. `_degrade.py`
carries six one-factor `A_*` probes (each L2 with exactly one field advanced to its L3 value, sharing
L2's `noise_key` so the noise realization is held fixed). They are reachable via `--levels` but
deliberately **not** in `LADDER`, so the recorded gate numbers stay comparable:

```bash
python benchmarks/degrade_ladder.py --levels L2_photocopy,A_contrast,A_blur,A_scale,A_skew,A_noise,A_jpeg,L3_fax
```

Both bookends are in that run on purpose, as a reproduction check on §3 itself — the recorded table's
provenance was in doubt, since the PyMuPDF this environment had at the time segfaults in the flag
detector. **It reproduced exactly**, per-document, on 1.28.2 (11 / 783 / 369 at `L2_photocopy`), and
again after the ladder was refactored to render each document once. So §3's table stands.

Full run, all six probes over all three documents (24 pages, 2170 known strikes), RapidOCR:

| rung | Copyright (34) | FDIC (1732) | Gretna (404) | **TOTAL (2170)** |
|---|---|---|---|---|
| `L2_photocopy` *(baseline)* | 35.3% \| 11 | 60.5% \| 783 | 92.8% \| 369 | **66.1% \| 1163** |
| `A_contrast` | 17.6% \| 6 | 44.6% \| 537 | 92.3% \| 371 | 53.0% \| 914 |
| **`A_blur`** | 2.9% \| 1 | 1.1% \| 13 | 8.9% \| 21 | **2.6% \| 35** |
| **`A_scale`** | 0.0% \| 0 | 5.0% \| 62 | 35.4% \| 123 | **10.6% \| 185** |
| `A_skew` | 32.4% \| 10 | 61.1% \| 791 | 89.6% \| 359 | 65.9% \| 1160 |
| `A_noise` | 26.5% \| 8 | 63.0% \| 857 | 91.8% \| 365 | **67.8% \| 1230** |
| `A_jpeg` | 29.4% \| 9 | 41.6% \| 488 | 90.6% \| 360 | 50.5% \| 857 |
| `L3_fax` *(all six at once)* | 0.0% \| 0 | 0.0% \| 0 | 2.0% \| 5 | 0.4% \| 5 |

**`blur` and `scale` are the whole cliff.** Each alone takes the aggregate from 1163 predicted to 35
and 185 — most of the way to L3's 5 — while `skew` is inert (1160) and `contrast` and `jpeg` are
merely moderate. Read Gretna for the cleanest signal, since Copyright is already near the floor at L0
and exaggerates everything: there, advancing `contrast` all the way to its L3 value changes
**nothing** (92.8% → 92.3%, 369 → 371) while `blur` alone collapses it to 8.9%.

That kills cause 2, and with it the planned escape. The hypothesis was that global `Contrast` lifts
ink uniformly until the detector finds no edges; the measurement says **edge destruction — blur and
resampling — does the damage.** So a gamma / local-threshold refade into the "readable band"
**would not rescue this ladder and should not be built.** `A_contrast` *is* the readable band (faded
to L3's contrast, still sharp), and its Gretna count is **371 against GT 404 — still under ground
truth.** `A_scale` also answers §3.1's resolution question directly, which is why the separate
resolution comparison is no longer carried here.

**The one residual lead, recorded honestly: `A_noise` is the only factor that pushes the count UP**
(1163 → 1230, recall 66.1% → 67.8%). Every other factor reduces detections; sensor noise adds them.
It is the only axis of this ladder that moves *toward* the over-flagging regime, and it was never
isolated above L3's `noise_sigma=6.0`, since L4 raises it to 9.0 but simultaneously applies the blur
and scale that annihilate the page.

⚠ **Do not oversell that.** The effect is small (+5.8% count) and recall rises alongside it, so it is
mostly finding more real strikes, not manufacturing confusers. And **cause 3 caps it regardless**: no
amount of speckle draws a printed table rule onto a page of typeset prose. A noise-forward ladder is
the only version of this route not yet falsified, which is a reason to record the lead — not a reason
to reopen the route ahead of §2's rows 1–3.

**No probe, at any setting, ever exceeded ground truth.** The highest count on the board is 1230
against GT 2170. The ladder has no over-flagging regime in it.

---

## §4 What counts as a usable positive

Guardrails set before collection, for the same reason as §1.

- **A strike is a mark through the x-height that deletes text.** An underline is not a strike
  (`B7` owns that distinction; the IOC `SOUTH,ARATU` dotted fill-in is an underline). A glyph
  crossbar is not a strike — the cursive `ft` ligature in `ANT-BULK-HONDURAS` is the cautionary case.
- **Label per `training/README.md`: `"struck"` / `"clean"`, not `1` / `0`**, which crashes the
  trainer.
- **Verify positives by eye at high dpi before counting them.** `ground-truth.json` was verified by
  rendering every candidate at 420 dpi; anything less has already produced a wrong label here.
- **PHI split rule:** public/redistributable sources → `manifest.json` with a sha256, files
  git-ignored under `corpus/` and re-downloadable. Anything with patient or client data → **never a
  manifest entry**; it goes under `benchmarks/private/` (git-ignored wholesale) and is never
  committed, quoted verbatim, or sent to a cloud API without checking that first.
- **Dumping crops for training: run with the veto OFF.** A vetoed line never reaches the CNN, so
  dumping under `ruled_forms()` silently drops the very rules and fragments wanted as negatives.
  `dump_crops` takes `di_result=`, so it reuses cached DI results rather than re-OCRing.

---

## §5 Why negatives, not positives, are the binding constraint

Worth restating because it inverts the intuition and it is what makes the data problem tractable:

- The zero-FP bar means the model's job is overwhelmingly **to not fire**. Negatives bound
  correctness; positives only stop a collapse to "never fire" (which scores a perfect zero FP and
  zero utility) and make recall measurable.
- **Useful asymmetry:** what a strike *looks like* — a line through the x-height — generalizes across
  document types. What *confuses* the detector — table rules, scan degradation, glyph crossbars — is
  domain-specific. **So positives can come from anywhere; negatives must look like SOFs.** This is
  the sentence that unblocks the pull: the positives never needed to be SOFs.
- Negatives are **already in hand**: every clean document in the 315-PDF sweep is a hard negative in
  the failing regime. The 28-FP residual is a characterized *sample* of that asset, not the whole.
- **Abstention** (struck / clean / **abstain**, with a conformal guarantee on the abstention rate) is
  the mechanism that can actually meet the bar, because it converts silent data loss into flagged
  review. Machinery exists: `calibration.py`, `conformal_threshold`, and `p_hi` is already a
  split-conformal threshold. It needs a calibration set representative of deployment — again, mass
  negatives.
