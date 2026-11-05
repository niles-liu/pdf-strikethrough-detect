# Training StrikeNet

The package ships the strike/clean CNN as `strike_verdict_cnn.onnx` (79k params). This directory
holds the script that trains a replacement for it on your own labeled documents. The shipped
weights predate the script, were not produced by it, and cannot be reproduced from this repository:
their training data is not recorded here ([model card](MODEL_CARD.md#provenance-and-training)). A
model the script trains records a digest of its labeled set and its settings in its meta; keep the
labeled directory, since the digest identifies the data but cannot restore it.

## The loop

```bash
# 1. Export the crops the detector actually scored, as a labeling set:
pdf-strikethrough detect scan.pdf --ocr rapidocr --dump-crops crops_out/

# 2. Label them: open crops_out/crops.jsonl and set each row's "label" to "struck" or "clean"
#    (the crop PNG is next to it under crops_out/crops/). Rows left unlabeled are skipped.

# 3. Train + calibrate + export ONNX:
python training/train_strikenet.py crops_out/ -o model_out/ --epochs 40

# 4. Use the new model:
PDF_STRIKETHROUGH_MODEL_DIR=model_out/ pdf-strikethrough detect scan.pdf --ocr rapidocr
#    or in code: pdf_strikethrough.cnn.set_model_dir("model_out/")
#    or from a URL with digest verification: pdf_strikethrough.cnn.ensure_model(url, sha256, ...)
```

Steps 1–2 are also how a "contribute a failing page" bug report becomes training data.

## Notes

- **Preprocessing can't drift.** Crops are re-standardized through the same `cnn.std_crop` the
  detector uses at inference, and the export records the crop geometry (`crop_h/crop_w/pad_x/pad_y`)
  into the meta, and `cnn._check_geometry` refuses to load a model whose geometry disagrees with the
  code constants.
- **Thresholds are matched to the model the package loads.** The script scores the validation crops
  with the new model and with a reference: the model the package loads, which is the packaged one
  unless `PDF_STRIKETHROUGH_MODEL_DIR` or `cnn.set_model_dir` points at another. `p_hi` is the
  highest threshold at which the new model calls at least as many struck validation crops struck
  as the reference does at its own `p_hi`, in each source (next note). `p_lo` is the lowest at
  which it calls at least as many clean validation crops clean as the reference does at its own
  `p_lo`, capped at `p_hi`. So on the validation crops the new model calls at least as many
  strikes struck as the reference does, in every source, and training mostly changes how many
  clean crops it calls struck. The script prints both models' counts. Both thresholds use the
  scores the detector compares with them: each model's scores for the validation crops, rounded
  to 3 decimals as the detector rounds them. Before it trains, the script stops if the reference
  does not load, gives a validation crop no finite score, or calls none of the struck validation
  crops struck. It also stops if the validation split holds fewer than 19 struck crops (more when
  `--alpha` is below 0.05) or no clean one, or if the training split has no struck crop or no
  clean crop. When the match puts `p_lo` at or above `p_hi`, `p_lo` equals `p_hi` and the model
  calls no word unsure; the script says so.
- **Sources.** A row may carry a `source`: the kind of crop, such as `"print"` or
  `"handwriting"`, not the document or page it came from (name those in `group`). Any JSON value
  will do, compared as JSON text, so `"Print"` and `"print"` are two sources. As for `group`,
  null, NaN, blank text, `[]` and `{}` count as none, and those rows form a source of their own.
  Any other NaN or Infinity, such as one inside a list, stops the script. `p_hi` is matched in each
  source that has struck validation crops. Each match is a bound on `p_hi`, and the lowest bound
  is kept, so on the validation crops no source loses recall. A source sets no bound when the
  reference calls none of its validation strikes struck, or when the split holds out none of its
  strikes, which is likely for a source in few groups; the script names it either way. Each source
  can only lower `p_hi`, and a source with few struck validation crops sets a noisy bound that is
  often the lowest, so one crop can set `p_hi` for every source and raise the number of clean
  crops called struck. Since every source can bound `p_hi`, label only crops like the ones the
  detector will score: a generated or augmented strike can land in the validation split like any
  other crop and bound `p_hi` there. `p_lo` is matched on all the clean crops at once.
- **What the match promises.** It holds on the validation crops. On new crops it holds only on
  average, and only to within a small shortfall, even for crops exchangeable with the validation
  ones: in a source with n struck validation crops, the new model's expected recall can trail the
  reference's by up to 1/(n + 1) (5 points when n is 19), and on one split it can trail by twice
  that or more. The validation crops come from groups the model did not train on (see the split
  below), but crops of one page or writer tend to look alike, so with few held-out groups the new
  model can trail further on new pages. The match holds the new model to the reference's recall,
  not more: if the reference misses many of your strikes, the new model misses about as many. To
  catch more strikes at the cost of more clean words called struck, pass a lower `cnn_p_hi` when
  you detect, with `ScanConfig(cnn_p_hi=...)`. The meta's `training.calibration.conformal.p_hi` is
  one choice, when it is below `p_hi`. Matching to an earlier retrain adds its shortfall to the
  new model's whenever the validation crops change, as they do when you label more pages or change
  `--seed`, so recall can drift down over retrains. To match to the packaged model, unset
  `PDF_STRIKETHROUGH_MODEL_DIR` when you train. `ScanConfig.recall_first()` and
  `precision_first()` set `cnn_p_hi` to 0.50 and 0.97, values on the packaged model's scale, so
  with a retrained model pass `cnn_p_hi` yourself.
- **The split-conformal pair is only reported.** The meta also records a split-conformal pair
  from the new model's validation scores. Its `p_hi` is `calibration.conformal_threshold` at
  `--alpha`: on average, at most `alpha` of struck crops exchangeable with the validation ones
  score below it, and a new page's crops generally are not exchangeable with them. Its `p_lo` is
  the script's mirror of that on the clean scores (the k-th highest, k from the clean count),
  capped at `p_hi`. Neither sets the thresholds.
- **The split holds out whole groups.** A row's group is its `group` field if set, else its
  `page` within its crop's folder, else the row alone. A `group` can be any JSON value, such as a
  document name or a writer's ID; null, NaN, blank text, `[]` and `{}` count as not set.
  `dump_crops` rows carry the page, so a dump splits by page. The script holds out
  `max(1, int(G * --val-frac))` of the G groups, drawn with `--seed`, and prints the split before
  training. Since `--val-frac` is a share of groups, the share of crops held out varies with the
  groups' sizes. A one-page dump is a single group, so holding it out leaves nothing to train on:
  label crops from more pages, or give each row a `group` of its own, such as its line number, to
  split by crop, which can put crops of one page on both sides. A page where some rows have a
  `group` and others do not can land on both sides too; the script prints a note.
- **Several dumps.** To train on several dumps, put each dump's crops in a folder of its own in
  one directory, such as `a/crops/` and `b/crops/`, and concatenate their `crops.jsonl` rows into
  one, with each `crop` path prefixed to match (`a/crops/p0003_00012.png`). A page is told apart
  by its crop's folder, so page 3 of one dump and page 3 of another stay separate groups. If two
  dumps cover the same pages, give each row a `group` naming its document and page, such as
  `["scan.pdf", 3]`, so a page dumped twice stays on one side.
- **The model records what it was trained on.** The exported meta carries a `training` block: the
  sha256 of the labeled set (one `"<crop sha256> <label>"` line per labeled row in `crops.jsonl`
  order, so it changes with any crop file, label or row order, but not with file names), the
  struck/clean counts, the validation size, how the split grouped the rows, the hyperparameters
  (`--epochs`, `--val-frac`, `--batch`, `--lr`, `--seed`, `--alpha`) and the package, numpy and
  torch versions. The grouping is `split_by` (how many rows took their group from `group`, from
  their page, or stood alone, as `row`), `n_groups`, `n_val_groups` (how many groups are held out)
  and `groups_sha256` (the sha256 of one line per labeled row with its group's number, groups
  numbered in order of first appearance). The numpy version is there because numpy draws the
  split, and its random stream can change between versions. `calibration` records how the
  thresholds were set. `rule` states the rule. `reference` holds the reference's `version`,
  `sha256`, `external_data` (when it has any), `p_hi` and `p_lo`. `p_hi.by_source` has one entry
  per source with labeled strikes: `source` (null for the rows without one); `struck`, its struck
  validation crops (0 when the split holds none out); `groups`, how many groups they come from;
  `reference_calls`, how many of them the reference calls struck; and `matched`, the bound they
  set (null when there is none). `p_hi.set_by` is the source that set `p_hi`, and `p_hi.pooled`
  the bound on all sources at once, reported only. `p_lo` holds `clean`, the clean validation
  crops; `reference_calls`, how many of them the reference calls clean; and `matched`, the match
  before the cap. `called_struck` counts, for the `struck` and the `clean` validation crops, how
  many each model (`reference`, `new`) calls struck. `conformal` holds the split-conformal pair:
  `alpha`, `p_hi`, `p_lo`, and `mirror`, which is `p_lo` before the cap (null with too few clean
  crops). `pdf_strikethrough.cnn.get_model_meta()` returns it with the rest.
- **A failed check changes nothing.** The ONNX goes to a temporary file. It replaces the output
  directory's model only once ONNX Runtime scores it like the trained net and its thresholds pass
  the loader's check (`0 <= p_lo <= p_hi <= 1`). If the replace itself fails, the old model and
  its meta stay. An export stopped after the replace leaves the new ONNX without a usable meta
  (none, or a partly written one, which the loader refuses), never beside the old meta.
- **Dev-only dependencies.** Training needs `torch`, and exporting needs `onnx`, which torch's ONNX
  exporter imports (`pip install torch onnx`; the script checks for both before it trains). The
  package itself runs on ONNX Runtime with no torch dependency.
- **A meaningful model needs a real corpus.** A handful of crops from one PDF will overfit: this
  is the pipeline, not a substitute for collecting a labeled set.
