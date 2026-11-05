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
- **Thresholds are calibrated, not guessed.** `p_hi` is a split-conformal threshold on held-out
  struck-word probabilities (`--alpha` sets the recall floor, `1 - alpha`). `p_lo` mirrors `p_hi`
  on the clean class (the k-th highest clean validation score, k from the clean count), capped at
  `p_hi`, so it carries no floor of its own. See `pdf_strikethrough.calibration`. Both are set on
  the scores the detector compares with them: the exported model's scores for the validation
  crops, rounded to 3 decimals as the detector rounds them. `p_hi`'s floor holds on average for
  new crops exchangeable with the validation crops. The validation crops come from groups the
  model did not train on (next note), so for a dump, whose rows carry the page, the thresholds are
  set on pages the model has not seen. Still, a new page's crops are generally not exchangeable
  with them, because crops of one page tend to look alike. So struck crops from few held-out pages
  count for less than their number suggests: with few such pages, recall on new pages can fall
  below the floor, and on new documents more so. Before it trains, the script stops if the
  validation split holds fewer struck crops than `--alpha` needs (19 at the default 0.05) or no
  clean ones, or if the training split has no struck crop or no clean crop. With 1 to 18 clean
  validation crops at the default `--alpha` it trains anyway and `p_lo` equals `p_hi`, as it does
  when too many clean validation crops score at or above `p_hi`; then the model calls no word
  unsure, and the script says so.
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
  split, and its random stream can change between versions. `pdf_strikethrough.cnn.get_model_meta()`
  returns it with the rest.
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
