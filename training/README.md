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
  into the meta — `cnn._check_geometry` refuses to load a model whose geometry disagrees with the
  code constants.
- **Thresholds are calibrated, not guessed.** `p_hi` is a split-conformal threshold on held-out
  struck-word probabilities (`--alpha` sets the guaranteed recall floor, `1 - alpha`); `p_lo`
  mirrors it on the clean class. See `pdf_strikethrough.calibration`. The floor holds for new crops
  exchangeable with the validation crops. The split is random over crops, not documents, so the
  validation crops come from documents the model also trained on; on any document outside the
  labeled set, recall can fall below the floor. The script stops when the validation split holds
  fewer struck crops than `--alpha` needs (19 at the default 0.05), or no clean ones.
- **The model records what it was trained on.** The exported meta carries a `training` block: the
  sha256 of the labeled set (one `"<crop sha256> <label>"` line per labeled row in `crops.jsonl`
  order, so it changes with any crop file, label or row order, but not with file names), the
  struck/clean counts, the validation size, the hyperparameters (`--epochs`, `--val-frac`,
  `--batch`, `--lr`, `--seed`, `--alpha`) and the package and torch versions.
  `pdf_strikethrough.cnn.get_model_meta()` returns it with the rest.
- **A failed export changes nothing.** The ONNX goes to a temporary file and replaces the output
  directory's model only once ONNX Runtime scores it like the trained net.
- **Dev-only dependencies.** Training needs `torch`, and exporting needs `onnx`, which torch's ONNX
  exporter imports (`pip install torch onnx`; the script checks for both before it trains). The
  package itself runs on ONNX Runtime with no torch dependency.
- **A meaningful model needs a real corpus.** A handful of crops from one PDF will overfit — this
  is the pipeline, not a substitute for collecting a labeled set (see `benchmarks/`).
