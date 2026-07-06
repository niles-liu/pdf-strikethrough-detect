# Training StrikeNet

The package ships the strike/clean CNN as `strike_verdict_cnn.onnx` (79k params). This directory
holds the script that produces it, so the shipped weights are reproducible and you can fine-tune on
your own documents.

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
  mirrors it on the clean class. See `pdf_strikethrough.calibration`.
- **Dev-only dependency.** Training needs `torch` (`pip install torch`); the package itself runs on
  ONNX Runtime with no torch dependency.
- **A meaningful model needs a real corpus.** A handful of crops from one PDF will overfit — this
  is the pipeline, not a substitute for collecting a labeled set (see `benchmarks/`).
