---
license: mit
library_name: onnxruntime
tags:
  - strikethrough-detection
  - document-ai
  - redline
  - pdf
  - onnx
pipeline_tag: image-classification
---

<!--
HF MODEL CARD for StrikeNet. Published copy lives at https://huggingface.co/niles-liu/strikenet
(as that repo's README.md). This is the source of record; keep the two in sync when the model or
numbers change. The weights every release has shipped (version "v3-corpus") are what is hosted.
-->

# StrikeNet — strike/clean word classifier

StrikeNet is the 79k-parameter CNN that [`pdf-strikethrough-detect`](https://github.com/niles-liu/pdf-strikethrough-detect)
uses to resolve pixel-ambiguous strikethrough cases on **scanned** document pages. A thin strike
over an ascender-less word is pixel-identical to an ordinary glyph chain; geometry alone cannot
separate them, so a learned model casts the deciding vote. On born-digital PDFs the package reads
vector strokes and annotations directly and never invokes this model.

- **Input:** one standardized grey word crop, `32`×`160` (ink-positive,
  height-normalized). Produced by `pdf_strikethrough.cnn.std_crop`; the exact geometry is recorded
  in `strike_verdict_cnn.meta.json` and enforced at load time (`cnn._check_geometry`) so training
  and inference preprocessing cannot drift.
- **Output:** a single logit → sigmoid probability that the word is struck.
- **Decision thresholds** (in the meta): `p_hi = 0.85`, `p_lo = 0.15`. A word scoring `≥ p_hi` is
  struck, `≤ p_lo` is clean, and the `[p_lo, p_hi)` band is "unsure" and deferred to geometry. How
  the shipped values were chosen is not recorded (see [Provenance](#provenance-and-training)); the
  training script sets a retrained model's thresholds from data: `p_hi` as a split-conformal
  threshold on held-out struck-word probabilities (a recall floor of `1 − alpha` for crops like the
  validation set; see `training/README.md`), `p_lo` mirrored on the clean class. Override per-call
  with `ScanConfig.recall_first(cnn_p_hi=…)` / `precision_first(cnn_p_lo=…)`.

## How it fits the pipeline

Scanned page → OCR words + morphological stroke geometry (`lines.py`) → StrikeNet adjudicates the
`auto`/`review` tier crops → struck words, char-level partial resolution, struck-aware markdown.
Full architecture: the project [README](https://github.com/niles-liu/pdf-strikethrough-detect#how-it-works).

## Evaluation

Measured on the project's reproducible **10-document public regulatory-redline corpus** (US
Copyright Office, FDIC, CEQ ×2, EPA ×3, California CCPA/CPPA, Gretna LA development code; 55.2k
struck words, each `sha256`-pinned). Reproduce with `benchmarks/scanned_recovery.py`.

| Metric | Value | Source |
|---|---|---|
| Scanned-path strike recovery (RapidOCR) | **96%** | `scanned_recovery.py`, 3 docs / 24 pages / 2,170 known strikes |
| Scanned-path strike recovery (Azure DI) | **96%** with `confidence_free()`; 91% with the default DI calibration | same harness, Azure Document Intelligence words |
| Native vector detections independently confirmed by the flag signal | 99.8% | `confirmation_rate.py` (context; native path, not this model) |

The two scanned-path figures measure the whole pipeline (stroke geometry, OCR words and this model
together), not the model on its own. Whether these documents overlap the model's training data is
not recorded (see [Provenance](#provenance-and-training)). No held-out precision/recall figure for
the model itself exists yet: it needs labeled crops the model cannot have seen, such as from
documents published after July 2026, or a retrain on a recorded labeled set.

## Usage

The package runs on ONNX Runtime with no torch dependency. Download is digest-verified before the
graph is ever loaded — a tampered host cannot swap the model.

```python
import pdf_strikethrough as st

st.ensure_model(
    "https://huggingface.co/niles-liu/strikenet/resolve/main/strike_verdict_cnn.onnx",
    "fac2c51baaa75ee782196bdfe7452638cb48c7deddb21163b1ac6a0a72ae4457",
    meta_url="https://huggingface.co/niles-liu/strikenet/resolve/main/strike_verdict_cnn.meta.json",
    meta_sha256="4388b14715bfb1f51b56bb6c463d8f5c0847533316297890e70ac0b2234405d4",
)
assert "p_hi" in st.get_model_meta()          # thresholds + crop geometry now loaded
result = st.detect_pdf("scanned-redline.pdf", ocr=st.rapidocr_backend())
```

## Provenance and training

**The shipped weights cannot be reproduced from the package's repository.** The ONNX the package
ships (`version: v3-corpus`, also hosted at `niles-liu/strikenet`) is byte-identical in every
release since the first (0.4.0, July 2026). It predates `training/train_strikenet.py`, which did not
produce it. Its training set, train/validation split and training settings are not recorded in the
package's repository or any release, and neither is how its thresholds (0.85 / 0.15) were chosen:
the first release's code calls them guardrails, nothing more.

**Training a replacement.** `training/train_strikenet.py` trains StrikeNet on a labeled crop set the
detector exports, sets `p_hi` / `p_lo` from held-out data, and records in the meta, under
`training`, what the model was trained on: a digest of the labeled crops and their labels, the class
counts, the validation size, the hyperparameters (the seed that draws the split among them) and the
package and torch versions. The digest identifies the labeled set but cannot restore it, so keep
the labeled directory. The failing-page → better-model loop is one command per step:

```bash
pdf-strikethrough detect scan.pdf --ocr rapidocr --dump-crops crops_out/   # export scored crops
# label crops_out/crops.jsonl: set each row's "label" to "struck" or "clean"
python training/train_strikenet.py crops_out/ -o model_out/ --epochs 40    # train + calibrate + ONNX
```

Full loop and label format: [`training/README.md`](https://github.com/niles-liu/pdf-strikethrough-detect/blob/main/training/README.md).

## Limitations

- **Handwritten / freehand strikes** are the main known gap: untested, and nothing records that the
  model saw any in training, so a wavy pen scribble or heavy cross-out may be missed.
- **Horizontal, left-to-right text only.** Vertical writing modes (CJK/Mongolian) and RTL strike
  axes are out of scope.
- The model only sees word crops the geometry stage flags as candidates; it does not itself find
  words on the page.

## License

MIT, matching the parent package.
