# Training Guide

Complete step-by-step instructions to reproduce the 1st-place result (0.98734).

---

## Environment Setup

```bash
conda create -n railway python=3.10
conda activate railway
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Requirements: CUDA-capable GPU with ≥16 GB VRAM (tested on H100). Mixed-precision (AMP) is used throughout so a single GPU is sufficient.

---

## Dataset Layout

Place the dataset in the same directory as the notebook:

```
medium_dataset/
├── train_set/
│   ├── excavator/
│   ├── train/
│   ├── train_tracks/
│   └── workers/
└── test_set/          ← unlabeled images for submission
```

The notebook uses `torchvision.datasets.ImageFolder` — folder names become class labels automatically.

---

## Notebook: `train_dinov3.ipynb`

Open the notebook and run cells in the order described below.

### Cell 0 — Imports

Imports all required libraries. Sets random seeds for reproducibility.

### Cell 1 — Constants

Key hyperparameters — change these if needed:

| Constant | Value | Notes |
|----------|-------|-------|
| `IM_SIZE` | 518 | DINOv3 ViT-L native input size |
| `BATCH_SIZE` | 8 | Fits 16 GB VRAM with AMP + gradient checkpointing |
| `VAL_SPLIT` | 0.15 | 15% of train_set held out for validation |
| `TEST_SPLIT` | 0.15 | 15% held out as internal labeled test |
| `MODEL_NAME` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | Downloaded automatically from HuggingFace |

### Cell 2 — Class Distribution

Plots the class imbalance. Workers and excavators dominate (~45% each); train and train_tracks are minority classes (~7% each). A `WeightedRandomSampler` corrects for this in the DataLoader.

### Cell 3 — Focal Loss

Custom loss with `gamma=2.0` and `label_smoothing=0.1`. The focal factor `(1-p_t)^gamma` suppresses gradient from high-confidence correct predictions, concentrating training signal on the hard excavator↔workers boundary.

### Cell 4 — Transforms & Splits

**Training augmentations:**
- `RandomResizedCrop(518, scale=(0.2, 1.0))` — the 0.2 lower bound is critical: it forces the model to classify objects that occupy only 20% of the image, matching real-world tiny-worker scenes
- `RandomHorizontalFlip`, `RandomVerticalFlip(p=0.2)`
- `ColorJitter(0.3, 0.3, 0.2, 0.1)`
- `RandomGrayscale(p=0.05)`

**10-view TTA transforms** (applied at inference only):

| # | Augmentation |
|---|-------------|
| 1 | Center crop — original view |
| 2 | Center crop + horizontal flip |
| 3 | Center crop + vertical flip |
| 4 | Center crop + H+V flip |
| 5 | Tight crop (85–100% scale) |
| 6 | Tight crop + horizontal flip |
| 7 | Zoom-in crop (65–85% scale) |
| 8 | Zoom-in crop + horizontal flip |
| 9 | Rotate 90° |
| 10 | Rotate 270° |

**Data split:** Stratified 3-way split using `sklearn.model_selection.train_test_split` with `SEED=42`. Class balance is preserved in all three splits.

### Cell 5 — TTA Datasets

- `TTALabeledDataset` — wraps the labeled test subset, returns `[N_TTA, C, H, W]` tensor per image
- `TTAUnlabeledDataset` — wraps the unlabeled test directory, returns `[N_TTA, C, H, W]` + filename

### Cell 6 — Batch Visualization

Sanity check: displays a grid of training images with augmentations applied. Verify class labels are correct before training.

### Cell 7 — Model Definition

```python
backbone = AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')
# CLS token [B, 1024] → Linear → BN → GELU → Dropout → Linear(4)
backbone.head = nn.Sequential(...)
```

`dino_forward(model, images)` extracts `outputs.last_hidden_state[:, 0]` (the CLS token) and passes it through `model.head`.

### Cell 8 — Trainer

Shared training loop used by all 3 stages:
- Optimizer: `AdamW` with `CosineAnnealingLR`
- Loss: `FocalLoss` with class weights from the sampler
- AMP: `GradScaler` + `autocast('cuda')`
- Gradient clipping: `clip_grad_norm_(1.0)`
- **Saves on best validation accuracy** (not loss)
- Early stopping with configurable `patience`

### Cell 9 — Metrics Plot

Plots loss / accuracy / F1 curves for train and val. Call after each stage.

---

## Stage 1 — Head-Only Training

**Cell 10** — run this cell:

```
Frozen:    entire DINOv3 backbone (307M params)
Trainable: classification head only (~530K params)
LR:        1e-3
Epochs:    20 (early stop patience=8)
Saves to:  output_dinov3_finetune/dinov3_s1_best.pth
```

Expected result: val_acc ≈ 0.98–0.99 within 10–12 epochs.

---

## Stage 2 — Last 6 Blocks Fine-tuning

**Cell 12** — run this cell:

```
Unfrozen:  model.layer[-6:]  (blocks 18–23 of 24) + model.norm
LR:        1e-5  (100× lower than Stage 1)
Epochs:    20 (early stop patience=8)
Saves to:  output_dinov3_finetune/dinov3_s2_best.pth
```

Why these blocks? Blocks 18–23 encode high-level semantic features — the layer where "worker standing next to excavator" vs "just excavator" is distinguished. Lower LR prevents catastrophic forgetting of the pretrained representations.

Expected result: val_acc ≈ 0.993–0.995.

---

## Stage 3 — Last 12 Blocks Fine-tuning

**Cell 13** — run this cell:

```
Unfrozen:  model.layer[-12:]  (blocks 12–23 of 24) + model.norm
LR:        5e-6  (6× lower than Stage 2)
Epochs:    15 (early stop patience=6)
Saves to:  output_dinov3_finetune/dinov3_s3_best.pth
```

Stage 3 opens up 6 additional mid-level semantic blocks. These encode the intermediate features needed to distinguish workers-in-heavy-context from pure machinery scenes. The even lower LR ensures S2 gains are preserved while allowing further refinement.

Expected result: val_acc ≈ 0.993–0.995 (matches or slightly improves S2).

---

## Evaluation on Labeled Test Set

**Cell 14** — defines `predict_tta_ensemble(models, loader)`:
- Runs all 10 TTA views through each model in the ensemble
- Averages `weight × softmax` across views and models
- Returns argmax predictions

**Cell 15** — loads S2 (weight=0.4) and S3 (weight=0.6) and evaluates on the internal labeled test:

```
Expected:  ~99% accuracy, 3 misclassifications (all excavator↔workers)
```

If S3 val_acc is lower than S2, adjust weights: `(0.6, 0.4)` or `(1.0, 0.0)`.

---

## Submission

**Cell 17** — runs the S2+S3 ensemble with 10-view TTA on the unlabeled test set:

```
Output: output_dinov3_finetune/submission_s2s3_ensemble_tta10.csv
Format: ID (filename), Label (0–3)
```

---

## Training Timeline (H100)

| Stage | Time/epoch | Total |
|-------|-----------|-------|
| Stage 1 (head only) | ~32s | ~5–6 min |
| Stage 2 (6 blocks) | ~46s | ~4 min |
| Stage 3 (12 blocks) | ~60s | ~2 min |
| TTA inference | — | ~3 min |

---

## Reproducing the Exact Result

```
SEED = 42
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
Ensemble weights: S2=0.4, S3=0.6
TTA views: 10
```

The stratified split is deterministic given `SEED=42`. The model checkpoint selection (best val_acc) is deterministic given the same data split and seed.

---

## Experiment Log

See [`experiment_log.txt`](experiment_log.txt) for full metric history across all experiments, including the ConvNeXt baseline and intermediate DINOv3 runs.
