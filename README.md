# Breast Cancer Grading — Deep Learning Pipeline

Automatic grading of breast cancer histopathology images (Grade I / II / III) using a
4-experiment CNN pipeline with progressive reporting. Charts and Excel update after every
single model finishes so results are always available mid-run.

---

## Architecture Overview

| Stage | Status | Description |
|-------|--------|-------------|
| **Stage 1 — Core Pipeline** | ✅ Complete | Exp1–Exp4: AlexNet, VGG16, ResNet50 with ReLU → GELU → Attention → Fusion |
| **Stage 2 — Data Pipeline V2** | 🔄 In Progress | Group-aware patient-level split, pre-augmentation pipeline |
| **Stage 3 — Mask-Guided Attention** | ⏳ Planned | Spatial masks from HoVer-Net / U-Net segmentation outputs |

**Experiment chain:**
```
ImageNet → Exp1 (ReLU, no attention)     ← independent baseline
ImageNet → Exp2 (GELU, no attention)     ← fair GELU comparison
Exp2.pth → Exp3 (GELU + attention)       ← backbone reused, only attention trains
Exp3.pth → Exp4 (3-CNN fusion + GELU + attention)  ← all backbones frozen
```

---

## Project Structure

```
CancerGrading/
├── config.py                    # All toggles live here
├── main.py                      # Entry point — numbered menu 1-7
├── requirements.txt
│
├── data/
│   └── loader.py                # DataLoader factory (60/20/20 stratified split_ using k-fold split + augmentation)
│
├── models/
│   ├── attention.py             # SEBlock, CBAM, MaskGuidedAttention stub
│   ├── alexnet.py               # CustomAlexNet (pretrained, configurable act + attn)
│   ├── vgg16.py                 # CustomVGG16
│   ├── resnet50.py              # CustomResNet50
│   └── fusion.py                # FusionModel — AlexNet + VGG16 + ResNet50 concat head
│
├── utils/
│   ├── logger.py                # Console + file logging
│   ├── trainer.py               # train(), evaluate(), run_or_load() — 3-tier cache
│   ├── results_saver.py         # In-memory accumulator
│   ├── reporter.py              # DiagnosticCompiler → 4-sheet Excel
│   ├── plotter.py               # Training curves, CM heatmap, per-class bar, comparison
│   └── nottingham_validator.py  # Nottingham grading validation utilities
│
├── experiments/
│   ├── exp1_baseline.py         # CNN Classic (ReLU)
│   ├── exp2_gelu.py             # CNN + GELU
│   ├── exp3_attention.py        # CNN + GELU + Attention (SE-Block)
│   └── exp4_fusion.py           # Fusion CNN + GELU + Attention (SE-Block)
│
└── outputs/
    ├── *_eval.json              # Per-model metrics cache
    ├── *_meta.json              # Per-model training history
    ├── *.pth                    # Checkpoints
    ├── predictions_*.json       # Per-sample prediction records
    └── reports/
        ├── breast_cancer_grading_diagnostic_report_YYYYMMDD.xlsx
        ├── training_{Exp}_{Model}.png
        ├── cm_{Exp}_{Model}.png
        ├── per_class_{Exp}_{Model}.png
        └── model_comparison.png
```

---

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Prepare dataset**

Organize histopathology images into ImageFolder structure:
```
your_dataset/
├── Grade1/   *.bmp  *.png  *.jpg
├── Grade2/
└── Grade3/
```

**3. Configure (optional)**

Edit `config.py` to set paths and hyperparameters before running:
```python
INPUT_DIR      = "path/to/your_dataset"   # leave blank to be prompted
OUTPUT_DIR     = "outputs"
ATTENTION_TYPE = "SE"                     # "SE" or "CBAM"
BATCH_SIZE     = 8                        # reduce to 4 if out of memory
EPOCHS         = 50
```

**4. Run**
```bash
python main.py
```

**5. Select an option**
```
1. Experiment 1 - Baseline CNN (ReLU)
2. Experiment 2 - CNN + GELU
3. Experiment 3 - CNN + GELU + Attention (SE-Block)
4. Experiment 4 - Feature Fusion (3 CNNs) + GELU + Attention (SE-Block)
5. Run All  Exp1 → Exp2 → Exp3 → Exp4
6. Regenerate Excel report only
7. Regenerate all charts only
```

Choose **5** to run the full pipeline. Results are written to `outputs/reports/` as each
model finishes — you can open the Excel file mid-run to see progress.

---

## Experiment Details

### Exp1 — Baseline CNN (ReLU)
- Models: AlexNet, VGG16, ResNet50 (each trained independently)
- Activation: ReLU (torchvision default)
- Attention: None
- Init: ImageNet pretrained

### Exp2 — CNN + GELU
- Same as Exp1 but all activations replaced with GELU
- Trains independently from ImageNet (fair comparison against Exp1)

### Exp3 — CNN + GELU + Attention (SE-Block)
- Adds attention module (type set by `ATTENTION_TYPE` in `config.py`)
- Backbone warm-started from Exp2 checkpoint (`strict=False`) — only the attention block
  trains from scratch, saving the expensive VGG16/ResNet50 training time
- Placement: after `features` block for AlexNet/VGG16; after each stage for ResNet50
- **Auto-dependency:** triggers Exp2 automatically if its checkpoint is missing

### Exp4 — Fusion CNN + GELU + Attention (SE-Block)
- Concatenates penultimate features: AlexNet(4096) + VGG16(4096) + ResNet50(2048) = **10,240**
- **LayerNorm(10240)** normalises heterogeneous backbone features before the fusion head
- Fusion head (v2): `LayerNorm → Linear(10240→4096) → GELU → Dropout(0.2) → Linear(4096→2048) → GELU → Dropout(0.2) → Linear(2048→3)`
  - Deeper 3-layer MLP with moderate dropout (0.2) for more capacity without over-regularising
  - Original v1 head: `Linear(10240→1024) → GELU → Dropout(0.5) → Linear(1024→3)`
- **Class-weighted CrossEntropyLoss** computed from training label frequencies to combat Grade I under-prediction
- Uses `HEAD_LR` (default `1e-3`) — higher learning rate suitable when only the fusion head trains
- All three backbones loaded from Exp3 checkpoints and **frozen** — only head trains
- **Auto-dependency:** triggers Exp3 → Exp2 chain automatically if checkpoints are missing

---

## Three-Tier Caching

Re-running the same experiment is safe and fast:

| Tier | Condition | Behaviour |
|------|-----------|-----------|
| 1 | `*_eval.json` exists | Load metrics instantly — skip model entirely |
| 2 | `*.pth` + `*_meta.json` exist | Skip training — run evaluate() only |
| 3 | Nothing found | Full train + evaluate |

Set `FORCE_RERUN = True` in `config.py` to bypass all tiers and retrain from scratch.

---

## Attention Options

Change `ATTENTION_TYPE` in `config.py` to switch between attention strategies:

| Value | Description | Stage |
|-------|-------------|-------|
| `"SE"` | Squeeze-and-Excitation channel recalibration | Stage 1 |
| `"CBAM"` | Channel + spatial attention in sequence | Stage 1 |
| `"HoVerNet"` | Spatial mask from HoVer-Net tumor nuclei detection | Stage 3 (stub) |
| `"MaskMitosis"` | Spatial mask from MaskMitosis mitotic figure detection | Stage 3 (stub) |
| `"DKSUNet"` | Spatial mask from DKS-DoubleU-Net tubular nuclei segmentation | Stage 3 (stub) |
| `"UNet"` | Spatial mask from U-Net epithelial nuclei segmentation | Stage 3 (stub) |

Exp1 and Exp2 always use `None` regardless of this setting.

---

## Outputs

| File | When generated |
|------|---------------|
| `training_{Exp}_{Model}.png` | Immediately after each model trains |
| `cm_{Exp}_{Model}.png` | Immediately after each model evaluates |
| `per_class_{Exp}_{Model}.png` | Immediately after each model evaluates |
| `model_comparison.png` | Updated after every single model |
| `breast_cancer_grading_diagnostic_report_YYYYMMDD.xlsx` | Updated after every single model |

The Excel workbook has 4 sheets:
- **Synthesis & Conclusion** — best model summary and clinical notes
- **Overall Performance** — accuracy, macro F1, weighted F1 per model
- **Confusion Matrices** — per-model grade confusion tables
- **Detailed Process** — per-grade F1 scores for all models

---

## Standalone Regeneration

If `outputs/*.json` files already exist from a previous run:
```bash
python main.py   # then select:
# Option 6 — Regenerate Excel report only
# Option 7 — Regenerate all charts only
```

---

## Stage 2 — Data Pipeline V2 (In Progress)

The original data pipeline (`data/loader.py`) uses a simple 70/15/15 random split with
on-the-fly augmentation. Stage 2 introduces improvements:

- **Group-aware split:** Patients are split as units — all images from the same patient
  go to the same split (train/val/test), preventing data leakage
- **Pre-augmentation only on train set:** No on-the-fly augmentation to avoid double
  augmenting images that were already pre-augmented by external tools
- **Val & test sets contain only original images** for unbiased evaluation
- **Configurable augmentation** via `config.py`:
  - Geometric: `AUG_HFLIP`, `AUG_VFLIP`, `AUG_ROTATION`, `AUG_TRANSLATION`
  - Photometric: `AUG_BRIGHTNESS`, `AUG_CONTRAST`, `AUG_SATURATION`, `AUG_HUE`

---

## Requirements

| Package | Minimum Version |
|---------|----------------|
| torch | 2.0.0 |
| torchvision | 0.15.0 |
| tqdm | 4.65.0 |
| scikit-learn | 1.2.0 |
| numpy | 1.24.0 |
| pandas | 2.0.0 |
| openpyxl | 3.1.0 |
| matplotlib | 3.7.0 |

GPU is recommended. CPU training is supported but slow — reduce `BATCH_SIZE` to 4 if needed.

---

## Stage 3 — Future Work

Stage 3 will replace the Stage 1 attention modules with spatial masks derived from
segmentation networks running on the same histopathology patches:

- **HoVer-Net** → tumor nuclei mask
- **MaskMitosis** → mitotic figure mask
- **DKS-DoubleU-Net** → tubular nuclei mask
- **U-Net** → epithelial nuclei mask

The `MaskGuidedAttention` stub in `models/attention.py` is already in place.
A `mask_loader` alongside the image loader will be added to `data/loader.py` when Stage 3 begins.