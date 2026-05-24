# CancerGrading — New Project Planning

**Goal:** Build a deep learning system that helps doctors automatically grade breast cancer
histopathology images (Grade I / II / III) through a clean, reproducible 4-experiment pipeline
with doctor-ready report and chart outputs generated automatically as each experiment finishes.

**Reference architecture (uploaded diagrams):**
- (a) WSI → Tile extraction + annotation (doctor-labelled patches)
- (b) Multi-ROI segmentation: tumor nuclei (HoVer-Net), mitotic nuclei (MaskMitosis),
  tubular nuclei (DKS-DoubleU-Net), epithelial nuclei (U-Net) — *Phase 2 attention source*
- (c) Deep feature extraction via multi-stream CNNs → KDC fusion → grade output
- (d) Handcrafted features: Cell Architecture, Cell Morphology, COrE, CCG, Haralick, QR — *Phase 2*
- (e) Fused feature vector → FC → Grade I / II / III prediction

**Phase 1 (this project):** CNN-based grading using 4 experiments; attention = SE-Block or CBAM.
**Phase 2 (future):** Attention maps derived from segmentation masks (HoVer-Net, U-Net etc.) as
  shown in diagram (b) — the segmentation output feeds spatial masks into the attention module.

---

## Project Structure

```
CancerGrading/
├── config.py                   # Central config — all toggles live here
├── main.py                     # Entry point — numbered menu (1-7)
│                               #   1. Experiment 1 - Baseline CNN (ReLU)
│                               #   2. Experiment 2 - CNN + GELU
│                               #   3. Experiment 3 - CNN + GELU + Attention
│                               #   4. Experiment 4 - Feature Fusion (3 CNNs) + GELU + Attention
│                               #   5. Experiment 5 - Run All Exp1-Exp4
│                               #   6. Experiment 6 - Regenerate Excel report only
│                               #   7. Experiment 7 - Regenerate all charts only
│
├── data/
│   └── loader.py               # DataLoader factory (train/val/test split + augmentation)
│
├── models/
│   ├── attention.py            # SEBlock + CBAM (Phase 1); mask-guided stubs (Phase 2)
│   ├── alexnet.py              # CustomAlexNet (pretrained, configurable act + attn)
│   ├── vgg16.py                # CustomVGG16  (pretrained, configurable act + attn)
│   ├── resnet50.py             # CustomResNet50 (pretrained, configurable act + attn)
│   └── fusion.py               # FusionModel (AlexNet + VGG16 + ResNet50 concat head)
│
├── utils/
│   ├── trainer.py              # Training loop, early stopping, checkpointing
│   │                           # Returns: train_acc_history[], val_acc_history[]
│   ├── results_saver.py        # Collects metrics across all experiments (in-memory)
│   ├── reporter.py             # ← generate_clinical_report logic moved here as callable util
│   │                           #   DiagnosticCompiler class, called after each model finishes
│   ├── plotter.py              # ← graphic.py logic moved here as callable util
│   │                           #   save_training_curve(), save_comparison(), save_cm_heatmap()
│   │                           #   save_per_class_bar() — called after each model finishes
│   └── logger.py               # Console + file logging
│
├── experiments/
│   ├── exp1_baseline.py        # CNN Classic (ReLU, no attention)
│   ├── exp2_gelu.py            # CNN + GELU
│   ├── exp3_attention.py       # CNN + GELU + Attention (SE or CBAM via config)
│   └── exp4_fusion.py          # Fusion CNN + GELU + Attention
│
└── outputs/
    ├── *_eval.json                   # Per-model full metric cache (flat in outputs/)
    ├── *_meta.json                   # Per-model training metadata cache
    ├── *.pth                         # Per-model checkpoint weights
    ├── predictions_*.json            # Per-model per-image prediction records
    └── reports/                      # Auto-created on first report/chart call
        ├── breast_cancer_grading_diagnostic_report_YYYYMMDD.xlsx
        ├── training_Exp1_AlexNet.png
        ├── training_Exp1_VGG16.png
        ├── ...                       # One chart per model, saved immediately after it trains
        ├── model_comparison.png      # Updated after every model finishes
        └── confusion_matrix_*.png
```

---

## config.py — All Toggles

```python
import torch

# ── Paths ──────────────────────────────────────────────────────────────────
OUTPUT_DIR   = "outputs"          # Override in Colab: "/content/drive/MyDrive/Outputs"
INPUT_DIR    = ""                 # Override in Colab: "/content/drive/MyDrive/Dataset"

# ── Data ───────────────────────────────────────────────────────────────────
IMAGE_SIZE   = (479, 640)         # Exact size of the histopathology images in the dataset
NUM_CLASSES  = 3                  # Grade I, Grade II, Grade III
BATCH_SIZE   = 8                  # Reduce to 4 if OOM on CPU
TRAIN_SPLIT  = 0.7
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.15
RANDOM_SEED  = 42

# ── Training ───────────────────────────────────────────────────────────────
EPOCHS        = 50
PATIENCE      = 5
LEARNING_RATE = 1e-4

# ── Model flags ────────────────────────────────────────────────────────────
USE_PRETRAINED = True             # ImageNet pretrained weights

# ATTENTION_TYPE — controls Exp3 and Exp4 only (Exp1 and Exp2 always use None)
#
# Phase 1 options (implemented):
#   None          — no attention (Exp1, Exp2 baseline)
#   "SE"          — Squeeze-and-Excitation channel recalibration (default)
#   "CBAM"        — Convolutional Block Attention Module (channel + spatial)
#
# Phase 2 options (future — attention map sourced from segmentation masks):
#   "HoVerNet"    — spatial mask from HoVer-Net tumor nuclei detection
#   "MaskMitosis" — spatial mask from MaskMitosis mitotic figure detection
#   "DKSUNet"     — spatial mask from DKS-DoubleU-Net tubular nuclei segmentation
#   "UNet"        — spatial mask from U-Net epithelial nuclei segmentation
#
ATTENTION_TYPE = "SE"

# FORCE_RERUN — True = ignore all caches and retrain everything from scratch
FORCE_RERUN  = False

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Key design:** `ATTENTION_TYPE` is the single toggle. Switching from SE to CBAM (or to a Phase 2
mask-guided type) requires changing exactly one line. Exp1 and Exp2 always pass `attention=None`
regardless of what `ATTENTION_TYPE` is set to.

---

## Experiment Matrix

| Exp | Name | Activation | Attention | Weight Init | Models |
|-----|------|-----------|-----------|-------------|--------|
| 1 | CNN Classic | ReLU | None | ImageNet | AlexNet, VGG16, ResNet50 |
| 2 | CNN + GELU | GELU | None | ImageNet | AlexNet, VGG16, ResNet50 |
| 3 | CNN + GELU + Attention | GELU | `ATTENTION_TYPE` from config | **Exp2 checkpoint** (backbone) + random (attention) | AlexNet, VGG16, ResNet50 |
| 4 | Fusion CNN + GELU + Attention | GELU | `ATTENTION_TYPE` from config | **Exp3 checkpoint** (frozen) | AlexNet+VGG16+ResNet50 fused |

**Weight init chain:**
```
ImageNet → Exp1 (independent, ReLU baseline)
ImageNet → Exp2 (independent, GELU — fair comparison against Exp1)
Exp2.pth → Exp3 (backbone reused, only attention block trains from scratch)
Exp3.pth → Exp4 (all three backbones frozen, only fusion head trains)
```
Exp1 and Exp2 start independently from ImageNet so their comparison is scientifically clean.
Exp3 reusing Exp2 is valid because the backbone architecture is identical — only the attention
module is new. This avoids retraining the expensive VGG16 and ResNet50 backbones from scratch.

---

## Experiment Details

### Exp1 — CNN Classic

- Models: AlexNet, VGG16, ResNet50 (each trained separately)
- Activation: ReLU (torchvision default)
- Attention: None
- After each model finishes: `plotter.save_training_curve()` + `reporter.update()` called automatically
- Output files per model:
  - `Exp1_AlexNet_eval.json`, `Exp1_AlexNet_meta.json`, `Exp1_AlexNet.pth`
  - `predictions_Exp1_AlexNet.json`
  - `reports/training_Exp1_AlexNet.png`

### Exp2 — CNN + GELU

- Same as Exp1 but ReLU replaced with GELU in all conv and FC layers
- After each model: same automatic reporter + plotter calls

### Exp3 — CNN + GELU + Attention

- Adds attention module; type read from `config.ATTENTION_TYPE`
- Placement:
  - AlexNet / VGG16: after `features` block, before avgpool
  - ResNet50: after each of layer1, layer2, layer3, layer4
- **Weight loading:** loads `Exp2_<ModelName>.pth` for backbone + classifier weights;
  attention block initializes randomly and is the only part that trains from scratch.
- **Auto-dependency:** if `Exp2_<ModelName>.pth` is missing, Exp3 automatically triggers
  Exp2 for that model first, then continues. No manual intervention needed.
- **Why:** VGG16 and ResNet50 are expensive — backbone is already converged from Exp2,
  only the small attention module needs to learn. Saves the majority of training time.
- Phase 2 note: When `ATTENTION_TYPE` is `"HoVerNet"` etc., the model receives a segmentation mask
  (H × W binary map) and uses it as a spatial weight multiplied onto the feature map — this is the
  mechanism shown in diagram (b)+(c) where ROI maps feed into the KDC fusion path

### Exp4 — Fusion CNN + GELU + Attention

- Penultimate features from all three Exp3 models concatenated: AlexNet(4096) + VGG16(4096) + ResNet50(2048) = 10,240
- Fusion head: `Linear(10240, 1024) → GELU → Dropout(0.5) → Linear(1024, 3)`
- Base model weights loaded from Exp3 checkpoints (all three frozen during fusion training)
- **Auto-dependency:** if any `Exp3_<ModelName>.pth` is missing, Exp4 triggers Exp3 for
  that model first — which in turn triggers Exp2 if needed. Full chain resolves automatically.
- Single model, no loop — runs once, generates one eval.json

---

## Three-Tier Caching (carried over from Santika)

```
Tier 1 — Full cache:   *_eval.json exists          → load metrics instantly, skip model entirely
Tier 2 — Checkpoint:   *.pth + *_meta.json exists  → skip training, run evaluate() only
Tier 3 — No cache:     nothing exists               → full train + evaluate
```

Set `FORCE_RERUN = True` in config to bypass all tiers.

**Auto-dependency resolution — full chain:**

```
User runs any experiment → system resolves dependencies automatically before training

Exp1 selected:
  └─ no dependencies → train from ImageNet directly

Exp2 selected:
  └─ no dependencies → train from ImageNet directly

Exp3 selected (per model: AlexNet, VGG16, ResNet50):
  └─ needs Exp2_<Model>.pth
        Found     → load backbone + classifier, random-init attention, train
        Not found → auto-run Exp2 for this model first, then continue Exp3

Exp4 selected:
  └─ needs Exp3_AlexNet.pth + Exp3_VGG16.pth + Exp3_ResNet50.pth
        Each found     → load frozen into fusion model
        Any not found  → auto-run Exp3 for missing model
                              └─ Exp3 auto-run needs Exp2_<Model>.pth
                                    Not found → auto-run Exp2 for that model first
```

Code pattern inside each experiment's `run()` function:
```python
# exp3_attention.py — before training each model
for name, model_fn in models:
    exp2_path = os.path.join(OUTPUT_DIR, f"Exp2_{name}.pth")
    if not os.path.exists(exp2_path):
        print(f"  [DEPENDENCY] Exp2_{name}.pth not found — running Exp2 first...")
        from experiments.exp2_gelu import run as run_exp2
        run_exp2(reporter)   # saves Exp2_<name>.pth as a side effect

    model = model_fn()
    model.load_state_dict(torch.load(exp2_path), strict=False)  # strict=False skips missing attention keys
    # ... train Exp3 normally
```

`strict=False` is the key — it loads all matching keys (backbone + classifier) and silently
skips the new attention block keys, which remain randomly initialized and learn during Exp3 training.

Option 5 (Run All) still executes Exp1 → Exp2 → Exp3 → Exp4 in order, so dependency
resolution never triggers when running the full pipeline — it only activates when a
single experiment is selected in isolation.

---

## utils/reporter.py — Clinical Report as a Utility

The `DiagnosticCompiler` class from `outputs/generate_clinical_report.py` moves here.
Two key changes make it a live utility rather than a post-hoc script:

### `update(exp, model_name, eval_data, pred_records)` — called after each model
```python
class DiagnosticCompiler:
    def update(self, exp: str, model_name: str,
               eval_data: dict, pred_records: list) -> None:
        """
        Appends one model's results to internal state.
        Called immediately after each model's evaluate() returns.
        Writes a fresh Excel file each time so the report is always current.
        """
        self._ingest_eval(exp, model_name, eval_data)
        self._ingest_predictions(exp, model_name, pred_records)
        self.build_excel()   # Overwrites previous version — always up to date
```

### `build_excel()` — unchanged from current implementation
Same 4-sheet structure: Synthesis & Conclusion, Overall Performance, Confusion Matrices,
Detailed Process. Saves to `outputs/reports/breast_cancer_grading_diagnostic_report_YYYYMMDD.xlsx`.

### New field: `activation` and `attention` stored in eval.json
Each experiment writes these into its eval cache so the reporter reads them dynamically:
```python
# Inside each experiment, add to the eval.json dump:
{
    ...existing fields...,
    "activation": "GELU",
    "attention": config.ATTENTION_TYPE or "None"
}

# Inside reporter.py, replace hardcoded logic with:
activation = eval_data.get("activation", "ReLU")
attention  = eval_data.get("attention", "None")
```

### Standalone regeneration — via `main.py` menu option 6
Since there is no standalone script in `outputs/`, manual regeneration goes through `main.py`:
```
6. Experiment 6 - Regenerate Excel report only  → DiagnosticCompiler.load_data() + build_excel()
7. Experiment 7 - Regenerate all charts only    → reads all *_eval.json, calls plotter functions
```
This keeps all Python files inside the project root and `utils/` — nothing lives in `outputs/`.

---

## utils/plotter.py — Charts as a Utility

Functions from `outputs/graphic.py` move here. All functions accept real data as arguments
(no more hardcoded values) and auto-save to `outputs/reports/`.

### `save_training_curve(exp, model_name, train_acc, val_acc, train_loss, val_loss)`
Called immediately after `trainer.train()` returns history arrays.
```python
# Saves: outputs/reports/training_{exp}_{model_name}.png
# Shows: accuracy curve (left) + loss curve (right) — same layout as current graphic.py
```

### `save_cm_heatmap(exp, model_name, cm)`
Called immediately after `trainer.evaluate()` returns.
```python
# cm is list-of-lists from eval.json (np.array(cm) inside function)
# Saves: outputs/reports/cm_{exp}_{model_name}.png
# Custom pink→purple colormap (same as current graphic.py)
```

### `save_per_class_bar(exp, model_name, per_class_metrics)`
New function — called after evaluate() returns `per_class_metrics`.
```python
# Shows: grouped bar chart of Precision / Recall / F1 for Grade I, II, III
# Saves: outputs/reports/per_class_{exp}_{model_name}.png
# Doctor view: at a glance, which grade is hardest to detect?
```

### `save_comparison(output_dir)` — called after every model finishes
```python
# Reads all *_eval.json in output_dir, re-plots the full comparison bar chart
# Updates: outputs/reports/model_comparison.png
# So after Exp1/AlexNet trains, comparison chart shows 1 bar.
# After all 10 models finish, it shows all 10 bars side by side.
```

Standalone chart regeneration is also handled through `main.py` menu option 8 (above).
No Python files are placed inside `outputs/` — that folder is data-only (json, pth, png, xlsx).

---

## How Each Experiment Calls the Utils

Pattern inside every `experiments/expN_*.py`, after evaluate() returns:

```python
from utils.reporter import DiagnosticCompiler
from utils.plotter  import save_training_curve, save_cm_heatmap, save_per_class_bar, save_comparison

# --- after trainer.train() ---
save_training_curve(EXP_PREFIX, name,
    train_acc_history, val_acc_history,
    train_loss_history, val_loss_history)

# --- after trainer.evaluate() + metrics computed ---
reporter.update(EXP_PREFIX, name, eval_dict, pred_records)
save_cm_heatmap(EXP_PREFIX, name, cm)
save_per_class_bar(EXP_PREFIX, name, per_class_metrics)
save_comparison(OUTPUT_DIR)   # refreshes the comparison chart with what's done so far

print(f"  [REPORT] Charts and Excel updated for {name}")
```

`reporter` is a single `DiagnosticCompiler` instance created in `main.py` and passed into
each experiment's `run(reporter)` function — same as the current `ResultsSaver` pattern.

---

## Run All — Progressive Update Flow

When user selects option 5 (Run All Exp1-Exp4) in `main.py`:

```
Exp1 AlexNet  trains → evaluate → reporter.update() → charts saved → Excel updated
Exp1 VGG16    trains → evaluate → reporter.update() → charts saved → Excel updated
Exp1 ResNet50 trains → evaluate → reporter.update() → charts saved → Excel updated
Exp2 AlexNet  trains → ...
...
Exp4 Fusion   trains → evaluate → reporter.update() → charts saved → Excel finalized
```

At any point the user can open `outputs/reports/breast_cancer_grading_diagnostic_report_*.xlsx`
and see results for all models that have finished so far. If training crashes mid-run, the
Excel already contains all previously-completed models — nothing is lost.

---

## data/loader.py — Key Design Decisions

```python
def get_loaders(data_root, image_size, batch_size, seed):
    """
    Returns (train_loader, val_loader, test_loader, class_names)
    Dataset folder must follow ImageFolder structure:
        data_root/
            Grade1/  *.bmp *.png *.jpg
            Grade2/
            Grade3/
    """
    train_transform = transforms.Compose([
        transforms.Resize(image_size),         # (479, 640) — exact image dimensions
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Split indices with fixed seed, apply val_transform to val and test subsets
```

**IMAGE_SIZE = (479, 640):** This matches the actual pixel dimensions of the histopathology
images in the dataset — no resize distortion. Resize is still called in case any image
is slightly off-spec.

**Normalization:** ImageNet stats `[0.485, 0.456, 0.406]` — required for pretrained weights.
The old `modelsold/` used `[0.5, 0.5, 0.5]` which is why it trained from scratch.

---

## Attention Module — Phase 1 vs Phase 2 Design

### Phase 1 (SE and CBAM — implement now)

`models/attention.py`:
```python
class SEBlock(nn.Module):
    """Channel recalibration: squeeze → excitation → scale."""

class CBAM(nn.Module):
    """Channel attention + spatial attention in sequence."""
```

Both take a feature map tensor `(B, C, H, W)` and return the same shape.

### Phase 2 (mask-guided spatial attention — stub now, implement later)

When `ATTENTION_TYPE = "HoVerNet"` etc., the attention module receives **two** inputs:
- Feature map `(B, C, H, W)` from the CNN backbone
- Segmentation mask `(B, 1, H, W)` from the corresponding detector (resized to match)

```python
class MaskGuidedAttention(nn.Module):
    """
    Uses binary segmentation mask as a spatial prior.
    Upsamples mask to feature map resolution, multiplies channel-wise.
    Phase 2 only — requires running HoVer-Net / U-Net first to produce masks.
    """
    def forward(self, features, mask):
        mask_resized = F.interpolate(mask, features.shape[2:])
        return features * mask_resized + features  # residual so gradient flows
```

The trainer will need a `mask_loader` alongside the image loader for Phase 2.
Document this here so the data pipeline can be planned correctly when Phase 2 starts.

---

## Doctor-Facing Output Summary

| Output | File | When generated |
|--------|------|---------------|
| Training curve | `reports/training_{Exp}_{Model}.png` | Immediately after each model trains |
| CM heatmap | `reports/cm_{Exp}_{Model}.png` | Immediately after each model evaluates |
| Per-class bar | `reports/per_class_{Exp}_{Model}.png` | Immediately after each model evaluates |
| Comparison bar | `reports/model_comparison.png` | Updated after every single model |
| Excel workbook | `reports/breast_cancer_grading_diagnostic_report_YYYYMMDD.xlsx` | Updated after every single model |

**Clinical significance of each chart:**
- Training curve — doctor sees if the model actually learned or just memorized
- CM heatmap — which grade gets confused with which (Grade III→Grade I = most dangerous error)
- Per-class bar — which grade is hardest; informs where more data is needed
- Comparison bar — justifies the final model choice to a clinical committee

---

## Build Order (new machine)

```
1. pip install torch torchvision tqdm scikit-learn pandas openpyxl matplotlib
2. Dataset → data_root/Grade1/*.bmp, data_root/Grade2/*.bmp, data_root/Grade3/*.bmp
3. Set INPUT_DIR in config.py  (or leave blank to be prompted)
4. python main.py              → numbered menu appears
5. Option 5 → Run All Exp1-Exp4  → trains all experiments, reports update after each model
6. outputs/reports/           → contains Excel + all charts when done
```

Standalone regeneration (if `outputs/*.json` already exist from a previous run):
```
python main.py → Option 6   # Regenerate Excel report only (reads existing eval.json)
python main.py → Option 7   # Regenerate all charts only  (reads existing eval.json)
```

---

## Differences from Current Project (Santika)

| Aspect | Santika | CancerGrading |
|--------|---------|---------------|
| Image size | 479×640 | **479×640 (same)** |
| Report generation | Manual — run separately after all experiments | **Automatic — updates after each model** |
| Graphic generation | Manual standalone script with hardcoded data | **Automatic — live data from eval.json** |
| Attention toggle | Hardcoded per experiment file | **Single `ATTENTION_TYPE` in config** |
| Attention options | SE, CBAM | **SE, CBAM + Phase 2 mask-guided stubs** |
| `activation`/`attention` in eval.json | No | **Yes — reporter reads them dynamically** |
| reporter/plotter location | `outputs/` standalone scripts | **`utils/` only — no Python in `outputs/`** |
| Dataset split | Single folder shuffle | **70/15/15 deterministic split** |

---

*Plan created: 2026-05-21 | Status: Ready to build*
