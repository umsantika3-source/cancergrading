# Breast Cancer Grading System

Deep learning experiments comparing CNN architectures for automated breast cancer histopathology grading (Grade I, II, III).

## Experiment Matrix

| # | Name | Activation | Attention | Models |
|---|------|-----------|-----------|--------|
| 1 | Baseline CNN | ReLU | — | AlexNet, VGG16, ResNet50 |
| 2 | CNN + GELU | GELU | — | AlexNet, VGG16, ResNet50 |
| 3 | CNN + SE-Block | ReLU | SE-Block | AlexNet, VGG16, ResNet50 |
| 4 | Feature Fusion | ReLU | — | AlexNet + VGG16 + ResNet50 (ensemble) |
| 5 | Proposed | GELU | CBAM | AlexNet, VGG16, ResNet50 |

## Requirements

- Python 3.8+
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

## Dataset Structure

```
dataset_root/
├── train/
│   ├── Grade I/
│   ├── Grade II/
│   └── Grade III/
├── val/
│   └── ...
└── test/
    └── ...
```

## Running Locally

```bash
python main.py
```

You will be prompted to enter the dataset path and select an experiment (1–5, or 6 for all).

**To force retrain** even if a checkpoint already exists, set in `config.py`:
```python
FORCE_RERUN = True
```

## Running on Google Colab

1. Open `colab_runner.ipynb` in Google Colab
2. Edit the two path cells to point to your Google Drive folders
3. Run all cells — training is automatically skipped for any model that already has a saved checkpoint

See [Colab Setup](#colab-setup) below for the full walkthrough.

## Output Files

All outputs are saved to `OUTPUT_DIR` (default: `outputs/`, configurable in `config.py`):

| File | Description |
|------|-------------|
| `results_YYYYMMDD_HHMMSS.xlsx` | Full results — Summary, Per-Class Metrics, Confusion Matrices |
| `Exp1_AlexNet.pth` | Saved model weights (checkpoint) |
| `Exp1_AlexNet_meta.json` | Train/val metrics paired with the checkpoint |
| `predictions_Exp1_AlexNet.json` | Per-image path + true label + predicted label |
| `cm_exp1_AlexNet.png` | Confusion matrix image |

## Colab Setup

### Step 1 — Prepare Google Drive

Create these two folders in your Drive:
- `MyDrive/SantikaDataset/` — upload your dataset here (train/val/test structure)
- `MyDrive/SantikaOutputs/` — checkpoints and results will be saved here

### Step 2 — Push this repo to GitHub

```bash
git init
git config --local user.name "YourGitHubUsername"
git config --local user.email "your@email.com"
git remote add origin https://github.com/YourAccount/SantikaCancer.git
git add .
git commit -m "Initial commit"
git push -u origin main
```

### Step 3 — Open colab_runner.ipynb

Upload `colab_runner.ipynb` to Colab or open directly from GitHub, then follow the instructions inside each cell.

## Project Structure

```
Santika/
├── config.py              # All hyperparameters and paths
├── main.py                # Interactive CLI runner (local use)
├── colab_runner.ipynb     # Google Colab notebook
├── requirements.txt
├── models/
│   ├── alexnet.py
│   ├── vgg16.py
│   ├── resnet50.py
│   ├── fusion.py
│   └── attention.py       # SEBlock, CBAM
├── experiments/
│   ├── exp1_baseline.py
│   ├── exp2_gelu.py
│   ├── exp3_attention.py
│   ├── exp4_fusion.py
│   └── exp5_proposed.py
└── utils/
    ├── data_loader.py
    ├── trainer.py
    ├── metrics.py
    ├── results_saver.py
    └── logger.py
```
