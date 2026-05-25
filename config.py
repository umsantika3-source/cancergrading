import torch

# ── Paths ──────────────────────────────────────────────────────────────────
OUTPUT_DIR   = "outputs"          # Override in Colab: "/content/drive/MyDrive/Outputs"
INPUT_DIR    = "D:/MyProject/Document/BuNurul/buSantika/kanker"                 # Override in Colab: "/content/drive/MyDrive/Dataset"

# ── Data ───────────────────────────────────────────────────────────────────
IMAGE_SIZE   = (224, 224)         # Exact size of the histopathology images in the dataset
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
