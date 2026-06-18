import torch

# ── Paths ──────────────────────────────────────────────────────────────────
OUTPUT_DIR   = "kfoldoutput"          # Override in Colab: "/content/drive/MyDrive/Outputs"
INPUT_DIR    = "kfoldset"            # Override in Colab: "/content/drive/MyDrive/Dataset"
INPUT_DIR_V2 = "kfoldset_v2"         # V2: group-aware split + pre-augmented (future)

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

# ── Augmentation (geometric) ───────────────────────────────────────────────
# Applied on-the-fly in data/dataloader.py (current) or pre-augmented in org_kfold_v2.py (future).
AUG_HFLIP       = True       # RandomHorizontalFlip
AUG_VFLIP       = True       # RandomVerticalFlip
AUG_ROTATION    = 15         # ±15° rotation. Options: 15, 30, 90 (Future: [−30, 30])
AUG_RANDOM_CROP = False      # Future: RandomResizedCrop(image_size, scale=(0.8, 1.0))
AUG_TRANSLATION = 0.05       # Translation/shift fraction for RandomAffine (Future: increase to 0.1-0.15)

# ── Augmentation (photometric / color) ─────────────────────────────────────
AUG_BRIGHTNESS = 0.2         # ColorJitter brightness factor
AUG_CONTRAST   = 0.2         # ColorJitter contrast factor
AUG_SATURATION = 0.1         # ColorJitter saturation factor
AUG_HUE        = 0.05        # ColorJitter hue factor

# ── Augmentation (stain — histopathology specific) ─────────────────────────
# FUTURE FEATURE — requires staintools / torchstain / torchvision extra dependencies
# AUG_STAIN        = False      # Enable H&E stain jitter / HED augmentation
# AUG_STAIN_METHOD = "HED"      # Options: "HED", "Macenko", "Vahadane"
# AUG_STAIN_SIGMA  = 0.2        # Standard deviation for stain perturbation

# ── Evaluation metrics ─────────────────────────────────────────────────────
USE_BALANCED_ACC = True       # Compute balanced_accuracy_score (better for imbalanced data)

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