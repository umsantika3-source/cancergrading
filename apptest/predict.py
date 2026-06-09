"""
CancerGrading — Prediction Engine
Loads trained model checkpoints and runs inference on single/batch images.
All dependencies are self-contained within ./apptest/
"""
import math
import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ── Local model imports (copied from parent project) ────────────────────
from models.alexnet  import CustomAlexNet
from models.vgg16    import CustomVGG16
from models.vgg19    import CustomVGG19
from models.resnet50 import CustomResNet50
from models.fusion   import FusionModel

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["GRADE1", "GRADE2", "GRADE3"]
NUM_CLASSES = 3
IMAGE_SIZE  = (224, 224)

# ── Out-of-Distribution (OOD) Detection via Energy Score ─────────────────
# Extended labels for non-breast-cell or uncertain inputs
CLASS_NAMES_EXTENDED = CLASS_NAMES + ["UNKNOWN", "NOT_A_CELL"]

# Energy-based OOD detection:
#
#   Energy score = -log( sum( exp(logits_i) ) )
#
#   - In-distribution (breast cells): logits are large magnitude
#     → energy is very negative (e.g. < -5)
#   - Out-of-distribution (text / random): logits are near zero
#     → energy is close to 0 or even positive
#
#   ENERGY_THRESHOLD:
#     Energy >= threshold → NOT_A_CELL (outside training distribution)
#     Energy <  threshold → In-distribution candidate → check softmax confidence
#
#   SOFTMAX_CONFIDENCE_HIGH:
#     Max prob >= this → GRADE1/2/3 (confident prediction)
#     Max prob <  this → UNKNOWN (uncertain which grade)
#
#   Note on ENERGY_THRESHOLD:
#     - Breast cell logits are typically large magnitude (e.g. 5-20),
#       giving strongly negative energy (e.g. -5 to -20)
#     - OOD images produce near-zero logits, giving energy close to 0
#     - Start with -5.0, tune if cells are still rejected (make more
#       negative = e.g. -8.0) or if OOD images slip through (make less
#       negative = e.g. -3.0)
ENERGY_THRESHOLD         = -5.0   # tune this: more negative = stricter
SOFTMAX_CONFIDENCE_HIGH  = 0.85   # only used when energy says "in-distribution"

# Directory where .pth checkpoint files are stored
# Falls back to env var (set by run_app.py for PyInstaller builds) or default
_CHECKPOINT_DIR_ENV = os.environ.get("APP_CHECKPOINT_DIR")
if _CHECKPOINT_DIR_ENV:
    CHECKPOINT_DIR = _CHECKPOINT_DIR_ENV
else:
    CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────
# Image transforms (must match training preprocessing)
# ─────────────────────────────────────────────────────────────────────────

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

predict_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    normalize,
])

# ─────────────────────────────────────────────────────────────────────────
# Model registry — maps display name → (architecture builder, checkpoint_name)
# ─────────────────────────────────────────────────────────────────────────

def _build_alexnet(activation: str = "GELU", attention: Optional[str] = "SE",
                   pretrained: bool = False):
    return CustomAlexNet(NUM_CLASSES, activation=activation,
                         attention_type=attention, pretrained=pretrained)

def _build_vgg16(activation: str = "GELU", attention: Optional[str] = "SE",
                  pretrained: bool = False):
    return CustomVGG16(NUM_CLASSES, activation=activation,
                       attention_type=attention, pretrained=pretrained)

def _build_vgg19(activation: str = "GELU", attention: Optional[str] = "SE",
                  pretrained: bool = False):
    return CustomVGG19(NUM_CLASSES, activation=activation,
                       attention_type=attention, pretrained=pretrained)

def _build_resnet50(activation: str = "GELU", attention: Optional[str] = "SE",
                     pretrained: bool = False):
    return CustomResNet50(NUM_CLASSES, activation=activation,
                          attention_type=attention, pretrained=pretrained)

def _build_fusion(attention: Optional[str] = "SE"):
    return FusionModel(NUM_CLASSES, attention_type=attention)


MODEL_REGISTRY: Dict[str, Dict] = {
    # ── Best performers ──
    "Exp4_Fusion (Fusion+GELU+SE)": {
        "builder": lambda: _build_fusion(attention="SE"),
        "checkpoint": "Exp4_Fusion.pth",
        "cached_model": None,
    },
    "ExpVGG19_Fusion_GELU_SE (VGG19 Fusion+GELU+SE)": {
        "builder": lambda: _build_fusion(attention="SE"),
        "checkpoint": "ExpVGG19_Fusion_GELU_SE.pth",
        "cached_model": None,
    },
    # ── VGG19 variants ──
    "ExpVGG19_VGG19_GELU_SE (VGG19+GELU+SE)": {
        "builder": lambda: _build_vgg19(activation="GELU", attention="SE"),
        "checkpoint": "ExpVGG19_VGG19_GELU_SE.pth",
        "cached_model": None,
    },
    "ExpVGG19_VGG19_GELU (VGG19+GELU)": {
        "builder": lambda: _build_vgg19(activation="GELU", attention=None),
        "checkpoint": "ExpVGG19_VGG19_GELU.pth",
        "cached_model": None,
    },
    "ExpVGG19_VGG19_ReLU (VGG19+ReLU)": {
        "builder": lambda: _build_vgg19(activation="ReLU", attention=None),
        "checkpoint": "ExpVGG19_VGG19_ReLU.pth",
        "cached_model": None,
    },
    # ── Exp3: GELU + SE attention ──
    "Exp3_AlexNet (AlexNet+GELU+SE)": {
        "builder": lambda: _build_alexnet(activation="GELU", attention="SE"),
        "checkpoint": "Exp3_AlexNet.pth",
        "cached_model": None,
    },
    "Exp3_VGG16 (VGG16+GELU+SE)": {
        "builder": lambda: _build_vgg16(activation="GELU", attention="SE"),
        "checkpoint": "Exp3_VGG16.pth",
        "cached_model": None,
    },
    "Exp3_ResNet50 (ResNet50+GELU+SE)": {
        "builder": lambda: _build_resnet50(activation="GELU", attention="SE"),
        "checkpoint": "Exp3_ResNet50.pth",
        "cached_model": None,
    },
    # ── Exp2: GELU only ──
    "Exp2_AlexNet (AlexNet+GELU)": {
        "builder": lambda: _build_alexnet(activation="GELU", attention=None),
        "checkpoint": "Exp2_AlexNet.pth",
        "cached_model": None,
    },
    "Exp2_VGG16 (VGG16+GELU)": {
        "builder": lambda: _build_vgg16(activation="GELU", attention=None),
        "checkpoint": "Exp2_VGG16.pth",
        "cached_model": None,
    },
    "Exp2_ResNet50 (ResNet50+GELU)": {
        "builder": lambda: _build_resnet50(activation="GELU", attention=None),
        "checkpoint": "Exp2_ResNet50.pth",
        "cached_model": None,
    },
    # ── Exp1: Baseline ReLU ──
    "Exp1_AlexNet (AlexNet+ReLU)": {
        "builder": lambda: _build_alexnet(activation="ReLU", attention=None),
        "checkpoint": "Exp1_AlexNet.pth",
        "cached_model": None,
    },
    "Exp1_VGG16 (VGG16+ReLU)": {
        "builder": lambda: _build_vgg16(activation="ReLU", attention=None),
        "checkpoint": "Exp1_VGG16.pth",
        "cached_model": None,
    },
    "Exp1_ResNet50 (ResNet50+ReLU)": {
        "builder": lambda: _build_resnet50(activation="ReLU", attention=None),
        "checkpoint": "Exp1_ResNet50.pth",
        "cached_model": None,
    },
}


def find_checkpoint(checkpoint_name: str) -> Optional[str]:
    """Return full path to a checkpoint file, or None if not found."""
    path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    return path if os.path.isfile(path) else None


def get_available_models() -> List[str]:
    """Return list of model names whose checkpoints exist on disk."""
    available = []
    for name, cfg in MODEL_REGISTRY.items():
        if find_checkpoint(cfg["checkpoint"]):
            available.append(name)
    return available


def load_model(model_name: str) -> nn.Module:
    """
    Load a model by display name, return it in eval mode on the correct device.
    Uses a simple cache to avoid re-loading every call.
    """
    cfg = MODEL_REGISTRY.get(model_name)
    if cfg is None:
        raise ValueError(f"Unknown model: {model_name}")

    # Return cached model if available
    if cfg["cached_model"] is not None:
        return cfg["cached_model"]

    print(f"  [Loading] {model_name} ...")
    model = cfg["builder"]()
    ckpt_path = find_checkpoint(cfg["checkpoint"])
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Checkpoint '{cfg['checkpoint']}' not found in {CHECKPOINT_DIR}"
        )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    # Cache the model
    cfg["cached_model"] = model
    print(f"  [Loaded] {model_name} (device={DEVICE})")
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a normalized 4D tensor (1, C, H, W).
    Handles grayscale → RGB conversion.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = predict_transform(image)          # (C, H, W)
    return tensor.unsqueeze(0)                 # (1, C, H, W)


# ─────────────────────────────────────────────────────────────────────────
# OOD Detection via Energy Score
# ─────────────────────────────────────────────────────────────────────────

def compute_energy_score(logits: np.ndarray) -> float:
    """
    Compute the energy score from raw logits using proper logsumexp.

    Energy = -log( sum_i exp(logit_i) )

    Uses the numerically stable log-sum-exp:
        logsumexp = max(logits) + log( sum_i exp(logits_i - max(logits)) )

    Then returns -logsumexp, which preserves the true magnitude of the logits.

    Args:
        logits: 1D numpy array of raw logits (before softmax), length = num_classes

    Returns:
        A scalar energy score. Lower (more negative) = in-distribution.
        Higher (near 0 or positive) = out-of-distribution.

    Example:
        Breast cells:   logits=[12, 2, -1] → energy ≈ -12.2  → in-distribution
        OOD (text/cat): logits=[0.3, -0.1, 0.5] → energy ≈ -0.7  → NOT_A_CELL
    """
    max_logit = logits.max()
    shifted = logits - max_logit
    logsumexp = max_logit + math.log(np.sum(np.exp(shifted)))
    energy = -logsumexp
    return energy


def compute_normalized_entropy(probs: np.ndarray) -> float:
    """
    Compute normalized Shannon entropy of a probability distribution.

    Returns a value in [0, 1]:
        - 0.0 → all mass on a single class (perfect confidence)
        - 1.0 → uniform distribution (maximum uncertainty)
    """
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps))
    entropy /= np.log(len(probs))  # normalize to [0, 1]
    return float(entropy)


def classify_ood(
    logits: np.ndarray,
) -> Tuple[int, str, float, float]:
    """
    Determine whether the input is a known grade, UNKNOWN, or NOT_A_CELL
    using energy-based OOD detection as the primary filter.

    Decision logic:

        1. Compute energy score from raw logits.
        2. If energy >= ENERGY_THRESHOLD:
             → Out-of-distribution → NOT_A_CELL
        3. If energy < ENERGY_THRESHOLD:
             → In-distribution candidate → compute softmax confidence
             a. If max probability >= SOFTMAX_CONFIDENCE_HIGH:
                  → Confident → GRADE1/2/3
             b. Else:
                  → Uncertain → UNKNOWN

    Args:
        logits: 1D numpy array of raw logits (length = num_classes)

    Returns:
        (pred_idx_extended, pred_label_extended, max_probability, energy_score)
        where pred_idx_extended indexes into CLASS_NAMES_EXTENDED
    """
    # Compute energy
    energy = compute_energy_score(logits)
    max_prob = float(torch.softmax(torch.from_numpy(logits), dim=0).numpy().max())

    # ── Step 1: Energy-based OOD detection ─────────────────────────
    if energy >= ENERGY_THRESHOLD:
        # Out-of-distribution → NOT_A_CELL
        return 4, "NOT_A_CELL", max_prob, energy

    # ── Step 2: In-distribution → assess softmax confidence ─────────
    if max_prob >= SOFTMAX_CONFIDENCE_HIGH:
        idx = int(logits.argmax())
        return idx, CLASS_NAMES_EXTENDED[idx], max_prob, energy

    # In-distribution but uncertain which grade → UNKNOWN
    return 3, "UNKNOWN", max_prob, energy


# ─────────────────────────────────────────────────────────────────────────
# Prediction Functions
# ─────────────────────────────────────────────────────────────────────────

def predict(
    image: Image.Image,
    model_name: str,
    use_ood_detection: bool = True,
) -> Tuple[int, str, Dict[str, float], torch.Tensor]:
    """
    Run prediction on a single image, with energy-based OOD detection.

    Args:
        image: PIL Image
        model_name: key into MODEL_REGISTRY
        use_ood_detection: if True, applies energy-based OOD to detect
                           NOT_A_CELL / UNKNOWN inputs.
                           If False, legacy mode (always assigns a grade).

    Returns:
        (pred_class_idx, pred_class_label, confidence_dict, raw_probabilities)

        When use_ood_detection=True:
            - pred_class_idx may be 0-2 (grade), 3 (UNKNOWN), or 4 (NOT_A_CELL)
            - pred_class_label is the corresponding CLASS_NAMES_EXTENDED entry
            - confidence_dict has the 3 original grade keys plus "entropy" and "energy"

        When use_ood_detection=False (legacy mode):
            - pred_class_idx is 0-2
            - pred_class_label is "GRADE1"/"GRADE2"/"GRADE3"
            - confidence_dict has the 3 original grade keys
    """
    model = load_model(model_name)
    input_tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)              # (1, 3)
        probs  = torch.softmax(logits, dim=1)     # (1, 3)

    logits_np = logits.cpu().numpy().flatten()    # (3,)
    probs_np  = probs.cpu().numpy().flatten()     # (3,)

    if use_ood_detection:
        pred_idx_ext, pred_label, max_prob, energy = classify_ood(logits_np)
        pred_idx = pred_idx_ext

        entropy_val = compute_normalized_entropy(probs_np)
        confidence_dict = {
            cls: float(probs_np[i])
            for i, cls in enumerate(CLASS_NAMES)
        }
        confidence_dict["entropy"] = entropy_val
        confidence_dict["energy"]  = energy
    else:
        # Legacy mode: always assign a grade
        pred_idx = int(probs_np.argmax())
        pred_label = CLASS_NAMES[pred_idx]
        confidence_dict = {
            cls: float(probs_np[i])
            for i, cls in enumerate(CLASS_NAMES)
        }

    return pred_idx, pred_label, confidence_dict, probs.cpu()


def predict_batch(
    images: List[Image.Image],
    model_name: str,
    use_ood_detection: bool = True,
) -> List[Tuple[int, str, Dict[str, float], torch.Tensor]]:
    """
    Run prediction on a batch of images, with energy-based OOD detection.

    Args:
        images: List of PIL Images
        model_name: key into MODEL_REGISTRY
        use_ood_detection: if True, applies energy-based OOD detection

    Returns:
        List of (pred_class_idx, pred_class_label, confidence_dict, raw_probs)
    """
    model = load_model(model_name)
    batch = torch.cat([preprocess_image(img) for img in images], dim=0).to(DEVICE)

    with torch.no_grad():
        logits = model(batch)                     # (B, 3)
        probs  = torch.softmax(logits, dim=1)     # (B, 3)

    results = []
    for i in range(len(images)):
        l = logits[i].cpu().numpy()               # (3,)
        p = probs[i].cpu().numpy()                # (3,)

        if use_ood_detection:
            pred_idx_ext, pred_label, max_prob, energy = classify_ood(l)
            pred_idx = pred_idx_ext
            entropy_val = compute_normalized_entropy(p)
            confidence_dict = {
                cls: float(p[j])
                for j, cls in enumerate(CLASS_NAMES)
            }
            confidence_dict["entropy"] = entropy_val
            confidence_dict["energy"]  = energy
        else:
            pred_idx = int(p.argmax())
            pred_label = CLASS_NAMES[pred_idx]
            confidence_dict = {
                cls: float(p[j])
                for j, cls in enumerate(CLASS_NAMES)
            }

        results.append((pred_idx, pred_label, confidence_dict, probs[i].cpu()))

    return results