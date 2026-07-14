"""
Experiment 4 — Feature Fusion (AlexNet + VGG16 + ResNet50) + GELU + Attention.
Loads Exp3 checkpoints into sub-models and freezes them.
Only the fusion head trains.
Auto-triggers Exp3 (which auto-triggers Exp2) for any missing checkpoint.

Improvements over v1:
  - LayerNorm before head to normalise heterogeneous backbone features
  - Deeper 3-layer MLP head (10240→4096→2048→3) with moderate dropout 0.2
  - Class-weighted loss to combat Grade I under-prediction
  - Higher LR (1e-3) for faster fusion-head convergence
"""
import os
import torch
import numpy as np
import config
from utils.trainer import run_or_load
from utils.plotter  import (save_training_curve, save_cm_heatmap,
                             save_per_class_bar, save_comparison)

EXP_PREFIX = "Exp4"
ACTIVATION  = "GELU"
MODEL_NAME  = "Fusion"


def _compute_class_weights(train_loader, num_classes, logger):
    """Inverse-frequency class weights from training labels."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, labels in train_loader:
        for lbl in labels:
            counts[lbl] += 1.0
    # Guard against zero-division
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    logger.info(f"  [WEIGHTS] Per-class sample counts: {counts.tolist()}")
    logger.info(f"  [WEIGHTS] Computed class weights: {weights.tolist()}")
    return torch.tensor(weights, dtype=torch.float32, device=config.DEVICE)


def run(reporter, logger):
    from data.dataloader_v2 import get_data_loaders
    from models.fusion    import FusionModel

    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        config.INPUT_DIR, config.IMAGE_SIZE, config.BATCH_SIZE, config.RANDOM_SEED
    )

    logger.info(f"  [DEBUG] class_names = {class_names}")

    attn_type  = config.ATTENTION_TYPE
    attn_label = attn_type if attn_type else "None"

    # Auto-dependency: need all three Exp3 checkpoints
    for backbone in ("AlexNet", "VGG16", "ResNet50"):
        exp3_path = os.path.join(config.OUTPUT_DIR, f"Exp3_{backbone}.pth")
        if not os.path.exists(exp3_path):
            logger.info(f"  [DEPENDENCY] Exp3_{backbone}.pth not found — running Exp3 first...")
            from experiments.exp3_attention import run as run_exp3
            run_exp3(reporter, logger)
            break  # run_exp3 handles all three backbones at once

    # Compute class weights from training data to handle imbalance
    class_weights = _compute_class_weights(train_loader, config.NUM_CLASSES, logger)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Force re-run of Exp4 if old (buggy) checkpoint exists
    old_ckpt = os.path.join(config.OUTPUT_DIR, f"{EXP_PREFIX}_{MODEL_NAME}.pth")
    if os.path.exists(old_ckpt):
        logger.info(f"  [CLEANUP] Removing old {EXP_PREFIX}_{MODEL_NAME}.pth to force retrain")
        os.remove(old_ckpt)
    old_eval = os.path.join(config.OUTPUT_DIR, f"{EXP_PREFIX}_{MODEL_NAME}_eval.json")
    if os.path.exists(old_eval):
        os.remove(old_eval)
    old_meta = os.path.join(config.OUTPUT_DIR, f"{EXP_PREFIX}_{MODEL_NAME}_meta.json")
    if os.path.exists(old_meta):
        os.remove(old_meta)

    def model_fn():
        fusion = FusionModel(num_classes=config.NUM_CLASSES, attention_type=attn_type)

        # Load Exp3 weights into each sub-model
        for backbone, sub in (("AlexNet",  fusion.alexnet),
                               ("VGG16",   fusion.vgg16),
                               ("ResNet50",fusion.resnet)):
            ckpt = os.path.join(config.OUTPUT_DIR, f"Exp3_{backbone}.pth")
            if os.path.exists(ckpt):
                try:
                    sub.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))
                except Exception as e:
                    logger.warning(f"  [WARN] {ckpt} corrupt/unreadable ({e}) "
                                   f"— {backbone} backbone starts random")
            else:
                logger.warning(f"  [WARN] {ckpt} still missing — backbone starts random")

        fusion.freeze_backbones()
        return fusion

    logger.info(f"=== {EXP_PREFIX}: Fusion + GELU + {attn_label} (LayerNorm + deeper head + class weights) ===")

    eval_d, preds, cm, per_class, ta_h, va_h, tl_h, vl_h = run_or_load(
        model_fn, train_loader, val_loader, test_loader, class_names,
        EXP_PREFIX, MODEL_NAME, logger,
        activation=ACTIVATION, attention=attn_label,
        criterion=criterion,
        lr=config.HEAD_LR,
    )

    save_training_curve(EXP_PREFIX, MODEL_NAME, ta_h, va_h, tl_h, vl_h)
    reporter.update(EXP_PREFIX, MODEL_NAME, eval_d, preds)
    save_cm_heatmap(EXP_PREFIX, MODEL_NAME, cm)
    save_per_class_bar(EXP_PREFIX, MODEL_NAME, per_class)
    save_comparison(config.OUTPUT_DIR)
    logger.info(f"  [REPORT] Charts and Excel updated for {MODEL_NAME}")
