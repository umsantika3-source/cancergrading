"""
Experiment 4 — Feature Fusion (AlexNet + VGG16 + ResNet50) + GELU + Attention.
Loads Exp3 checkpoints into sub-models and freezes them.
Only the fusion head trains.
Auto-triggers Exp3 (which auto-triggers Exp2) for any missing checkpoint.
"""
import os
import torch
import config
from utils.trainer import run_or_load
from utils.plotter  import (save_training_curve, save_cm_heatmap,
                             save_per_class_bar, save_comparison)

EXP_PREFIX = "Exp4"
ACTIVATION  = "GELU"
MODEL_NAME  = "Fusion"


def run(reporter, logger):
    from data.loader      import get_loaders
    from models.fusion    import FusionModel

    train_loader, val_loader, test_loader, class_names = get_loaders(
        config.INPUT_DIR, config.IMAGE_SIZE, config.BATCH_SIZE, config.RANDOM_SEED
    )

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

    def model_fn():
        fusion = FusionModel(num_classes=config.NUM_CLASSES, attention_type=attn_type)

        # Load Exp3 weights into each sub-model
        for backbone, sub in (("AlexNet",  fusion.alexnet),
                               ("VGG16",   fusion.vgg16),
                               ("ResNet50",fusion.resnet)):
            ckpt = os.path.join(config.OUTPUT_DIR, f"Exp3_{backbone}.pth")
            if os.path.exists(ckpt):
                sub.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))
            else:
                logger.warning(f"  [WARN] {ckpt} still missing — backbone starts random")

        fusion.freeze_backbones()
        return fusion

    logger.info(f"=== {EXP_PREFIX}: Fusion + GELU + {attn_label} ===")

    eval_d, preds, cm, per_class, ta_h, va_h, tl_h, vl_h = run_or_load(
        model_fn, train_loader, val_loader, test_loader, class_names,
        EXP_PREFIX, MODEL_NAME, logger,
        activation=ACTIVATION, attention=attn_label,
    )

    save_training_curve(EXP_PREFIX, MODEL_NAME, ta_h, va_h, tl_h, vl_h)
    reporter.update(EXP_PREFIX, MODEL_NAME, eval_d, preds)
    save_cm_heatmap(EXP_PREFIX, MODEL_NAME, cm)
    save_per_class_bar(EXP_PREFIX, MODEL_NAME, per_class)
    save_comparison(config.OUTPUT_DIR)
    logger.info(f"  [REPORT] Charts and Excel updated for {MODEL_NAME}")
