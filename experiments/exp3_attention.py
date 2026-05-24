"""
Experiment 3 — CNN + GELU + Attention.
Backbone warm-started from Exp2 checkpoints (strict=False).
Auto-triggers Exp2 if its checkpoint is missing.
"""
import os
import config
from utils.trainer import run_or_load
from utils.plotter  import (save_training_curve, save_cm_heatmap,
                             save_per_class_bar, save_comparison)

EXP_PREFIX = "Exp3"
ACTIVATION  = "GELU"


def run(reporter, logger):
    from data.loader      import get_loaders
    from models.alexnet   import CustomAlexNet
    from models.vgg16     import CustomVGG16
    from models.resnet50  import CustomResNet50

    train_loader, val_loader, test_loader, class_names = get_loaders(
        config.INPUT_DIR, config.IMAGE_SIZE, config.BATCH_SIZE, config.RANDOM_SEED
    )

    attn_type   = config.ATTENTION_TYPE
    attn_label  = attn_type if attn_type else "None"

    models_cfg = [
        ("AlexNet",  lambda: CustomAlexNet( config.NUM_CLASSES, activation=ACTIVATION,
                                            attention_type=attn_type,
                                            pretrained=config.USE_PRETRAINED)),
        ("VGG16",    lambda: CustomVGG16(   config.NUM_CLASSES, activation=ACTIVATION,
                                            attention_type=attn_type,
                                            pretrained=config.USE_PRETRAINED)),
        ("ResNet50", lambda: CustomResNet50(config.NUM_CLASSES, activation=ACTIVATION,
                                            attention_type=attn_type,
                                            pretrained=config.USE_PRETRAINED)),
    ]

    logger.info(f"=== {EXP_PREFIX}: CNN + GELU + {attn_label} ===")

    for name, model_fn in models_cfg:
        # Auto-dependency: need Exp2_<name>.pth to warm-start backbone
        exp2_path = os.path.join(config.OUTPUT_DIR, f"Exp2_{name}.pth")
        if not os.path.exists(exp2_path):
            logger.info(f"  [DEPENDENCY] Exp2_{name}.pth not found — running Exp2 first...")
            from experiments.exp2_gelu import run as run_exp2
            run_exp2(reporter, logger)

        eval_d, preds, cm, per_class, ta_h, va_h, tl_h, vl_h = run_or_load(
            model_fn, train_loader, val_loader, test_loader, class_names,
            EXP_PREFIX, name, logger,
            activation=ACTIVATION, attention=attn_label,
            pretrain_path=exp2_path,
        )

        save_training_curve(EXP_PREFIX, name, ta_h, va_h, tl_h, vl_h)
        reporter.update(EXP_PREFIX, name, eval_d, preds)
        save_cm_heatmap(EXP_PREFIX, name, cm)
        save_per_class_bar(EXP_PREFIX, name, per_class)
        save_comparison(config.OUTPUT_DIR)
        logger.info(f"  [REPORT] Charts and Excel updated for {name}")
