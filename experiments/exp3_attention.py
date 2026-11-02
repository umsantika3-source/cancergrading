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


def run(reporter, logger, data_loaders=None, fold_tag=""):
    tagged_prefix = f"{EXP_PREFIX}{fold_tag}"
    from models.alexnet   import CustomAlexNet
    from models.vgg16     import CustomVGG16
    from models.resnet50  import CustomResNet50

    if data_loaders is not None:
        train_loader, val_loader, test_loader, class_names = data_loaders
    else:
        from data.dataloader_v2 import get_data_loaders
        # train_loader, val_loader, test_loader, class_names = get_loaders(
        #     config.INPUT_DIR, config.IMAGE_SIZE, config.BATCH_SIZE, config.RANDOM_SEED
        # )
        train_loader, val_loader, test_loader, class_names = get_data_loaders(
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
            run_exp2(reporter, logger, data_loaders=data_loaders, fold_tag=fold_tag)

        eval_d, preds, cm, per_class, ta_h, va_h, tl_h, vl_h = run_or_load(
            model_fn, train_loader, val_loader, test_loader, class_names,
            tagged_prefix, name, logger,
            activation=ACTIVATION, attention=attn_label,
            pretrain_path=exp2_path,
        )

        save_training_curve(tagged_prefix, name, ta_h, va_h, tl_h, vl_h)
        reporter.update(tagged_prefix, name, eval_d, preds)
        save_cm_heatmap(tagged_prefix, name, cm)
        save_per_class_bar(tagged_prefix, name, per_class)
        save_comparison(config.OUTPUT_DIR)
        logger.info(f"  [REPORT] Charts and Excel updated for {name}")
