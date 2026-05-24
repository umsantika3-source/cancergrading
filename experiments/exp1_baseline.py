"""Experiment 1 — Baseline CNN (ReLU, no attention, ImageNet init)."""
import os
import config
from utils.trainer import run_or_load
from utils.plotter  import (save_training_curve, save_cm_heatmap,
                             save_per_class_bar, save_comparison)

EXP_PREFIX = "Exp1"
ACTIVATION  = "ReLU"
ATTENTION   = "None"


def run(reporter, logger):
    from data.loader      import get_loaders
    from models.alexnet   import CustomAlexNet
    from models.vgg16     import CustomVGG16
    from models.resnet50  import CustomResNet50

    train_loader, val_loader, test_loader, class_names = get_loaders(
        config.INPUT_DIR, config.IMAGE_SIZE, config.BATCH_SIZE, config.RANDOM_SEED
    )

    models_cfg = [
        ("AlexNet",  lambda: CustomAlexNet( config.NUM_CLASSES, activation=ACTIVATION,
                                            attention_type=None,
                                            pretrained=config.USE_PRETRAINED)),
        ("VGG16",    lambda: CustomVGG16(   config.NUM_CLASSES, activation=ACTIVATION,
                                            attention_type=None,
                                            pretrained=config.USE_PRETRAINED)),
        ("ResNet50", lambda: CustomResNet50(config.NUM_CLASSES, activation=ACTIVATION,
                                            attention_type=None,
                                            pretrained=config.USE_PRETRAINED)),
    ]

    logger.info(f"=== {EXP_PREFIX}: Baseline CNN (ReLU) ===")

    for name, model_fn in models_cfg:
        eval_d, preds, cm, per_class, ta_h, va_h, tl_h, vl_h = run_or_load(
            model_fn, train_loader, val_loader, test_loader, class_names,
            EXP_PREFIX, name, logger,
            activation=ACTIVATION, attention=ATTENTION,
        )

        save_training_curve(EXP_PREFIX, name, ta_h, va_h, tl_h, vl_h)
        reporter.update(EXP_PREFIX, name, eval_d, preds)
        save_cm_heatmap(EXP_PREFIX, name, cm)
        save_per_class_bar(EXP_PREFIX, name, per_class)
        save_comparison(config.OUTPUT_DIR)
        logger.info(f"  [REPORT] Charts and Excel updated for {name}")
