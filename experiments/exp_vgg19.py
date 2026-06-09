"""Experiment VGG19 - Complete comparison of ReLU, GELU, Attention, and Fusion variants."""
import os
import torch
import config
from utils.trainer import run_or_load
from utils.plotter import (save_training_curve, save_cm_heatmap,
                           save_per_class_bar, save_comparison)

EXP_PREFIX = "ExpVGG19"


def run(reporter, logger):
    from data.loader import get_loaders
    from data.dataloader import get_data_loaders
    from models.vgg19 import CustomVGG19
    from models.alexnet import CustomAlexNet
    from models.resnet50 import CustomResNet50
    from models.fusion import FusionModel

    # Load data
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        config.INPUT_DIR, config.IMAGE_SIZE, config.BATCH_SIZE, config.RANDOM_SEED
    )

    attn_type = config.ATTENTION_TYPE
    attn_label = attn_type if attn_type else "None"

    # Define three VGG19 variants matching Exp1, Exp2, Exp3 patterns
    models_cfg = [
        # Exp1 style: Baseline ReLU, no attention
        ("VGG19_ReLU", lambda: CustomVGG19(
            config.NUM_CLASSES, 
            activation="ReLU",
            attention_type=None,
            pretrained=config.USE_PRETRAINED
        )),
        
        # Exp2 style: GELU, no attention  
        ("VGG19_GELU", lambda: CustomVGG19(
            config.NUM_CLASSES, 
            activation="GELU",
            attention_type=None,
            pretrained=config.USE_PRETRAINED
        )),
        
        # Exp3 style: GELU + Attention (using config.ATTENTION_TYPE)
        (f"VGG19_GELU_{attn_label}", lambda: CustomVGG19(
            config.NUM_CLASSES, 
            activation="GELU",
            attention_type=attn_type,
            pretrained=config.USE_PRETRAINED
        )),
    ]

    # Log what we're running
    logger.info(f"\n{'='*60}")
    logger.info(f"=== {EXP_PREFIX}: Testing VGG19 Variants ===")
    logger.info(f"{'='*60}")
    logger.info(f"  1. VGG19_ReLU (Baseline - same as Exp1)")
    logger.info(f"  2. VGG19_GELU (GELU only - same as Exp2)")
    logger.info(f"  3. VGG19_GELU_{attn_label} (GELU + Attention - same as Exp3)")

    # Run VGG19 variants
    for name, model_fn in models_cfg:
        # Determine which experiment pattern this follows for logging
        if "ReLU" in name:
            activation_log = "ReLU"
            attention_log = "None"
        elif "GELU" in name and attn_label == "None":
            activation_log = "GELU"
            attention_log = "None"
        else:
            activation_log = "GELU"
            attention_log = attn_label
            
        eval_d, preds, cm, per_class, ta_h, va_h, tl_h, vl_h = run_or_load(
            model_fn, train_loader, val_loader, test_loader, class_names,
            EXP_PREFIX, name, logger,
            activation=activation_log, 
            attention=attention_log,
        )

        save_training_curve(EXP_PREFIX, name, ta_h, va_h, tl_h, vl_h)
        reporter.update(EXP_PREFIX, name, eval_d, preds)
        save_cm_heatmap(EXP_PREFIX, name, cm)
        save_per_class_bar(EXP_PREFIX, name, per_class)
        save_comparison(config.OUTPUT_DIR)
        logger.info(f"  [REPORT] Charts and Excel updated for {name}")

    # ==================== FUSION EXPERIMENT (Exp4) ====================
    logger.info(f"\n{'='*60}")
    logger.info(f"=== {EXP_PREFIX}: Fusion (AlexNet + VGG19 + ResNet50) + GELU + {attn_label} ===")
    logger.info(f"{'='*60}")

    # First, ensure we have checkpoints for all three backbones using the correct naming
    # We'll train AlexNet and ResNet50 with the same ExpVGG19 prefix for consistency
    fusion_ready = True
    
    # Check and train AlexNet if needed
    alexnet_ckpt = os.path.join(config.OUTPUT_DIR, f"Exp3_AlexNet.pth")
    if not os.path.exists(alexnet_ckpt):
        logger.info(f"  [TRAINING] AlexNet checkpoint not found. Training now...")
        alexnet_model = lambda: CustomAlexNet(config.NUM_CLASSES, activation="GELU",
                                               attention_type=attn_type, 
                                               pretrained=config.USE_PRETRAINED)
        run_or_load(alexnet_model, train_loader, val_loader, test_loader, class_names,
                   "Exp3", "AlexNet", logger, 
                   activation="GELU", attention=attn_label)
        fusion_ready = False
    else:
        logger.info(f"  [OK] Found AlexNet checkpoint: {os.path.basename(alexnet_ckpt)}")
    
    # Check and train ResNet50 if needed
    resnet_ckpt = os.path.join(config.OUTPUT_DIR, f"Exp3__ResNet50.pth")
    if not os.path.exists(resnet_ckpt):
        logger.info(f"  [TRAINING] ResNet50 checkpoint not found. Training now...")
        resnet_model = lambda: CustomResNet50(config.NUM_CLASSES, activation="GELU",
                                               attention_type=attn_type,
                                               pretrained=config.USE_PRETRAINED)
        run_or_load(resnet_model, train_loader, val_loader, test_loader, class_names,
                   "Exp3", "ResNet50", logger,
                   activation="GELU", attention=attn_label)
        fusion_ready = False
    else:
        logger.info(f"  [OK] Found ResNet50 checkpoint: {os.path.basename(resnet_ckpt)}")
    
    # VGG19 attention variant was already trained above
    vgg19_ckpt = os.path.join(config.OUTPUT_DIR, f"{EXP_PREFIX}_VGG19_GELU_{attn_label}.pth")
    if os.path.exists(vgg19_ckpt):
        logger.info(f"  [OK] Found VGG19 checkpoint: {os.path.basename(vgg19_ckpt)}")
    else:
        logger.warning(f"  [WARNING] VGG19 checkpoint not found at {vgg19_ckpt}")
    
    # Define fusion model with warm-start from checkpoints
    def fusion_model_fn():
        fusion = FusionModel(num_classes=config.NUM_CLASSES, attention_type=attn_type)
        
        # Load AlexNet weights
        if os.path.exists(alexnet_ckpt):
            try:
                fusion.alexnet.load_state_dict(torch.load(alexnet_ckpt, map_location=config.DEVICE))
                logger.info(f"  [LOAD] AlexNet weights loaded from {os.path.basename(alexnet_ckpt)}")
            except Exception as e:
                logger.warning(f"  [WARN] Failed to load AlexNet weights: {e}")
        else:
            logger.warning(f"  [WARN] AlexNet checkpoint missing - using random weights")
        
        # Load VGG19 weights
        if os.path.exists(vgg19_ckpt):
            try:
                fusion.vgg19.load_state_dict(torch.load(vgg19_ckpt, map_location=config.DEVICE))
                logger.info(f"  [LOAD] VGG19 weights loaded from {os.path.basename(vgg19_ckpt)}")
            except Exception as e:
                logger.warning(f"  [WARN] Failed to load VGG19 weights: {e}")
        else:
            logger.warning(f"  [WARN] VGG19 checkpoint missing - using random weights")
        
        # Load ResNet50 weights
        if os.path.exists(resnet_ckpt):
            try:
                fusion.resnet.load_state_dict(torch.load(resnet_ckpt, map_location=config.DEVICE))
                logger.info(f"  [LOAD] ResNet50 weights loaded from {os.path.basename(resnet_ckpt)}")
            except Exception as e:
                logger.warning(f"  [WARN] Failed to load ResNet50 weights: {e}")
        else:
            logger.warning(f"  [WARN] ResNet50 checkpoint missing - using random weights")
        
        fusion.freeze_backbones()
        return fusion
    
    # Run fusion experiment
    fusion_name = f"Fusion_GELU_{attn_label}"
    eval_d, preds, cm, per_class, ta_h, va_h, tl_h, vl_h = run_or_load(
        fusion_model_fn, train_loader, val_loader, test_loader, class_names,
        EXP_PREFIX, fusion_name, logger,
        activation="GELU", attention=attn_label,
    )
    
    save_training_curve(EXP_PREFIX, fusion_name, ta_h, va_h, tl_h, vl_h)
    reporter.update(EXP_PREFIX, fusion_name, eval_d, preds)
    save_cm_heatmap(EXP_PREFIX, fusion_name, cm)
    save_per_class_bar(EXP_PREFIX, fusion_name, per_class)
    save_comparison(config.OUTPUT_DIR)
    logger.info(f"  [REPORT] Charts and Excel updated for {fusion_name}")
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"=== {EXP_PREFIX}: ALL EXPERIMENTS COMPLETED ===")
    logger.info(f"{'='*60}")
    logger.info(f"  Models trained/evaluated:")
    logger.info(f"  - VGG19_ReLU")
    logger.info(f"  - VGG19_GELU")
    logger.info(f"  - VGG19_GELU_{attn_label}")
    logger.info(f"  - AlexNet_GELU_{attn_label} (for fusion)")
    logger.info(f"  - ResNet50_GELU_{attn_label} (for fusion)")
    logger.info(f"  - {fusion_name} (Fusion ensemble)")
    logger.info(f"{'='*60}")