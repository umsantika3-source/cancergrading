import os
import sys
import glob
import json
import numpy as np

import config
from utils.logger import get_logger
from utils.reporter import DiagnosticCompiler


def _prompt_data_dir(logger):
    if config.INPUT_DIR:
        return config.INPUT_DIR
    path = input("\nEnter dataset path (folder containing Grade1/Grade2/Grade3 subfolders):\n> ").strip()
    if not os.path.isdir(path):
        logger.error(f"Directory not found: {path}")
        sys.exit(1)
    return path


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger = get_logger()

    print("\n" + "=" * 57)
    print("   Breast Cancer Grading — Experiment Pipeline")
    print("=" * 57)
    print("  1. Experiment 1 - Baseline CNN (ReLU)")
    print("  2. Experiment 2 - CNN + GELU")
    print("  3. Experiment 3 - CNN + GELU + Attention")
    print("  4. Experiment 4 - Feature Fusion (3 CNNs) + GELU + Attention")
    print("  5. Run All  Exp1 → Exp2 → Exp3 → Exp4")
    print("  6. Regenerate Excel report only")
    print("  7. Regenerate all charts only")
    print("=" * 57)

    choice = input("Select option (1-7): ").strip()

    reporter = DiagnosticCompiler()
    reporter.load_data()   # pre-populate with any already-completed experiments

    if choice in ("1", "2", "3", "4", "5"):
        config.INPUT_DIR = _prompt_data_dir(logger)

        # ── Structure validation gate ──
        from data.structure_validator import validate_structure
        ok, msg = validate_structure(
            config.INPUT_DIR, config.DATA_STRUCTURE, config.K_FOLDS
        )
        if not ok:
            logger.error(msg)
            sys.exit(1)
        logger.info(f"✓ Structure validation passed ({config.DATA_STRUCTURE})")

        from experiments.exp1_baseline  import run as run_exp1
        from experiments.exp2_gelu      import run as run_exp2
        from experiments.exp3_attention import run as run_exp3
        from experiments.exp4_fusion    import run as run_exp4

        run_map = {
            "1": run_exp1,
            "2": run_exp2,
            "3": run_exp3,
            "4": run_exp4,
        }

        def _run_selected(run_fn, fold_tag=""):
            if choice == "5":
                run_exp1(reporter, logger, data_loaders=data_loaders, fold_tag=fold_tag)
                run_exp2(reporter, logger, data_loaders=data_loaders, fold_tag=fold_tag)
                run_exp3(reporter, logger, data_loaders=data_loaders, fold_tag=fold_tag)
                run_exp4(reporter, logger, data_loaders=data_loaders, fold_tag=fold_tag)
            else:
                run_fn(reporter, logger, data_loaders=data_loaders, fold_tag=fold_tag)

        if config.DATA_STRUCTURE == "kfold5":
            fold_numbers = (
                range(1, config.K_FOLDS + 1)
                if config.ITERATE_ALL_FOLDS
                else [config.CURRENT_FOLD]
            )
            from data.kfold_loader import get_kfold_data_loaders
            for fold_num in fold_numbers:
                data_loaders = get_kfold_data_loaders(
                    config.INPUT_DIR,
                    fold_num,
                    config.IMAGE_SIZE,
                    config.BATCH_SIZE,
                    config.RANDOM_SEED,
                )
                fold_tag = f"_fold_{fold_num}"
                logger.info(
                    f"\n{'=' * 57}\n  Fold {fold_num} / {config.K_FOLDS}\n{'=' * 57}"
                )
                _run_selected(run_map[choice], fold_tag=fold_tag)
        else:
            # Old flat structure — single run, no fold tag
            data_loaders = None  # experiments will fall back to dataloader_v2
            _run_selected(run_map[choice])

    elif choice == "6":
        logger.info("Regenerating Excel report from existing eval.json files...")
        reporter.load_data()
        reporter.build_excel()
        logger.info("Done — check outputs/reports/")

    elif choice == "7":
        logger.info("Regenerating all charts from existing eval.json files...")
        from utils.plotter import (save_training_curve, save_cm_heatmap,
                                   save_per_class_bar, save_comparison)
        for eval_path in sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "*_eval.json"))):
            with open(eval_path) as f:
                ed = json.load(f)
            exp   = ed.get("exp", "?")
            model = ed.get("model", "?")
            cm = np.array(ed.get("confusion_matrix", []))
            if cm.size:
                save_cm_heatmap(exp, model, cm)
            pcm = ed.get("per_class_metrics", {})
            if isinstance(pcm, dict) and pcm:
                save_per_class_bar(exp, model, pcm)
            meta_path = os.path.join(config.OUTPUT_DIR, f"{exp}_{model}_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                save_training_curve(exp, model,
                                    meta["train_acc_history"], meta["val_acc_history"],
                                    meta["train_loss_history"], meta["val_loss_history"])
        save_comparison(config.OUTPUT_DIR)
        logger.info("Done — check outputs/reports/")
    
    elif choice == "8":
        config.INPUT_DIR = _prompt_data_dir(logger)

        # ── Structure validation gate ──
        from data.structure_validator import validate_structure
        ok, msg = validate_structure(
            config.INPUT_DIR, config.DATA_STRUCTURE, config.K_FOLDS
        )
        if not ok:
            logger.error(msg)
            sys.exit(1)
        logger.info(f"✓ Structure validation passed ({config.DATA_STRUCTURE})")

        from experiments.exp_vgg19    import run as run_expvgg19
        run_expvgg19(reporter, logger)
		
    else:
        print("Invalid option. Please enter a number 1–7.")


if __name__ == "__main__":
    main()
