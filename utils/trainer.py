import os
import json
import time

import torch
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report,
                              balanced_accuracy_score)
from tqdm import tqdm

import config


# ─────────────────────────────────────────────────────────────────────────────
# Core training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, exp_prefix, model_name, logger):
    """
    Trains model with early stopping and checkpointing.
    Returns (train_acc_h, val_acc_h, train_loss_h, val_loss_h).
    Best checkpoint saved to outputs/{exp_prefix}_{model_name}.pth.
    """
    device    = config.DEVICE
    model     = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    ckpt_path = os.path.join(config.OUTPUT_DIR, f"{exp_prefix}_{model_name}.pth")
    meta_path = os.path.join(config.OUTPUT_DIR, f"{exp_prefix}_{model_name}_meta.json")

    best_val_acc    = 0.0
    patience_count  = 0
    train_acc_h, val_acc_h         = [], []
    train_loss_h, val_loss_h       = [], []

    start_time = time.time()

    for epoch in range(config.EPOCHS):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader,
                                   desc=f"  [{exp_prefix}/{model_name}] "
                                        f"Ep {epoch+1}/{config.EPOCHS}",
                                   leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * inputs.size(0)
            preds     = model(inputs).argmax(1)
            correct  += preds.eq(labels).sum().item()
            total    += labels.size(0)

        train_loss = run_loss / total
        train_acc  = correct  / total
        val_loss, val_acc = _val_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_acc_h.append(train_acc);  val_acc_h.append(val_acc)
        train_loss_h.append(train_loss); val_loss_h.append(val_loss)

        logger.info(f"  [{exp_prefix}/{model_name}] Ep {epoch+1:02d}: "
                    f"train {train_acc:.4f} | val {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= config.PATIENCE:
                logger.info(f"  [{exp_prefix}/{model_name}] Early stop at epoch {epoch+1}")
                break

    training_time = round(time.time() - start_time, 1)

    meta = {
        "exp": exp_prefix, "model": model_name,
        "best_val_acc": best_val_acc, "epochs_trained": epoch + 1,
        "training_time_seconds": training_time,
        "train_acc_history": train_acc_h, "val_acc_history": val_acc_h,
        "train_loss_history": train_loss_h, "val_loss_history": val_loss_h,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return train_acc_h, val_acc_h, train_loss_h, val_loss_h


def _val_epoch(model, loader, criterion, device):
    model.eval()
    run_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            out    = model(inputs)
            loss   = criterion(out, labels)
            run_loss += loss.item() * inputs.size(0)
            correct  += out.argmax(1).eq(labels).sum().item()
            total    += labels.size(0)
    return run_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, test_loader, class_names, exp_prefix, model_name):
    """
    Runs inference on test_loader.
    Returns (eval_dict, pred_records, cm_array, per_class_metrics).
    Saves eval.json to outputs/.
    """
    device    = config.DEVICE
    model     = model.to(device).eval()
    criterion = torch.nn.CrossEntropyLoss()

    all_preds, all_labels, all_paths = [], [], []
    total_test_loss = 0.0
    total = 0

    # Attempt to resolve per-image filenames from the Subset dataset
    try:
        subset    = test_loader.dataset
        img_ds    = subset.dataset
        img_idxs  = list(subset.indices)
        has_paths = True
    except AttributeError:
        has_paths = False

    with torch.no_grad():
        item_ptr = 0
        for inputs, labels in test_loader:
            inputs_d, labels_d = inputs.to(device), labels.to(device)
            out  = model(inputs_d)
            loss = criterion(out, labels_d)
            total_test_loss += loss.item() * inputs.size(0)
            batch_preds = out.argmax(1).cpu().numpy()
            all_preds.extend(batch_preds)
            all_labels.extend(labels.numpy())

            if has_paths:
                for i in range(len(labels)):
                    orig = img_idxs[item_ptr + i]
                    all_paths.append(os.path.basename(img_ds.imgs[orig][0]))
            item_ptr += len(labels)
            total    += len(labels)

    test_loss = total_test_loss / total if total > 0 else 0.0

    acc    = accuracy_score(all_labels, all_preds)
    cm     = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, output_dict=True)

    per_class = {
        cls: {
            "precision": report[cls]["precision"],
            "recall":    report[cls]["recall"],
            "f1":        report[cls]["f1-score"],
        }
        for cls in class_names
    }

    eval_dict = {
        "exp":              exp_prefix,
        "model":            model_name,
        "accuracy":         acc,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class,
        "macro_f1":         report["macro avg"]["f1-score"],
        "weighted_f1":      report["weighted avg"]["f1-score"],
        "macro_precision":  report["macro avg"]["precision"],
        "macro_recall":     report["macro avg"]["recall"],
        "test_loss":        round(test_loss, 6),
    }
    if config.USE_BALANCED_ACC:
        eval_dict["balanced_accuracy"] = round(balanced_accuracy_score(all_labels, all_preds), 6)

    pred_records = []
    for i, (lbl, pred) in enumerate(zip(all_labels, all_preds)):
        rec = {"label": int(lbl), "pred": int(pred)}
        if has_paths and i < len(all_paths):
            rec["path"] = all_paths[i]
        pred_records.append(rec)

    return eval_dict, pred_records, cm, per_class


# ─────────────────────────────────────────────────────────────────────────────
# 3-tier caching helper  (used by all experiment files)
# ─────────────────────────────────────────────────────────────────────────────

def run_or_load(model_fn, train_loader, val_loader, test_loader,
                class_names, exp_prefix, model_name, logger,
                activation="ReLU", attention="None",
                pretrain_path=None):
    """
    Tier 1 — eval.json exists       → load metrics instantly, skip model entirely.
    Tier 2 — .pth + _meta.json      → skip training, run evaluate() only.
    Tier 3 — nothing                → full train + evaluate.

    Returns:
        (eval_dict, pred_records, cm, per_class,
         train_acc_h, val_acc_h, train_loss_h, val_loss_h)
    """
    eval_path = os.path.join(config.OUTPUT_DIR, f"{exp_prefix}_{model_name}_eval.json")
    ckpt_path = os.path.join(config.OUTPUT_DIR, f"{exp_prefix}_{model_name}.pth")
    meta_path = os.path.join(config.OUTPUT_DIR, f"{exp_prefix}_{model_name}_meta.json")
    pred_path = os.path.join(config.OUTPUT_DIR, f"predictions_{exp_prefix}_{model_name}.json")

    # ── Tier 1 ─────────────────────────────────────────────────────────────
    if not config.FORCE_RERUN and os.path.exists(eval_path):
        logger.info(f"  [{exp_prefix}/{model_name}] Tier 1 — full cache hit")
        with open(eval_path) as f:
            eval_dict = json.load(f)
        meta  = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

        # Patch fields that may be absent in old-format eval.json files
        eval_dict["exp"]   = exp_prefix
        eval_dict["model"] = model_name
        eval_dict.setdefault("activation", activation)
        eval_dict.setdefault("attention",  attention)
        # Old field renames
        _OLD = [("test_acc","accuracy"),("precision","macro_precision"),
                ("recall","macro_recall"),("f1","macro_f1"),
                ("cm","confusion_matrix"),("train_time","training_time_seconds"),
                ("train_acc","train_acc_final"),("val_acc","val_acc_final"),
                ("train_loss","train_loss_final"),("val_loss","val_loss_final")]
        for old, new in _OLD:
            if old in eval_dict and new not in eval_dict:
                eval_dict[new] = eval_dict[old]
        # per_class_metrics list → dict
        per_raw = eval_dict.get("per_class_metrics", {})
        if isinstance(per_raw, list):
            eval_dict["per_class_metrics"] = {
                item["Class"]: {"precision": item.get("Precision",0),
                                "recall":    item.get("Recall",   0),
                                "f1":        item.get("F1",        0)}
                for item in per_raw if "Class" in item
            }
        # Fill summary meta fields if missing (old eval.json didn't embed them)
        ta_h = meta.get("train_acc_history", [])
        va_h = meta.get("val_acc_history",   [])
        tl_h = meta.get("train_loss_history",[])
        vl_h = meta.get("val_loss_history",  [])
        eval_dict.setdefault("epochs_trained",        meta.get("epochs_trained", 0))
        eval_dict.setdefault("training_time_seconds", meta.get("training_time_seconds",
                                                               meta.get("train_time", 0)))
        eval_dict.setdefault("best_val_acc",          meta.get("best_val_acc",
                                                               meta.get("val_acc", 0)))
        eval_dict.setdefault("train_acc_final",  ta_h[-1] if ta_h else
                                                 meta.get("train_acc", 0))
        eval_dict.setdefault("val_acc_final",    va_h[-1] if va_h else
                                                 meta.get("val_acc", 0))
        eval_dict.setdefault("train_loss_final", tl_h[-1] if tl_h else
                                                 meta.get("train_loss", 0))
        eval_dict.setdefault("val_loss_final",   vl_h[-1] if vl_h else
                                                 meta.get("val_loss", 0))

        preds = json.load(open(pred_path)) if os.path.exists(pred_path) else []
        cm  = np.array(eval_dict.get("confusion_matrix", []))
        per = eval_dict.get("per_class_metrics", {})
        return (eval_dict, preds, cm, per, ta_h, va_h, tl_h, vl_h)

    model = model_fn()

    # ── Tier 2 ─────────────────────────────────────────────────────────────
    _tier2_ok = False
    if not config.FORCE_RERUN and os.path.exists(ckpt_path) and os.path.exists(meta_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
            meta = json.load(open(meta_path))
            ta_h = meta["train_acc_history"]
            va_h = meta["val_acc_history"]
            tl_h = meta["train_loss_history"]
            vl_h = meta["val_loss_history"]
            _tier2_ok = True
            logger.info(f"  [{exp_prefix}/{model_name}] Tier 2 — checkpoint, skipping training")
        except Exception as e:
            logger.warning(f"  [{exp_prefix}/{model_name}] Tier 2 checkpoint corrupt/unreadable "
                           f"({e}) — falling through to Tier 3 full training")

    # ── Tier 3 ─────────────────────────────────────────────────────────────
    if not _tier2_ok:
        logger.info(f"  [{exp_prefix}/{model_name}] Tier 3 — full training")
        if pretrain_path and os.path.exists(pretrain_path):
            try:
                model.load_state_dict(
                    torch.load(pretrain_path, map_location=config.DEVICE), strict=False
                )
                logger.info(f"    Loaded pretrain weights: {pretrain_path} (strict=False)")
            except Exception as e:
                logger.warning(f"    Pretrain weights corrupt/unreadable ({e}) "
                               f"— training from scratch without warm-start")
        ta_h, va_h, tl_h, vl_h = train(
            model, train_loader, val_loader, exp_prefix, model_name, logger
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
        meta = json.load(open(meta_path))

    eval_dict, pred_records, cm, per_class = evaluate(
        model, test_loader, class_names, exp_prefix, model_name
    )
    eval_dict["activation"] = activation
    eval_dict["attention"]  = attention

    # Embed summary meta fields so eval.json is self-contained for the reporter
    eval_dict["epochs_trained"]        = meta.get("epochs_trained", 0)
    eval_dict["training_time_seconds"] = meta.get("training_time_seconds", 0)
    eval_dict["best_val_acc"]          = meta.get("best_val_acc", 0)
    eval_dict["train_acc_final"]       = ta_h[-1] if ta_h else 0
    eval_dict["val_acc_final"]         = va_h[-1] if va_h else 0
    eval_dict["train_loss_final"]      = tl_h[-1] if tl_h else 0
    eval_dict["val_loss_final"]        = vl_h[-1] if vl_h else 0

    with open(eval_path, "w") as f:
        json.dump(eval_dict, f, indent=2)
    with open(pred_path, "w") as f:
        json.dump(pred_records, f, indent=2)

    return eval_dict, pred_records, cm, per_class, ta_h, va_h, tl_h, vl_h
