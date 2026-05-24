import os
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import config


def _report_dir():
    d = os.path.join(config.OUTPUT_DIR, "reports")
    os.makedirs(d, exist_ok=True)
    return d


def _versioned_path(path):
    """If path already exists, returns path_2.ext, path_3.ext, … until free."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{base}_{n}{ext}"):
        n += 1
    return f"{base}_{n}{ext}"


def _save_path(filename):
    """Versioned path inside the reports directory."""
    return _versioned_path(os.path.join(_report_dir(), filename))


def _safe(name):
    """Strip characters that are illegal in Windows filenames."""
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name


def save_training_curve(exp, model_name, train_acc, val_acc, train_loss, val_loss):
    """Training accuracy + loss curves side by side. Saved after trainer.train()."""
    epochs = range(1, len(train_acc) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_acc, label="Train", color="#9400D3")
    ax1.plot(epochs, val_acc,   label="Val",   color="#FF69B4")
    ax1.set_title(f"{exp} {model_name} — Accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_loss, label="Train", color="#9400D3")
    ax2.plot(epochs, val_loss,   label="Val",   color="#FF69B4")
    ax2.set_title(f"{exp} {model_name} — Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(_save_path(f"training_{exp}_{model_name}.png"), dpi=150)
    plt.close()


def save_cm_heatmap(exp, model_name, cm):
    """Confusion-matrix heatmap with pink→purple colormap."""
    if not isinstance(cm, np.ndarray):
        cm = np.array(cm)

    cmap   = mcolors.LinearSegmentedColormap.from_list(
        "pinkpurple", ["#FFE4E1", "#DA70D6", "#9400D3"])
    labels = ["Grade I", "Grade II", "Grade III"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {exp} {model_name}")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(_save_path(f"cm_{exp}_{model_name}.png"), dpi=150)
    plt.close()


def save_per_class_bar(exp, model_name, per_class_metrics):
    """Grouped bar: Precision / Recall / F1 per grade — doctor view."""
    if not isinstance(per_class_metrics, dict) or not per_class_metrics:
        return
    classes   = list(per_class_metrics.keys())
    x         = np.arange(len(classes))
    width     = 0.25

    precision = [per_class_metrics[c]["precision"] for c in classes]
    recall    = [per_class_metrics[c]["recall"]    for c in classes]
    f1        = [per_class_metrics[c]["f1"]        for c in classes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#FF69B4")
    ax.bar(x,         recall,    width, label="Recall",    color="#DA70D6")
    ax.bar(x + width, f1,        width, label="F1",        color="#9400D3")

    ax.set_title(f"Per-Class Metrics — {exp} {model_name}")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.15)
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(_save_path(f"per_class_{exp}_{model_name}.png"), dpi=150)
    plt.close()


def save_comparison(output_dir):
    """
    Master entry point — rebuilds all comparison charts after every model finishes.
    1. Overall accuracy bar chart (existing)
    2. Option A: per-model series across experiments (series_AlexNet.png, etc.)
    3. Option B: per-experiment overview of all models (overview_Exp1.png, etc.)
    """
    eval_files = sorted(glob.glob(os.path.join(output_dir, "*_eval.json")))
    if not eval_files:
        return

    _save_overall_bar(eval_files)
    _save_model_series(output_dir)
    _save_exp_overview(output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_overall_bar(eval_files):
    """Overall accuracy bar for every completed model, sorted by exp+model."""
    data = []
    for path in eval_files:
        with open(path) as f:
            d = json.load(f)
        label = f"{d.get('exp','?')}_{d.get('model','?')}"
        data.append((label, d.get("accuracy", 0.0)))

    labels, accs = zip(*data)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    bars = ax.bar(x, accs, color="#9400D3", alpha=0.85)
    ax.set_title("All Models — Test Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(_save_path("model_comparison.png"), dpi=150)
    plt.close()


def _save_model_series(output_dir):
    """
    Option A — one image per backbone showing training curves across Exp1/Exp2/Exp3.
    Saved as: reports/series_{model_name}.png
    Layout: each column = one experiment, rows = accuracy / loss.
    """
    backbone_names = ["AlexNet", "VGG16", "ResNet50"]

    for model_name in backbone_names:
        # Collect all experiments that have a meta file for this model
        entries = []
        for meta_path in sorted(glob.glob(
                os.path.join(output_dir, f"*_{model_name}_meta.json"))):
            with open(meta_path) as f:
                meta = json.load(f)
            exp = meta.get("exp", os.path.basename(meta_path).split(f"_{model_name}")[0])
            entries.append((exp, meta))

        if not entries:
            continue

        n = len(entries)
        fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle(f"{model_name} — Progress Across Experiments",
                     fontsize=13, fontweight="bold", color="#9400D3")

        for col, (exp, meta) in enumerate(entries):
            ta = meta.get("train_acc_history", [])
            va = meta.get("val_acc_history",   [])
            tl = meta.get("train_loss_history", [])
            vl = meta.get("val_loss_history",   [])
            ep = range(1, len(ta) + 1)

            attn  = meta.get("attention",  "")
            act   = meta.get("activation", "")
            label = exp
            if act:  label += f"\n{act}"
            if attn and attn not in ("None", ""): label += f" + {attn}"

            # Accuracy row
            ax_acc = axes[0][col]
            ax_acc.plot(ep, ta, color="#9400D3", label="Train")
            ax_acc.plot(ep, va, color="#FF69B4", label="Val",  linestyle="--")
            ax_acc.set_title(label, fontsize=10)
            ax_acc.set_ylabel("Accuracy" if col == 0 else "")
            ax_acc.set_ylim(0, 1.05); ax_acc.legend(fontsize=8)
            ax_acc.grid(True, alpha=0.3)

            # Loss row
            ax_loss = axes[1][col]
            ax_loss.plot(ep, tl, color="#9400D3", label="Train")
            ax_loss.plot(ep, vl, color="#FF69B4", label="Val",  linestyle="--")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss" if col == 0 else "")
            ax_loss.legend(fontsize=8); ax_loss.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(_save_path(f"series_{_safe(model_name)}.png"), dpi=150)
        plt.close()


def _save_exp_overview(output_dir):
    """
    Option B — one image per experiment showing all models side by side.
    Saved as: reports/overview_{exp}.png
    Layout: each column = one model, rows = accuracy / loss.
    """
    # Discover all unique experiments from meta files
    exps_seen = {}
    for meta_path in sorted(glob.glob(os.path.join(output_dir, "*_meta.json"))):
        with open(meta_path) as f:
            meta = json.load(f)
        exp   = meta.get("exp")
        model = meta.get("model")
        if not exp or not model:   # skip incomplete/foreign meta files
            continue
        exps_seen.setdefault(exp, {})[model] = meta

    for exp, models_dict in exps_seen.items():
        entries = list(models_dict.items())   # [(model_name, meta), ...]
        n = len(entries)
        if n == 0:
            continue

        fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle(f"{exp} — All Models Compared",
                     fontsize=13, fontweight="bold", color="#9400D3")

        for col, (model_name, meta) in enumerate(entries):
            ta = meta.get("train_acc_history",  [])
            va = meta.get("val_acc_history",    [])
            tl = meta.get("train_loss_history", [])
            vl = meta.get("val_loss_history",   [])
            ep = range(1, len(ta) + 1)

            # Accuracy row
            ax_acc = axes[0][col]
            ax_acc.plot(ep, ta, color="#9400D3", label="Train")
            ax_acc.plot(ep, va, color="#FF69B4", label="Val",  linestyle="--")
            ax_acc.set_title(model_name, fontsize=11, fontweight="bold")
            ax_acc.set_ylabel("Accuracy" if col == 0 else "")
            ax_acc.set_ylim(0, 1.05); ax_acc.legend(fontsize=8)
            ax_acc.grid(True, alpha=0.3)

            # Loss row
            ax_loss = axes[1][col]
            ax_loss.plot(ep, tl, color="#9400D3", label="Train")
            ax_loss.plot(ep, vl, color="#FF69B4", label="Val",  linestyle="--")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss" if col == 0 else "")
            ax_loss.legend(fontsize=8); ax_loss.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(_save_path(f"overview_{_safe(exp)}.png"), dpi=150)
        plt.close()
