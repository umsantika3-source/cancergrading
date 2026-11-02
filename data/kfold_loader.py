"""
kfold_loader.py — Data loader for the k-fold 5 (kfold5) directory structure.

Expected layout (from Others/kfold5_structure.png):
    {data_root}/
        fold_1/
            train/Grade_1/  (80% of Grade 1 images)
            train/Grade_2/
            train/Grade_3/
            val/Grade_1/    (remaining 20% of Grade 1)
            val/Grade_2/
            val/Grade_3/
        fold_2/ … fold_K/   (different 80/20 mix per fold)

There is NO test/ folder in this structure, so the returned test_loader
aliases the val_loader (the held-out fold serves as evaluation).

Transform: Resize + ToTensor + ImageNet normalization only
(no on-the-fly augmentation — folds are pre-split).
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


def get_kfold_data_loaders(
    data_root: str,
    fold_num: int,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 8,
    seed: int = 42,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Returns (train_loader, val_loader, test_loader, class_names) for a given fold.

    Args:
        data_root: Root directory containing fold_1/ … fold_K/.
        fold_num: Which fold to load (1-based, e.g. 1 … 5).
        image_size: Target image size (height, width).
        batch_size: Batch size for DataLoader.
        seed: Random seed (kept for API compatibility; not used here because
              splits are pre-defined).
        num_workers: Number of DataLoader workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names).
        test_loader is the same as val_loader because kfold5 has no test split.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    pin_memory = torch.cuda.is_available()

    fold_dir = os.path.join(data_root, f"fold_{fold_num}")
    train_dir = os.path.join(fold_dir, "train")
    val_dir = os.path.join(fold_dir, "val")

    missing = [d for d in [fold_dir, train_dir, val_dir] if not os.path.exists(d)]
    if missing:
        raise FileNotFoundError(
            f"Missing kfold5 directory(ies): {missing}\n"
            f"Expected: {fold_dir}/train/ and {fold_dir}/val/\n"
            f"Ensure config.INPUT_DIR points to the kfold_dataset root and "
            f"config.K_FOLDS covers fold_{fold_num}."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    # kfold5 structure has no test/ folder — use val as the evaluation split
    test_loader = val_loader

    print(f"  DataLoader KFold — fold_{fold_num}")
    print(f"    Train: {len(train_dataset)} images ({len(train_loader)} batches)")
    print(f"    Val:   {len(val_dataset)} images ({len(val_loader)} batches)")
    print(f"    Test:  (alias of val — no test/ folder in kfold5 structure)")
    print(f"    Classes: {train_dataset.classes}")

    return train_loader, val_loader, test_loader, train_dataset.classes