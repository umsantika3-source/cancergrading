"""
dataloader_v2.py — Data loader for PRE-AUGMENTED dataset (no on-the-fly augmentation).

This loader is designed to work with datasets that have ALREADY been augmented
during the k-fold split process (org_kfold_v2.py). It applies ONLY:
    - Resize to target size
    - Convert to Tensor
    - ImageNet normalization (mean & std)

No random flips, rotations, color jitter, etc. to avoid DOUBLE AUGMENTATION.

Expected folder structure:
    {data_root}/
        train/GRADE1/  (original + augmented images)
        train/GRADE2/
        train/GRADE3/
        val/GRADE1/    (original only)
        val/GRADE2/
        val/GRADE3/
        test/GRADE1/   (original only)
        test/GRADE2/
        test/GRADE3/
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


def get_data_loaders(
    data_root: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 8,
    seed: int = 42,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Returns (train_loader, val_loader, test_loader, class_names).
    
    All splits use the SAME transform (Resize + ToTensor + Normalize).
    No augmentation on-the-fly — data is already pre-augmented.
    """
    # ImageNet normalization (standard for most pretrained models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Single transform for ALL splits (train/val/test)
    # No augmentation — data already pre-augmented
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize
    ])
    
    pin_memory = torch.cuda.is_available()
    
    train_dir = os.path.join(data_root, 'train')
    val_dir   = os.path.join(data_root, 'val')
    test_dir  = os.path.join(data_root, 'test')
    
    # Verify folders exist
    missing = [d for d in [train_dir, val_dir, test_dir] if not os.path.exists(d)]
    if missing:
        raise FileNotFoundError(
            f"Missing split folders: {missing}\n"
            f"Ensure {data_root} contains train/, val/, test/ subfolders."
        )
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    print(f"  DataLoader V2 — using pre-augmented dataset (no on-the-fly augmentation)")
    print(f"    Train: {len(train_dataset)} images ({len(train_loader)} batches)")
    print(f"    Val:   {len(val_dataset)} images")
    print(f"    Test:  {len(test_dataset)} images")
    print(f"    Classes: {train_dataset.classes}")
    
    return train_loader, val_loader, test_loader, train_dataset.classes