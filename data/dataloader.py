import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional, List

def get_data_loaders(
    data_root: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 42,
    use_pre_split: bool = True,  # Default: use existing train/val/test folders
    train_val_test_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    num_workers: int = 4,  # Better performance
    augment_train: bool = True
):
    """
    Flexible data loader that works with both pre-split folders and auto-split.
    
    Args:
        data_root: Root directory containing images
        image_size: Target image size (height, width)
        batch_size: Batch size for DataLoader
        seed: Random seed for reproducibility (auto-split only)
        use_pre_split: If True, look for train/val/test subfolders
        train_val_test_ratio: Split ratios (auto-split only)
        num_workers: Number of DataLoader workers
        augment_train: Apply data augmentation to training set
    """
    
    # ImageNet normalization (standard for most pretrained models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms (stronger augmentation for better generalization)
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # Useful for medical/OCR
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        normalize
    ]) if augment_train else transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize
    ])
    
    # Validation/Test transforms (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize
    ])
    
    pin_memory = torch.cuda.is_available()
    
    # Case 1: Use pre-split folders (RECOMMENDED for your case)
    if use_pre_split:
        train_dir = os.path.join(data_root, 'train')
        val_dir = os.path.join(data_root, 'val')
        test_dir = os.path.join(data_root, 'test')
        
        # Check if folders exist
        if all([os.path.exists(train_dir), os.path.exists(val_dir), os.path.exists(test_dir)]):
            print(f"✓ Using pre-split folders from {data_root}")
            
            train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
            val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
            test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)
            
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
            
            print(f"  Train: {len(train_dataset)} images")
            print(f"  Val: {len(val_dataset)} images")
            print(f"  Test: {len(test_dataset)} images")
            print(f"  Classes: {train_dataset.classes}")
            
            return train_loader, val_loader, test_loader, train_dataset.classes
        
        else:
            print(f"⚠ Pre-split folders not found, falling back to auto-split...")
            # Fall through to auto-split
    
    # Case 2: Auto-split from single folder
    print(f"✓ Using auto-split (ratio={train_val_test_ratio}) from {data_root}")
    
    full_dataset = datasets.ImageFolder(data_root, transform=None)  # Load without transform
    n = len(full_dataset)
    
    # Deterministic split
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    n_train = int(n * train_val_test_ratio[0])
    n_val = int(n * train_val_test_ratio[1])
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Apply transforms via wrapper datasets
    class TransformSubset(Subset):
        def __init__(self, dataset, indices, transform=None):
            super().__init__(dataset, indices)
            self.transform = transform
        
        def __getitem__(self, idx):
            x, y = self.dataset[self.indices[idx]]
            if self.transform:
                x = self.transform(x)
            return x, y
    
    train_dataset = TransformSubset(full_dataset, train_idx, transform=train_transform)
    val_dataset = TransformSubset(full_dataset, val_idx, transform=val_test_transform)
    test_dataset = TransformSubset(full_dataset, test_idx, transform=val_test_transform)
    
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
    
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    print(f"  Classes: {full_dataset.classes}")
    
    return train_loader, val_loader, test_loader, full_dataset.classes
