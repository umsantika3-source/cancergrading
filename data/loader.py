import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_loaders(data_root, image_size, batch_size, seed):
    """
    Returns (train_loader, val_loader, test_loader, class_names).
    Expects ImageFolder structure: data_root/Grade1/, Grade2/, Grade3/
    Split: 70% train / 15% val / 15% test — deterministic with fixed seed.
    """
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full   = datasets.ImageFolder(data_root, transform=train_transform)
    val_ds = datasets.ImageFolder(data_root, transform=val_transform)
    tst_ds = datasets.ImageFolder(data_root, transform=val_transform)

    n = len(full)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(Subset(full,   train_idx), batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=pin_memory)
    val_loader   = DataLoader(Subset(val_ds, val_idx),   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader  = DataLoader(Subset(tst_ds, test_idx),  batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, full.classes
