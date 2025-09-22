from pathlib import Path
from typing import Tuple, List
from torch.utils.data import DataLoader
from torchvision import datasets

from .transforms import train_transforms, val_transforms

def _build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 4, pin_memory: bool = True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

def get_datasets(
    data_dir: str | Path = "data",
    img_size: int = 224,
    augment: bool = True,
):
    data_dir = Path(data_dir)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transforms(img_size, augment))
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=val_transforms(img_size))
    class_names: List[str] = train_ds.classes
    return train_ds, val_ds, class_names

def get_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    augment: bool = True,
):
    train_ds, val_ds, class_names = get_datasets(data_dir, img_size, augment)

    train_loader = _build_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = _build_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, class_names

def get_test_loader(
    data_dir: str | Path = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
):
    data_dir = Path(data_dir)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=val_transforms(img_size))
    test_loader = _build_loader(test_ds, batch_size, shuffle=False, num_workers=num_workers)
    return test_loader, test_ds.classes
