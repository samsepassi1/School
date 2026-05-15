"""MNIST data loading utilities.

The generative models expect images in the [-1, 1] range so they can use a
Tanh output activation, which is the standard choice for GAN / diffusion
training. We therefore normalize with mean=0.5, std=0.5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MNIST_MEAN = (0.5,)
MNIST_STD = (0.5,)

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data_cache"


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),  # -> [0, 1], shape (1, 28, 28)
            transforms.Normalize(MNIST_MEAN, MNIST_STD),  # -> [-1, 1]
        ]
    )


def get_mnist_datasets(
    root: str | Path = DEFAULT_ROOT,
    download: bool = True,
) -> Tuple[datasets.MNIST, datasets.MNIST]:
    """Return (train, test) MNIST datasets with the standard transform."""
    transform = _build_transform()
    root = str(Path(root))
    train_ds = datasets.MNIST(root=root, train=True, transform=transform, download=download)
    test_ds = datasets.MNIST(root=root, train=False, transform=transform, download=download)
    return train_ds, test_ds


def get_mnist_loaders(
    batch_size: int = 128,
    root: str | Path = DEFAULT_ROOT,
    num_workers: int = 2,
    download: bool = True,
    pin_memory: bool | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader)."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_ds, test_ds = get_mnist_datasets(root=root, download=download)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Map a tensor from [-1, 1] back to [0, 1] for visualization."""
    return (x.clamp(-1, 1) + 1) / 2
