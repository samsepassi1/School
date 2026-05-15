import math
import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, transforms

from .helpers import compute_mean_and_std, get_data_location


def get_data_loaders(
    batch_size: int = 32,
    valid_size: float = 0.2,
    num_workers: int = -1,
    limit: int = -1,
):
    """
    Build train, validation and test DataLoaders for the landmark dataset.

    Parameters
    ----------
    batch_size : int
        Mini-batch size for all three loaders.
    valid_size : float
        Fraction of the training set to hold out for validation (0-1).
    num_workers : int
        Number of subprocesses for data loading. -1 = use all CPUs.
    limit : int
        If positive, limit each split to this many samples (useful for debugging).
    """
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()

    data_loaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location())
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean.tolist()}, std: {std.tolist()}")

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
            ]
        ),
    }

    train_data = datasets.ImageFolder(
        str(base_path / "train"), transform=data_transforms["train"]
    )
    valid_data = datasets.ImageFolder(
        str(base_path / "train"), transform=data_transforms["valid"]
    )

    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )

    test_data = datasets.ImageFolder(
        str(base_path / "test"), transform=data_transforms["test"]
    )

    if limit > 0:
        test_indices = torch.arange(min(limit, len(test_data)))
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
        data_loaders["test"] = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
        )
    else:
        data_loaders["test"] = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """Show a few examples from the training data loader with their labels."""
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    class_names = data_loaders["train"].dataset.classes

    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1.0 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )
    images = invTrans(images)

    images = torch.clamp(images, 0, 1)
    images = images.numpy().transpose((0, 2, 3, 1))

    fig = plt.figure(figsize=(3 * max_n, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(class_names[labels[idx].item()].split(".")[-1].replace("_", " "))
    return fig


# ---------- tests ----------

import pytest


@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2, num_workers=0, limit=200)


def test_data_loaders_keys(data_loaders):
    assert set(data_loaders.keys()) == {"train", "valid", "test"}


def test_data_loaders_output_type(data_loaders):
    images, labels = next(iter(data_loaders["train"]))
    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)


def test_data_loaders_output_shape(data_loaders):
    images, labels = next(iter(data_loaders["train"]))
    assert images.shape[0] == 2
    assert labels.shape[0] == 2


def test_visualize_one_batch(data_loaders):
    visualize_one_batch(data_loaders, max_n=2)
