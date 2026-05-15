"""Plotting helpers for sample inspection and comparisons."""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a (B, 1, H, W) or (B, H, W) tensor in [-1, 1] to a NumPy array in [0, 1]."""
    arr = x.detach().cpu()
    if arr.dim() == 4 and arr.size(1) == 1:
        arr = arr.squeeze(1)
    arr = (arr.clamp(-1, 1) + 1) / 2
    return arr.numpy()


def plot_image_grid(
    images: torch.Tensor,
    labels: Sequence[int] | None = None,
    *,
    n_cols: int = 8,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Plot a flat grid of images. Returns the matplotlib Figure."""
    imgs = _to_numpy(images)
    n = imgs.shape[0]
    n_rows = int(np.ceil(n / n_cols))
    figsize = figsize or (n_cols * 1.0, n_rows * 1.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols][i % n_cols]
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
            if labels is not None:
                ax.set_title(str(int(labels[i])), fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_class_grid(
    images_per_class: dict[int, torch.Tensor],
    *,
    title: str | None = None,
    n_per_class: int | None = None,
):
    """Plot one row per class. images_per_class[c] should be (n, 1, H, W)."""
    classes = sorted(images_per_class.keys())
    n_per_class = n_per_class or min(int(images_per_class[c].size(0)) for c in classes)

    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(n_per_class, len(classes)))
    if len(classes) == 1:
        axes = np.array([axes])
    if n_per_class == 1:
        axes = axes.reshape(-1, 1)

    for r, c in enumerate(classes):
        imgs = _to_numpy(images_per_class[c][:n_per_class])
        for j in range(n_per_class):
            ax = axes[r][j]
            ax.imshow(imgs[j], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        axes[r][0].set_ylabel(str(c), rotation=0, labelpad=12, fontsize=10)
        axes[r][0].axis("on")
        axes[r][0].set_xticks([])
        axes[r][0].set_yticks([])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_comparison_grid(
    samples_by_source: dict[str, dict[int, torch.Tensor]],
    *,
    classes: Iterable[int] | None = None,
    title: str | None = None,
):
    """Plot a "Real vs cGAN vs Diffusion" style grid.

    Args:
        samples_by_source: {"Real": {0: tensor, 1: tensor, ...}, "cGAN": {...}, ...}
                           Each tensor is (n, 1, H, W); we display n_per_class=1.
        classes: ordered list of class labels (columns). Defaults to sorted union.
    """
    sources = list(samples_by_source.keys())
    if classes is None:
        classes = sorted(set().union(*[d.keys() for d in samples_by_source.values()]))
    classes = list(classes)

    fig, axes = plt.subplots(
        len(sources),
        len(classes),
        figsize=(len(classes), len(sources)),
        squeeze=False,
    )

    for r, src in enumerate(sources):
        for c_idx, c in enumerate(classes):
            ax = axes[r][c_idx]
            ax.axis("off")
            if c in samples_by_source[src]:
                img = _to_numpy(samples_by_source[src][c][:1])[0]
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(str(c), fontsize=10)
        axes[r][0].set_ylabel(src, rotation=0, labelpad=24, fontsize=11)
        axes[r][0].axis("on")
        axes[r][0].set_xticks([])
        axes[r][0].set_yticks([])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_training_history(history: dict[str, list[float]], *, title: str | None = None):
    """Plot one curve per key in `history`."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in history.items():
        ax.plot(values, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
