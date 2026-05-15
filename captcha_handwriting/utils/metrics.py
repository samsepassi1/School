"""Quantitative evaluation: Frechet Inception Distance (FID).

We extract 2048-D pool3 features from a pretrained InceptionV3, then compute
FID as: ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2 * sqrtm(C_r @ C_g)).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class InceptionFeatures(nn.Module):
    """InceptionV3 pool3 (2048-D) feature extractor for FID."""

    def __init__(self) -> None:
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        net = models.inception_v3(weights=weights, aux_logits=True)
        net.fc = nn.Identity()
        # Inception applies its own resizing internally only if transform_input=True;
        # we handle resizing/normalization ourselves so we can feed grayscale MNIST.
        net.transform_input = False
        net.eval()
        self.net = net

    @staticmethod
    def _prepare(x: torch.Tensor) -> torch.Tensor:
        """Take images in [-1, 1] (grayscale or RGB) and produce Inception-ready
        299x299 RGB tensors with ImageNet normalization."""
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W); got {tuple(x.shape)}")
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        # [-1, 1] -> [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = _IMAGENET_MEAN.to(x.device, dtype=x.dtype)
        std = _IMAGENET_STD.to(x.device, dtype=x.dtype)
        return (x - mean) / std

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare(x)
        return self.net(x)


@torch.no_grad()
def _activations(
    extractor: InceptionFeatures,
    images: Iterable[torch.Tensor] | torch.Tensor,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Compute activations for an iterable (or tensor) of images."""
    device = torch.device(device)
    extractor.to(device).eval()

    if isinstance(images, torch.Tensor):
        chunks = images.split(batch_size, dim=0)
    else:
        chunks = list(images)

    feats: list[np.ndarray] = []
    for chunk in chunks:
        chunk = chunk.to(device)
        out = extractor(chunk).cpu().numpy()
        feats.append(out)
    return np.concatenate(feats, axis=0)


def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Stable matrix square root via scipy."""
    from scipy.linalg import sqrtm  # imported lazily so the module loads without scipy
    sqrt_m, _ = sqrtm(matrix, disp=False)
    if np.iscomplexobj(sqrt_m):
        sqrt_m = sqrt_m.real
    return sqrt_m


def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    *,
    extractor: InceptionFeatures | None = None,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
    eps: float = 1e-6,
) -> float:
    """Compute FID between two batches of images in [-1, 1].

    Args:
        real_images: (N, C, H, W) tensor of reference samples.
        fake_images: (M, C, H, W) tensor of generated samples.
        extractor:   Optional shared InceptionFeatures (avoids reloading weights).
    """
    if extractor is None:
        extractor = InceptionFeatures()

    real_feats = _activations(extractor, real_images, batch_size=batch_size, device=device)
    fake_feats = _activations(extractor, fake_images, batch_size=batch_size, device=device)

    mu_r, mu_g = real_feats.mean(axis=0), fake_feats.mean(axis=0)
    cov_r = np.cov(real_feats, rowvar=False)
    cov_g = np.cov(fake_feats, rowvar=False)

    diff = mu_r - mu_g

    # Add a tiny diagonal to keep sqrtm well-conditioned on small batches.
    offset = np.eye(cov_r.shape[0]) * eps
    cov_sqrt = _matrix_sqrt((cov_r + offset) @ (cov_g + offset))

    fid = float(diff @ diff + np.trace(cov_r + cov_g - 2.0 * cov_sqrt))
    return fid
