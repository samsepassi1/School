"""Conditional GAN for class-conditional MNIST digit generation.

The Generator concatenates a class embedding to the latent noise vector and
maps the result to a 1x28x28 image with a Tanh output. The Discriminator
fuses a learned label embedding with the image as an additional input
channel and predicts whether the (image, label) pair is real or generated.

Both networks share the same `num_classes` and `embed_dim` so the same
integer label can index the same embedding semantics.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Maps (z, y) -> 1x28x28 image in [-1, 1]."""

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        embed_dim: int = 50,
        img_size: int = 28,
        feature_maps: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.img_size = img_size

        self.label_emb = nn.Embedding(num_classes, embed_dim)

        # Project the combined (noise, label) vector to a 7x7 feature map,
        # then upsample 7x7 -> 14x14 -> 28x28 with strided transposed convs.
        self.project = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, feature_maps * 4 * 7 * 7),
            nn.BatchNorm1d(feature_maps * 4 * 7 * 7),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(inplace=True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True),
            # refine to a single output channel
            nn.Conv2d(feature_maps, 1, kernel_size=3, padding=1),
            nn.Tanh(),  # output in [-1, 1] to match the normalized training data
        )

        self._feature_maps = feature_maps

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.size(1) != self.latent_dim:
            raise ValueError(f"z must be (B, {self.latent_dim}); got {tuple(z.shape)}")
        if labels.dim() != 1:
            raise ValueError(f"labels must be 1D; got {tuple(labels.shape)}")

        y = self.label_emb(labels)              # (B, embed_dim)
        x = torch.cat([z, y], dim=1)            # (B, latent_dim + embed_dim)
        x = self.project(x)                     # (B, fm*4 * 7 * 7)
        x = x.view(-1, self._feature_maps * 4, 7, 7)
        x = self.upsample(x)                    # (B, 1, 28, 28)
        return x


class Discriminator(nn.Module):
    """Maps (image, y) -> real/fake logit.

    The label is embedded into a (num_classes * img_size * img_size) projection
    that is reshaped into an extra image-shaped channel and concatenated with
    the input image, so the convolutional stack sees both inputs jointly.
    """

    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 28,
        feature_maps: int = 64,
        embed_dim: int = 50,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.label_to_map = nn.Linear(embed_dim, img_size * img_size)

        self.net = nn.Sequential(
            # input: 2 x 28 x 28  (image channel + label channel)
            nn.Conv2d(2, feature_maps, 4, stride=2, padding=1),         # 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 3, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_maps * 4 * 4 * 4, 1),
            # Note: we return raw logits and use BCEWithLogitsLoss in training.
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if img.dim() != 4 or img.size(1) != 1:
            raise ValueError(f"img must be (B, 1, H, W); got {tuple(img.shape)}")
        if labels.dim() != 1:
            raise ValueError(f"labels must be 1D; got {tuple(labels.shape)}")

        b = img.size(0)
        y = self.label_emb(labels)                       # (B, embed_dim)
        y = self.label_to_map(y)                         # (B, H*W)
        y = y.view(b, 1, self.img_size, self.img_size)   # (B, 1, H, W)
        x = torch.cat([img, y], dim=1)                   # (B, 2, H, W)
        x = self.net(x)
        return self.head(x).view(b)                      # (B,) logits
