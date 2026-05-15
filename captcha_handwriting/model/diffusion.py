"""Conditional U-Net for class-conditional diffusion on MNIST.

The model predicts the noise added to a clean image at diffusion step `t`,
conditioned on a class label `y`. We use:

* Sinusoidal time-step embeddings projected through a small MLP.
* A label embedding added to the time embedding so every residual block
  receives a joint conditioning vector.
* GroupNorm + SiLU residual blocks with FiLM-style conditioning (the
  conditioning vector is projected to a per-channel bias added between two
  convolutions).
* Skip connections between matching resolutions in the down and up paths.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal time-step embedding, matching the original DDPM paper.

    Args:
        t:   (B,) integer or float tensor of diffusion steps.
        dim: embedding dimension (should be even).

    Returns:
        (B, dim) tensor of embeddings.
    """
    if dim % 2 != 0:
        raise ValueError(f"timestep_embedding dim must be even, got {dim}")
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(0, half, device=device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    """Conv + GroupNorm + SiLU residual block with time/label conditioning."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8) -> None:
        super().__init__()
        groups_in = math.gcd(groups, in_ch)
        groups_out = math.gcd(groups, out_ch)

        self.norm1 = nn.GroupNorm(groups_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.cond_proj = nn.Linear(cond_dim, out_ch)

        self.norm2 = nn.GroupNorm(groups_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # FiLM-style additive conditioning broadcast across spatial dims.
        h = h + self.cond_proj(F.silu(cond))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class ConditionalUNet(nn.Module):
    """Encoder-decoder UNet for class-conditional noise prediction on 28x28 images.

    The 28x28 grid is downsampled twice (28 -> 14 -> 7) and upsampled symmetrically
    back to 28x28. Skip connections concatenate the matching down-path features
    at each up step so high-frequency detail is preserved.
    """

    def __init__(
        self,
        num_classes: int = 10,
        base_channels: int = 64,
        time_dim: int = 128,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.time_dim = time_dim

        # Time and label embeddings combined into a single conditioning vector.
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.label_emb = nn.Embedding(num_classes, time_dim)

        ch1, ch2, ch3 = base_channels, base_channels * 2, base_channels * 2

        self.stem = nn.Conv2d(in_channels, ch1, kernel_size=3, padding=1)

        # Down path
        self.down1 = ResidualBlock(ch1, ch1, time_dim)
        self.downsample1 = Downsample(ch1)           # 28 -> 14
        self.down2 = ResidualBlock(ch1, ch2, time_dim)
        self.downsample2 = Downsample(ch2)           # 14 -> 7
        self.down3 = ResidualBlock(ch2, ch3, time_dim)

        # Bottleneck
        self.mid1 = ResidualBlock(ch3, ch3, time_dim)
        self.mid2 = ResidualBlock(ch3, ch3, time_dim)

        # Up path. Each block receives skip-concatenated features, hence 2x channels.
        self.up3 = ResidualBlock(ch3 + ch3, ch2, time_dim)
        self.upsample2 = Upsample(ch2)               # 7 -> 14
        self.up2 = ResidualBlock(ch2 + ch2, ch1, time_dim)
        self.upsample1 = Upsample(ch1)               # 14 -> 28
        self.up1 = ResidualBlock(ch1 + ch1, ch1, time_dim)

        self.out_norm = nn.GroupNorm(math.gcd(8, ch1), ch1)
        self.out_conv = nn.Conv2d(ch1, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict the noise that was added to produce `x` at step `t` for class `y`.

        Args:
            x: (B, 1, 28, 28) noisy image.
            t: (B,) diffusion timestep indices.
            y: (B,) integer class labels.

        Returns:
            (B, 1, 28, 28) noise prediction.
        """
        # Conditioning: sinusoidal time -> MLP, plus class embedding.
        t_emb = timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        y_emb = self.label_emb(y)
        cond = t_emb + y_emb

        # Stem
        h0 = self.stem(x)                  # (B, c1, 28, 28)

        # Down path
        d1 = self.down1(h0, cond)          # (B, c1, 28, 28)   skip
        x_ = self.downsample1(d1)          # (B, c1, 14, 14)
        d2 = self.down2(x_, cond)          # (B, c2, 14, 14)   skip
        x_ = self.downsample2(d2)          # (B, c2, 7, 7)
        d3 = self.down3(x_, cond)          # (B, c3, 7, 7)     skip

        # Bottleneck
        m = self.mid1(d3, cond)
        m = self.mid2(m, cond)             # (B, c3, 7, 7)

        # Up path with skip concatenations
        u = self.up3(torch.cat([m, d3], dim=1), cond)    # (B, c2, 7, 7)
        u = self.upsample2(u)                            # (B, c2, 14, 14)
        u = self.up2(torch.cat([u, d2], dim=1), cond)    # (B, c1, 14, 14)
        u = self.upsample1(u)                            # (B, c1, 28, 28)
        u = self.up1(torch.cat([u, d1], dim=1), cond)    # (B, c1, 28, 28)

        out = self.out_conv(F.silu(self.out_norm(u)))    # (B, 1, 28, 28)
        return out
