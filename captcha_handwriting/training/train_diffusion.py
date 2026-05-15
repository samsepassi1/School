"""Training and sampling utilities for the Conditional Diffusion model.

Implements the DDPM forward / reverse processes:

* Forward:  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
* Reverse:  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1 - alpha_bar_t) * eps_theta)
            + sigma_t * z          (z = 0 when t == 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.diffusion import ConditionalUNet
from utils.checkpoint import save_checkpoint


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule from the original DDPM paper."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Improved-DDPM cosine schedule (Nichol & Dhariwal, 2021)."""
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps) / num_timesteps
    alphas_cumprod = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=0.999)


@dataclass
class DiffusionSchedule:
    """Container for all precomputed schedule tensors."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor

    @classmethod
    def from_betas(cls, betas: torch.Tensor) -> "DiffusionSchedule":
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        return cls(
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
            posterior_variance=posterior_variance,
        )

    def to(self, device: torch.device | str) -> "DiffusionSchedule":
        return DiffusionSchedule(
            **{k: v.to(device) for k, v in self.__dict__.items()}
        )

    @property
    def num_timesteps(self) -> int:
        return self.betas.shape[0]


def _extract(values: torch.Tensor, t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Gather schedule entries at `t` and reshape for broadcasting."""
    out = values.gather(0, t)
    return out.view(t.shape[0], *([1] * (len(shape) - 1)))


# ---------------------------------------------------------------------------
# Forward process
# ---------------------------------------------------------------------------

def q_sample(
    x_start: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x_t given a clean x_0 by adding scaled Gaussian noise."""
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_acp = _extract(schedule.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_acp = _extract(schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    x_t = sqrt_acp * x_start + sqrt_one_minus_acp * noise
    return x_t, noise


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_diffusion(
    model: ConditionalUNet,
    train_loader: DataLoader,
    *,
    epochs: int = 20,
    lr: float = 2e-4,
    num_timesteps: int = 1000,
    schedule: str = "linear",
    device: torch.device | str = "cuda",
    checkpoint_dir: str | Path | None = "checkpoints/diffusion",
    checkpoint_every: int = 5,
    log_every: int = 100,
    on_epoch_end: Callable | None = None,
) -> dict:
    """Train the noise-prediction UNet with MSE loss against the true noise."""
    device = torch.device(device)
    model.to(device)

    if schedule == "linear":
        betas = linear_beta_schedule(num_timesteps)
    elif schedule == "cosine":
        betas = cosine_beta_schedule(num_timesteps)
    else:
        raise ValueError(f"Unknown schedule: {schedule!r}")

    sched = DiffusionSchedule.from_betas(betas).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"loss": []}
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        running, steps = 0.0, 0
        for step, (imgs, labels) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bs = imgs.size(0)

            # 1) sample a random timestep per example
            t = torch.randint(0, num_timesteps, (bs,), device=device, dtype=torch.long)

            # 2) forward diffusion: add noise to the clean image
            x_t, noise = q_sample(imgs, t, sched)

            # 3) predict the noise and minimize MSE
            pred = model(x_t, t, labels)
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            steps += 1
            if log_every and step % log_every == 0:
                print(f"  epoch {epoch:3d} | step {step:4d} | loss {loss.item():.4f}")

        avg = running / max(steps, 1)
        history["loss"].append(avg)
        print(f"epoch {epoch:3d}/{epochs} | mse={avg:.4f}")

        if on_epoch_end is not None:
            on_epoch_end(epoch, model, sched)

        if checkpoint_dir is not None and (
            epoch % checkpoint_every == 0 or epoch == epochs
        ):
            save_checkpoint(
                checkpoint_dir / f"diffusion_epoch_{epoch:03d}.pt",
                models={"unet": model},
                optimizers={"opt": optimizer},
                extra={
                    "epoch": epoch,
                    "num_timesteps": num_timesteps,
                    "schedule": schedule,
                },
            )

    return history


# ---------------------------------------------------------------------------
# Reverse / sampling process
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_images(
    model: ConditionalUNet,
    labels: torch.Tensor,
    schedule: DiffusionSchedule,
    *,
    image_size: int = 28,
    channels: int = 1,
    device: torch.device | str | None = None,
    clip_denoised: bool = True,
    return_trajectory: bool = False,
) -> torch.Tensor | list[torch.Tensor]:
    """Run the reverse diffusion process from pure Gaussian noise.

    Args:
        model:      Trained ConditionalUNet.
        labels:     (B,) integer class labels to condition on.
        schedule:   Same schedule the model was trained against.
        image_size: spatial size of the output image (28 for MNIST).
        return_trajectory: if True returns a list of (B, C, H, W) snapshots
                           from t=T-1 .. t=0, otherwise just the final x_0.
    """
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    schedule = schedule.to(device)
    labels = labels.to(device)
    b = labels.shape[0]

    x = torch.randn(b, channels, image_size, image_size, device=device)
    traj = [x.clone()] if return_trajectory else None

    model.eval()
    T = schedule.num_timesteps
    for i in reversed(range(T)):
        t = torch.full((b,), i, device=device, dtype=torch.long)

        beta_t = _extract(schedule.betas, t, x.shape)
        alpha_t = _extract(schedule.alphas, t, x.shape)
        sqrt_one_minus_acp = _extract(schedule.sqrt_one_minus_alphas_cumprod, t, x.shape)

        eps = model(x, t, labels)

        # Posterior mean (eps-parameterization)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / sqrt_one_minus_acp) * eps)

        if i > 0:
            posterior_var = _extract(schedule.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(posterior_var) * noise
        else:
            x = mean

        if clip_denoised:
            # Optional: keep samples inside the training data range.
            x = x.clamp(-1.0, 1.0)

        if return_trajectory:
            traj.append(x.clone())

    return traj if return_trajectory else x
