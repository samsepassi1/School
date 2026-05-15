"""Adversarial training loop for the Conditional GAN."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.cgan import Generator, Discriminator
from utils.checkpoint import save_checkpoint


def _sample_noise_and_labels(
    batch_size: int,
    latent_dim: int,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.randn(batch_size, latent_dim, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    return z, y


def train_cgan(
    generator: Generator,
    discriminator: Discriminator,
    train_loader: DataLoader,
    *,
    epochs: int = 20,
    lr: float = 2e-4,
    beta1: float = 0.5,
    beta2: float = 0.999,
    latent_dim: int = 100,
    num_classes: int = 10,
    device: torch.device | str = "cuda",
    checkpoint_dir: str | Path | None = "checkpoints/cgan",
    checkpoint_every: int = 5,
    log_every: int = 100,
    on_epoch_end: Callable | None = None,
) -> dict:
    """Train a conditional GAN with the standard non-saturating BCE loss.

    Returns a dict containing the per-epoch loss history.
    """
    device = torch.device(device)
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    history = {"d_loss": [], "g_loss": []}

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        epoch_d, epoch_g, steps = 0.0, 0.0, 0
        for step, (real_imgs, real_labels) in enumerate(train_loader, start=1):
            real_imgs = real_imgs.to(device, non_blocking=True)
            real_labels = real_labels.to(device, non_blocking=True)
            bs = real_imgs.size(0)

            # ---------------------------------------------------------------
            # 1) Discriminator update — push real toward 1 and fake toward 0.
            # ---------------------------------------------------------------
            opt_d.zero_grad(set_to_none=True)

            real_targets = torch.ones(bs, device=device)
            fake_targets = torch.zeros(bs, device=device)

            d_real_logits = discriminator(real_imgs, real_labels)
            d_loss_real = criterion(d_real_logits, real_targets)

            z, fake_labels = _sample_noise_and_labels(bs, latent_dim, num_classes, device)
            with torch.no_grad():
                fake_imgs = generator(z, fake_labels)
            d_fake_logits = discriminator(fake_imgs, fake_labels)
            d_loss_fake = criterion(d_fake_logits, fake_targets)

            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_loss.backward()
            opt_d.step()

            # ---------------------------------------------------------------
            # 2) Generator update — non-saturating loss, pretend fakes are real.
            # ---------------------------------------------------------------
            opt_g.zero_grad(set_to_none=True)
            z, gen_labels = _sample_noise_and_labels(bs, latent_dim, num_classes, device)
            gen_imgs = generator(z, gen_labels)
            g_logits = discriminator(gen_imgs, gen_labels)
            g_loss = criterion(g_logits, torch.ones(bs, device=device))
            g_loss.backward()
            opt_g.step()

            epoch_d += float(d_loss.item())
            epoch_g += float(g_loss.item())
            steps += 1

            if log_every and step % log_every == 0:
                print(
                    f"  epoch {epoch:3d} | step {step:4d} | "
                    f"D {d_loss.item():.4f} | G {g_loss.item():.4f}"
                )

        avg_d = epoch_d / max(steps, 1)
        avg_g = epoch_g / max(steps, 1)
        history["d_loss"].append(avg_d)
        history["g_loss"].append(avg_g)
        print(f"epoch {epoch:3d}/{epochs} | D={avg_d:.4f} | G={avg_g:.4f}")

        if on_epoch_end is not None:
            on_epoch_end(epoch, generator, discriminator)

        if checkpoint_dir is not None and (
            epoch % checkpoint_every == 0 or epoch == epochs
        ):
            save_checkpoint(
                checkpoint_dir / f"cgan_epoch_{epoch:03d}.pt",
                models={"generator": generator, "discriminator": discriminator},
                optimizers={"opt_g": opt_g, "opt_d": opt_d},
                extra={"epoch": epoch, "latent_dim": latent_dim, "num_classes": num_classes},
            )

    return history
