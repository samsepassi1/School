"""Regenerate the four project notebooks from inline cell definitions.

Run from the project root:
    python build_notebooks.py

Each notebook is built from a list of (kind, source) tuples below.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text: str) -> tuple[str, str]:
    return ("markdown", text)


def code(text: str) -> tuple[str, str]:
    return ("code", text)


def build_notebook(cells: list[tuple[str, str]]) -> dict:
    nb_cells = []
    for kind, src in cells:
        if kind == "markdown":
            nb_cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": src.splitlines(keepends=True),
                }
            )
        elif kind == "code":
            nb_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": src.splitlines(keepends=True),
                }
            )
        else:
            raise ValueError(kind)

    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# 00 — Data Preparation
# ---------------------------------------------------------------------------

NB_00 = [
    md(
        "# 00 · Data Preparation\n\n"
        "Load MNIST, apply the standard normalization to `[-1, 1]`, and "
        "visualize a batch of digits. The transform here is the same one used "
        "to train the cGAN and the diffusion model, so the generators learn to "
        "produce images in the same range that a `Tanh` activation outputs."
    ),
    code(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "# Allow `import data`, `import model`, etc. when running the notebook\n"
        "# from the project root.\n"
        "ROOT = Path.cwd()\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n"
        "\n"
        "import torch\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from data.dataloader import get_mnist_loaders\n"
        "from utils.visualize import plot_image_grid\n"
        "\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "print('device:', device)"
    ),
    md("## Load MNIST"),
    code(
        "train_loader, test_loader = get_mnist_loaders(batch_size=64, num_workers=0)\n"
        "print(f'train batches: {len(train_loader)} | test batches: {len(test_loader)}')\n"
        "\n"
        "images, labels = next(iter(train_loader))\n"
        "print('batch shape:', images.shape, '| dtype:', images.dtype)\n"
        "print('value range:', images.min().item(), '..', images.max().item())\n"
        "print('labels:', labels[:16].tolist())"
    ),
    md("## Visualize a batch\n\nThe images are in `[-1, 1]`; `plot_image_grid` shifts them back to `[0, 1]` for display."),
    code(
        "fig = plot_image_grid(images[:32], labels=labels[:32].tolist(), n_cols=8, title='MNIST sample batch')\n"
        "plt.show()"
    ),
    md(
        "## Per-class samples\n\n"
        "Confirm class balance and that the conditioning labels match the images. "
        "If the dataset is imbalanced (it isn't for MNIST), the generators may "
        "still memorize the majority classes well — something to keep in mind "
        "for non-MNIST extensions."
    ),
    code(
        "import collections\n"
        "\n"
        "samples_per_class = collections.defaultdict(list)\n"
        "for img, lab in zip(images, labels):\n"
        "    if len(samples_per_class[int(lab)]) < 1:\n"
        "        samples_per_class[int(lab)].append(img)\n"
        "    if len(samples_per_class) == 10 and all(len(v) >= 1 for v in samples_per_class.values()):\n"
        "        break\n"
        "\n"
        "# Fill in any missing classes from the next few batches.\n"
        "if not all(c in samples_per_class for c in range(10)):\n"
        "    for img, lab in zip(*next(iter(train_loader))):\n"
        "        samples_per_class.setdefault(int(lab), []).append(img)\n"
        "\n"
        "grid = torch.stack([samples_per_class[c][0] for c in range(10)])\n"
        "fig = plot_image_grid(grid, labels=list(range(10)), n_cols=10, title='One sample per class')\n"
        "plt.show()"
    ),
    md(
        "**Next:**\n"
        "* `01_cGAN_training.ipynb` — train the Conditional GAN\n"
        "* `02_diffusion_training.ipynb` — train the Conditional Diffusion model\n"
        "* `03_evaluation.ipynb` — FID + downstream classifier comparison"
    ),
]


# ---------------------------------------------------------------------------
# 01 — cGAN training
# ---------------------------------------------------------------------------

NB_01 = [
    md(
        "# 01 · Conditional GAN Training\n\n"
        "Train a class-conditional GAN on MNIST. The Generator concatenates a "
        "learned class embedding with the latent noise vector, and the "
        "Discriminator fuses the same embedding with the image as an extra "
        "input channel. Loss is the standard BCE with logits — D pushes real "
        "logits toward 1 and fake logits toward 0, G uses the non-saturating "
        "form (label = 1 for its own samples)."
    ),
    code(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "ROOT = Path.cwd()\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n"
        "\n"
        "import torch\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from data.dataloader import get_mnist_loaders\n"
        "from model.cgan import Generator, Discriminator\n"
        "from training.train_cgan import train_cgan\n"
        "from utils.visualize import plot_image_grid, plot_training_history\n"
        "\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "print('device:', device)"
    ),
    md("## Hyperparameters"),
    code(
        "LATENT_DIM = 100\n"
        "NUM_CLASSES = 10\n"
        "BATCH_SIZE = 128\n"
        "EPOCHS = 20            # bump to 30-50 for stronger samples\n"
        "LR = 2e-4\n"
        "BETAS = (0.5, 0.999)   # standard DCGAN choice"
    ),
    md("## Build the data + models"),
    code(
        "train_loader, _ = get_mnist_loaders(batch_size=BATCH_SIZE, num_workers=2)\n"
        "\n"
        "G = Generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)\n"
        "D = Discriminator(num_classes=NUM_CLASSES).to(device)\n"
        "\n"
        "n_params_g = sum(p.numel() for p in G.parameters())\n"
        "n_params_d = sum(p.numel() for p in D.parameters())\n"
        "print(f'G params: {n_params_g:,} | D params: {n_params_d:,}')\n"
        "\n"
        "# Quick shape sanity check before training\n"
        "z = torch.randn(4, LATENT_DIM, device=device)\n"
        "y = torch.randint(0, NUM_CLASSES, (4,), device=device)\n"
        "fake = G(z, y)\n"
        "logits = D(fake, y)\n"
        "print('fake:', fake.shape, '| D logits:', logits.shape)"
    ),
    md(
        "## Train\n\n"
        "Checkpoints are written under `checkpoints/cgan/` every 5 epochs and at "
        "the final epoch. Training prints the average D / G loss per epoch."
    ),
    code(
        "history = train_cgan(\n"
        "    G,\n"
        "    D,\n"
        "    train_loader,\n"
        "    epochs=EPOCHS,\n"
        "    lr=LR,\n"
        "    beta1=BETAS[0],\n"
        "    beta2=BETAS[1],\n"
        "    latent_dim=LATENT_DIM,\n"
        "    num_classes=NUM_CLASSES,\n"
        "    device=device,\n"
        "    checkpoint_dir='checkpoints/cgan',\n"
        "    log_every=200,\n"
        ")"
    ),
    code(
        "fig = plot_training_history(history, title='cGAN losses')\n"
        "plt.show()"
    ),
    md(
        "## Class-conditional samples\n\n"
        "Generate one row per digit (0–9). If the Generator has learned the "
        "label conditioning, each row should look like the corresponding digit."
    ),
    code(
        "G.eval()\n"
        "n_per_class = 8\n"
        "with torch.no_grad():\n"
        "    z = torch.randn(NUM_CLASSES * n_per_class, LATENT_DIM, device=device)\n"
        "    y = torch.arange(NUM_CLASSES, device=device).repeat_interleave(n_per_class)\n"
        "    samples = G(z, y).cpu()\n"
        "\n"
        "fig = plot_image_grid(samples, labels=y.cpu().tolist(), n_cols=n_per_class,\n"
        "                     title='cGAN: 8 samples per digit class')\n"
        "plt.show()"
    ),
    md(
        "## Bonus: latent-space walk\n\n"
        "Hold the class fixed and linearly interpolate between two noise vectors. "
        "A well-trained Generator should produce a smooth transition rather "
        "than jumping discretely between samples."
    ),
    code(
        "def latent_walk(generator, label, steps=10, latent_dim=LATENT_DIM, device=device):\n"
        "    z0 = torch.randn(1, latent_dim, device=device)\n"
        "    z1 = torch.randn(1, latent_dim, device=device)\n"
        "    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)\n"
        "    z = (1 - alphas) * z0 + alphas * z1\n"
        "    y = torch.full((steps,), label, dtype=torch.long, device=device)\n"
        "    with torch.no_grad():\n"
        "        return generator(z, y).cpu()\n"
        "\n"
        "walk = latent_walk(G, label=7, steps=10)\n"
        "fig = plot_image_grid(walk, n_cols=10, title='Latent walk for class 7')\n"
        "plt.show()"
    ),
]


# ---------------------------------------------------------------------------
# 02 — Diffusion training
# ---------------------------------------------------------------------------

NB_02 = [
    md(
        "# 02 · Conditional Diffusion Training\n\n"
        "Train a class-conditional UNet to predict the noise added at a random "
        "diffusion step. Sampling runs the reverse DDPM update from `t=T-1` to "
        "`t=0`. The same `DiffusionSchedule` object is used for both training "
        "(forward `q_sample`) and inference (`sample_images`)."
    ),
    code(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "ROOT = Path.cwd()\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n"
        "\n"
        "import torch\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from data.dataloader import get_mnist_loaders\n"
        "from model.diffusion import ConditionalUNet\n"
        "from training.train_diffusion import (\n"
        "    linear_beta_schedule, cosine_beta_schedule,\n"
        "    DiffusionSchedule, train_diffusion, sample_images,\n"
        ")\n"
        "from utils.visualize import plot_image_grid, plot_training_history\n"
        "\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "print('device:', device)"
    ),
    md("## Hyperparameters"),
    code(
        "NUM_CLASSES = 10\n"
        "BATCH_SIZE = 128\n"
        "EPOCHS = 15            # 20-30 gives noticeably crisper digits\n"
        "LR = 2e-4\n"
        "NUM_TIMESTEPS = 1000\n"
        "SCHEDULE = 'linear'    # try 'cosine' for the stand-out variant"
    ),
    md("## Build the data + model"),
    code(
        "train_loader, _ = get_mnist_loaders(batch_size=BATCH_SIZE, num_workers=2)\n"
        "\n"
        "unet = ConditionalUNet(num_classes=NUM_CLASSES, base_channels=64, time_dim=128).to(device)\n"
        "print(f'UNet params: {sum(p.numel() for p in unet.parameters()):,}')\n"
        "\n"
        "# Shape sanity check before training\n"
        "x = torch.randn(4, 1, 28, 28, device=device)\n"
        "t = torch.randint(0, NUM_TIMESTEPS, (4,), device=device)\n"
        "y = torch.randint(0, NUM_CLASSES, (4,), device=device)\n"
        "out = unet(x, t, y)\n"
        "print('pred shape:', out.shape, '(should equal input shape)')"
    ),
    md(
        "## Train\n\n"
        "Training samples a random timestep `t` per example, adds the matching "
        "amount of Gaussian noise via `q_sample`, and minimizes MSE between the "
        "predicted noise and the actual noise."
    ),
    code(
        "history = train_diffusion(\n"
        "    unet,\n"
        "    train_loader,\n"
        "    epochs=EPOCHS,\n"
        "    lr=LR,\n"
        "    num_timesteps=NUM_TIMESTEPS,\n"
        "    schedule=SCHEDULE,\n"
        "    device=device,\n"
        "    checkpoint_dir='checkpoints/diffusion',\n"
        "    log_every=200,\n"
        ")"
    ),
    code(
        "fig = plot_training_history(history, title='Diffusion MSE')\n"
        "plt.show()"
    ),
    md(
        "## Sample class-conditional digits\n\n"
        "Run the reverse diffusion loop starting from pure Gaussian noise. We "
        "request 8 samples per class so each row in the grid corresponds to "
        "one digit identity."
    ),
    code(
        "betas = linear_beta_schedule(NUM_TIMESTEPS) if SCHEDULE == 'linear' else cosine_beta_schedule(NUM_TIMESTEPS)\n"
        "schedule = DiffusionSchedule.from_betas(betas).to(device)\n"
        "\n"
        "n_per_class = 8\n"
        "labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(n_per_class)\n"
        "samples = sample_images(unet, labels, schedule).cpu()\n"
        "\n"
        "fig = plot_image_grid(samples, labels=labels.cpu().tolist(), n_cols=n_per_class,\n"
        "                     title='Diffusion: 8 samples per digit class')\n"
        "plt.show()"
    ),
    md(
        "## Bonus: visualize the denoising trajectory\n\n"
        "Plot snapshots from the reverse process so you can see noise gradually "
        "resolving into a digit. We subsample 10 evenly spaced snapshots from "
        "the full T-step chain."
    ),
    code(
        "label = torch.tensor([3], device=device)\n"
        "traj = sample_images(unet, label, schedule, return_trajectory=True)\n"
        "indices = torch.linspace(0, len(traj) - 1, 10).long().tolist()\n"
        "snapshots = torch.stack([traj[i][0] for i in indices])\n"
        "fig = plot_image_grid(snapshots, n_cols=10, title='Reverse diffusion (class 3, T=0 ... T=1000)')\n"
        "plt.show()"
    ),
]


# ---------------------------------------------------------------------------
# 03 — Evaluation
# ---------------------------------------------------------------------------

NB_03 = [
    md(
        "# 03 · Evaluation\n\n"
        "We compare the two trained generators on two complementary axes:\n\n"
        "1. **Fidelity & diversity** — Frechet Inception Distance (FID) against the real MNIST test set.\n"
        "2. **Utility for downstream tasks** — train a small CNN on *synthetic-only* data and report its accuracy on the *real* MNIST test set.\n"
        "\n"
        "We also produce a side-by-side `Real vs cGAN vs Diffusion` grid for a "
        "qualitative comparison."
    ),
    code(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "ROOT = Path.cwd()\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n"
        "\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "from torch.utils.data import DataLoader, TensorDataset\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from data.dataloader import get_mnist_loaders\n"
        "from model.cgan import Generator\n"
        "from model.diffusion import ConditionalUNet\n"
        "from training.train_diffusion import (\n"
        "    linear_beta_schedule, DiffusionSchedule, sample_images,\n"
        ")\n"
        "from utils.checkpoint import load_checkpoint\n"
        "from utils.metrics import compute_fid, InceptionFeatures\n"
        "from utils.visualize import plot_comparison_grid\n"
        "\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "print('device:', device)"
    ),
    md("## Load the trained models\n\nReplace the checkpoint paths with whichever epoch you trained to."),
    code(
        "NUM_CLASSES = 10\n"
        "LATENT_DIM = 100\n"
        "NUM_TIMESTEPS = 1000\n"
        "\n"
        "G = Generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)\n"
        "unet = ConditionalUNet(num_classes=NUM_CLASSES, base_channels=64, time_dim=128).to(device)\n"
        "\n"
        "# Pick the latest checkpoints from each training run\n"
        "cgan_ckpts = sorted(Path('checkpoints/cgan').glob('cgan_epoch_*.pt'))\n"
        "diff_ckpts = sorted(Path('checkpoints/diffusion').glob('diffusion_epoch_*.pt'))\n"
        "print('cgan ckpt :', cgan_ckpts[-1] if cgan_ckpts else 'MISSING')\n"
        "print('diff ckpt :', diff_ckpts[-1] if diff_ckpts else 'MISSING')\n"
        "\n"
        "load_checkpoint(cgan_ckpts[-1], models={'generator': G}, map_location=device)\n"
        "load_checkpoint(diff_ckpts[-1], models={'unet': unet}, map_location=device)\n"
        "G.eval(); unet.eval()"
    ),
    md("## Generate large synthetic batches per class"),
    code(
        "@torch.no_grad()\n"
        "def cgan_generate(n_per_class: int = 200) -> tuple[torch.Tensor, torch.Tensor]:\n"
        "    \"\"\"Return (N, 1, 28, 28) images and (N,) labels from the cGAN.\"\"\"\n"
        "    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(n_per_class)\n"
        "    z = torch.randn(labels.size(0), LATENT_DIM, device=device)\n"
        "    return G(z, labels).cpu(), labels.cpu()\n"
        "\n"
        "betas = linear_beta_schedule(NUM_TIMESTEPS)\n"
        "schedule = DiffusionSchedule.from_betas(betas).to(device)\n"
        "\n"
        "@torch.no_grad()\n"
        "def diffusion_generate(n_per_class: int = 100, batch: int = 200) -> tuple[torch.Tensor, torch.Tensor]:\n"
        "    \"\"\"Return synthetic samples from the diffusion model.\n"
        "\n"
        "    Diffusion sampling is expensive (T denoising steps per sample), so we\n"
        "    generate in batches of `batch` images at a time.\n"
        "    \"\"\"\n"
        "    all_imgs, all_labels = [], []\n"
        "    full_labels = torch.arange(NUM_CLASSES).repeat_interleave(n_per_class)\n"
        "    for i in range(0, full_labels.size(0), batch):\n"
        "        chunk = full_labels[i : i + batch].to(device)\n"
        "        imgs = sample_images(unet, chunk, schedule).cpu()\n"
        "        all_imgs.append(imgs)\n"
        "        all_labels.append(chunk.cpu())\n"
        "    return torch.cat(all_imgs), torch.cat(all_labels)\n"
        "\n"
        "# Use a smaller diffusion sample count if you're CPU-bound.\n"
        "cgan_imgs, cgan_labels = cgan_generate(n_per_class=200)\n"
        "diff_imgs, diff_labels = diffusion_generate(n_per_class=100)\n"
        "print('cgan samples :', cgan_imgs.shape)\n"
        "print('diff samples :', diff_imgs.shape)"
    ),
    md("## Qualitative comparison: Real vs cGAN vs Diffusion"),
    code(
        "_, test_loader = get_mnist_loaders(batch_size=512, num_workers=0)\n"
        "real_imgs, real_labels = next(iter(test_loader))\n"
        "\n"
        "def first_per_class(images: torch.Tensor, labels: torch.Tensor, classes=range(10)) -> dict[int, torch.Tensor]:\n"
        "    by_class = {}\n"
        "    for c in classes:\n"
        "        mask = labels == c\n"
        "        if mask.any():\n"
        "            by_class[c] = images[mask][:1]\n"
        "    return by_class\n"
        "\n"
        "panels = {\n"
        "    'Real':      first_per_class(real_imgs, real_labels),\n"
        "    'cGAN':      first_per_class(cgan_imgs, cgan_labels),\n"
        "    'Diffusion': first_per_class(diff_imgs, diff_labels),\n"
        "}\n"
        "fig = plot_comparison_grid(panels, classes=range(10), title='Real vs cGAN vs Diffusion')\n"
        "plt.show()"
    ),
    md(
        "## Quantitative: FID\n\n"
        "We compute FID against a fixed reference sample of real MNIST test "
        "images. Lower is better — FID values for MNIST-like models typically "
        "sit in the 5–50 range depending on training budget."
    ),
    code(
        "# Build a fixed real reference set the same size as our synthetic batches.\n"
        "n_ref = min(2000, real_imgs.size(0))\n"
        "real_ref = real_imgs[:n_ref]\n"
        "print('real reference:', real_ref.shape)\n"
        "\n"
        "extractor = InceptionFeatures().to(device).eval()\n"
        "\n"
        "fid_cgan = compute_fid(real_ref, cgan_imgs, extractor=extractor, device=device, batch_size=64)\n"
        "fid_diff = compute_fid(real_ref, diff_imgs, extractor=extractor, device=device, batch_size=64)\n"
        "\n"
        "print(f'FID cGAN      : {fid_cgan:8.3f}')\n"
        "print(f'FID Diffusion : {fid_diff:8.3f}')"
    ),
    md(
        "### Analysis\n\n"
        "* **Higher fidelity:** the model with the lower FID score produces samples whose Inception features more closely match the real distribution.\n"
        "* **Diversity:** GANs are known to *mode collapse* (producing too few distinct images per class), which inflates FID even when individual samples look sharp. Diffusion models tend to be more diverse but slower to sample.\n"
        "* **Practical suitability:** for *large-scale* data generation, cGAN sampling is one forward pass per image while diffusion needs `T` passes — typically two to three orders of magnitude slower. If both FIDs are acceptable, the cGAN is the obvious choice for a CAPTCHA pipeline that needs millions of images.\n"
        "\n"
        "Fill in the actual observed FID values and the qualitative read of the comparison grid above when reporting results."
    ),
    md(
        "## Downstream utility — CNN trained on synthetic, evaluated on real\n\n"
        "If the synthetic samples really capture the structure of MNIST, a "
        "classifier trained on them alone should still generalize to the real "
        "test set. We train one small CNN per synthetic source and report top-1 "
        "accuracy on the real MNIST test set."
    ),
    code(
        "class SimpleCNN(nn.Module):\n"
        "    def __init__(self, num_classes: int = 10) -> None:\n"
        "        super().__init__()\n"
        "        self.features = nn.Sequential(\n"
        "            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 14x14\n"
        "            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), # 7x7\n"
        "        )\n"
        "        self.head = nn.Sequential(\n"
        "            nn.Flatten(),\n"
        "            nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),\n"
        "            nn.Linear(128, num_classes),\n"
        "        )\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.head(self.features(x))\n"
        "\n"
        "\n"
        "def train_classifier(images: torch.Tensor, labels: torch.Tensor,\n"
        "                    *, epochs: int = 5, batch_size: int = 128) -> SimpleCNN:\n"
        "    ds = TensorDataset(images, labels)\n"
        "    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)\n"
        "    model = SimpleCNN().to(device)\n"
        "    opt = torch.optim.Adam(model.parameters(), lr=1e-3)\n"
        "    for ep in range(1, epochs + 1):\n"
        "        model.train()\n"
        "        total, correct, loss_sum = 0, 0, 0.0\n"
        "        for x, y in loader:\n"
        "            x, y = x.to(device), y.to(device)\n"
        "            logits = model(x)\n"
        "            loss = F.cross_entropy(logits, y)\n"
        "            opt.zero_grad(); loss.backward(); opt.step()\n"
        "            total += y.size(0)\n"
        "            correct += (logits.argmax(1) == y).sum().item()\n"
        "            loss_sum += loss.item() * y.size(0)\n"
        "        print(f'  ep {ep}: loss={loss_sum / total:.4f} acc={correct / total:.4f}')\n"
        "    return model\n"
        "\n"
        "\n"
        "@torch.no_grad()\n"
        "def evaluate(model: SimpleCNN, loader: DataLoader) -> float:\n"
        "    model.eval()\n"
        "    total, correct = 0, 0\n"
        "    for x, y in loader:\n"
        "        x, y = x.to(device), y.to(device)\n"
        "        correct += (model(x).argmax(1) == y).sum().item()\n"
        "        total += y.size(0)\n"
        "    return correct / total"
    ),
    code(
        "print('--- training on cGAN samples ---')\n"
        "clf_cgan = train_classifier(cgan_imgs, cgan_labels, epochs=5)\n"
        "acc_cgan = evaluate(clf_cgan, test_loader)\n"
        "print(f'cGAN-trained CNN on real MNIST test: {acc_cgan:.4f}')\n"
        "\n"
        "print('--- training on Diffusion samples ---')\n"
        "clf_diff = train_classifier(diff_imgs, diff_labels, epochs=5)\n"
        "acc_diff = evaluate(clf_diff, test_loader)\n"
        "print(f'Diffusion-trained CNN on real MNIST test: {acc_diff:.4f}')"
    ),
    md(
        "## Summary\n\n"
        "| Metric                         | cGAN | Diffusion |\n"
        "|--------------------------------|------|-----------|\n"
        "| FID (lower better)             | …    | …         |\n"
        "| Downstream test accuracy       | …    | …         |\n"
        "| Sampling cost / image          | 1 forward pass | T forward passes |\n"
        "\n"
        "Fill in the numeric cells with your run's results, then decide which "
        "model best fits SuperCognition's CAPTCHA pipeline: typically the cGAN "
        "wins on throughput while diffusion wins on diversity. For an "
        "operational system you would combine both — diffusion for a curated "
        "high-quality seed set, cGAN for bulk augmentation."
    ),
    md(
        "## Optional stand-out extension: multi-digit CAPTCHA composition\n\n"
        "Stitch four single-digit samples together into a CAPTCHA-style image."
    ),
    code(
        "def make_captcha(digits: str, source: str = 'cgan') -> torch.Tensor:\n"
        "    labels = torch.tensor([int(c) for c in digits], device=device)\n"
        "    if source == 'cgan':\n"
        "        with torch.no_grad():\n"
        "            z = torch.randn(labels.size(0), LATENT_DIM, device=device)\n"
        "            imgs = G(z, labels)\n"
        "    elif source == 'diffusion':\n"
        "        imgs = sample_images(unet, labels, schedule)\n"
        "    else:\n"
        "        raise ValueError(source)\n"
        "    return torch.cat([img for img in imgs.squeeze(1)], dim=1)  # concatenate horizontally\n"
        "\n"
        "captcha = make_captcha('7392', source='cgan')\n"
        "plt.figure(figsize=(6, 2))\n"
        "plt.imshow(((captcha.clamp(-1, 1) + 1) / 2).cpu(), cmap='gray', vmin=0, vmax=1)\n"
        "plt.axis('off')\n"
        "plt.title('cGAN CAPTCHA: 7392')\n"
        "plt.show()"
    ),
]


def main() -> None:
    targets = {
        "00_data_preparation.ipynb": NB_00,
        "01_cGAN_training.ipynb": NB_01,
        "02_diffusion_training.ipynb": NB_02,
        "03_evaluation.ipynb": NB_03,
    }
    for fname, cells in targets.items():
        nb = build_notebook(cells)
        out = HERE / fname
        out.write_text(json.dumps(nb, indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
