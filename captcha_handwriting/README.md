# Synthetic Handwriting Generator for CAPTCHA Systems

Generate synthetic, class-conditional MNIST-style digits for a next-generation
CAPTCHA classifier. The project compares two conditional generative models:

* A **Conditional GAN (cGAN)** — fast sampling, one forward pass per image.
* A **Conditional Diffusion** model — higher diversity, many denoising steps per image.

Both are trained on MNIST, then evaluated with **FID** against the real test
set and with a **downstream CNN** trained on synthetic-only data and tested on
real MNIST.

## Layout

```
captcha_handwriting/
|-- data/
|   `-- dataloader.py            MNIST loader, normalizes to [-1, 1]
|-- model/
|   |-- cgan.py                  Generator + Discriminator
|   `-- diffusion.py             ConditionalUNet, timestep_embedding, ResidualBlock
|-- training/
|   |-- train_cgan.py            adversarial training loop, BCE w/ logits
|   `-- train_diffusion.py       linear/cosine schedule, q_sample, train_diffusion, sample_images
|-- utils/
|   |-- checkpoint.py            save_checkpoint / load_checkpoint
|   |-- metrics.py               InceptionV3 features + FID
|   `-- visualize.py             plot_image_grid / plot_class_grid / plot_comparison_grid
|-- 00_data_preparation.ipynb    download + visualize MNIST
|-- 01_cGAN_training.ipynb       train cGAN, plot class-conditional samples
|-- 02_diffusion_training.ipynb  train diffusion model, plot samples + trajectory
|-- 03_evaluation.ipynb          FID + downstream classifier accuracy
|-- build_notebooks.py           regenerate the .ipynb files from source
|-- requirements.txt
`-- README.md
```

## Setup

```bash
pip install -r requirements.txt
```

A GPU is strongly recommended for diffusion training and FID. The code falls
back to CPU automatically if no CUDA device is available — useful for
debugging but not for full training.

## How to run

Open the notebooks in order:

1. `00_data_preparation.ipynb` — sanity-check the MNIST loader.
2. `01_cGAN_training.ipynb` — train cGAN, view class-conditional samples.
3. `02_diffusion_training.ipynb` — train diffusion model, view denoising trajectory.
4. `03_evaluation.ipynb` — load both checkpoints, compute FID, train downstream CNN.

Checkpoints are written under `checkpoints/cgan/` and `checkpoints/diffusion/`.
`03_evaluation.ipynb` picks the highest-epoch checkpoint from each directory.

## Rubric mapping

| Rubric criterion                                                  | Where it's covered |
|-------------------------------------------------------------------|--------------------|
| cGAN: noise + class embedding concatenated in G                   | `model/cgan.py` :: `Generator.forward` |
| cGAN: label fused with image input in D                           | `model/cgan.py` :: `Discriminator.forward` |
| Generator output uses Tanh -> 1x28x28 in [-1, 1]                  | `model/cgan.py` last layer |
| Conditional UNet with time-step + class embedding, skip connections | `model/diffusion.py` :: `ConditionalUNet` |
| BCE adversarial training, separate D / G steps                    | `training/train_cgan.py` :: `train_cgan` |
| Grid of generated digits 0-9                                      | `01_cGAN_training.ipynb` final cells |
| Forward noising via linear / cosine schedule                      | `training/train_diffusion.py` :: `linear_beta_schedule`, `cosine_beta_schedule`, `q_sample` |
| Reverse iterative denoising T -> 0                                | `training/train_diffusion.py` :: `sample_images` |
| MSE noise-prediction loss                                         | `training/train_diffusion.py` :: `train_diffusion` |
| FID for both models                                               | `03_evaluation.ipynb` FID section |
| Side-by-side Real vs cGAN vs Diffusion (>= 5 digits)              | `03_evaluation.ipynb` comparison grid (10 digits) |
| Analysis of fidelity and diversity                                | `03_evaluation.ipynb` analysis markdown cell |
| CNN trained on synthetic only, evaluated on real MNIST            | `03_evaluation.ipynb` downstream utility section |
| Final accuracy reported                                           | `03_evaluation.ipynb` summary cell |

## Stand-out extensions

* **Latent walk** — `01_cGAN_training.ipynb` interpolates between two latent
  vectors with a fixed class label.
* **Cosine schedule** — `cosine_beta_schedule` ready to drop into
  `train_diffusion(... schedule='cosine')` for an FID comparison.
* **Multi-digit CAPTCHA** — `03_evaluation.ipynb` includes a `make_captcha`
  helper that composes four digits with whichever generator you pick.
* **Denoising trajectory** — `02_diffusion_training.ipynb` plots 10 snapshots
  from `t=T` down to `t=0` so you can see noise resolving into a digit.
