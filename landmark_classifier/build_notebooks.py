"""
Script that builds the three project notebooks from inline cell definitions.
Keeping the cells as Python literals here makes them easy to review and edit.
"""

import json
from pathlib import Path


def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": src.splitlines(keepends=True),
    }


def nb(cells):
    return {
        "cells": cells,
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
# Notebook 1: CNN from scratch
# ---------------------------------------------------------------------------

PART1 = [
    md(
        "# Convolutional Neural Networks\n"
        "## Project: Landmark Classification & Tagging for Social Media\n\n"
        "**Part 1 of 3: build a CNN from scratch.**\n\n"
        "In this notebook you will:\n"
        "1. Build a data pipeline (`src/data.py`) and visualize it.\n"
        "2. Define a CNN from scratch in `src/model.py`.\n"
        "3. Define loss & optimizer in `src/optimization.py`.\n"
        "4. Implement training in `src/train.py`.\n"
        "5. Train and evaluate the model, then export it to TorchScript.\n"
    ),
    md("## 0. Set up the environment"),
    code(
        "%load_ext autoreload\n"
        "%autoreload 2\n"
        "\n"
        "from src.helpers import setup_env\n"
        "setup_env()\n"
    ),
    md(
        "## 1. Data\n"
        "Open `src/data.py` and complete the `get_data_loaders` and "
        "`visualize_one_batch` functions. Then run the tests below."
    ),
    code(
        "!pytest -vv src/data.py --no-header -x\n"
    ),
    md(
        "### 1.1 Visualize a batch\n"
        "Use the data loaders to grab a batch from the training set and "
        "display a few examples with their labels."
    ),
    code(
        "%matplotlib inline\n"
        "from src.data import get_data_loaders, visualize_one_batch\n"
        "\n"
        "data_loaders = get_data_loaders(batch_size=32)\n"
        "fig = visualize_one_batch(data_loaders, max_n=5)\n"
    ),
    md(
        "**Question 1**: Describe your data preprocessing and augmentation procedure.\n\n"
        "**Answer**: All splits use `Resize(256)` to keep aspect ratio while making the "
        "shorter side 256 pixels, then a 224×224 crop (random for train, center for "
        "valid/test) — 224 is the canonical input size for ImageNet-style CNNs. The "
        "training pipeline adds three augmentations between the crop and `ToTensor()`: "
        "`RandomHorizontalFlip(p=0.5)` (landmarks look fine mirrored), `RandomRotation(15)` "
        "(camera tilt invariance) and `ColorJitter` (different times of day and weather). "
        "Finally, all splits normalize by the per-channel mean and std computed once over "
        "the training set so the inputs are centered and scaled."
    ),
    md(
        "## 2. Model\n"
        "Open `src/model.py` and implement the `MyModel` class.\n\n"
        "**Question 2**: Outline the steps you took to arrive at your final architecture.\n\n"
        "**Answer**: I used a VGG-style backbone: 5 sequential blocks of "
        "(Conv 3×3 → BN → ReLU → Conv 3×3 → BN → ReLU → MaxPool 2×2). Each block "
        "doubles the channel count (3 → 32 → 64 → 128 → 256 → 512) and halves the "
        "spatial size (224 → 112 → 56 → 28 → 14 → 7). Two stacked 3×3 convs per block "
        "match the receptive field of a 5×5 conv with fewer parameters. BatchNorm "
        "stabilizes training and lets me use a higher learning rate. After the "
        "feature extractor an `AdaptiveAvgPool2d(1)` produces a 512-dim embedding, "
        "which is fed into a small MLP head (`Linear 512→256 → BN → ReLU → Dropout → "
        "Linear 256→num_classes`). Dropout in the head is the main regularizer against "
        "overfitting on the relatively small landmark dataset."
    ),
    code("!pytest -vv src/model.py --no-header -x\n"),
    md("## 3. Loss and optimizer\nComplete `src/optimization.py`."),
    code("!pytest -vv src/optimization.py --no-header -x\n"),
    md("## 4. Train and validate\nComplete `src/train.py`."),
    code("!pytest -vv src/train.py --no-header -x\n"),
    md("### 4.1 Putting it all together"),
    code(
        "import torch\n"
        "from src.data import get_data_loaders\n"
        "from src.model import MyModel\n"
        "from src.optimization import get_loss, get_optimizer\n"
        "from src.train import optimize\n"
        "\n"
        "batch_size = 64\n"
        "valid_size = 0.2\n"
        "num_epochs = 35\n"
        "num_classes = 50\n"
        "dropout = 0.4\n"
        "learning_rate = 0.01\n"
        "opt = 'sgd'\n"
        "weight_decay = 1e-4\n"
        "\n"
        "data_loaders = get_data_loaders(batch_size=batch_size, valid_size=valid_size)\n"
        "model = MyModel(num_classes=num_classes, dropout=dropout)\n"
        "optimizer = get_optimizer(model, optimizer=opt, learning_rate=learning_rate, momentum=0.9, weight_decay=weight_decay)\n"
        "loss = get_loss()\n"
        "\n"
        "optimize(\n"
        "    data_loaders,\n"
        "    model,\n"
        "    optimizer,\n"
        "    loss,\n"
        "    n_epochs=num_epochs,\n"
        "    save_path='checkpoints/best_val_loss.pt',\n"
        "    interactive_tracking=True,\n"
        ")\n"
    ),
    md("### 4.2 Test the model"),
    code(
        "model.load_state_dict(torch.load('checkpoints/best_val_loss.pt'))\n"
        "from src.train import one_epoch_test\n"
        "_ = one_epoch_test(data_loaders['test'], model, loss)\n"
    ),
    md(
        "## 5. Export with TorchScript\n"
        "Complete the `Predictor` class in `src/predictor.py` (it must apply "
        "`self.transforms`, run the model and apply `softmax(dim=1)`)."
    ),
    code("!pytest -vv src/predictor.py --no-header -x\n"),
    code(
        "from src.predictor import Predictor\n"
        "from src.helpers import compute_mean_and_std\n"
        "\n"
        "model.load_state_dict(torch.load('checkpoints/best_val_loss.pt'))\n"
        "mean, std = compute_mean_and_std()\n"
        "class_names = data_loaders['train'].dataset.classes\n"
        "\n"
        "predictor = Predictor(model, class_names=class_names, mean=mean, std=std).cpu()\n"
        "scripted_predictor = torch.jit.script(predictor)\n"
        "scripted_predictor.save('checkpoints/original_exported.pt')\n"
    ),
    md("### 5.1 Reload the exported model and compute the confusion matrix"),
    code(
        "import torch\n"
        "from src.helpers import plot_confusion_matrix\n"
        "\n"
        "model_reloaded = torch.jit.load('checkpoints/original_exported.pt')\n"
        "\n"
        "preds, truths = [], []\n"
        "with torch.no_grad():\n"
        "    for images, labels in data_loaders['test']:\n"
        "        # Predictor expects uint8 images so reverse the normalization the\n"
        "        # test loader applied.\n"
        "        from torchvision import transforms\n"
        "        invTrans = transforms.Compose([\n"
        "            transforms.Normalize(mean=[0., 0., 0.], std=1.0/std),\n"
        "            transforms.Normalize(mean=-mean, std=[1., 1., 1.]),\n"
        "        ])\n"
        "        uint_imgs = (invTrans(images).clamp(0, 1) * 255).to(torch.uint8)\n"
        "        probs = model_reloaded(uint_imgs)\n"
        "        preds.append(probs.argmax(dim=1))\n"
        "        truths.append(labels)\n"
        "preds = torch.cat(preds)\n"
        "truths = torch.cat(truths)\n"
        "fig = plot_confusion_matrix(preds, truths)\n"
    ),
]


# ---------------------------------------------------------------------------
# Notebook 2: Transfer learning
# ---------------------------------------------------------------------------

PART2 = [
    md(
        "# Convolutional Neural Networks\n"
        "## Project: Landmark Classification & Tagging for Social Media\n\n"
        "**Part 2 of 3: transfer learning.**"
    ),
    md("## 0. Setup"),
    code(
        "%load_ext autoreload\n"
        "%autoreload 2\n"
        "\n"
        "from src.helpers import setup_env\n"
        "setup_env()\n"
    ),
    md(
        "## 1. Transfer-learning architecture\n"
        "Complete `src/transfer.py` then run the tests."
    ),
    code("!pytest -vv src/transfer.py --no-header -x\n"),
    md(
        "**Question**: Why is the chosen architecture suitable for this task?\n\n"
        "**Answer**: I chose **ResNet-50** pretrained on ImageNet. Three reasons:\n"
        "1. ImageNet contains many photographs of buildings, monuments and "
        "natural scenes that share low- and mid-level features (edges, textures, "
        "windows, sky, brickwork) with landmark photos, so the frozen backbone is "
        "already a strong feature extractor for this domain.\n"
        "2. ResNet-50's residual connections let gradients flow through 50 layers "
        "without vanishing, giving us a 2048-dim feature embedding that captures "
        "high-level semantic content (\"tower\", \"arch\", \"dome\").\n"
        "3. With only ~5k training images per landmark dataset, training a deep "
        "network from scratch overfits badly. Freezing the backbone and training "
        "only the new linear head means we fit ~100k parameters instead of "
        "~25M — perfectly matched to the available data."
    ),
    md("## 2. Train and validate"),
    code(
        "import torch\n"
        "from src.data import get_data_loaders\n"
        "from src.transfer import get_model_transfer_learning\n"
        "from src.optimization import get_loss, get_optimizer\n"
        "from src.train import optimize, one_epoch_test\n"
        "\n"
        "batch_size = 64\n"
        "num_epochs = 15\n"
        "num_classes = 50\n"
        "learning_rate = 0.001\n"
        "weight_decay = 1e-4\n"
        "\n"
        "data_loaders = get_data_loaders(batch_size=batch_size)\n"
        "model_transfer = get_model_transfer_learning('resnet50', n_classes=num_classes)\n"
        "\n"
        "optimizer = get_optimizer(model_transfer, optimizer='adam', learning_rate=learning_rate, weight_decay=weight_decay)\n"
        "loss = get_loss()\n"
        "\n"
        "optimize(\n"
        "    data_loaders,\n"
        "    model_transfer,\n"
        "    optimizer,\n"
        "    loss,\n"
        "    n_epochs=num_epochs,\n"
        "    save_path='checkpoints/model_transfer.pt',\n"
        "    interactive_tracking=True,\n"
        ")\n"
    ),
    md("## 3. Test"),
    code(
        "model_transfer.load_state_dict(torch.load('checkpoints/model_transfer.pt'))\n"
        "_ = one_epoch_test(data_loaders['test'], model_transfer, loss)\n"
    ),
    md("## 4. Export with TorchScript"),
    code(
        "from src.predictor import Predictor\n"
        "from src.helpers import compute_mean_and_std\n"
        "\n"
        "mean, std = compute_mean_and_std()\n"
        "class_names = data_loaders['train'].dataset.classes\n"
        "\n"
        "predictor = Predictor(model_transfer.cpu(), class_names=class_names, mean=mean, std=std)\n"
        "scripted = torch.jit.script(predictor)\n"
        "scripted.save('checkpoints/transfer_exported.pt')\n"
        "\n"
        "# Sanity check: reload it.\n"
        "_ = torch.jit.load('checkpoints/transfer_exported.pt')\n"
        "print('Saved checkpoints/transfer_exported.pt')\n"
    ),
]


# ---------------------------------------------------------------------------
# Notebook 3: App
# ---------------------------------------------------------------------------

PART3 = [
    md(
        "# Convolutional Neural Networks\n"
        "## Project: Landmark Classification & Tagging for Social Media\n\n"
        "**Part 3 of 3: a simple inference app.**\n"
    ),
    md("## Load the exported model"),
    code(
        "import torch\n"
        "import torchvision.transforms as T\n"
        "from PIL import Image\n"
        "from IPython.display import display\n"
        "\n"
        "learn_inf = torch.jit.load('checkpoints/transfer_exported.pt')\n"
    ),
    md(
        "## A tiny inference helper\n"
        "Drop an image into `static_images/` and pass its path."
    ),
    code(
        "def predict_landmark(img_path, top_k=3):\n"
        "    img = Image.open(img_path).convert('RGB')\n"
        "    tensor = T.functional.pil_to_tensor(img).unsqueeze(0)\n"
        "    probs = learn_inf(tensor).squeeze()\n"
        "    classes = list(learn_inf.class_names)\n"
        "    top = torch.topk(probs, k=top_k)\n"
        "    display(img.resize((300, 300)))\n"
        "    for p, idx in zip(top.values.tolist(), top.indices.tolist()):\n"
        "        name = classes[idx].split('.')[-1].replace('_', ' ')\n"
        "        print(f'{name:40s} {p*100:6.2f}%')\n"
        "    return classes[top.indices[0].item()]\n"
    ),
    md("## Run on an out-of-set image"),
    code("predict_landmark('static_images/test_image.jpg')\n"),
    md(
        "## (Optional) ipywidgets UI\n"
        "Run the cell below to get an upload widget."
    ),
    code(
        "from ipywidgets import FileUpload, Output, VBox, Label\n"
        "from io import BytesIO\n"
        "\n"
        "btn_upload = FileUpload(accept='image/*', multiple=False)\n"
        "out_pl = Output()\n"
        "lbl_pred = Label()\n"
        "\n"
        "def on_upload(change):\n"
        "    out_pl.clear_output()\n"
        "    for name, fileinfo in btn_upload.value.items() if isinstance(btn_upload.value, dict) else ((f['name'], f) for f in btn_upload.value):\n"
        "        img = Image.open(BytesIO(fileinfo['content'])).convert('RGB')\n"
        "        with out_pl:\n"
        "            display(img.resize((300, 300)))\n"
        "        tensor = T.functional.pil_to_tensor(img).unsqueeze(0)\n"
        "        probs = learn_inf(tensor).squeeze()\n"
        "        top = torch.topk(probs, k=3)\n"
        "        classes = list(learn_inf.class_names)\n"
        "        lbl_pred.value = ' | '.join(\n"
        "            f\"{classes[i].split('.')[-1].replace('_',' ')} ({p*100:.1f}%)\"\n"
        "            for p, i in zip(top.values.tolist(), top.indices.tolist())\n"
        "        )\n"
        "btn_upload.observe(on_upload, names='value')\n"
        "VBox([btn_upload, out_pl, lbl_pred])\n"
    ),
    md(
        "## Submission archive\n"
        "Run the cell below to bundle all notebooks and source files into a "
        "`submission_<timestamp>.tar.gz` for upload."
    ),
    code(
        "import datetime, tarfile, os, glob\n"
        "\n"
        "ts = datetime.datetime.now().strftime('%Y-%m-%dT%Hh%Mm')\n"
        "out = f'submission_{ts}.tar.gz'\n"
        "with tarfile.open(out, 'w:gz') as tar:\n"
        "    for path in [\n"
        "        'Project_Landmarks_Part1_CNNfromScratch__starter.ipynb',\n"
        "        'Project_Landmarks_Part2_TransferLearning__starter.ipynb',\n"
        "        'Project_Landmarks_Part3_App__starter.ipynb',\n"
        "    ]:\n"
        "        if os.path.exists(path):\n"
        "            tar.add(path)\n"
        "    for path in glob.glob('src/*.py'):\n"
        "        tar.add(path)\n"
        "    for path in glob.glob('checkpoints/*.pt'):\n"
        "        tar.add(path)\n"
        "print('Wrote', out)\n"
    ),
]


def main():
    here = Path(__file__).resolve().parent
    targets = [
        ("Project_Landmarks_Part1_CNNfromScratch__starter.ipynb", PART1),
        ("Project_Landmarks_Part2_TransferLearning__starter.ipynb", PART2),
        ("Project_Landmarks_Part3_App__starter.ipynb", PART3),
    ]
    for name, cells in targets:
        path = here / name
        path.write_text(json.dumps(nb(cells), indent=1))
        print("wrote", path)


if __name__ == "__main__":
    main()
