import os
import random
import urllib.request
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm


def setup_env():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU *NOT* available. Will use CPU (slow).")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    download_data()

    if "DATA_LOCATION" not in os.environ:
        os.environ["DATA_LOCATION"] = str(get_data_location())


def get_data_location():
    """
    Find the location of the dataset, raising an exception if it cannot be found.
    """
    candidates = [
        Path("landmark_images"),
        Path("/data/DLND/C2/landmark_images"),
        Path(__file__).resolve().parent.parent / "landmark_images",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise IOError(
        "Cannot find the landmark_images dataset. Please download it and place it in "
        "the project root or set DATA_LOCATION env var."
    )


def download_data():
    """
    Download the landmark dataset if it is not already present. Silently noop
    if the data is found locally.
    """
    try:
        get_data_location()
        return
    except IOError:
        pass

    url = (
        "https://udacity-dlnfd.s3-us-west-1.amazonaws.com/"
        "datasets/landmark_images.zip"
    )
    print(f"Downloading dataset from {url} ...")
    with urllib.request.urlopen(url) as response:
        with ZipFile(BytesIO(response.read())) as zf:
            zf.extractall(".")
    print("Done.")


def compute_mean_and_std():
    """
    Compute per-channel mean and std of the training dataset. Cached to disk so
    subsequent calls are fast.
    """
    cache_file = "mean_and_std.pt"
    if os.path.exists(cache_file):
        d = torch.load(cache_file)
        return d["mean"], d["std"]

    folder = get_data_location()
    ds = datasets.ImageFolder(
        str(folder / "train"),
        transform=transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
        ),
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0)

    mean = torch.zeros(3)
    var = torch.zeros(3)
    npix = 0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing mean", ncols=80):
        for c in range(3):
            mean[c] += images[:, c, :, :].mean()
            var[c] += images[:, c, :, :].var()
        npix += 1

    mean = mean / npix
    std = torch.sqrt(var / npix)

    torch.save({"mean": mean, "std": std}, cache_file)
    return mean, std


def after_subplot(ax, group_name, x_label):
    """Add titles, labels and legend to a subplot for the training tracker."""
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.legend(loc="center right")
    if group_name.lower() == "loss":
        ax.set_ylim([None, 4.5])


def plot_confusion_matrix(pred, truth):
    """Render a confusion matrix from prediction / truth tensors."""
    import pandas as pd
    import seaborn as sns

    gt = pd.Series(truth.numpy() if torch.is_tensor(truth) else truth, name="Ground Truth")
    pr = pd.Series(pred.numpy() if torch.is_tensor(pred) else pred, name="Predicted")

    confusion = pd.crosstab(gt, pr, normalize="index")

    fig, sub = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        confusion,
        annot=True,
        fmt=".2f",
        ax=sub,
        cbar=False,
        cmap="Blues",
        square=True,
    )
    sub.set_xlabel("Predicted")
    sub.set_ylabel("Truth")
    fig.tight_layout()
    return fig
