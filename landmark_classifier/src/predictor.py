from typing import List

import torch
import torch.nn as nn
import torchvision.transforms as T


class Predictor(nn.Module):
    """
    Inference wrapper that bundles preprocessing + model + softmax so the
    whole thing can be exported as a self-contained TorchScript module.
    """

    def __init__(self, model: nn.Module, class_names: List[str], mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model.eval()
        self.class_names = class_names
        self.transforms = nn.Sequential(
            T.Resize([256]),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean.tolist(), std.tolist()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            x = self.model(x)
            x = nn.functional.softmax(x, dim=1)
            return x


# ---------- tests ----------

import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2, num_workers=0, limit=200)


def test_predictor(data_loaders):
    from .helpers import compute_mean_and_std
    from .model import MyModel

    mean, std = compute_mean_and_std()
    model = MyModel(num_classes=50)
    classes = data_loaders["train"].dataset.classes

    predictor = Predictor(model, class_names=classes, mean=mean, std=std)

    images, _ = next(iter(data_loaders["train"]))
    images = (images * 255).to(torch.uint8)

    out = predictor(images)
    assert out.shape == (2, 50)
    assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-5), (
        "Predictor output rows should sum to 1 (softmax)"
    )


def test_predictor_scriptable():
    from .helpers import compute_mean_and_std
    from .model import MyModel

    mean, std = compute_mean_and_std()
    model = MyModel(num_classes=50)
    predictor = Predictor(model, class_names=["a"] * 50, mean=mean, std=std)
    scripted = torch.jit.script(predictor)
    assert scripted is not None
