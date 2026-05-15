import torch
import torch.nn as nn


class MyModel(nn.Module):
    """
    A from-scratch CNN for landmark classification.

    Architecture follows a VGG-style pattern: five blocks of
    (Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool)
    that progressively downsample the 224x224 input to 7x7 while increasing
    channel depth from 3 -> 512, followed by an MLP classifier head.
    """

    def __init__(self, num_classes: int = 50, dropout: float = 0.5) -> None:
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),     # 224 -> 112
            conv_block(32, 64),    # 112 -> 56
            conv_block(64, 128),   # 56 -> 28
            conv_block(128, 256),  # 28 -> 14
            conv_block(256, 512),  # 14 -> 7
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# ---------- tests ----------

import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2, num_workers=0, limit=200)


def test_model_construction():
    model = MyModel(num_classes=23, dropout=0.3)
    assert isinstance(model, nn.Module)


def test_model_output_shape():
    model = MyModel(num_classes=23)
    model.eval()
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 23), f"Expected (2, 23) but got {out.shape}"


def test_model_no_softmax():
    """The forward method must return raw logits (no softmax)."""
    model = MyModel(num_classes=10)
    model.eval()
    out = model(torch.randn(4, 3, 224, 224))
    sums = out.sum(dim=1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3), (
        "Model output rows sum to 1 — looks like softmax was applied in forward()."
    )
