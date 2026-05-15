import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


def get_model_transfer_learning(model_name: str = "resnet18", n_classes: int = 50) -> nn.Module:
    """
    Load a torchvision model pretrained on ImageNet, freeze its parameters,
    and replace its final classification head with a fresh `n_classes` linear
    layer so it can be fine-tuned on the landmark dataset.
    """
    if hasattr(models, model_name):
        model_transfer = getattr(models, model_name)(weights="IMAGENET1K_V1")
    else:
        raise ValueError(
            f"Model {model_name!r} not found in torchvision.models. "
            f"Try one of: resnet18, resnet50, vgg16, efficientnet_b0 ..."
        )

    for p in model_transfer.parameters():
        p.requires_grad = False

    if hasattr(model_transfer, "fc"):
        in_features = model_transfer.fc.in_features
        model_transfer.fc = nn.Linear(in_features, n_classes)
    elif hasattr(model_transfer, "classifier"):
        classifier = model_transfer.classifier
        if isinstance(classifier, nn.Linear):
            in_features = classifier.in_features
            model_transfer.classifier = nn.Linear(in_features, n_classes)
        else:
            last_idx = None
            for i, layer in enumerate(classifier):
                if isinstance(layer, nn.Linear):
                    last_idx = i
            if last_idx is None:
                raise RuntimeError(
                    f"Could not find a Linear layer in classifier of {model_name}"
                )
            in_features = classifier[last_idx].in_features
            classifier[last_idx] = nn.Linear(in_features, n_classes)
    else:
        raise RuntimeError(
            f"Don't know how to replace the head of {model_name}; "
            f"please extend get_model_transfer_learning to handle it."
        )

    return model_transfer


# ---------- tests ----------

import pytest


@pytest.fixture(scope="session")
def model():
    return get_model_transfer_learning("resnet18", n_classes=23)


def test_returns_module(model):
    assert isinstance(model, nn.Module)


def test_backbone_is_frozen(model):
    for name, p in model.named_parameters():
        if name.startswith("fc.") or name.startswith("classifier."):
            continue
        assert not p.requires_grad, f"Backbone parameter {name} is not frozen"


def test_head_is_trainable(model):
    head_params = [p for p in model.fc.parameters()]
    assert all(p.requires_grad for p in head_params)


def test_output_shape(model):
    model.eval()
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 23), f"Expected (2, 23) but got {out.shape}"
