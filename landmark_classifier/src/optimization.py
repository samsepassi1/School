import torch
import torch.nn as nn
import torch.optim


def get_loss() -> nn.Module:
    """Cross-entropy loss for multiclass classification."""
    return nn.CrossEntropyLoss()


def get_optimizer(
    model: nn.Module,
    optimizer: str = "SGD",
    learning_rate: float = 0.01,
    momentum: float = 0.5,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Build the requested optimizer for `model`."""
    if optimizer.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if optimizer.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer!r}")


# ---------- tests ----------

import pytest

from .model import MyModel


@pytest.fixture(scope="session")
def fake_model():
    return MyModel(num_classes=10)


def test_get_loss_is_cross_entropy():
    loss = get_loss()
    assert isinstance(loss, nn.CrossEntropyLoss), (
        f"Expected CrossEntropyLoss but got {type(loss).__name__}"
    )


def test_get_optimizer_sgd(fake_model):
    opt = get_optimizer(fake_model, optimizer="SGD")
    assert isinstance(opt, torch.optim.SGD)


def test_get_optimizer_adam(fake_model):
    opt = get_optimizer(fake_model, optimizer="adam")
    assert isinstance(opt, torch.optim.Adam)


def test_get_optimizer_sets_lr(fake_model):
    opt = get_optimizer(fake_model, optimizer="SGD", learning_rate=0.123)
    assert opt.defaults["lr"] == pytest.approx(0.123)


def test_get_optimizer_sets_momentum(fake_model):
    opt = get_optimizer(fake_model, optimizer="SGD", momentum=0.85)
    assert opt.defaults["momentum"] == pytest.approx(0.85)


def test_get_optimizer_sets_weight_decay(fake_model):
    opt = get_optimizer(fake_model, optimizer="adam", weight_decay=1e-3)
    assert opt.defaults["weight_decay"] == pytest.approx(1e-3)


def test_get_optimizer_uses_model_params(fake_model):
    opt = get_optimizer(fake_model)
    p_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    assert p_ids == {id(p) for p in fake_model.parameters()}


def test_get_optimizer_raises_on_bad_name(fake_model):
    with pytest.raises(ValueError):
        get_optimizer(fake_model, optimizer="nope")
