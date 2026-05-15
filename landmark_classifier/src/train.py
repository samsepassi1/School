import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm

from .helpers import after_subplot


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """Run one epoch of training."""
    if torch.cuda.is_available():
        model = model.cuda()

    model.train()
    train_loss = 0.0
    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()

        train_loss = train_loss + (
            (1.0 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """Run one validation epoch (no gradients)."""
    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        model.eval()
        valid_loss = 0.0
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss_value = loss(output, target)

            valid_loss = valid_loss + (
                (1.0 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )

    return valid_loss


def optimize(
    data_loaders,
    model,
    optimizer,
    loss,
    n_epochs,
    save_path,
    interactive_tracking: bool = False,
):
    """Full training loop with checkpointing and LR-on-plateau scheduling."""
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_loss_min = None
    logs = {}

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(data_loaders["train"], model, optimizer, loss)
        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        print(
            f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} "
            f"\tValidation Loss: {valid_loss:.6f}"
        )

        if valid_loss_min is None or (
            (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):
            print(
                f"New minimum validation loss: {valid_loss:.6f}. Saving model ..."
            )
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

        scheduler.step(valid_loss)

        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]
            liveloss.update(logs)
            liveloss.send()


def one_epoch_test(test_dataloader, model, loss):
    """Run the test set, returning the average loss."""
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        model.eval()
        for batch_idx, (data, target) in tqdm(
            enumerate(test_dataloader),
            desc="Testing",
            total=len(test_dataloader),
            leave=True,
            ncols=80,
        ):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            logits = model(data)
            loss_value = loss(logits, target)

            test_loss = test_loss + (
                (1.0 / (batch_idx + 1)) * (loss_value.data.item() - test_loss)
            )

            pred = logits.data.max(1, keepdim=True)[1]
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)

    print(f"Test Loss: {test_loss:.6f}\n")
    print(f"\nTest Accuracy: {100.0 * correct / total:.2f}% ({int(correct)}/{int(total)})")
    return test_loss


# ---------- tests ----------

import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2, num_workers=0, limit=200)


@pytest.fixture(scope="session")
def optim_objects():
    from .model import MyModel
    from .optimization import get_loss, get_optimizer

    model = MyModel(num_classes=50, dropout=0.5)
    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    for _ in range(2):
        lt = train_one_epoch(data_loaders["train"], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):
    model, loss, _ = optim_objects
    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"


def test_optimize(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    with tempfile.TemporaryDirectory() as tmp:
        optimize(
            data_loaders, model, optimizer, loss, n_epochs=2, save_path=f"{tmp}/m.pt"
        )


def test_one_epoch_test(data_loaders, optim_objects):
    model, loss, _ = optim_objects
    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
