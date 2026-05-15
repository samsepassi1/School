"""Save/load model + optimizer state to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch


def save_checkpoint(
    path: str | Path,
    *,
    models: Mapping[str, torch.nn.Module],
    optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
    extra: dict | None = None,
) -> None:
    """Persist model and optimizer state dictionaries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"models": {name: m.state_dict() for name, m in models.items()}}
    if optimizers:
        payload["optimizers"] = {name: o.state_dict() for name, o in optimizers.items()}
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    models: Mapping[str, torch.nn.Module] | None = None,
    optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> dict:
    """Restore state dicts into the supplied modules / optimizers.

    Returns the raw payload so callers can read `extra` metadata.
    """
    payload = torch.load(str(path), map_location=map_location)

    if models:
        for name, module in models.items():
            module.load_state_dict(payload["models"][name], strict=strict)

    if optimizers and "optimizers" in payload:
        for name, opt in optimizers.items():
            if name in payload["optimizers"]:
                opt.load_state_dict(payload["optimizers"][name])

    return payload
