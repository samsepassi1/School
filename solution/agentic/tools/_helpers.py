"""Shared helpers for tool implementations."""

from __future__ import annotations

import json
from typing import Any


def ok(payload: dict[str, Any]) -> str:
    return json.dumps({"status": "ok", **payload}, default=str)


def err(message: str, **extra: Any) -> str:
    return json.dumps({"status": "error", "error": message, **extra})
