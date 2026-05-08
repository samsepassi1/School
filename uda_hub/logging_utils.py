"""Structured logging helpers.

Every agent emits human-readable log lines via ``logging`` AND appends a
machine-readable entry to ``state['log']`` for the demo notebook to display.
This module configures the standard library ``logging`` so output is uniform.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any


# A correlation id (e.g. ticket_id) for the current run.
_run_log_var: ContextVar[list[dict[str, Any]] | None] = ContextVar("run_log", default=None)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO, *, json_lines: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if json_lines:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-5s %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root = logging.getLogger()
    # Avoid duplicate handlers if called repeatedly (notebook reloads).
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)
    # Quiet noisy libraries.
    for noisy in ("httpx", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_run_log() -> list[dict[str, Any]]:
    log = _run_log_var.get()
    if log is None:
        log = []
        _run_log_var.set(log)
    return log
