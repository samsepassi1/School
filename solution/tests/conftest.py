"""Test fixtures — make ``solution/`` importable and isolate per-test data."""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

import pytest


SOLUTION_ROOT = Path(__file__).resolve().parent.parent
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))


@pytest.fixture()
def tmp_env(monkeypatch):
    """Point UDA-Hub paths at a fresh temp dir, then reload modules so the
    frozen Settings dataclass picks up the new values."""
    tmp = Path(tempfile.mkdtemp(prefix="uda_test_"))
    (tmp / "core").mkdir()
    (tmp / "external").mkdir()
    (tmp / "models").mkdir()

    monkeypatch.setenv("UDA_CORE_DB",            str(tmp / "core" / "uda_hub.db"))
    monkeypatch.setenv("UDA_EXTERNAL_DB",        str(tmp / "external" / "cultpass.db"))
    monkeypatch.setenv("UDA_CHECKPOINT_DB",      str(tmp / "core" / "ck.db"))
    monkeypatch.setenv("UDA_LONGTERM_DB",        str(tmp / "core" / "lt.db"))
    monkeypatch.setenv("UDA_VECTORSTORE",        str(tmp / "models" / "kb"))
    monkeypatch.setenv("UDA_CULTPASS_ARTICLES",
                       str(SOLUTION_ROOT / "data" / "external" / "cultpass_articles.jsonl"))

    for mod in (
        "agentic.config",
        "data.core.db",
        "data.external.db",
        "agentic.memory",
    ):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])

    return tmp
