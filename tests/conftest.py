"""Shared pytest fixtures.

Adds the project root to ``sys.path`` so tests can import ``agents``,
``types``, ``gradio_app``, etc. without a package install.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Tests must never reach a real LLM.
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
