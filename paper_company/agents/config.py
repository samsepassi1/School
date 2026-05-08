"""Shared configuration for all pydantic-ai agents.

Reads ``PAPER_LLM_MODEL`` (default ``gpt-4o-mini``) and exposes the model
identifier in the form pydantic-ai expects (``openai:<model>``).
"""

from __future__ import annotations

import os

MODEL_NAME = os.environ.get("PAPER_LLM_MODEL", "gpt-4o-mini")
PYDANTIC_AI_MODEL = f"openai:{MODEL_NAME}"

#: When True, agents are constructed lazily and Agent.run is bypassed by the
#: deterministic Python pipeline in ``agents/deterministic.py``. Useful when no
#: OpenAI API key is available.
USE_DETERMINISTIC = os.environ.get("PAPER_USE_DETERMINISTIC", "0") == "1"
