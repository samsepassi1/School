"""Shared helpers for the Pydantic-AI moderation agents."""

from __future__ import annotations

import os
from functools import lru_cache

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


# Pydantic-AI's GoogleProvider eagerly validates the API key when an Agent is
# constructed. We want module imports to work even before the user has set up
# their .env so the eval scripts can print a friendly message and tests can
# stub the model out entirely. If nothing is set, fall back to a placeholder
# — actual run() calls will still fail loudly without a real key.
if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "placeholder-set-GOOGLE_API_KEY-in-.env"
elif not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


@lru_cache(maxsize=1)
def moderation_model_name() -> str:
    """Model id used by the moderation agents.

    Reads ``MODERATION_MODEL`` from the env, defaulting to a fast Gemini
    multimodal model. Returned as a Pydantic-AI model string so we can switch
    providers without touching the agents.
    """
    raw = os.getenv("MODERATION_MODEL", "gemini-2.0-flash").strip()
    if ":" in raw:
        return raw
    return f"google-gla:{raw}"


@lru_cache(maxsize=1)
def customer_model_name() -> str:
    raw = os.getenv("CUSTOMER_MODEL", "gemini-2.0-flash").strip()
    if ":" in raw:
        return raw
    return f"google-gla:{raw}"
