"""Single LLM factory so the model and temperature live in one place."""

from __future__ import annotations

from functools import lru_cache

from agentic.config import settings


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.llm_model, temperature=temperature)
