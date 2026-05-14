"""Tests for the text moderation agent."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic_ai import models
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

from agents.text_agent import moderate_text, text_agent
from schemas.moderation_result import ModerationResult


# Prevent any accidental network use.
models.ALLOW_MODEL_REQUESTS = False


def _canned_response(payload: dict[str, object]):
    import json

    async def _call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # When pydantic-ai expects structured output, returning JSON in a
        # TextPart works for tool-less output validation.
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    return _call


def test_agent_is_configured() -> None:
    assert text_agent.output_type is ModerationResult
    assert text_agent.name == "text_moderation_agent"


def test_clean_message_passes() -> None:
    function_model = FunctionModel(
        _canned_response(
            {
                "contains_pii": False,
                "is_unfriendly": False,
                "is_unprofessional": False,
                "rationale": "Message is polite and contains no PII.",
            }
        )
    )
    with text_agent.override(model=function_model):
        result = text_agent.run_sync("Hello, how can I help you today?").output
    assert isinstance(result, ModerationResult)
    assert result.flagged is False


def test_pii_is_flagged() -> None:
    function_model = FunctionModel(
        _canned_response(
            {
                "contains_pii": True,
                "is_unfriendly": False,
                "is_unprofessional": False,
                "rationale": "Reveals customer email and phone number.",
            }
        )
    )
    with text_agent.override(model=function_model):
        result = text_agent.run_sync(
            "Hi John, please call us at 555-1234 or email john@x.com"
        ).output
    assert result.contains_pii is True
    assert result.flagged is True
    assert result.rationale


def test_rude_message_is_flagged() -> None:
    function_model = FunctionModel(
        _canned_response(
            {
                "contains_pii": False,
                "is_unfriendly": True,
                "is_unprofessional": True,
                "rationale": "Tone is dismissive and uses slang.",
            }
        )
    )
    with text_agent.override(model=function_model):
        result = text_agent.run_sync("ugh, dude, not my problem").output
    assert result.is_unfriendly is True
    assert result.is_unprofessional is True


@pytest.mark.asyncio
async def test_moderate_text_helper_returns_model() -> None:
    function_model = FunctionModel(
        _canned_response(
            {
                "contains_pii": False,
                "is_unfriendly": False,
                "is_unprofessional": False,
                "rationale": "OK",
            }
        )
    )
    with text_agent.override(model=function_model):
        result = await moderate_text("Hello there.")
    assert isinstance(result, ModerationResult)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-vv"]))
