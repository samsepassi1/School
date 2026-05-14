"""Tests for the audio moderation agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agents.audio_agent import audio_agent, moderate_audio
from tests._sample_media import TINY_WAV
from schemas.moderation_result import AudioModerationResult

models.ALLOW_MODEL_REQUESTS = False


def _canned_response(payload: dict[str, object]):
    async def _call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    return _call


def test_agent_is_configured() -> None:
    assert audio_agent.output_type is AudioModerationResult
    assert audio_agent.name == "audio_moderation_agent"


@pytest.mark.asyncio
async def test_clean_audio_passes() -> None:
    function_model = FunctionModel(
        _canned_response(
            {
                "contains_pii": False,
                "is_unfriendly": False,
                "is_unprofessional": False,
                "is_low_quality": False,
                "rationale": "Clear, friendly voicenote.",
            }
        )
    )
    with audio_agent.override(model=function_model):
        result = await moderate_audio(TINY_WAV, filename="hello.wav")
    assert isinstance(result, AudioModerationResult)
    assert result.flagged is False


@pytest.mark.asyncio
async def test_unprofessional_audio_is_flagged() -> None:
    function_model = FunctionModel(
        _canned_response(
            {
                "contains_pii": False,
                "is_unfriendly": True,
                "is_unprofessional": True,
                "is_low_quality": True,
                "rationale": "Distorted audio, dismissive tone, slang.",
            }
        )
    )
    with audio_agent.override(model=function_model):
        result = await moderate_audio(TINY_WAV, filename="rant.wav")
    assert result.is_unfriendly is True
    assert result.is_unprofessional is True
    assert result.is_low_quality is True
    assert result.flagged is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-vv"]))
