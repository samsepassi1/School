"""Tests for the video moderation agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agents.video_agent import moderate_video, video_agent
from tests._sample_media import TINY_MP4
from schemas.moderation_result import VideoModerationResult

models.ALLOW_MODEL_REQUESTS = False


def _canned_response(payload: dict[str, object]):
    async def _call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    return _call


def test_agent_is_configured() -> None:
    assert video_agent.output_type is VideoModerationResult
    assert video_agent.name == "video_moderation_agent"


@pytest.mark.asyncio
async def test_safe_video_passes() -> None:
    function_model = FunctionModel(
        _canned_response(
            {
                "contains_pii": False,
                "is_unfriendly": False,
                "is_unprofessional": False,
                "is_disturbing": False,
                "is_low_quality": False,
                "rationale": "A short product demo, nothing flagged.",
            }
        )
    )
    with video_agent.override(model=function_model):
        result = await moderate_video(TINY_MP4, filename="demo.mp4")
    assert isinstance(result, VideoModerationResult)
    assert result.flagged is False


@pytest.mark.asyncio
async def test_unsafe_video_is_flagged() -> None:
    function_model = FunctionModel(
        _canned_response(
            {
                "contains_pii": False,
                "is_unfriendly": True,
                "is_unprofessional": True,
                "is_disturbing": True,
                "is_low_quality": False,
                "rationale": "Hostile, mocking and graphic.",
            }
        )
    )
    with video_agent.override(model=function_model):
        result = await moderate_video(TINY_MP4, filename="rude.mp4")
    assert result.flagged is True
    assert result.is_disturbing is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-vv"]))
