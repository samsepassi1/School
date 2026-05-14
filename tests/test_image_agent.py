"""Tests for the image moderation agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic_ai import BinaryContent, models
from pydantic_ai.messages import (
    BinaryContent as _BinaryContent,  # noqa: F401  (parity check)
    ModelMessage,
    ModelResponse,
    TextPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agents.image_agent import image_agent, moderate_image
from tests._sample_media import TINY_PNG
from schemas.moderation_result import ImageModerationResult

models.ALLOW_MODEL_REQUESTS = False


def _canned_response(payload: dict[str, object]):
    async def _call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    return _call


def test_agent_is_configured() -> None:
    assert image_agent.output_type is ImageModerationResult
    assert image_agent.name == "image_moderation_agent"


def test_binary_content_is_imported() -> None:
    bc = BinaryContent(data=TINY_PNG, media_type="image/png")
    assert bc.data == TINY_PNG
    assert bc.media_type == "image/png"


@pytest.mark.asyncio
async def test_safe_image_passes() -> None:
    function_model = FunctionModel(
        _canned_response(
            {
                "contains_pii": False,
                "is_unfriendly": False,
                "is_unprofessional": False,
                "is_disturbing": False,
                "is_low_quality": False,
                "rationale": "A neutral product photograph.",
            }
        )
    )
    with image_agent.override(model=function_model):
        result = await moderate_image(TINY_PNG, filename="ok.png")
    assert isinstance(result, ImageModerationResult)
    assert result.flagged is False


@pytest.mark.asyncio
async def test_disturbing_image_is_flagged() -> None:
    function_model = FunctionModel(
        _canned_response(
            {
                "contains_pii": False,
                "is_unfriendly": False,
                "is_unprofessional": False,
                "is_disturbing": True,
                "is_low_quality": False,
                "rationale": "Graphic content not suitable for customer.",
            }
        )
    )
    with image_agent.override(model=function_model):
        result = await moderate_image(TINY_PNG, filename="bad.png")
    assert result.is_disturbing is True
    assert result.flagged is True


@pytest.mark.asyncio
async def test_low_quality_image_is_flagged() -> None:
    function_model = FunctionModel(
        _canned_response(
            {
                "contains_pii": False,
                "is_unfriendly": False,
                "is_unprofessional": False,
                "is_disturbing": False,
                "is_low_quality": True,
                "rationale": "Blurry and dark, unusable.",
            }
        )
    )
    with image_agent.override(model=function_model):
        result = await moderate_image(TINY_PNG, filename="blurry.png")
    assert result.is_low_quality is True
    assert result.flagged is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-vv"]))
