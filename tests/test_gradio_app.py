"""Tests for the Gradio chat front-end."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import gradio_app
from agents.customer_agent import customer_agent
from agents.image_agent import image_agent
from agents.text_agent import text_agent
from tests._sample_media import TINY_PNG

models.ALLOW_MODEL_REQUESTS = False


def _canned(payload: dict[str, object] | str):
    body = payload if isinstance(payload, str) else json.dumps(payload)

    async def _call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=body)])

    return _call


def test_demo_object_exists() -> None:
    """The Gradio Blocks object should be importable at module scope."""
    assert gradio_app.demo is not None
    assert hasattr(gradio_app.demo, "launch")


def test_classify_media() -> None:
    assert gradio_app._classify_media("foo.png") == "image"
    assert gradio_app._classify_media("foo.JPG") == "image"
    assert gradio_app._classify_media("foo.mp3") == "audio"
    assert gradio_app._classify_media("foo.wav") == "audio"
    assert gradio_app._classify_media("foo.mp4") == "video"
    assert gradio_app._classify_media("foo.webm") == "video"


def test_format_block_includes_rationale() -> None:
    from schemas.moderation_result import ModerationResult

    r = ModerationResult(
        contains_pii=True, rationale="leaks an email"
    )
    msg = gradio_app._format_block(r, "message")
    assert "blocked" in msg.lower()
    assert "leaks an email" in msg
    assert "PII" in msg


@pytest.mark.asyncio
async def test_clean_message_gets_customer_reply() -> None:
    clean = _canned(
        {
            "contains_pii": False,
            "is_unfriendly": False,
            "is_unprofessional": False,
            "rationale": "OK",
        }
    )
    reply = _canned("I'm furious about my Power Widget Pro!")
    with (
        text_agent.override(model=FunctionModel(clean)),
        customer_agent.override(model=FunctionModel(reply)),
    ):
        history, session_id = await gradio_app.respond(
            {"text": "Hello, how can I help?", "files": []},
            [],
            "",
        )
    assert session_id.startswith("conv-")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello, how can I help?"}
    assert history[1]["role"] == "assistant"
    assert "Power Widget Pro" in history[1]["content"]


@pytest.mark.asyncio
async def test_flagged_message_is_blocked() -> None:
    rude = _canned(
        {
            "contains_pii": False,
            "is_unfriendly": True,
            "is_unprofessional": True,
            "rationale": "Tone is rude and dismissive.",
        }
    )
    # Customer agent should NOT be called when message is blocked, but we
    # still provide an override that would explode if called.
    customer = _canned("should-not-be-called")
    with (
        text_agent.override(model=FunctionModel(rude)),
        customer_agent.override(model=FunctionModel(customer)),
    ):
        history, session_id = await gradio_app.respond(
            {"text": "ugh, not my problem dude", "files": []},
            [],
            "session-1",
        )
    assert session_id == "session-1"
    # user msg + assistant block notice = 2 entries
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert "blocked" in history[1]["content"].lower()
    assert "rude" in history[1]["content"].lower()


@pytest.mark.asyncio
async def test_image_attachment_is_moderated(tmp_path: Path) -> None:
    clean_text = _canned(
        {
            "contains_pii": False,
            "is_unfriendly": False,
            "is_unprofessional": False,
            "rationale": "OK",
        }
    )
    flagged_image = _canned(
        {
            "contains_pii": False,
            "is_unfriendly": False,
            "is_unprofessional": False,
            "is_disturbing": True,
            "is_low_quality": False,
            "rationale": "Graphic, not suitable.",
        }
    )
    customer = _canned("should-not-be-called")

    img_path = tmp_path / "bad.png"
    img_path.write_bytes(TINY_PNG)

    with (
        text_agent.override(model=FunctionModel(clean_text)),
        image_agent.override(model=FunctionModel(flagged_image)),
        customer_agent.override(model=FunctionModel(customer)),
    ):
        history, _ = await gradio_app.respond(
            {"text": "Look at this", "files": [str(img_path)]},
            [],
            "",
        )

    block_entries = [h for h in history if h["role"] == "assistant"]
    assert block_entries, "expected a block notice"
    assert "blocked" in block_entries[0]["content"].lower()


def test_end_conversation_resets_state() -> None:
    new_history, new_session, status = gradio_app.end_conversation(
        "session-abc", [{"role": "user", "content": "hi"}]
    )
    assert new_history == []
    assert new_session.startswith("conv-")
    assert "ended" in status.lower()


def test_submit_feedback_runs_without_error() -> None:
    out = gradio_app.submit_feedback("Great session!", "session-xyz", [])
    assert "feedback" in out.lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-vv"]))
