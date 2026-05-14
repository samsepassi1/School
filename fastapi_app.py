"""FastAPI backend exposing the moderation agents as HTTP endpoints."""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from agents.audio_agent import moderate_audio
from agents.image_agent import moderate_image
from agents.text_agent import moderate_text
from agents.video_agent import moderate_video
from tracing import tracer
from schemas.moderation_result import (
    AudioModerationResult,
    ImageModerationResult,
    ModerationResult,
    VideoModerationResult,
)

app = FastAPI(
    title="ACME Moderation API",
    description="Multimodal content moderation for customer-service messages.",
    version="0.1.0",
)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(api_key_header)) -> str:
    expected = os.getenv("USER_API_KEY")
    if not expected:
        return api_key or "anonymous"
    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


class TextRequest(BaseModel):
    text: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/moderate-text", response_model=ModerationResult)
async def api_moderate_text(
    payload: TextRequest, _: str = Depends(require_api_key)
) -> ModerationResult:
    with tracer.start_as_current_span("moderate_text") as span:
        span.set_attribute("moderation.media", "text")
        span.set_attribute("moderation.input_length", len(payload.text))
        result = await moderate_text(payload.text)
        span.set_attribute("moderation.flagged", result.flagged)
        return result


@app.post("/api/v1/moderate-image", response_model=ImageModerationResult)
async def api_moderate_image(
    file: UploadFile = File(...), _: str = Depends(require_api_key)
) -> ImageModerationResult:
    data = await file.read()
    with tracer.start_as_current_span("moderate_image") as span:
        span.set_attribute("moderation.media", "image")
        span.set_attribute("moderation.filename", file.filename or "")
        result = await moderate_image(
            data, filename=file.filename, media_type=file.content_type
        )
        span.set_attribute("moderation.flagged", result.flagged)
        return result


@app.post("/api/v1/moderate-video", response_model=VideoModerationResult)
async def api_moderate_video(
    file: UploadFile = File(...), _: str = Depends(require_api_key)
) -> VideoModerationResult:
    data = await file.read()
    with tracer.start_as_current_span("moderate_video") as span:
        span.set_attribute("moderation.media", "video")
        span.set_attribute("moderation.filename", file.filename or "")
        result = await moderate_video(
            data, filename=file.filename, media_type=file.content_type
        )
        span.set_attribute("moderation.flagged", result.flagged)
        return result


@app.post("/api/v1/moderate-audio", response_model=AudioModerationResult)
async def api_moderate_audio(
    file: UploadFile = File(...), _: str = Depends(require_api_key)
) -> AudioModerationResult:
    data = await file.read()
    with tracer.start_as_current_span("moderate_audio") as span:
        span.set_attribute("moderation.media", "audio")
        span.set_attribute("moderation.filename", file.filename or "")
        result = await moderate_audio(
            data, filename=file.filename, media_type=file.content_type
        )
        span.set_attribute("moderation.flagged", result.flagged)
        return result
