"""Video moderation agent.

Uses Pydantic-AI's multimodal ``BinaryContent`` to pass raw video bytes to
Gemini and returns a ``VideoModerationResult``.
"""

from __future__ import annotations

from pydantic_ai import Agent, BinaryContent

from schemas.moderation_result import VideoModerationResult

from ._common import moderation_model_name

VIDEO_MODERATION_SYSTEM_PROMPT = """\
You are a video moderation agent for ACME Enterprise customer service.

A trainee customer-service agent is about to send the following video to a
customer. Inspect it carefully and decide whether it is appropriate.

Flag the video on every dimension that applies:

- contains_pii: true if the video shows personally identifiable information
  (faces tied to names, ID documents, credit cards, license plates, etc.).
- is_unfriendly: true if the video carries an unfriendly, hostile or
  mocking message toward the customer.
- is_unprofessional: true if the video is inappropriate for a professional
  customer-service exchange.
- is_disturbing: true if the video contains disturbing, violent or sexual
  content.
- is_low_quality: true if the video is so blurry, dark or pixelated that it
  cannot serve its communicative purpose.

Always provide a short ``rationale`` (one or two sentences) describing what
you see and why you flagged (or didn't flag) the video.
"""


video_agent: Agent[None, VideoModerationResult] = Agent(
    moderation_model_name(),
    output_type=VideoModerationResult,
    system_prompt=VIDEO_MODERATION_SYSTEM_PROMPT,
    name="video_moderation_agent",
)


def _guess_media_type(filename: str | None) -> str:
    if not filename:
        return "video/mp4"
    lowered = filename.lower()
    if lowered.endswith(".webm"):
        return "video/webm"
    if lowered.endswith(".mov"):
        return "video/quicktime"
    if lowered.endswith(".avi"):
        return "video/x-msvideo"
    if lowered.endswith(".mkv"):
        return "video/x-matroska"
    return "video/mp4"


async def moderate_video(
    video_bytes: bytes,
    *,
    filename: str | None = None,
    media_type: str | None = None,
) -> VideoModerationResult:
    """Run the video moderation agent on raw bytes and return the result."""
    resolved_media_type = media_type or _guess_media_type(filename)
    binary = BinaryContent(data=video_bytes, media_type=resolved_media_type)
    result = await video_agent.run(
        ["Please moderate this customer-service video.", binary]
    )
    return result.output
