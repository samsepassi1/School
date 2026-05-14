"""Image moderation agent.

Uses Pydantic-AI's multimodal ``BinaryContent`` so we can pass raw image
bytes straight to Gemini and get back a structured
``ImageModerationResult``.
"""

from __future__ import annotations

from pydantic_ai import Agent, BinaryContent

from schemas.moderation_result import ImageModerationResult

from ._common import moderation_model_name

IMAGE_MODERATION_SYSTEM_PROMPT = """\
You are a visual moderation agent for ACME Enterprise customer service.

A trainee customer-service agent is about to send the following image to a
customer. Inspect the image and decide whether it is appropriate.

Flag the image on every dimension that applies:

- contains_pii: set to true if the image displays personally identifiable
  information (faces with names, ID documents, credit cards, addresses on
  letters or packages, license plates clearly readable, etc.).
- is_unfriendly: set to true if the image carries an unfriendly, hostile or
  taunting message (memes mocking the customer, rude gestures, etc.).
- is_unprofessional: set to true if the image is unprofessional for a
  customer-service exchange (party photos, irrelevant memes, selfies).
- is_disturbing: set to true if the image is disturbing, graphic, violent,
  sexual, or otherwise inappropriate.
- is_low_quality: set to true if the image is so blurry, dark, pixelated or
  cropped that it cannot serve its communicative purpose.

Always provide a short ``rationale`` (one or two sentences) describing what
you see and why you flagged (or didn't flag) the image.
"""


image_agent: Agent[None, ImageModerationResult] = Agent(
    moderation_model_name(),
    output_type=ImageModerationResult,
    system_prompt=IMAGE_MODERATION_SYSTEM_PROMPT,
    name="image_moderation_agent",
)


def _guess_media_type(filename: str | None) -> str:
    if not filename:
        return "image/jpeg"
    lowered = filename.lower()
    if lowered.endswith(".png"):
        return "image/png"
    if lowered.endswith(".gif"):
        return "image/gif"
    if lowered.endswith(".webp"):
        return "image/webp"
    if lowered.endswith(".bmp"):
        return "image/bmp"
    return "image/jpeg"


async def moderate_image(
    image_bytes: bytes,
    *,
    filename: str | None = None,
    media_type: str | None = None,
) -> ImageModerationResult:
    """Run the image moderation agent on raw bytes and return the result."""
    resolved_media_type = media_type or _guess_media_type(filename)
    binary = BinaryContent(data=image_bytes, media_type=resolved_media_type)
    result = await image_agent.run(
        ["Please moderate this customer-service image.", binary]
    )
    return result.output
