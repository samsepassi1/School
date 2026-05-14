"""Audio moderation agent.

Uses Pydantic-AI's multimodal ``BinaryContent`` to pass raw audio bytes to
Gemini and returns an ``AudioModerationResult``.
"""

from __future__ import annotations

from pydantic_ai import Agent, BinaryContent

from schemas.moderation_result import AudioModerationResult

from ._common import moderation_model_name

AUDIO_MODERATION_SYSTEM_PROMPT = """\
You are an audio moderation agent for ACME Enterprise customer service.

A trainee customer-service agent is about to send the following audio clip
to a customer. Listen carefully and decide whether it is appropriate.

Flag the clip on every dimension that applies:

- contains_pii: true if the audio contains personally identifiable
  information (full names, addresses, phone numbers, emails, account
  numbers, etc.).
- is_unfriendly: true if the tone of voice is rude, sarcastic or hostile
  toward the customer.
- is_unprofessional: true if the speaker uses slang, profanity, careless
  phrasing or anything else inappropriate in a customer-service exchange.
- is_low_quality: true if the audio is so noisy, distorted or quiet that it
  cannot serve its communicative purpose.

Always provide a short ``rationale`` (one or two sentences) describing what
you heard and why you flagged (or didn't flag) the clip.
"""


audio_agent: Agent[None, AudioModerationResult] = Agent(
    moderation_model_name(),
    output_type=AudioModerationResult,
    system_prompt=AUDIO_MODERATION_SYSTEM_PROMPT,
    name="audio_moderation_agent",
)


def _guess_media_type(filename: str | None) -> str:
    if not filename:
        return "audio/mpeg"
    lowered = filename.lower()
    if lowered.endswith(".wav"):
        return "audio/wav"
    if lowered.endswith(".ogg"):
        return "audio/ogg"
    if lowered.endswith(".flac"):
        return "audio/flac"
    if lowered.endswith(".m4a") or lowered.endswith(".mp4"):
        return "audio/mp4"
    if lowered.endswith(".aac"):
        return "audio/aac"
    return "audio/mpeg"


async def moderate_audio(
    audio_bytes: bytes,
    *,
    filename: str | None = None,
    media_type: str | None = None,
) -> AudioModerationResult:
    """Run the audio moderation agent on raw bytes and return the result."""
    resolved_media_type = media_type or _guess_media_type(filename)
    binary = BinaryContent(data=audio_bytes, media_type=resolved_media_type)
    result = await audio_agent.run(
        ["Please moderate this customer-service audio clip.", binary]
    )
    return result.output
