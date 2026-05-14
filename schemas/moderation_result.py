"""Structured output schemas for the moderation agents.

Every moderation agent (text, image, video, audio) returns one of these
models. The base ``ModerationResult`` carries the flags every modality
shares (PII, unfriendly, unprofessional, rationale). Image / video / audio
add their own modality-specific flags.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModerationResult(BaseModel):
    """Base moderation output used by the text agent and inherited by others."""

    contains_pii: bool = Field(
        default=False,
        description=(
            "True when the content contains personally identifiable information "
            "such as full names, addresses, phone numbers, emails, government IDs, "
            "credit card numbers, etc."
        ),
    )
    is_unfriendly: bool = Field(
        default=False,
        description=(
            "True when the content uses rude, sarcastic, hostile or otherwise "
            "unfriendly language toward the customer."
        ),
    )
    is_unprofessional: bool = Field(
        default=False,
        description=(
            "True when the content is unprofessional for a customer service "
            "context (slang, profanity, careless wording, opinions, etc.)."
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "Short natural-language explanation of why the content was flagged "
            "(or why it was deemed acceptable)."
        ),
    )

    @property
    def flagged(self) -> bool:
        """True when *any* moderation flag is set."""
        return bool(self.contains_pii or self.is_unfriendly or self.is_unprofessional)


class ImageModerationResult(ModerationResult):
    """Moderation output for images."""

    is_disturbing: bool = Field(
        default=False,
        description=(
            "True when the image contains disturbing, graphic, violent or "
            "otherwise inappropriate visuals that should not be sent to a "
            "customer."
        ),
    )
    is_low_quality: bool = Field(
        default=False,
        description=(
            "True when the image is too low-quality, blurry, dark or otherwise "
            "unusable in a professional customer service exchange."
        ),
    )

    @property
    def flagged(self) -> bool:
        return bool(super().flagged or self.is_disturbing or self.is_low_quality)


class VideoModerationResult(ImageModerationResult):
    """Moderation output for videos. Same shape as images for now."""


class AudioModerationResult(ModerationResult):
    """Moderation output for audio."""

    is_low_quality: bool = Field(
        default=False,
        description=(
            "True when the audio is too noisy, distorted or quiet to be "
            "acceptable in a customer service exchange."
        ),
    )

    @property
    def flagged(self) -> bool:
        return bool(super().flagged or self.is_low_quality)
