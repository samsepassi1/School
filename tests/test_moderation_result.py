"""Tests for the ``ModerationResult`` Pydantic models."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic import ValidationError

from schemas.moderation_result import (
    AudioModerationResult,
    ImageModerationResult,
    ModerationResult,
    VideoModerationResult,
)


REQUIRED_FIELDS = {"contains_pii", "is_unfriendly", "is_unprofessional", "rationale"}


def test_required_fields_present() -> None:
    fields = set(ModerationResult.model_fields)
    missing = REQUIRED_FIELDS - fields
    assert not missing, f"missing required fields: {missing}"


def test_default_values_are_safe() -> None:
    r = ModerationResult()
    assert r.contains_pii is False
    assert r.is_unfriendly is False
    assert r.is_unprofessional is False
    assert r.rationale == ""
    assert r.flagged is False


def test_field_types() -> None:
    fields = ModerationResult.model_fields
    assert fields["contains_pii"].annotation is bool
    assert fields["is_unfriendly"].annotation is bool
    assert fields["is_unprofessional"].annotation is bool
    assert fields["rationale"].annotation is str


def test_flagged_is_true_when_any_flag_set() -> None:
    assert ModerationResult(contains_pii=True).flagged is True
    assert ModerationResult(is_unfriendly=True).flagged is True
    assert ModerationResult(is_unprofessional=True).flagged is True


def test_rejects_unknown_types() -> None:
    with pytest.raises(ValidationError):
        ModerationResult(contains_pii="not a bool")  # type: ignore[arg-type]


def test_image_result_adds_visual_flags() -> None:
    r = ImageModerationResult()
    assert r.is_disturbing is False
    assert r.is_low_quality is False
    assert r.flagged is False
    assert ImageModerationResult(is_disturbing=True).flagged is True
    assert ImageModerationResult(is_low_quality=True).flagged is True


def test_image_result_is_subclass_of_base() -> None:
    assert issubclass(ImageModerationResult, ModerationResult)
    fields = set(ImageModerationResult.model_fields)
    assert REQUIRED_FIELDS.issubset(fields)


def test_video_result_is_subclass_of_image() -> None:
    assert issubclass(VideoModerationResult, ImageModerationResult)


def test_audio_result_has_quality_flag() -> None:
    r = AudioModerationResult()
    assert r.is_low_quality is False
    assert AudioModerationResult(is_low_quality=True).flagged is True
    assert issubclass(AudioModerationResult, ModerationResult)


def test_roundtrip_json() -> None:
    r = ModerationResult(
        contains_pii=True,
        is_unfriendly=False,
        is_unprofessional=True,
        rationale="leaks an email and uses slang",
    )
    payload = r.model_dump_json()
    parsed = ModerationResult.model_validate_json(payload)
    assert parsed == r


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-vv"]))
