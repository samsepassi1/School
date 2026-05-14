"""Audio moderation evals."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.audio_agent import moderate_audio
from evals._common import EvalCase, ExpectedFlags, run_eval

TEST_DATA = ROOT / "evals" / "test_data"


CASES: list[EvalCase] = [
    EvalCase(
        name="professional_voicenote",
        inputs=TEST_DATA / "professional_audio.mp3",
        expected=ExpectedFlags(is_low_quality=False),
    ),
    EvalCase(
        name="noisy_voicenote",
        inputs=TEST_DATA / "noisy_audio.mp3",
        expected=ExpectedFlags(is_low_quality=True),
    ),
    EvalCase(
        name="rude_voicenote",
        inputs=TEST_DATA / "rude_audio.mp3",
        expected=ExpectedFlags(is_unfriendly=True, is_unprofessional=True),
    ),
]


async def task(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"missing test asset: {path}. Drop a file at that location to run this case."
        )
    return await moderate_audio(path.read_bytes(), filename=path.name)


def main() -> int:
    return run_eval(title="Audio moderation evals", cases=CASES, task=task)


if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
