"""Video moderation evals."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.video_agent import moderate_video
from evals._common import EvalCase, ExpectedFlags, run_eval

TEST_DATA = ROOT / "evals" / "test_data"


CASES: list[EvalCase] = [
    EvalCase(
        name="professional_demo",
        inputs=TEST_DATA / "professional_video.mp4",
        expected=ExpectedFlags(is_disturbing=False, is_low_quality=False),
    ),
    EvalCase(
        name="low_quality_demo",
        inputs=TEST_DATA / "low_quality_video.mp4",
        expected=ExpectedFlags(is_low_quality=True),
    ),
    EvalCase(
        name="disturbing_clip",
        inputs=TEST_DATA / "disturbing_video.mp4",
        expected=ExpectedFlags(is_disturbing=True),
    ),
]


async def task(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"missing test asset: {path}. Drop a file at that location to run this case."
        )
    return await moderate_video(path.read_bytes(), filename=path.name)


def main() -> int:
    return run_eval(title="Video moderation evals", cases=CASES, task=task)


if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
