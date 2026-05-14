"""Image moderation evals.

Run with:

    uv run evals/image/test_cases.py

Cases reference files under ``evals/test_data/``. If a file is missing for a
case it will be skipped at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.image_agent import moderate_image
from evals._common import EvalCase, ExpectedFlags, run_eval

TEST_DATA = ROOT / "evals" / "test_data"


CASES: list[EvalCase] = [
    EvalCase(
        name="professional_product_image",
        inputs=TEST_DATA / "professional_image.jpg",
        expected=ExpectedFlags(is_disturbing=False, is_low_quality=False),
    ),
    EvalCase(
        name="blurry_image",
        inputs=TEST_DATA / "blurry_image.jpg",
        expected=ExpectedFlags(is_low_quality=True),
    ),
    EvalCase(
        name="document_with_pii",
        inputs=TEST_DATA / "id_document.jpg",
        expected=ExpectedFlags(contains_pii=True),
    ),
    EvalCase(
        name="disturbing_image",
        inputs=TEST_DATA / "disturbing_image.jpg",
        expected=ExpectedFlags(is_disturbing=True),
    ),
]


async def task(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"missing test asset: {path}. Drop a file at that location to run this case."
        )
    return await moderate_image(path.read_bytes(), filename=path.name)


def main() -> int:
    return run_eval(title="Image moderation evals", cases=CASES, task=task)


if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
