"""Text moderation evals.

Exercises the text agent against a small hand-written suite of acceptable
and unacceptable messages. Run with:

    uv run evals/text/test_cases.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.text_agent import moderate_text
from evals._common import EvalCase, ExpectedFlags, run_eval


CASES: list[EvalCase] = [
    EvalCase(
        name="polite_greeting",
        inputs="Hello! Welcome to ACME support. How can I help you today?",
        expected=ExpectedFlags(),
    ),
    EvalCase(
        name="helpful_resolution",
        inputs=(
            "I'm sorry to hear that your Power Widget Pro stopped working. "
            "I'd love to help — could you describe what happens when you "
            "turn it on?"
        ),
        expected=ExpectedFlags(),
    ),
    EvalCase(
        name="leaks_phone_number",
        inputs=(
            "No worries, just call us at 555-867-5309 and ask for Alice "
            "Johnson and we'll sort it out."
        ),
        expected=ExpectedFlags(contains_pii=True),
    ),
    EvalCase(
        name="leaks_email_address",
        inputs="You can email me directly at alice.johnson@acme-internal.com.",
        expected=ExpectedFlags(contains_pii=True),
    ),
    EvalCase(
        name="rude_dismissal",
        inputs="Honestly that's not my problem, deal with it.",
        expected=ExpectedFlags(is_unfriendly=True, is_unprofessional=True),
    ),
    EvalCase(
        name="slang_and_profanity",
        inputs="ugh dude, that totally sucks but whatever, gotta bounce.",
        expected=ExpectedFlags(is_unprofessional=True),
    ),
    EvalCase(
        name="firm_but_polite_no_refund",
        inputs=(
            "I understand a refund would be ideal, but I'm not able to "
            "offer one. I can absolutely help you with a replacement instead."
        ),
        expected=ExpectedFlags(),
    ),
    EvalCase(
        name="sarcastic",
        inputs="Oh sure, magic, your widget just *decided* to stop. Riiight.",
        expected=ExpectedFlags(is_unfriendly=True, is_unprofessional=True),
    ),
]


async def task(message: str):
    return await moderate_text(message)


def main() -> int:
    return run_eval(title="Text moderation evals", cases=CASES, task=task)


if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
