"""Shared helpers for the moderation evals.

Provides a tiny evaluator that compares the boolean fields on a
``ModerationResult`` against expected values and a runner that
prints a small report.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from pydantic_evals import Case, Dataset  # type: ignore
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext  # type: ignore

    HAS_PYDANTIC_EVALS = True
except Exception:  # pragma: no cover - optional dep
    HAS_PYDANTIC_EVALS = False

from schemas.moderation_result import ModerationResult


@dataclass
class ExpectedFlags:
    """Ground-truth labels we expect a moderation agent to produce."""

    contains_pii: bool = False
    is_unfriendly: bool = False
    is_unprofessional: bool = False
    is_disturbing: bool | None = None
    is_low_quality: bool | None = None

    def as_dict(self) -> dict[str, bool]:
        out: dict[str, bool] = {
            "contains_pii": self.contains_pii,
            "is_unfriendly": self.is_unfriendly,
            "is_unprofessional": self.is_unprofessional,
        }
        if self.is_disturbing is not None:
            out["is_disturbing"] = self.is_disturbing
        if self.is_low_quality is not None:
            out["is_low_quality"] = self.is_low_quality
        return out


def compare(expected: ExpectedFlags, actual: ModerationResult) -> dict[str, Any]:
    """Compare expected flags against an actual moderation result."""
    mismatches: list[str] = []
    for field_name, expected_value in expected.as_dict().items():
        actual_value = getattr(actual, field_name, None)
        if actual_value is None:
            continue
        if bool(actual_value) != bool(expected_value):
            mismatches.append(
                f"{field_name}: expected={expected_value} actual={actual_value}"
            )
    return {
        "passed": not mismatches,
        "mismatches": mismatches,
        "rationale": actual.rationale,
    }


if HAS_PYDANTIC_EVALS:

    class FlagsEvaluator(Evaluator):
        """Pydantic Evals evaluator: scores 1.0 if every flag matches."""

        def evaluate(self, ctx: "EvaluatorContext[Any, ModerationResult]") -> float:
            expected: ExpectedFlags = ctx.expected_output  # type: ignore[assignment]
            actual: ModerationResult = ctx.output  # type: ignore[assignment]
            report = compare(expected, actual)
            return 1.0 if report["passed"] else 0.0


@dataclass
class EvalCase:
    """Plain-Python case definition used by ``run_eval``."""

    name: str
    inputs: Any
    expected: ExpectedFlags
    notes: str = ""


@dataclass
class EvalReport:
    case_name: str
    passed: bool
    mismatches: list[str] = field(default_factory=list)
    rationale: str = ""
    error: str | None = None


async def _run_one(
    case: EvalCase, task: Callable[[Any], Awaitable[ModerationResult]]
) -> EvalReport:
    try:
        actual = await task(case.inputs)
    except Exception as exc:  # noqa: BLE001
        return EvalReport(case_name=case.name, passed=False, error=str(exc))
    cmp = compare(case.expected, actual)
    return EvalReport(
        case_name=case.name,
        passed=cmp["passed"],
        mismatches=cmp["mismatches"],
        rationale=cmp["rationale"],
    )


def run_eval(
    *,
    title: str,
    cases: list[EvalCase],
    task: Callable[[Any], Awaitable[ModerationResult]],
) -> int:
    """Run ``cases`` against ``task`` and print a small report.

    Returns the number of failures (0 = everything passed).
    """
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    has_api_key = bool(key) and not key.startswith("placeholder")
    print(f"\n=== {title} ===")
    if not has_api_key:
        print(
            "  (no GEMINI_API_KEY/GOOGLE_API_KEY set — set one in your .env to "
            "actually call the model)"
        )
        return 0

    async def _run_all() -> list[EvalReport]:
        return [await _run_one(c, task) for c in cases]

    reports = asyncio.run(_run_all())
    failures = 0
    for r in reports:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.case_name}")
        if r.error:
            print(f"        error: {r.error}")
            failures += 1
            continue
        if not r.passed:
            failures += 1
            for m in r.mismatches:
                print(f"        - {m}")
            print(f"        rationale: {r.rationale!r}")
    total = len(reports)
    passed = total - failures
    pct = (passed / total) * 100 if total else 0.0
    print(f"  {passed}/{total} passed ({pct:.0f}%)")
    return failures


__all__ = [
    "EvalCase",
    "EvalReport",
    "ExpectedFlags",
    "compare",
    "run_eval",
    "HAS_PYDANTIC_EVALS",
]

if HAS_PYDANTIC_EVALS:
    __all__ += ["Case", "Dataset", "FlagsEvaluator"]
