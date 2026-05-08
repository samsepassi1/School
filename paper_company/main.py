"""Single-request CLI demo for the multi-agent system.

Usage::

    python main.py "Please send 50 reams of A4 paper for our office."

If no argument is provided, a baked-in sample request is used. This script
runs through the same orchestrator path as ``run_eval.py`` so it's a useful
sanity check before running the full evaluation.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

import project_starter as ps  # noqa: E402
from agents.orchestrator import run_request  # noqa: E402
from models.schemas import QuoteRequest  # noqa: E402

DEFAULT_TEXT = (
    "Please send 80 reams of letter paper for our quarterly office restock."
)


async def main() -> None:
    ps.init_database(reset=False)

    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    today = datetime.today().date()
    request = QuoteRequest(
        request_id="DEMO",
        request_text=text,
        job_description="single-request CLI demo",
        event_type="office",
        request_date=today.isoformat(),
        deadline=(today + timedelta(days=14)).isoformat(),
    )

    response = await run_request(request)

    print("=" * 60)
    print(f"Request:  {text}")
    print(f"Decision: {response.decision}")
    print(f"Total:    ${response.total_price:.2f}")
    print(f"Cash:     ${response.cash_balance_before:.2f} -> ${response.cash_balance_after:.2f}")
    print("-" * 60)
    print("Customer reply:")
    print(response.customer_reply)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
