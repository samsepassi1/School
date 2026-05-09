"""Batch evaluation runner.

Reads ``quote_requests_sample.csv``, processes each request through the
multi-agent system in order, and writes ``test_results.csv`` plus a short
summary to stdout.

Run modes
---------
* ``python run_eval.py`` — uses the LLM-driven pipeline (requires ``OPENAI_API_KEY``).
* ``PAPER_USE_DETERMINISTIC=1 python run_eval.py`` — uses the deterministic
  Python pipeline (no API key required); useful for offline evaluation and
  reproducible CI.
"""

from __future__ import annotations

import asyncio
import csv
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import project_starter as ps  # noqa: E402
from agents.orchestrator import run_request  # noqa: E402
from models.schemas import QuoteRequest, QuoteResponse  # noqa: E402

INPUT_CSV = Path(__file__).parent / "quote_requests_sample.csv"
OUTPUT_CSV = Path(__file__).parent / "test_results.csv"

OUTPUT_COLUMNS = [
    "request_id",
    "request_text",
    "decision",
    "total_price",
    "fulfilled",
    "cash_balance_before",
    "cash_balance_after",
    "items_summary",
    "reasoning",
    "customer_reply_excerpt",
]

#: Substrings that should never appear in a customer-facing reply. The eval
#: flags any row whose customer_reply contains one of these, as a regression
#: check on the "no internal info leak" rubric criterion. Covers both
#: cost/margin leaks and machine-generated logic phrases.
SENSITIVE_TERMS = (
    r"\bsupplier\b",
    r"\bmargin\b",
    r"\bsupplier_price\b",
    r"\binternal cost\b",
    r"\binternal\b",
    r"customer prefers immediate fulfillment",
    r"— declining",
    r"\bviable_restock\b",
    r"\bnot_viable\b",
)
SENSITIVE_RE = re.compile("|".join(SENSITIVE_TERMS), re.I)


def _excerpt(text: str, length: int = 280) -> str:
    """One-line excerpt of the customer reply for the CSV."""
    flat = " ".join(text.split())
    return flat if len(flat) <= length else flat[: length - 3] + "..."


async def main() -> int:
    ps.init_database(reset=True)

    df = pd.read_csv(INPUT_CSV)
    rows: list[dict] = []
    leaks: list[str] = []

    for _, row in df.iterrows():
        request = QuoteRequest(
            request_id=str(row["request_id"]),
            request_text=str(row["request_text"]),
            job_description=str(row.get("job_description", "")),
            event_type=str(row.get("event_type", "general")),
            request_date=str(row["request_date"]),
            deadline=str(row["deadline"]),
        )
        response: QuoteResponse = await run_request(request)
        if SENSITIVE_RE.search(response.customer_reply):
            leaks.append(request.request_id)
        rows.append(
            {
                "request_id": response.request_id,
                "request_text": request.request_text,
                "decision": response.decision,
                "total_price": f"{response.total_price:.2f}",
                "fulfilled": "yes" if response.decision == "fulfilled" else "no",
                "cash_balance_before": f"{response.cash_balance_before:.2f}",
                "cash_balance_after": f"{response.cash_balance_after:.2f}",
                "items_summary": response.items_summary,
                "reasoning": response.reasoning,
                "customer_reply_excerpt": _excerpt(response.customer_reply),
            }
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    fulfilled = sum(1 for r in rows if r["decision"] == "fulfilled")
    declined = sum(1 for r in rows if r["decision"] == "declined")
    final_cash = ps.get_cash_balance("9999-12-31")
    final_report = ps.generate_financial_report("9999-12-31")

    mode = "deterministic" if os.environ.get("PAPER_USE_DETERMINISTIC", "0") == "1" else "llm"
    print(f"\n=== Beaver's Choice eval summary (mode={mode}) ===")
    print(f"  total requests: {len(rows)}")
    print(f"  fulfilled:      {fulfilled}")
    print(f"  declined:       {declined}")
    print(f"  final cash:     ${final_cash:,.2f}")
    print(f"  inventory $:    ${final_report['inventory_value']:,.2f}")
    print(f"  total assets:   ${final_report['total_assets']:,.2f}")
    if leaks:
        print(f"  WARNING: customer replies contained sensitive terms in: {leaks}")
    print(f"  results written to {OUTPUT_CSV}")

    rubric_ok = fulfilled >= 3 and declined >= 1 and not leaks
    return 0 if rubric_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
