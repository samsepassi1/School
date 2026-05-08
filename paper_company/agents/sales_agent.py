"""Sales agent.

Decides fulfill vs decline, writes transactions on fulfillment, and produces
the customer-facing reply. Tools wrap ``create_transaction``,
``get_cash_balance``, and ``generate_financial_report``.
"""

from __future__ import annotations

from pydantic_ai import Agent

from models.schemas import SalesOutcome
from tools.sales_tools import (
    tool_create_transaction,
    tool_get_cash_balance,
    tool_financial_report,
)
from .config import PYDANTIC_AI_MODEL


SYSTEM_PROMPT = """You are the sales closer for Beaver's Choice Paper Company.
You receive a JSON object with the quote, the inventory report, the request
date, the deadline, and a pre-computed decision flag (fulfill or decline)
from the orchestrator. Your job is:

1. Read cash_balance_before with tool_get_cash_balance(as_of_date=request_date).

2. If decision == "fulfilled":
   - For each line in quote.lines, call
     tool_create_transaction(item_name=sku, transaction_type="sale",
                             units=quantity, price=line_total, date=request_date)
     and collect the returned transaction id.
   - Read cash_balance_after with tool_get_cash_balance(as_of_date=request_date).

3. If decision == "declined":
   - Do NOT call tool_create_transaction.
   - cash_balance_after = cash_balance_before.

4. Write a customer_reply (2–5 sentences) that:
   - thanks the customer for the request,
   - lists each item with quantity and line total,
   - states the final total,
   - notes any bulk discount tier or urgency premium that was applied,
   - if declined, explains the reason in plain language (e.g. "stock would not
     arrive before your deadline").
   - NEVER mentions supplier prices, margins, internal SKU codes the customer
     didn't ask about, or internal system errors.

5. Set reasoning to a short internal note (one sentence) describing why the
   decision was made.
"""


sales_agent = Agent(
    PYDANTIC_AI_MODEL,
    result_type=SalesOutcome,
    system_prompt=SYSTEM_PROMPT,
    defer_model_check=True,
)

sales_agent.tool_plain(tool_create_transaction)
sales_agent.tool_plain(tool_get_cash_balance)
sales_agent.tool_plain(tool_financial_report)
