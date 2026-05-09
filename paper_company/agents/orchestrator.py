"""Orchestrator agent.

Sequentially invokes parser → inventory → quoting → fulfillment. The
fulfill/decline branch is computed deterministically in Python (not by an
LLM) so the eval is reproducible — the LLM is responsible for parsing free
text, querying tools, and writing the customer-facing reply.

For environments without an OpenAI API key, set
``PAPER_USE_DETERMINISTIC=1`` to bypass the LLM entirely; the orchestrator
will route through ``agents.deterministic.run_request`` instead.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from models.schemas import (
    InventoryReport,
    LineItem,
    ParsedItems,
    Quote,
    QuoteRequest,
    QuoteResponse,
    SalesOutcome,
)
from project_starter import get_catalog
from . import deterministic
from .config import USE_DETERMINISTIC
from .inventory_agent import inventory_agent
from .parser_agent import parser_agent
from .quoting_agent import quoting_agent
from .sales_agent import sales_agent


def _catalog_prices() -> dict[str, float]:
    return {row["item_name"]: row["unit_price"] for row in get_catalog()}


async def run_request_with_llm(request: QuoteRequest) -> QuoteResponse:
    """LLM-driven pipeline (one agent per step)."""
    catalog_prices = _catalog_prices()

    parsed: ParsedItems = (await parser_agent.run(request.request_text)).data

    request_dt = datetime.strptime(request.request_date, "%Y-%m-%d")
    deadline_dt = datetime.strptime(request.deadline, "%Y-%m-%d")
    urgency_days = (deadline_dt - request_dt).days

    if not parsed.items or parsed.parse_confidence == 0:
        return _early_decline(
            request,
            reasoning="parser could not extract any catalog items from the request",
            customer_reply=(
                "Thank you for your interest in Beaver's Choice. We were unable to "
                "interpret the items requested — could you list specific products "
                "with quantities?"
            ),
            items_summary="",
        )

    if urgency_days < 0:
        return _early_decline(
            request,
            reasoning="deadline before request_date",
            customer_reply=(
                "Thank you for your interest. The deadline you provided is before "
                "the request date, so we are unable to schedule this order."
            ),
            items_summary="; ".join(f"{i.sku} x{i.quantity}" for i in parsed.items),
        )

    inventory_input = {
        "items": [{"sku": i.sku, "quantity": i.quantity} for i in parsed.items],
        "request_date": request.request_date,
        "deadline": request.deadline,
    }
    inventory: InventoryReport = (
        await inventory_agent.run(json.dumps(inventory_input))
    ).data

    quoting_input = {
        "items": [
            {"sku": i.sku, "quantity": i.quantity, "unit_price": catalog_prices.get(i.sku, 0.0)}
            for i in parsed.items
        ],
        "event_type": request.event_type,
        "job_description": request.job_description,
        "urgency_days": urgency_days,
        "inventory_report": inventory.model_dump(),
    }
    quote: Quote = (await quoting_agent.run(json.dumps(quoting_input))).data

    short_lines = [l for l in inventory.lines if l.shortfall > 0]
    if short_lines:
        decision = "declined"
        viable = all(l.viable_by_deadline for l in short_lines)
        decline_kind = "viable_restock" if viable else "not_viable"
        internal_reason = (
            f"{decline_kind}: shortfalls "
            + ("within" if viable else "past")
            + " supplier ETA — "
            + ", ".join(
                f"{l.sku} short by {l.shortfall} (ETA {l.eta_if_reordered})"
                for l in short_lines
            )
        )
    else:
        decision = "fulfilled"
        decline_kind = ""
        internal_reason = ""

    sales_input = {
        "decision": decision,
        "decline_kind": decline_kind,
        "internal_reason": internal_reason,
        "quote": quote.model_dump(),
        "inventory_report": inventory.model_dump(),
        "request_date": request.request_date,
        "deadline": request.deadline,
    }
    outcome: SalesOutcome = (await sales_agent.run(json.dumps(sales_input))).data

    return QuoteResponse(
        request_id=request.request_id,
        decision=outcome.decision,
        total_price=quote.total_price if outcome.decision == "fulfilled" else 0.0,
        customer_reply=outcome.customer_reply,
        reasoning=outcome.reasoning,
        cash_balance_before=outcome.cash_balance_before,
        cash_balance_after=outcome.cash_balance_after,
        items_summary="; ".join(f"{l.sku} x{l.quantity}" for l in quote.lines),
    )


def _early_decline(
    request: QuoteRequest,
    *,
    reasoning: str,
    customer_reply: str,
    items_summary: str,
) -> QuoteResponse:
    from project_starter import get_cash_balance
    cash = get_cash_balance(request.request_date)
    return QuoteResponse(
        request_id=request.request_id,
        decision="declined",
        total_price=0.0,
        customer_reply=customer_reply,
        reasoning=reasoning,
        cash_balance_before=cash,
        cash_balance_after=cash,
        items_summary=items_summary,
    )


async def run_request(request: QuoteRequest) -> QuoteResponse:
    """Entry point used by ``main.py`` and ``run_eval.py``.

    Routes to the LLM pipeline by default, or to the deterministic pipeline
    if ``PAPER_USE_DETERMINISTIC=1``.
    """
    if USE_DETERMINISTIC:
        return deterministic.run_request(request, _catalog_prices())
    return await run_request_with_llm(request)
