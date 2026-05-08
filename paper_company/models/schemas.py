"""Pydantic IO models exchanged between agents.

Each specialist agent has a typed ``output_type`` so the orchestrator can
pass structured objects between calls without re-parsing free text.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QuoteRequest(BaseModel):
    """A single customer quote request — orchestrator entry point."""

    request_id: str
    request_text: str
    job_description: str = ""
    event_type: str = "general"
    request_date: str = Field(..., description="YYYY-MM-DD when the request arrived")
    deadline: str = Field(..., description="YYYY-MM-DD by which goods must be delivered")


class LineItem(BaseModel):
    sku: str
    quantity: int = Field(gt=0)


class ParsedItems(BaseModel):
    """Output of the parser agent."""

    items: list[LineItem] = Field(default_factory=list)
    parse_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    notes: str = ""


class InventoryLine(BaseModel):
    sku: str
    needed: int
    on_hand: int
    shortfall: int
    eta_if_reordered: str | None = None
    viable_by_deadline: bool = True


class InventoryReport(BaseModel):
    """Output of the inventory agent."""

    lines: list[InventoryLine]
    all_available: bool
    restock_needed: bool


class QuoteLine(BaseModel):
    sku: str
    quantity: int
    unit_price: float
    line_subtotal: float
    discount_pct: float = 0.0
    line_total: float


class Quote(BaseModel):
    """Output of the quoting agent."""

    lines: list[QuoteLine]
    subtotal: float
    bulk_discount_total: float
    urgency_premium_pct: float
    urgency_premium_total: float
    total_price: float
    pricing_rationale: str


class SalesOutcome(BaseModel):
    """Output of the sales agent."""

    decision: Literal["fulfilled", "declined"]
    customer_reply: str
    recorded_transaction_ids: list[int] = Field(default_factory=list)
    cash_balance_before: float
    cash_balance_after: float
    reasoning: str


class QuoteResponse(BaseModel):
    """Final orchestrator output — also serialized into ``test_results.csv``."""

    request_id: str
    decision: Literal["fulfilled", "declined"]
    total_price: float
    customer_reply: str
    reasoning: str
    cash_balance_before: float
    cash_balance_after: float
    items_summary: str
