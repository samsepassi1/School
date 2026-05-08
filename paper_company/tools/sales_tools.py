"""Tool wrappers used by the SalesAgent."""

from __future__ import annotations

from typing import Any

import project_starter as ps


def tool_create_transaction(
    item_name: str,
    transaction_type: str,
    units: int,
    price: float,
    date: str,
) -> int:
    """Record a sale or stock_orders transaction and return its id.

    ``transaction_type`` must be either ``"sale"`` (cash in) or
    ``"stock_orders"`` (cash out). ``price`` is the *total* line value, not
    a unit price. ``date`` must be YYYY-MM-DD.
    """
    return ps.create_transaction(item_name, transaction_type, units, price, date)


def tool_get_cash_balance(as_of_date: str) -> float:
    """Current company cash balance as of ``as_of_date`` (YYYY-MM-DD)."""
    return ps.get_cash_balance(as_of_date)


def tool_financial_report(as_of_date: str) -> dict[str, Any]:
    """Cash + inventory snapshot as of ``as_of_date`` (YYYY-MM-DD).

    Returns ``cash_balance``, ``inventory_value``, ``total_assets``, and a
    breakdown of inventory units by SKU.
    """
    return ps.generate_financial_report(as_of_date)
