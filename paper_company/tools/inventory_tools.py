"""Tool wrappers used by the InventoryAgent.

Each function is registered on the agent via ``agent.tool_plain``. They
delegate to helpers in ``project_starter`` and add docstrings tuned for the
LLM (which sees the docstring as the tool description).
"""

from __future__ import annotations

import project_starter as ps


def tool_get_all_inventory(as_of_date: str) -> dict[str, int]:
    """Return current stock for every paper SKU as of ``as_of_date``.

    ``as_of_date`` must be in YYYY-MM-DD format. Returns a mapping of
    item_name → units on hand.
    """
    return ps.get_all_inventory(as_of_date)


def tool_get_stock_level(item_name: str, as_of_date: str) -> int:
    """Return units on hand for a single SKU as of ``as_of_date`` (YYYY-MM-DD)."""
    return ps.get_stock_level(item_name, as_of_date)


def tool_get_supplier_eta(order_date: str, quantity: int) -> str:
    """Estimate the supplier delivery date if we restocked ``quantity`` units on ``order_date``.

    Returns a YYYY-MM-DD date string. Larger orders take longer.
    """
    return ps.get_supplier_delivery_date(order_date, quantity)
