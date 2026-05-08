"""Tool wrappers used by the QuotingAgent."""

from __future__ import annotations

from typing import Any

import project_starter as ps


def tool_search_quote_history(search_terms: list[str], limit: int = 5) -> list[dict[str, Any]]:
    """Look up similar past customer quotes by keyword.

    ``search_terms`` is a list of words/phrases (e.g. ``["wedding","invitations"]``).
    Returns up to ``limit`` matching historical quotes ordered most-recent first.
    Each row includes job_description, event_type, items, total_price, and date.
    """
    return ps.search_quote_history(search_terms, limit)


def tool_get_stock_level(item_name: str, as_of_date: str) -> int:
    """Sanity-check current stock for a SKU before quoting (YYYY-MM-DD date)."""
    return ps.get_stock_level(item_name, as_of_date)
