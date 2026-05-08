"""Inventory agent.

Given a list of needed items, checks current stock and queries supplier
ETAs for any shortfalls. Tools wrap ``get_all_inventory``,
``get_stock_level``, and ``get_supplier_delivery_date``.
"""

from __future__ import annotations

from pydantic_ai import Agent

from models.schemas import InventoryReport
from tools.inventory_tools import (
    tool_get_all_inventory,
    tool_get_stock_level,
    tool_get_supplier_eta,
)
from .config import PYDANTIC_AI_MODEL


SYSTEM_PROMPT = """You are the inventory analyst for Beaver's Choice Paper Company.
You receive a JSON object with the items the customer wants, the request
date, and the deadline. For each item:

1. Use tool_get_all_inventory(as_of_date) once at the start to get the current
   stock for every SKU.
2. For each requested SKU, compute shortfall = max(needed - on_hand, 0).
3. If shortfall > 0, call tool_get_supplier_eta(order_date=request_date,
   quantity=shortfall) to estimate when restock would arrive. Compare against
   the deadline: viable_by_deadline = (eta <= deadline).
4. Set restock_needed = True if any shortfall > 0.
5. Set all_available = True only if every line has shortfall == 0.

Return one InventoryLine per requested SKU in the same order as the input.
Do not omit lines, even if shortfall is 0.
"""


inventory_agent = Agent(
    PYDANTIC_AI_MODEL,
    result_type=InventoryReport,
    system_prompt=SYSTEM_PROMPT,
    defer_model_check=True,
)

inventory_agent.tool_plain(tool_get_all_inventory)
inventory_agent.tool_plain(tool_get_stock_level)
inventory_agent.tool_plain(tool_get_supplier_eta)
