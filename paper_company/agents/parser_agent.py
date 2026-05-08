"""Request-parser agent.

Extracts structured ``LineItem`` rows from free-text customer requests. Has
no DB tools — the catalog is injected into the system prompt so the LLM
can map customer language ("200 reams of A4 paper") to canonical SKUs.
"""

from __future__ import annotations

from pydantic_ai import Agent

from models.schemas import ParsedItems
from project_starter import get_catalog
from .config import PYDANTIC_AI_MODEL


def _build_catalog_block() -> str:
    rows = get_catalog()
    lines = ["Available SKUs (use the exact item_name in your output):"]
    for r in rows:
        lines.append(
            f"- {r['item_name']} (category: {r['category']}, list price: ${r['unit_price']:.2f})"
        )
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are the request parser for Beaver's Choice Paper Company.
Read the customer's free-text request and extract a list of line items
(SKU and integer quantity). Map customer language to canonical SKUs from
the catalog below. If the customer mentions a product not in the catalog,
omit it and lower your parse_confidence accordingly.

{_build_catalog_block()}

Output rules:
- Only emit SKUs that appear in the catalog above (exact item_name match).
- Quantities must be positive integers.
- Set parse_confidence in [0,1]: 1.0 = unambiguous, 0.5 = best guess, 0.0 = could not interpret.
- If the request is gibberish or asks for non-paper goods, return an empty items list and parse_confidence=0.
- Use the notes field to record assumptions (e.g. "interpreted '5 reams of A4' as 5 packs of A4_paper_500").
"""


parser_agent = Agent(
    PYDANTIC_AI_MODEL,
    result_type=ParsedItems,
    system_prompt=SYSTEM_PROMPT,
    defer_model_check=True,
)
