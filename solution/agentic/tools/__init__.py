"""Support-operation tools available to the Resolver agent.

Each tool wraps the database (core or external) behind a typed,
LangChain-compatible callable. They are written so they can also be exposed
through an MCP server (FastMCP) without code changes — see
``agentic/tools/mcp_server.py`` for the reference wrapper.
"""

from agentic.tools.uda_account import (
    lookup_account,
    process_refund,
    update_subscription,
    get_ticket_history,
)
from agentic.tools.cultpass import (
    cultpass_member_lookup,
    cultpass_list_bookings,
    cultpass_cancel_booking,
)


ALL_TOOLS = [
    # core (UDA-Hub) tools
    lookup_account,
    process_refund,
    update_subscription,
    get_ticket_history,
    # external (CultPass) tools
    cultpass_member_lookup,
    cultpass_list_bookings,
    cultpass_cancel_booking,
]
