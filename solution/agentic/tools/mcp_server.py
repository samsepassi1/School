"""Optional FastMCP wrapper around the UDA-Hub tools.

Run with::

    pip install fastmcp
    python -m agentic.tools.mcp_server          # stdio (default)

The same tool functions are reused — the only thing this file does is expose
them through the Model Context Protocol so external IDEs/clients can call
them directly. The agents themselves still call the LangChain ``@tool``
versions for in-process performance.
"""

from __future__ import annotations


def main() -> None:  # pragma: no cover (exercised manually)
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit(
            "fastmcp is not installed. `pip install fastmcp` to enable this server."
        ) from exc

    from agentic.tools import (
        cultpass_cancel_booking,
        cultpass_list_bookings,
        cultpass_member_lookup,
        get_ticket_history,
        lookup_account,
        process_refund,
        update_subscription,
    )

    mcp = FastMCP("uda-hub")
    for tool in (
        lookup_account,
        process_refund,
        update_subscription,
        get_ticket_history,
        cultpass_member_lookup,
        cultpass_list_bookings,
        cultpass_cancel_booking,
    ):
        mcp.tool(tool.name, tool.description)(tool.func)
    mcp.run()


if __name__ == "__main__":
    main()
