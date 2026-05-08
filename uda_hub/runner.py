"""Convenience helper to run a single ticket through the compiled graph."""

from __future__ import annotations

import logging
from typing import Any

from uda_hub.agents.state import AgentState


logger = logging.getLogger(__name__)


def run_ticket(app, ticket: dict[str, Any]) -> dict[str, Any]:
    """Invoke the compiled graph on one ticket.

    The ``thread_id`` field on the ticket controls short-term memory: reusing
    one resumes that conversation, picking a new one starts fresh.
    """
    thread_id = ticket.get("thread_id") or f"thread-{ticket['ticket_id']}"
    config = {"configurable": {"thread_id": thread_id}}

    initial: AgentState = {
        "ticket_id": ticket["ticket_id"],
        "user_id": ticket.get("user_id", ""),
        "thread_id": thread_id,
        "subject": ticket.get("subject", ""),
        "body": ticket.get("body", ""),
        "channel": ticket.get("channel", "web"),
        "urgency_in": ticket.get("urgency", "normal"),
        "messages": [],
        "log": [],
    }

    final_state = app.invoke(initial, config=config)
    logger.info(
        "run_ticket ticket=%s -> escalation=%s",
        ticket["ticket_id"],
        final_state.get("needs_escalation"),
    )
    return final_state
