"""Supervisor / orchestration nodes (hydrate + routing + memory writer)."""

from __future__ import annotations

import logging
from typing import Any, Literal

from agentic.agents.state import AgentState
from agentic.config import settings
from agentic.memory import (
    recall_customer_history,
    recall_preferences,
    remember_resolution,
)
from data.core import db


logger = logging.getLogger(__name__)


def hydrate_node(state: AgentState) -> dict[str, Any]:
    """Pull the customer's profile + history before classification."""
    user_id = state.get("user_id")
    profile = None
    if user_id:
        profile = db.fetch_one(
            """
            SELECT u.user_id, u.email, u.full_name, u.locale,
                   a.account_id, a.plan, a.status, a.external_member_id
              FROM User u JOIN Account a USING(account_id)
             WHERE u.user_id=?
            """,
            (user_id,),
        )

    if state.get("ticket_id") and user_id:
        existing = db.fetch_one(
            "SELECT 1 FROM Ticket WHERE ticket_id=?", (state["ticket_id"],)
        )
        if not existing:
            db.execute(
                "INSERT INTO Ticket(ticket_id,user_id,subject,body,status) VALUES (?,?,?,?,?)",
                (state["ticket_id"], user_id, state.get("subject", ""),
                 state.get("body", ""), "open"),
            )
            db.append_message(
                state["ticket_id"], role="customer", author=user_id,
                content=f"{state.get('subject','')}\n\n{state.get('body','')}",
            )
            db.upsert_ticket_metadata(
                state["ticket_id"],
                channel=state.get("channel", "web"),
                urgency=state.get("urgency_in", "normal"),
            )

    history = recall_customer_history(user_id, limit=5) if user_id else []
    prefs = recall_preferences(user_id) if user_id else {}
    log = list(state.get("log", []))
    log.append(
        f"hydrate -> profile={'yes' if profile else 'no'} "
        f"history={len(history)} prefs={list(prefs.keys())}"
    )
    return {
        "customer_profile": profile or {},
        "customer_history": history,
        "customer_preferences": prefs,
        "log": log,
    }


def supervisor_router(state: AgentState) -> Literal["resolver", "escalation"]:
    """Routing decision used as a conditional edge after retrieval."""
    classification = state.get("classification") or {}
    urgency = classification.get("urgency", state.get("urgency_in", "normal"))
    sentiment = classification.get("sentiment", "neutral")
    confidence = state.get("retrieval_confidence", 0.0)
    threshold = settings.confidence_threshold

    if urgency == "critical":
        logger.info("router -> escalation (urgency=critical)")
        return "escalation"
    if sentiment == "negative" and confidence < threshold:
        logger.info("router -> escalation (negative sentiment, conf=%.2f)", confidence)
        return "escalation"
    if confidence < threshold and not _looks_like_actionable(state):
        logger.info("router -> escalation (low conf=%.2f)", confidence)
        return "escalation"
    return "resolver"


def _looks_like_actionable(state: AgentState) -> bool:
    text = f"{state.get('subject','')} {state.get('body','')}".lower()
    actionables = ("refund", "charge", "downgrade", "upgrade", "cancel",
                   "pause", "booking", "ticket", "reservation")
    classification = state.get("classification") or {}
    if classification.get("category") in {"billing", "account", "bookings", "membership"} \
            and any(a in text for a in actionables):
        return True
    return False


def memory_writer_node(state: AgentState) -> dict[str, Any]:
    user_id = state.get("user_id")
    ticket_id = state.get("ticket_id")
    if not user_id or not ticket_id:
        return {}
    summary = {
        "ticket_id": ticket_id,
        "subject": state.get("subject"),
        "category": (state.get("classification") or {}).get("category"),
        "urgency": (state.get("classification") or {}).get("urgency"),
        "outcome": "escalated" if state.get("needs_escalation") else "resolved",
        "answer_preview": (state.get("answer") or "")[:160],
    }
    remember_resolution(user_id, ticket_id, summary)
    log = list(state.get("log", []))
    log.append("memory_writer -> stored long-term summary")
    return {"log": log}
