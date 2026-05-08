"""Classifier agent.

Reads the inbound ticket plus any short-term context from `state["messages"]`
and produces a structured classification (category / urgency / sentiment /
confidence). Persists the result into TicketMetadata for audit.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from uda_hub import db
from uda_hub.agents.state import AgentState
from uda_hub.llm import get_llm


logger = logging.getLogger(__name__)


class _ClassifySchema(BaseModel):
    category: str = Field(
        description="One of: billing, technical, account, security, privacy, other"
    )
    urgency: str = Field(description="One of: low, normal, high, critical")
    sentiment: str = Field(description="One of: positive, neutral, negative")
    confidence: float = Field(description="0..1 confidence in this classification", ge=0.0, le=1.0)
    reasoning: str = Field(description="One sentence justification")


SYSTEM = (
    "You are the Classifier agent in a customer-support AI system. "
    "Read the ticket and produce a structured classification. "
    "Be conservative with urgency: 'critical' is reserved for outages or data loss "
    "affecting paying customers. If the user is angry or has been billed incorrectly, "
    "sentiment is 'negative'. Confidence reflects how sure you are about the category."
)


def classifier_node(state: AgentState) -> dict[str, Any]:
    llm = get_llm().with_structured_output(_ClassifySchema)
    ticket_text = f"Subject: {state.get('subject','')}\n\nBody: {state.get('body','')}"
    history_hint = ""
    if state.get("customer_history"):
        recent = state["customer_history"][:3]
        history_hint = "\n\nRecent prior issues:\n" + "\n".join(
            f"- {h.get('subject','?')} ({h.get('outcome','?')})" for h in recent
        )

    result: _ClassifySchema = llm.invoke(
        [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=ticket_text + history_hint),
        ]
    )
    classification = result.model_dump()
    logger.info(
        "classifier ticket=%s category=%s urgency=%s sentiment=%s conf=%.2f",
        state.get("ticket_id"),
        classification["category"],
        classification["urgency"],
        classification["sentiment"],
        classification["confidence"],
    )

    # Persist to TicketMetadata so the audit log captures it.
    if state.get("ticket_id"):
        db.upsert_ticket_metadata(
            state["ticket_id"],
            channel=state.get("channel", "web"),
            urgency=classification["urgency"],
            category=classification["category"],
            sentiment=classification["sentiment"],
            confidence=classification["confidence"],
        )

    log = list(state.get("log", []))
    log.append(
        f"classifier -> {classification['category']}/{classification['urgency']}/"
        f"{classification['sentiment']} (conf={classification['confidence']:.2f})"
    )
    return {"classification": classification, "log": log}
