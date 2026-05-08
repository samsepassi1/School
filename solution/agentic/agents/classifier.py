"""Classifier agent — produces structured (category / urgency / sentiment / confidence)."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic.agents.state import AgentState
from agentic.llm import get_llm
from data.core import db


logger = logging.getLogger(__name__)


class _ClassifySchema(BaseModel):
    category: str = Field(
        description="One of: bookings, billing, technical, account, security, "
                    "membership, privacy, other"
    )
    urgency: str = Field(description="One of: low, normal, high, critical")
    sentiment: str = Field(description="One of: positive, neutral, negative")
    confidence: float = Field(ge=0.0, le=1.0,
                              description="0..1 confidence in this classification")
    reasoning: str = Field(description="One sentence justification")


SYSTEM = (
    "You are the Classifier agent for a Cult Pass support AI. "
    "Read the ticket and produce a structured classification. Be conservative "
    "with urgency: 'critical' is reserved for outages, fraud, or data loss. "
    "If the customer is angry or has been billed incorrectly, sentiment is "
    "'negative'. Confidence reflects how sure you are about the category."
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
    result: _ClassifySchema = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=ticket_text + history_hint),
    ])
    classification = result.model_dump()
    logger.info(
        "classifier ticket=%s category=%s urgency=%s sentiment=%s conf=%.2f",
        state.get("ticket_id"), classification["category"],
        classification["urgency"], classification["sentiment"],
        classification["confidence"],
    )
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
