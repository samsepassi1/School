"""Escalation agent.

Triggered when:
  - Retrieval confidence is below the threshold AND no actionable tool fits, OR
  - Classifier flags urgency=critical, OR
  - Resolver explicitly hands off (e.g. tool returned auto-cap exceeded).

Produces a human-readable escalation note: a concise summary, the steps already
attempted, suggested owner team, and a priority. Marks the ticket as escalated
in the database.
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


class _EscalationSchema(BaseModel):
    summary: str = Field(description="2-3 sentence summary of the case for the human agent")
    suggested_team: str = Field(
        description="One of: tier2_support, billing_ops, engineering, trust_safety, sales"
    )
    priority: str = Field(description="One of: P1, P2, P3, P4")
    next_steps: list[str] = Field(description="3-5 short bullet points")


SYSTEM = (
    "You are the Escalation agent. The automated system could not resolve this ticket. "
    "Write a brief handoff note for a human agent. Do NOT promise outcomes to the customer; "
    "instead, acknowledge the issue and set expectations on response time."
)


_TEAM_TO_TEMPLATE = {
    "billing_ops": "Our billing operations team will review your case and respond within 1 business day.",
    "engineering": "I've escalated this to our engineering on-call team — expect an update within 2 hours.",
    "tier2_support": "I've routed this to a senior support specialist who will follow up within 4 hours.",
    "trust_safety": "Our trust & safety team will investigate and respond within 1 business day.",
    "sales": "Our sales team will reach out within 1 business day to discuss options.",
}


def escalation_node(state: AgentState) -> dict[str, Any]:
    llm = get_llm().with_structured_output(_EscalationSchema)

    user_block = (
        f"Ticket {state.get('ticket_id')} from user {state.get('user_id')}\n"
        f"Subject: {state.get('subject','')}\n"
        f"Body: {state.get('body','')}\n"
        f"Classification: {state.get('classification', {})}\n"
        f"Retrieval confidence: {state.get('retrieval_confidence', 0):.2f}\n"
        f"Reason for escalation: {state.get('escalation_reason','low confidence / no resolution')}"
    )

    decision: _EscalationSchema = llm.invoke(
        [SystemMessage(content=SYSTEM), HumanMessage(content=user_block)]
    )
    decision_dict = decision.model_dump()
    customer_msg = _TEAM_TO_TEMPLATE.get(
        decision.suggested_team, "I've escalated this to a human agent who will follow up shortly."
    )
    full_reply = (
        "Thanks for reaching out — I want to make sure this gets the right attention.\n\n"
        f"{customer_msg}\n\n"
        "Reference: " + (state.get("ticket_id") or "(pending)")
    )

    logger.info(
        "escalation ticket=%s team=%s priority=%s",
        state.get("ticket_id"),
        decision.suggested_team,
        decision.priority,
    )

    if state.get("ticket_id"):
        db.append_message(
            state["ticket_id"],
            role="agent",
            author="escalation",
            content=f"[ESCALATED to {decision.suggested_team} / {decision.priority}] {decision.summary}",
        )
        db.append_message(
            state["ticket_id"], role="agent", author="escalation", content=full_reply
        )
        db.update_ticket_status(state["ticket_id"], "escalated")
        db.upsert_ticket_metadata(
            state["ticket_id"],
            routed_to="escalation",
            extra={"escalation": decision_dict},
        )

    log = list(state.get("log", []))
    log.append(
        f"escalation -> team={decision.suggested_team} priority={decision.priority}"
    )
    return {
        "answer": full_reply,
        "needs_escalation": True,
        "escalation_reason": decision.summary,
        "log": log,
    }
