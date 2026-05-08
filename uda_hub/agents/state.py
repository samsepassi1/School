"""Shared LangGraph state schema for UDA-Hub agents."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


Category = Literal["billing", "technical", "account", "security", "privacy", "other"]
Urgency = Literal["low", "normal", "high", "critical"]
Sentiment = Literal["positive", "neutral", "negative"]
RouteDecision = Literal["resolver", "escalation", "end"]


class Classification(TypedDict, total=False):
    category: Category
    urgency: Urgency
    sentiment: Sentiment
    confidence: float
    reasoning: str


class RetrievedDoc(TypedDict):
    article_id: str
    title: str
    category: str
    body: str
    score: float


class AgentState(TypedDict, total=False):
    # Identity
    ticket_id: str
    user_id: str
    thread_id: str

    # Inbound ticket
    subject: str
    body: str
    channel: str
    urgency_in: str

    # Conversation messages exchanged with the LLM (short-term memory inside one run)
    messages: Annotated[list[AnyMessage], add_messages]

    # Pulled from long-term memory at the start of a run
    customer_profile: dict[str, Any]
    customer_history: list[dict[str, Any]]
    customer_preferences: dict[str, Any]

    # Filled in by the classifier
    classification: Classification

    # Filled in by the retrieval agent
    retrieved: list[RetrievedDoc]
    retrieval_confidence: float

    # Routing decision from the supervisor
    route: RouteDecision

    # Output
    answer: str
    needs_escalation: bool
    escalation_reason: str

    # Audit trail (visible to the demo notebook)
    log: list[str]
