"""Shared LangGraph state schema."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


Category = Literal["billing", "technical", "account", "security", "privacy",
                   "bookings", "membership", "other"]
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

    # Inbound
    subject: str
    body: str
    channel: str
    urgency_in: str

    # Conversation (in-run, short-term inside a single graph invocation)
    messages: Annotated[list[AnyMessage], add_messages]

    # Pulled from long-term memory at the start
    customer_profile: dict[str, Any]
    customer_history: list[dict[str, Any]]
    customer_preferences: dict[str, Any]

    # Filled by classifier / retriever
    classification: Classification
    retrieved: list[RetrievedDoc]
    retrieval_confidence: float

    # Routing decision
    route: RouteDecision

    # Output
    answer: str
    needs_escalation: bool
    escalation_reason: str

    # Audit trail
    log: list[str]
