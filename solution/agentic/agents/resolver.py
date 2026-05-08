"""Resolver agent — composes a customer-facing reply, may call tools (ReAct)."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agentic.agents.state import AgentState
from agentic.llm import get_llm
from agentic.tools import ALL_TOOLS
from data.core import db


logger = logging.getLogger(__name__)


SYSTEM = """You are the Resolver agent for a Cult Pass customer-support AI.

Compose a clear, friendly reply for the customer using ONLY information from
the provided knowledge-base articles or from tool call results. Never invent
policy, prices, or product behaviour.

You may call tools to take action (refund, plan change, lookup, cancel
booking). Rules:
- Verify the customer with `lookup_account` before any account-changing action.
- For billing complaints (e.g. duplicate charges), use `cultpass_member_lookup`
  to confirm the underlying CultPass payments before refunding.
- If a tool returns status="error", do not retry the same call — explain the
  limitation in your reply and let the supervisor escalate if needed.
- Keep replies concise (3-6 sentences) and end with a clear next step.
- Cite article ids inline like [cp_kb_005] when you use a KB article.
"""


_TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def _format_kb(retrieved: list[dict]) -> str:
    if not retrieved:
        return "(no matching knowledge-base articles)"
    parts = []
    for d in retrieved:
        parts.append(
            f"[{d['article_id']}] {d['title']} (category={d['category']}, score={d['score']:.2f})\n{d['body']}"
        )
    return "\n\n---\n\n".join(parts)


def resolver_node(state: AgentState) -> dict[str, Any]:
    llm = get_llm().bind_tools(ALL_TOOLS)

    user_block = (
        f"Ticket {state.get('ticket_id','?')} from {state.get('user_id','?')}\n"
        f"Channel: {state.get('channel','web')}  Urgency: {state.get('urgency_in','normal')}\n"
        f"Subject: {state.get('subject','')}\n"
        f"Body: {state.get('body','')}"
    )
    classification = state.get("classification") or {}
    if classification:
        user_block += (
            f"\n\nClassification: {classification.get('category')} "
            f"({classification.get('urgency')}, sentiment={classification.get('sentiment')})"
        )
    history = state.get("customer_history") or []
    if history:
        user_block += "\n\nPrior tickets:\n" + "\n".join(
            f"- {h.get('subject','?')}: {h.get('outcome','?')}" for h in history[:3]
        )
    prefs = state.get("customer_preferences") or {}
    if prefs:
        user_block += f"\n\nCustomer preferences: {prefs}"
    user_block += "\n\nKnowledge base context:\n" + _format_kb(state.get("retrieved", []))

    messages: list = [SystemMessage(content=SYSTEM), HumanMessage(content=user_block)]

    tool_log: list[str] = []
    final_text = ""
    for _ in range(3):
        ai: AIMessage = llm.invoke(messages)
        messages.append(ai)
        tool_calls = getattr(ai, "tool_calls", []) or []
        if not tool_calls:
            final_text = (ai.content or "").strip()
            break
        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            tool = _TOOLS_BY_NAME.get(name)
            if tool is None:
                result = f'{{"status":"error","error":"unknown tool {name}"}}'
            else:
                try:
                    result = tool.invoke(args)
                except Exception as exc:  # pragma: no cover
                    result = f'{{"status":"error","error":"{exc}"}}'
            tool_log.append(f"{name}({args}) -> {str(result)[:120]}")
            messages.append(ToolMessage(content=str(result),
                                        tool_call_id=call["id"], name=name))
            if state.get("ticket_id"):
                db.append_message(
                    state["ticket_id"], role="tool", author=name,
                    content=f"{name}({args}) -> {result}",
                )
    else:
        final_text = "I couldn't complete this automatically — escalating to a human agent."

    log = list(state.get("log", []))
    if tool_log:
        log.append(f"resolver -> tools: {tool_log}")
    log.append("resolver -> drafted answer")

    if state.get("ticket_id") and final_text:
        db.append_message(state["ticket_id"], role="agent", author="resolver",
                          content=final_text)
        db.update_ticket_status(state["ticket_id"], "resolved")
        db.upsert_ticket_metadata(state["ticket_id"], routed_to="resolver")

    return {"answer": final_text, "needs_escalation": False, "log": log}
