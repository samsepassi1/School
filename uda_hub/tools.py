"""Support-operation tools exposed to the Resolver agent.

Each tool is a thin wrapper over the SQLite database. They are decorated with
``@tool`` so LangGraph / LangChain can bind them to an LLM. Validation and
error reporting are explicit so agents can recover gracefully.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from uda_hub import db


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ok(payload: dict[str, Any]) -> str:
    return json.dumps({"status": "ok", **payload}, default=str)


def _err(message: str, **extra: Any) -> str:
    return json.dumps({"status": "error", "error": message, **extra})


# --------------------------------------------------------------------------- #
# 1. Account & subscription lookup
# --------------------------------------------------------------------------- #

@tool
def lookup_account(user_id: str) -> str:
    """Return the user's profile, account, plan, and current billing status.

    Use this when you need to verify who the customer is or check their plan
    before taking another action (refunds, upgrades, etc.).
    """
    if not user_id or not user_id.startswith("usr_"):
        return _err("user_id must look like 'usr_xxx'", user_id=user_id)
    row = db.fetch_one(
        """
        SELECT u.user_id, u.email, u.full_name, u.locale,
               a.account_id, a.name AS account_name, a.plan,
               a.status AS account_status, a.balance_cents
          FROM User u JOIN Account a USING(account_id)
         WHERE u.user_id = ?
        """,
        (user_id,),
    )
    if row is None:
        return _err("user not found", user_id=user_id)
    logger.info(
        "tool.lookup_account user_id=%s -> plan=%s status=%s",
        user_id, row["plan"], row["account_status"],
    )
    return _ok({"account": row})


# --------------------------------------------------------------------------- #
# 2. Refund processing
# --------------------------------------------------------------------------- #

@tool
def process_refund(user_id: str, amount_cents: int, reason: str) -> str:
    """Issue a refund for the given user.

    Args:
      user_id: 'usr_xxx' identifier.
      amount_cents: positive integer in cents (e.g. 1999 for $19.99).
      reason: short human-readable justification (audit log).

    Refunds above $50 (5000 cents) are NOT auto-approved and the tool returns
    an error so the agent can escalate.
    """
    if amount_cents <= 0:
        return _err("amount_cents must be positive", amount_cents=amount_cents)
    if amount_cents > 5000:
        return _err(
            "amount exceeds auto-refund cap (5000 cents); please escalate",
            amount_cents=amount_cents,
        )
    user = db.fetch_one(
        "SELECT account_id FROM User WHERE user_id=?", (user_id,)
    )
    if user is None:
        return _err("user not found", user_id=user_id)

    # Decrease account balance (negative balance = credit owed to customer).
    db.execute(
        "UPDATE Account SET balance_cents = balance_cents - ? WHERE account_id=?",
        (amount_cents, user["account_id"]),
    )
    refund_id = f"rfd_{user_id}_{amount_cents}"
    logger.info(
        "tool.process_refund user_id=%s amount_cents=%s reason=%r -> %s",
        user_id, amount_cents, reason, refund_id,
    )
    return _ok(
        {
            "refund_id": refund_id,
            "user_id": user_id,
            "account_id": user["account_id"],
            "amount_cents": amount_cents,
            "reason": reason,
            "eta": "5-10 business days",
        }
    )


# --------------------------------------------------------------------------- #
# 3. Subscription change
# --------------------------------------------------------------------------- #

@tool
def update_subscription(user_id: str, new_plan: str) -> str:
    """Change the plan on the user's account.

    Args:
      user_id: 'usr_xxx' identifier.
      new_plan: one of 'free', 'basic', 'pro', 'enterprise'.

    Enterprise upgrades are NOT auto-approved (sales review required).
    """
    valid = {"free", "basic", "pro", "enterprise"}
    if new_plan not in valid:
        return _err(f"new_plan must be one of {sorted(valid)}", new_plan=new_plan)
    if new_plan == "enterprise":
        return _err("enterprise upgrades require sales approval; please escalate")
    user = db.fetch_one(
        "SELECT a.account_id, a.plan FROM User u JOIN Account a USING(account_id) WHERE u.user_id=?",
        (user_id,),
    )
    if user is None:
        return _err("user not found", user_id=user_id)
    db.execute(
        "UPDATE Account SET plan=? WHERE account_id=?", (new_plan, user["account_id"])
    )
    logger.info(
        "tool.update_subscription user_id=%s %s -> %s", user_id, user["plan"], new_plan
    )
    return _ok(
        {
            "user_id": user_id,
            "account_id": user["account_id"],
            "previous_plan": user["plan"],
            "new_plan": new_plan,
        }
    )


# --------------------------------------------------------------------------- #
# 4. Customer ticket history
# --------------------------------------------------------------------------- #

@tool
def get_ticket_history(user_id: str, limit: int = 5) -> str:
    """Return the user's most recent tickets and their resolution status."""
    rows = db.fetch_all(
        """
        SELECT t.ticket_id, t.subject, t.status, t.created_at,
               m.category, m.urgency, m.routed_to
          FROM Ticket t
          LEFT JOIN TicketMetadata m USING(ticket_id)
         WHERE t.user_id=?
         ORDER BY t.created_at DESC
         LIMIT ?
        """,
        (user_id, max(1, min(limit, 25))),
    )
    return _ok({"user_id": user_id, "tickets": rows})


# --------------------------------------------------------------------------- #
# Registry consumed by the Resolver agent
# --------------------------------------------------------------------------- #

ALL_TOOLS = [lookup_account, process_refund, update_subscription, get_ticket_history]
