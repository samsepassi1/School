"""Core UDA-Hub support tools (account lookup, refunds, plan changes, history).

These tools talk only to the core database (Account / User / Ticket / ...).
They are explicit about their authorisation rules (auto-refund cap,
enterprise/elite upgrades require sales) so the Resolver can read the rules
straight from the docstring and decide when to escalate.
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from agentic.tools._helpers import err, ok
from data.core import db


logger = logging.getLogger(__name__)


@tool
def lookup_account(user_id: str) -> str:
    """Return the user's profile, account, plan, and current billing status.

    Use this to verify who the customer is or check their plan before taking
    another action (refund, upgrade, ...).
    """
    if not user_id or not user_id.startswith("usr_"):
        return err("user_id must look like 'usr_xxx'", user_id=user_id)
    row = db.fetch_one(
        """
        SELECT u.user_id, u.email, u.full_name, u.locale,
               a.account_id, a.name AS account_name, a.plan,
               a.status AS account_status, a.balance_cents,
               a.external_member_id
          FROM User u JOIN Account a USING(account_id)
         WHERE u.user_id = ?
        """,
        (user_id,),
    )
    if row is None:
        return err("user not found", user_id=user_id)
    logger.info(
        "tool.lookup_account user_id=%s plan=%s status=%s",
        user_id, row["plan"], row["account_status"],
    )
    return ok({"account": row})


@tool
def process_refund(user_id: str, amount_cents: int, reason: str) -> str:
    """Issue a refund for the given user.

    Args:
      user_id: 'usr_xxx' identifier.
      amount_cents: positive integer in cents (e.g. 999 for $9.99).
      reason: short justification (audit log).

    Refunds above $50 (5000 cents) are NOT auto-approved — the tool returns an
    error so the agent can escalate.
    """
    if amount_cents <= 0:
        return err("amount_cents must be positive", amount_cents=amount_cents)
    if amount_cents > 5000:
        return err(
            "amount exceeds auto-refund cap (5000 cents); please escalate",
            amount_cents=amount_cents,
        )
    user = db.fetch_one("SELECT account_id FROM User WHERE user_id=?", (user_id,))
    if user is None:
        return err("user not found", user_id=user_id)
    db.execute(
        "UPDATE Account SET balance_cents = balance_cents - ? WHERE account_id=?",
        (amount_cents, user["account_id"]),
    )
    refund_id = f"rfd_{user_id}_{amount_cents}"
    logger.info("tool.process_refund user_id=%s amount=%s -> %s",
                user_id, amount_cents, refund_id)
    return ok({
        "refund_id": refund_id,
        "user_id": user_id,
        "account_id": user["account_id"],
        "amount_cents": amount_cents,
        "reason": reason,
        "eta": "5-10 business days",
    })


@tool
def update_subscription(user_id: str, new_plan: str) -> str:
    """Change the plan on the user's account.

    Args:
      user_id: 'usr_xxx' identifier.
      new_plan: one of 'classic', 'plus', 'elite'.

    Elite upgrades are NOT auto-approved (concierge/sales review required).
    """
    valid = {"classic", "plus", "elite"}
    if new_plan not in valid:
        return err(f"new_plan must be one of {sorted(valid)}", new_plan=new_plan)
    if new_plan == "elite":
        return err("elite upgrades require concierge approval; please escalate")
    user = db.fetch_one(
        "SELECT a.account_id, a.plan FROM User u JOIN Account a USING(account_id) WHERE u.user_id=?",
        (user_id,),
    )
    if user is None:
        return err("user not found", user_id=user_id)
    db.execute("UPDATE Account SET plan=? WHERE account_id=?", (new_plan, user["account_id"]))
    logger.info("tool.update_subscription user_id=%s %s -> %s",
                user_id, user["plan"], new_plan)
    return ok({
        "user_id": user_id,
        "account_id": user["account_id"],
        "previous_plan": user["plan"],
        "new_plan": new_plan,
    })


@tool
def get_ticket_history(user_id: str, limit: int = 5) -> str:
    """Return the user's most recent tickets and their resolution status."""
    rows = db.fetch_all(
        """
        SELECT t.ticket_id, t.subject, t.status, t.created_at,
               m.category, m.urgency, m.routed_to
          FROM Ticket t LEFT JOIN TicketMetadata m USING(ticket_id)
         WHERE t.user_id=?
         ORDER BY t.created_at DESC
         LIMIT ?
        """,
        (user_id, max(1, min(limit, 25))),
    )
    return ok({"user_id": user_id, "tickets": rows})
