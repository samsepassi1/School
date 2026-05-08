"""External-database tools that read/write the CultPass system.

These demonstrate that UDA-Hub can act on the customer's *external* system
(not just our own core DB). They are the natural fit for an MCP server in
production — see ``agentic/tools/__init__.py``.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from agentic.tools._helpers import err, ok
from data.core import db as core_db
from data.external import db as ext_db


logger = logging.getLogger(__name__)


def _resolve_member_id(user_id: str | None, member_id: str | None) -> str | None:
    if member_id:
        return member_id
    if user_id:
        row = core_db.fetch_one(
            "SELECT a.external_member_id "
            "FROM User u JOIN Account a USING(account_id) WHERE u.user_id=?",
            (user_id,),
        )
        if row:
            return row["external_member_id"]
    return None


@tool
def cultpass_member_lookup(user_id: str = "", member_id: str = "") -> str:
    """Fetch the CultPass member profile and recent payments.

    Pass either ``user_id`` (UDA-Hub side) or ``member_id`` (CultPass side).
    Use this when a billing question needs the underlying CultPass charge
    history (e.g. duplicate charges, failed payments).
    """
    mid = _resolve_member_id(user_id or None, member_id or None)
    if not mid:
        return err("could not resolve member_id from inputs",
                   user_id=user_id, member_id=member_id)
    member = ext_db.fetch_one("SELECT * FROM CultPassMember WHERE member_id=?", (mid,))
    if member is None:
        return err("member not found in CultPass", member_id=mid)
    payments = ext_db.fetch_all(
        "SELECT * FROM CultPassPayment WHERE member_id=? ORDER BY created_at DESC LIMIT 10",
        (mid,),
    )
    logger.info("tool.cultpass_member_lookup mid=%s payments=%d", mid, len(payments))
    return ok({"member": member, "payments": payments})


@tool
def cultpass_list_bookings(user_id: str = "", member_id: str = "", limit: int = 5) -> str:
    """List the member's most recent bookings (with event details)."""
    mid = _resolve_member_id(user_id or None, member_id or None)
    if not mid:
        return err("could not resolve member_id from inputs",
                   user_id=user_id, member_id=member_id)
    rows = ext_db.fetch_all(
        """
        SELECT b.booking_id, b.status, b.plus_one, b.booked_at,
               e.event_id, e.title, e.venue, e.city, e.starts_at, e.category
          FROM CultPassBooking b JOIN CultPassEvent e USING(event_id)
         WHERE b.member_id=?
         ORDER BY b.booked_at DESC
         LIMIT ?
        """,
        (mid, max(1, min(limit, 25))),
    )
    return ok({"member_id": mid, "bookings": rows})


@tool
def cultpass_cancel_booking(booking_id: str, reason: str = "customer_request") -> str:
    """Cancel a CultPass booking by id. Booking must be in 'confirmed' or
    'waitlisted' status. Past events cannot be cancelled."""
    if not booking_id:
        return err("booking_id is required")
    booking = ext_db.fetch_one("SELECT * FROM CultPassBooking WHERE booking_id=?", (booking_id,))
    if booking is None:
        return err("booking not found", booking_id=booking_id)
    if booking["status"] not in {"confirmed", "waitlisted"}:
        return err(f"booking status is '{booking['status']}', cannot cancel",
                   booking_id=booking_id)
    ext_db.execute(
        "UPDATE CultPassBooking SET status='cancelled' WHERE booking_id=?",
        (booking_id,),
    )
    logger.info("tool.cultpass_cancel_booking booking_id=%s reason=%s",
                booking_id, reason)
    return ok({
        "booking_id": booking_id,
        "previous_status": booking["status"],
        "new_status": "cancelled",
        "reason": reason,
    })
