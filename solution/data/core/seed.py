"""Seed the core UDA-Hub database (Account/User/Ticket/...) plus load
the Knowledge table from ``data/external/cultpass_articles.jsonl``.

Account rows are 1:1 with CultPass members so cross-DB joins on
``external_member_id`` are trivial.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentic.config import settings
from data.core import db


# Mirrors the CultPass member roster (data/external/seed.py::MEMBERS),
# adjusted to satisfy the rubric's Account schema. ``plan`` and ``status``
# follow the CultPass member tier so cross-DB queries stay aligned.
ACCOUNTS = [
    dict(account_id="acc_001", name="Alice Nguyen",   plan="plus",    status="active",   balance_cents=0,    external_member_id="cp_m_001"),
    dict(account_id="acc_002", name="Bob Martinez",   plan="classic", status="past_due", balance_cents=999,  external_member_id="cp_m_002"),
    dict(account_id="acc_003", name="Carla Schmidt",  plan="elite",   status="active",   balance_cents=0,    external_member_id="cp_m_003"),
    dict(account_id="acc_004", name="Diego Park",     plan="classic", status="active",   balance_cents=0,    external_member_id="cp_m_004"),
    dict(account_id="acc_005", name="Emi Tanaka",     plan="plus",    status="paused",   balance_cents=0,    external_member_id="cp_m_005"),
    dict(account_id="acc_006", name="Frank Li",       plan="classic", status="active",   balance_cents=0,    external_member_id="cp_m_006"),
]

USERS = [
    dict(user_id="usr_001", account_id="acc_001", email="alice@cultpass.test",  full_name="Alice Nguyen",  locale="en-GB"),
    dict(user_id="usr_002", account_id="acc_002", email="bob@cultpass.test",    full_name="Bob Martinez",  locale="es-ES"),
    dict(user_id="usr_003", account_id="acc_003", email="carla@cultpass.test",  full_name="Carla Schmidt", locale="de-DE"),
    dict(user_id="usr_004", account_id="acc_004", email="diego@cultpass.test",  full_name="Diego Park",    locale="pt-PT"),
    dict(user_id="usr_005", account_id="acc_005", email="emi@cultpass.test",    full_name="Emi Tanaka",    locale="ja-JP"),
    dict(user_id="usr_006", account_id="acc_006", email="frank@cultpass.test",  full_name="Frank Li",      locale="en-GB"),
]

PRIOR_TICKETS = [
    dict(
        ticket_id="tkt_seed_001",
        user_id="usr_001",
        subject="Password reset email not arriving",
        body="I requested a password reset twice and never got the email.",
        status="resolved",
        meta=dict(channel="email", urgency="normal", category="account", sentiment="neutral", routed_to="resolver"),
        messages=[
            ("customer", "I requested a password reset twice and never got the email."),
            ("agent",    "Please check your spam folder and ensure alice@cultpass.test is correct. We have re-sent the reset email."),
            ("customer", "Found it in spam — thanks!"),
        ],
    ),
    dict(
        ticket_id="tkt_seed_002",
        user_id="usr_002",
        subject="Charged twice for May membership",
        body="My card was charged for the monthly membership twice this month.",
        status="resolved",
        meta=dict(channel="web", urgency="high", category="billing", sentiment="negative", routed_to="resolver"),
        messages=[
            ("customer", "My card was charged for the monthly membership twice this month."),
            ("tool",     "process_refund: refunded 999 cents to acc_002"),
            ("agent",    "Refund issued for the duplicate charge. It will appear in 5-10 business days."),
        ],
    ),
    dict(
        ticket_id="tkt_seed_003",
        user_id="usr_003",
        subject="App keeps crashing on iOS 17",
        body="The Cult Pass app crashes whenever I open the Wallet to view a ticket.",
        status="escalated",
        meta=dict(channel="email", urgency="critical", category="technical", sentiment="negative", routed_to="escalation"),
        messages=[
            ("customer", "The Cult Pass app crashes whenever I open the Wallet to view a ticket."),
            ("agent",    "Escalating to mobile engineering — we're tracking this under MOB-118."),
        ],
    ),
]


def _load_articles_jsonl(path: str) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def seed_all(
    core_db_path: str | None = None,
    *,
    articles_path: str | None = None,
    reset: bool = True,
) -> dict[str, int]:
    if reset:
        db.reset_db(core_db_path)
    else:
        db.init_db(core_db_path)

    counts: dict[str, int] = {}
    counts["Account"] = db.insert_many("Account", ACCOUNTS, core_db_path)
    counts["User"] = db.insert_many("User", USERS, core_db_path)

    # Knowledge — loaded from the JSONL file shipped by CultPass.
    articles = _load_articles_jsonl(articles_path or settings.cultpass_articles_path)
    counts["Knowledge"] = db.insert_many("Knowledge", articles, core_db_path)

    # Prior tickets + metadata + messages (so returning customers have history).
    ticket_rows, meta_rows, message_rows = [], [], []
    for t in PRIOR_TICKETS:
        ticket_rows.append(dict(
            ticket_id=t["ticket_id"], user_id=t["user_id"],
            subject=t["subject"], body=t["body"], status=t["status"],
        ))
        meta_rows.append(dict(
            ticket_id=t["ticket_id"],
            channel=t["meta"]["channel"], urgency=t["meta"]["urgency"],
            category=t["meta"]["category"], sentiment=t["meta"]["sentiment"],
            confidence=0.9, routed_to=t["meta"]["routed_to"], extra_json=None,
        ))
        for role, content in t["messages"]:
            message_rows.append(dict(
                ticket_id=t["ticket_id"], role=role,
                author=t["user_id"] if role == "customer" else role,
                content=content,
            ))
    counts["Ticket"] = db.insert_many("Ticket", ticket_rows, core_db_path)
    counts["TicketMetadata"] = db.insert_many("TicketMetadata", meta_rows, core_db_path)
    counts["TicketMessage"] = db.insert_many("TicketMessage", message_rows, core_db_path)
    return counts
