"""Offline smoke tests — no LLM / network required.

Cover schema/seed for both DBs, knowledge ingest, keyword retrieval, every
tool's happy and error paths, the long-term store, the supervisor's routing
rules, and graph compilation.
"""

from __future__ import annotations

import json

import pytest


# --------------------------------------------------------------------------- #
# Database setup
# --------------------------------------------------------------------------- #

def test_external_seed(tmp_env):
    from data.external import db, seed

    counts = seed.seed_all(reset=True)
    assert counts["CultPassMember"] == 6
    assert counts["CultPassEvent"] >= 5
    assert counts["CultPassBooking"] >= 5
    assert counts["CultPassPayment"] >= 5

    # Bob has the seeded duplicate captured payment
    payments = db.fetch_all(
        "SELECT * FROM CultPassPayment WHERE member_id='cp_m_002' AND status='captured'"
    )
    assert len(payments) >= 2


def test_core_seed_loads_articles_jsonl(tmp_env):
    from data.core import db as core_db
    from data.core import seed as core_seed
    from data.external import seed as ext_seed

    ext_seed.seed_all(reset=True)
    counts = core_seed.seed_all(reset=True)

    for tbl in ("Account", "User", "Ticket", "TicketMetadata", "TicketMessage", "Knowledge"):
        n = core_db.fetch_one(f"SELECT COUNT(*) AS n FROM {tbl}")["n"]
        assert n > 0, f"{tbl} is empty"

    assert counts["Knowledge"] >= 14, "rubric requires ≥14 KB articles"
    cats = {r["category"] for r in core_db.fetch_all("SELECT DISTINCT category FROM Knowledge")}
    assert {"bookings", "billing", "account", "membership", "security"}.issubset(cats)


# --------------------------------------------------------------------------- #
# RAG fallback
# --------------------------------------------------------------------------- #

def test_keyword_retrieval(tmp_env):
    from data.core import seed as core_seed
    from data.external import seed as ext_seed
    from agentic.retrieval import keyword_retrieve

    ext_seed.seed_all(reset=True)
    core_seed.seed_all(reset=True)

    docs, best = keyword_retrieve("how do I enable two factor authentication", k=3)
    assert docs
    assert any(d["article_id"] == "cp_kb_012" for d in docs)
    assert 0.0 < best <= 1.0


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

def _seed_all(tmp_env):
    from data.core import seed as core_seed
    from data.external import seed as ext_seed

    ext_seed.seed_all(reset=True)
    core_seed.seed_all(reset=True)


def test_lookup_account(tmp_env):
    _seed_all(tmp_env)
    from agentic.tools import lookup_account

    out = json.loads(lookup_account.invoke({"user_id": "usr_001"}))
    assert out["status"] == "ok"
    assert out["account"]["account_id"] == "acc_001"
    assert out["account"]["external_member_id"] == "cp_m_001"


def test_process_refund_cap(tmp_env):
    _seed_all(tmp_env)
    from agentic.tools import process_refund

    over = json.loads(process_refund.invoke({"user_id": "usr_001", "amount_cents": 9999, "reason": "x"}))
    assert over["status"] == "error"
    ok = json.loads(process_refund.invoke({"user_id": "usr_002", "amount_cents": 999, "reason": "duplicate"}))
    assert ok["status"] == "ok"
    assert ok["amount_cents"] == 999


def test_update_subscription_rules(tmp_env):
    _seed_all(tmp_env)
    from agentic.tools import update_subscription

    invalid = json.loads(update_subscription.invoke({"user_id": "usr_001", "new_plan": "premium"}))
    assert invalid["status"] == "error"
    elite = json.loads(update_subscription.invoke({"user_id": "usr_001", "new_plan": "elite"}))
    assert elite["status"] == "error"
    ok = json.loads(update_subscription.invoke({"user_id": "usr_004", "new_plan": "plus"}))
    assert ok["status"] == "ok"
    assert ok["previous_plan"] == "classic"
    assert ok["new_plan"] == "plus"


def test_cultpass_member_lookup(tmp_env):
    _seed_all(tmp_env)
    from agentic.tools import cultpass_member_lookup

    out = json.loads(cultpass_member_lookup.invoke({"user_id": "usr_002"}))
    assert out["status"] == "ok"
    assert out["member"]["member_id"] == "cp_m_002"
    assert len(out["payments"]) >= 1

    miss = json.loads(cultpass_member_lookup.invoke({"user_id": "", "member_id": ""}))
    assert miss["status"] == "error"


def test_cultpass_cancel_booking(tmp_env):
    _seed_all(tmp_env)
    from agentic.tools import cultpass_cancel_booking

    cancel = json.loads(cultpass_cancel_booking.invoke({"booking_id": "cp_b_007", "reason": "test"}))
    assert cancel["status"] == "ok"
    assert cancel["new_status"] == "cancelled"

    twice = json.loads(cultpass_cancel_booking.invoke({"booking_id": "cp_b_007", "reason": "test"}))
    assert twice["status"] == "error"  # already cancelled


def test_cultpass_list_bookings(tmp_env):
    _seed_all(tmp_env)
    from agentic.tools import cultpass_list_bookings

    out = json.loads(cultpass_list_bookings.invoke({"user_id": "usr_001"}))
    assert out["status"] == "ok"
    assert len(out["bookings"]) >= 1


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #

def test_long_term_store_roundtrip(tmp_env):
    from agentic.memory import SqliteLongTermStore

    store = SqliteLongTermStore(path=str(tmp_env / "core" / "lt2.db"))
    store.put(("customer", "usr_001"), "ticket:abc", {"outcome": "resolved"})
    store.put(("customer", "usr_001"), "pref:locale", {"name": "locale", "value": "en-GB"})
    items = store.search(("customer", "usr_001"))
    assert {it.key for it in items} == {"ticket:abc", "pref:locale"}


# --------------------------------------------------------------------------- #
# Routing logic
# --------------------------------------------------------------------------- #

def test_supervisor_router_rules(tmp_env):
    from agentic.agents.supervisor import supervisor_router

    assert supervisor_router({
        "classification": {"urgency": "critical", "sentiment": "neutral"},
        "retrieval_confidence": 0.99,
    }) == "escalation"

    assert supervisor_router({
        "classification": {"urgency": "normal", "sentiment": "neutral"},
        "retrieval_confidence": 0.85,
    }) == "resolver"

    assert supervisor_router({
        "classification": {"urgency": "normal", "sentiment": "neutral"},
        "retrieval_confidence": 0.20,
        "subject": "weird question",
        "body": "no idea",
    }) == "escalation"

    # Low conf but obvious actionable verb in billing -> resolver
    assert supervisor_router({
        "classification": {"urgency": "normal", "sentiment": "neutral", "category": "billing"},
        "retrieval_confidence": 0.30,
        "subject": "please refund my charge",
        "body": "refund please",
    }) == "resolver"


# --------------------------------------------------------------------------- #
# Graph compilation
# --------------------------------------------------------------------------- #

def test_graph_compiles(tmp_env):
    from agentic.workflow import build_app

    app = build_app(with_checkpointer=False)
    mermaid = app.get_graph().draw_mermaid()
    for node in ("hydrate", "classifier", "retriever", "resolver", "escalation", "memory_writer"):
        assert node in mermaid
