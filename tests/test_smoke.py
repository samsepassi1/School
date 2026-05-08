"""Offline smoke tests — no LLM / network required.

Verify schema creation, seeding, keyword retrieval, tool execution, long-term
memory persistence, and the supervisor's routing rules. The agent nodes that
require an LLM are exercised indirectly via the routing logic and the
keyword-retrieval fallback.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_env(monkeypatch):
    """Point UDA-Hub paths at a fresh temp dir, then reload settings."""
    tmp = Path(tempfile.mkdtemp(prefix="uda_test_"))
    monkeypatch.setenv("UDA_DB_PATH", str(tmp / "uda.db"))
    monkeypatch.setenv("UDA_CHECKPOINT_PATH", str(tmp / "ck.db"))
    monkeypatch.setenv("UDA_LONGTERM_PATH", str(tmp / "lt.db"))
    monkeypatch.setenv("UDA_VECTORSTORE_PATH", str(tmp / "faiss"))

    # Force reimport so the dataclass settings pick up the new env.
    import importlib

    import uda_hub.config as config_mod
    importlib.reload(config_mod)
    import uda_hub.db as db_mod
    importlib.reload(db_mod)
    import uda_hub.memory as memory_mod
    importlib.reload(memory_mod)

    return tmp


def test_seed_creates_all_tables(tmp_env):
    from uda_hub import db, seed

    counts = seed.seed_all(reset=True)
    assert counts["Account"] == 5
    assert counts["User"] == 6
    assert counts["Knowledge"] >= 14, "rubric requires ≥14 KB articles"
    assert counts["Ticket"] == 3

    # Each KB article has a category and non-empty body
    rows = db.fetch_all("SELECT category, body FROM Knowledge")
    cats = {r["category"] for r in rows}
    assert {"billing", "account", "technical", "security"}.issubset(cats)
    assert all(r["body"] for r in rows)


def test_keyword_retrieval_finds_2fa(tmp_env):
    from uda_hub import seed
    from uda_hub.retrieval import keyword_retrieve

    seed.seed_all(reset=True)
    docs, best = keyword_retrieve("how do I enable two factor authentication", k=3)
    assert docs, "should return at least one match"
    assert any(d["article_id"] == "kb_005" for d in docs)
    assert 0.0 < best <= 1.0


def test_tool_lookup_account(tmp_env):
    import json

    from uda_hub import seed
    from uda_hub.tools import lookup_account

    seed.seed_all(reset=True)
    out = json.loads(lookup_account.invoke({"user_id": "usr_001"}))
    assert out["status"] == "ok"
    assert out["account"]["account_id"] == "acc_001"
    assert out["account"]["account_status"] in {"active", "past_due", "canceled", "paused"}


def test_tool_refund_cap(tmp_env):
    import json

    from uda_hub import seed
    from uda_hub.tools import process_refund

    seed.seed_all(reset=True)
    over_cap = json.loads(
        process_refund.invoke({"user_id": "usr_001", "amount_cents": 9999, "reason": "test"})
    )
    assert over_cap["status"] == "error"
    ok = json.loads(
        process_refund.invoke({"user_id": "usr_001", "amount_cents": 1999, "reason": "test"})
    )
    assert ok["status"] == "ok"
    assert ok["amount_cents"] == 1999


def test_long_term_store_roundtrip(tmp_env):
    from uda_hub.memory import SqliteLongTermStore

    store = SqliteLongTermStore(path=str(tmp_env / "lt2.db"))
    store.put(("customer", "usr_001"), "ticket:abc", {"outcome": "resolved"})
    store.put(("customer", "usr_001"), "pref:locale", {"name": "locale", "value": "en-US"})
    items = store.search(("customer", "usr_001"))
    assert {it.key for it in items} == {"ticket:abc", "pref:locale"}


def test_supervisor_router_rules(tmp_env):
    from uda_hub.agents.supervisor import supervisor_router

    # Critical urgency always escalates
    assert (
        supervisor_router(
            {
                "classification": {"urgency": "critical", "sentiment": "neutral"},
                "retrieval_confidence": 0.99,
            }
        )
        == "escalation"
    )
    # High confidence + neutral routes to resolver
    assert (
        supervisor_router(
            {
                "classification": {"urgency": "normal", "sentiment": "neutral"},
                "retrieval_confidence": 0.85,
            }
        )
        == "resolver"
    )
    # Low confidence + neutral, not actionable -> escalate
    assert (
        supervisor_router(
            {
                "classification": {"urgency": "normal", "sentiment": "neutral"},
                "retrieval_confidence": 0.20,
                "subject": "weird question about marketing",
                "body": "no idea",
            }
        )
        == "escalation"
    )
    # Low confidence but obvious refund verb -> resolver (tools handle it)
    assert (
        supervisor_router(
            {
                "classification": {"urgency": "normal", "sentiment": "neutral", "category": "billing"},
                "retrieval_confidence": 0.30,
                "subject": "please refund my charge",
                "body": "refund please",
            }
        )
        == "resolver"
    )


def test_graph_compiles_without_checkpointer(tmp_env):
    """We don't run the graph here (would need an LLM), only check it compiles."""
    from uda_hub.graph import build_app

    app = build_app(with_checkpointer=False)
    mermaid = app.get_graph().draw_mermaid()
    for node in ("hydrate", "classifier", "retriever", "resolver", "escalation", "memory_writer"):
        assert node in mermaid
