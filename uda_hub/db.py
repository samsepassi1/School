"""SQLite schema and data-access helpers for UDA-Hub.

Tables:
  - Account: tenant/company a user belongs to (CultPass-style billing customer).
  - User: end-customer profile.
  - Ticket: a support case.
  - TicketMetadata: structured metadata (channel, urgency, classification, ...).
  - TicketMessage: append-only conversation log for a ticket.
  - Knowledge: support articles consumed by the retrieval agent.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterable

from uda_hub.config import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS Account (
    account_id    TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    plan          TEXT NOT NULL CHECK (plan IN ('free','basic','pro','enterprise')),
    status        TEXT NOT NULL CHECK (status IN ('active','past_due','canceled','paused')),
    balance_cents INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS User (
    user_id     TEXT PRIMARY KEY,
    account_id  TEXT NOT NULL REFERENCES Account(account_id) ON DELETE CASCADE,
    email       TEXT UNIQUE NOT NULL,
    full_name   TEXT NOT NULL,
    locale      TEXT NOT NULL DEFAULT 'en-US',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS Ticket (
    ticket_id   TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES User(user_id) ON DELETE CASCADE,
    subject     TEXT NOT NULL,
    body        TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('open','in_progress','resolved','escalated','closed')),
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS TicketMetadata (
    ticket_id      TEXT PRIMARY KEY REFERENCES Ticket(ticket_id) ON DELETE CASCADE,
    channel        TEXT NOT NULL,            -- email, chat, web, twitter, ...
    urgency        TEXT NOT NULL,            -- low, normal, high, critical
    category       TEXT,                     -- billing, technical, account, ...
    sentiment      TEXT,                     -- positive, neutral, negative
    confidence     REAL,                     -- classifier confidence 0..1
    routed_to      TEXT,                     -- which agent ultimately handled it
    extra_json     TEXT                      -- free-form JSON metadata
);

CREATE TABLE IF NOT EXISTS TicketMessage (
    message_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id   TEXT NOT NULL REFERENCES Ticket(ticket_id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('customer','agent','system','tool')),
    author      TEXT,                        -- user_id or agent name
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS Knowledge (
    article_id  TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    category    TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '',    -- comma separated
    body        TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ticket_user      ON Ticket(user_id);
CREATE INDEX IF NOT EXISTS idx_ticket_status    ON Ticket(status);
CREATE INDEX IF NOT EXISTS idx_message_ticket   ON TicketMessage(ticket_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_cat    ON Knowledge(category);
"""


@contextmanager
def connect(db_path: str | None = None):
    """Yield a SQLite connection with FK enforcement and Row factory."""
    conn = sqlite3.connect(db_path or settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def reset_db(db_path: str | None = None) -> None:
    """Drop all UDA-Hub tables. Useful for rerunning the setup notebook."""
    with connect(db_path) as conn:
        for tbl in (
            "TicketMessage",
            "TicketMetadata",
            "Ticket",
            "Knowledge",
            "User",
            "Account",
        ):
            conn.execute(f"DROP TABLE IF EXISTS {tbl};")
        conn.executescript(SCHEMA)


def insert_many(table: str, rows: Iterable[dict[str, Any]], db_path: str | None = None) -> int:
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ",".join(["?"] * len(cols))
    sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    with connect(db_path) as conn:
        conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
    return len(rows)


def fetch_all(query: str, params: tuple = (), db_path: str | None = None) -> list[dict]:
    with connect(db_path) as conn:
        cur = conn.execute(query, params)
        return [dict(r) for r in cur.fetchall()]


def fetch_one(query: str, params: tuple = (), db_path: str | None = None) -> dict | None:
    with connect(db_path) as conn:
        cur = conn.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else None


def execute(query: str, params: tuple = (), db_path: str | None = None) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(query, params)
        return cur.rowcount


def append_message(
    ticket_id: str,
    role: str,
    content: str,
    author: str | None = None,
    db_path: str | None = None,
) -> None:
    execute(
        "INSERT INTO TicketMessage (ticket_id, role, author, content) VALUES (?,?,?,?)",
        (ticket_id, role, author, content),
        db_path,
    )


def update_ticket_status(ticket_id: str, status: str, db_path: str | None = None) -> None:
    execute(
        "UPDATE Ticket SET status=?, updated_at=datetime('now') WHERE ticket_id=?",
        (status, ticket_id),
        db_path,
    )


def upsert_ticket_metadata(
    ticket_id: str,
    *,
    channel: str | None = None,
    urgency: str | None = None,
    category: str | None = None,
    sentiment: str | None = None,
    confidence: float | None = None,
    routed_to: str | None = None,
    extra: dict | None = None,
    db_path: str | None = None,
) -> None:
    """Insert or merge metadata for a ticket."""
    existing = fetch_one(
        "SELECT * FROM TicketMetadata WHERE ticket_id=?", (ticket_id,), db_path
    )
    payload = {
        "channel": channel,
        "urgency": urgency,
        "category": category,
        "sentiment": sentiment,
        "confidence": confidence,
        "routed_to": routed_to,
        "extra_json": json.dumps(extra) if extra is not None else None,
    }
    if existing is None:
        cols = ["ticket_id"] + list(payload.keys())
        vals = [ticket_id] + list(payload.values())
        # Provide sensible defaults for NOT NULL columns when first inserting.
        for required in ("channel", "urgency"):
            idx = cols.index(required)
            if vals[idx] is None:
                vals[idx] = "web" if required == "channel" else "normal"
        placeholders = ",".join(["?"] * len(cols))
        execute(
            f"INSERT INTO TicketMetadata ({','.join(cols)}) VALUES ({placeholders})",
            tuple(vals),
            db_path,
        )
        return
    set_parts, params = [], []
    for key, val in payload.items():
        if val is not None:
            set_parts.append(f"{key}=?")
            params.append(val)
    if not set_parts:
        return
    params.append(ticket_id)
    execute(
        f"UPDATE TicketMetadata SET {', '.join(set_parts)} WHERE ticket_id=?",
        tuple(params),
        db_path,
    )
