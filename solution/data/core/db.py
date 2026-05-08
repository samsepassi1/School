"""Schema + helpers for the core UDA-Hub database.

Tables required by the rubric:
  * Account         — a tenant of UDA-Hub (CultPass is the first one).
                      Each tenant ships an external system that we link via
                      ``external_member_id`` so tools can join across DBs.
  * User            — an end customer who submits tickets. Linked to one Account.
  * Ticket          — a single support case.
  * TicketMetadata  — channel, urgency, classification, sentiment, routing.
  * TicketMessage   — append-only conversation log (customer / agent / tool / system).
  * Knowledge       — support articles (loaded from cultpass_articles.jsonl).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterable

from agentic.config import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS Account (
    account_id          TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    plan                TEXT NOT NULL CHECK (plan IN ('classic','plus','elite')),
    status              TEXT NOT NULL CHECK (status IN ('active','past_due','paused','cancelled')),
    balance_cents       INTEGER NOT NULL DEFAULT 0,
    external_member_id  TEXT UNIQUE,                 -- foreign id in CultPass
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
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
    channel        TEXT NOT NULL,
    urgency        TEXT NOT NULL,
    category       TEXT,
    sentiment      TEXT,
    confidence     REAL,
    routed_to      TEXT,
    extra_json     TEXT
);

CREATE TABLE IF NOT EXISTS TicketMessage (
    message_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id   TEXT NOT NULL REFERENCES Ticket(ticket_id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('customer','agent','system','tool')),
    author      TEXT,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS Knowledge (
    article_id  TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    category    TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '',
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
    conn = sqlite3.connect(db_path or settings.core_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    with connect(db_path) as c:
        c.executescript(SCHEMA)


def reset_db(db_path: str | None = None) -> None:
    with connect(db_path) as c:
        for tbl in ("TicketMessage", "TicketMetadata", "Ticket", "Knowledge", "User", "Account"):
            c.execute(f"DROP TABLE IF EXISTS {tbl};")
        c.executescript(SCHEMA)


def insert_many(table: str, rows: Iterable[dict[str, Any]], db_path: str | None = None) -> int:
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ",".join(["?"] * len(cols))
    sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    with connect(db_path) as c:
        c.executemany(sql, [tuple(r[k] for k in cols) for r in rows])
    return len(rows)


def fetch_all(query: str, params: tuple = (), db_path: str | None = None) -> list[dict]:
    with connect(db_path) as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def fetch_one(query: str, params: tuple = (), db_path: str | None = None) -> dict | None:
    with connect(db_path) as c:
        row = c.execute(query, params).fetchone()
        return dict(row) if row else None


def execute(query: str, params: tuple = (), db_path: str | None = None) -> int:
    with connect(db_path) as c:
        return c.execute(query, params).rowcount


# --------------------------------------------------------------------------- #
# Domain helpers used by the agents
# --------------------------------------------------------------------------- #

def append_message(ticket_id: str, role: str, content: str, author: str | None = None,
                   db_path: str | None = None) -> None:
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
    existing = fetch_one("SELECT * FROM TicketMetadata WHERE ticket_id=?", (ticket_id,), db_path)
    payload = dict(
        channel=channel, urgency=urgency, category=category, sentiment=sentiment,
        confidence=confidence, routed_to=routed_to,
        extra_json=json.dumps(extra) if extra is not None else None,
    )
    if existing is None:
        cols = ["ticket_id"] + list(payload.keys())
        vals = [ticket_id] + list(payload.values())
        for required in ("channel", "urgency"):
            idx = cols.index(required)
            if vals[idx] is None:
                vals[idx] = "web" if required == "channel" else "normal"
        placeholders = ",".join(["?"] * len(cols))
        execute(f"INSERT INTO TicketMetadata ({','.join(cols)}) VALUES ({placeholders})",
                tuple(vals), db_path)
        return
    set_parts, params = [], []
    for k, v in payload.items():
        if v is not None:
            set_parts.append(f"{k}=?")
            params.append(v)
    if not set_parts:
        return
    params.append(ticket_id)
    execute(f"UPDATE TicketMetadata SET {', '.join(set_parts)} WHERE ticket_id=?",
            tuple(params), db_path)
