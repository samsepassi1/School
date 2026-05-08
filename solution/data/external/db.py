"""Schema + helpers for the external CultPass database.

CultPass is UDA-Hub's first customer — a membership service that gives members
access to curated cultural events (music, theatre, film, art, food, talks).
This database lives in ``data/external/cultpass.db`` and is owned by CultPass;
UDA-Hub tools query it read-mostly to enrich tickets.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Iterable

from agentic.config import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS CultPassMember (
    member_id   TEXT PRIMARY KEY,
    full_name   TEXT NOT NULL,
    email       TEXT UNIQUE NOT NULL,
    tier        TEXT NOT NULL CHECK (tier IN ('classic','plus','elite')),
    status      TEXT NOT NULL CHECK (status IN ('active','past_due','paused','cancelled')),
    city        TEXT,
    joined_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS CultPassEvent (
    event_id    TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    venue       TEXT NOT NULL,
    city        TEXT NOT NULL,
    starts_at   TEXT NOT NULL,
    capacity    INTEGER NOT NULL,
    price_cents INTEGER NOT NULL,
    category    TEXT NOT NULL CHECK (category IN ('music','theatre','film','art','food','talk'))
);

CREATE TABLE IF NOT EXISTS CultPassBooking (
    booking_id  TEXT PRIMARY KEY,
    member_id   TEXT NOT NULL REFERENCES CultPassMember(member_id) ON DELETE CASCADE,
    event_id    TEXT NOT NULL REFERENCES CultPassEvent(event_id) ON DELETE CASCADE,
    status      TEXT NOT NULL CHECK (status IN ('confirmed','waitlisted','cancelled','attended','no_show')),
    plus_one    INTEGER NOT NULL DEFAULT 0,
    booked_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS CultPassPayment (
    payment_id   TEXT PRIMARY KEY,
    member_id    TEXT NOT NULL REFERENCES CultPassMember(member_id) ON DELETE CASCADE,
    amount_cents INTEGER NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('subscription','event','refund')),
    status       TEXT NOT NULL CHECK (status IN ('captured','failed','refunded','pending')),
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_booking_member ON CultPassBooking(member_id);
CREATE INDEX IF NOT EXISTS idx_payment_member ON CultPassPayment(member_id);
"""


@contextmanager
def connect(db_path: str | None = None):
    conn = sqlite3.connect(db_path or settings.external_db_path)
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
        for tbl in ("CultPassPayment", "CultPassBooking", "CultPassEvent", "CultPassMember"):
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
