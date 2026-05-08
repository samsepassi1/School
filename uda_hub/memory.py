"""Short-term and long-term memory for UDA-Hub.

* **Short-term (per-thread)**: a SQLite ``Checkpointer`` persists the LangGraph
  state for each ``thread_id``. Reusing the same ``thread_id`` resumes the
  conversation; using a new one starts fresh.

* **Long-term (cross-session)**: a tiny SQLite-backed key/value store keyed by
  ``(namespace, key)``. Namespaces look like ``("customer", user_id)`` so
  resolved-issue summaries and customer preferences survive across threads.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from uda_hub.config import settings


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Short-term: LangGraph checkpointer
# --------------------------------------------------------------------------- #

def get_checkpointer():
    """Return a SQLite-backed LangGraph checkpointer.

    ``langgraph-checkpoint-sqlite`` exposes both sync and context-manager
    variants. We want a long-lived sync saver that the compiled app can use.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
    return SqliteSaver(conn)


# --------------------------------------------------------------------------- #
# Long-term: simple SQLite KV store with a LangGraph-style API
# --------------------------------------------------------------------------- #

LONGTERM_SCHEMA = """
CREATE TABLE IF NOT EXISTS LongTerm (
    namespace   TEXT NOT NULL,            -- json-encoded tuple
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,            -- json
    updated_at  REAL NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS idx_longterm_ns ON LongTerm(namespace);
"""


@dataclass
class StoredItem:
    namespace: tuple[str, ...]
    key: str
    value: dict
    updated_at: float


class SqliteLongTermStore:
    """LangGraph-Store-compatible long-term memory backed by SQLite.

    Mirrors the subset of the langgraph store API we need: ``put``, ``get``,
    ``delete``, ``search``. Namespaces are tuples (encoded as JSON for storage).
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or settings.longterm_path
        with self._conn() as c:
            c.executescript(LONGTERM_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _ns(ns: Iterable[str]) -> str:
        return json.dumps(list(ns))

    def put(self, namespace: tuple[str, ...], key: str | None, value: dict) -> str:
        key = key or str(uuid.uuid4())
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO LongTerm(namespace,key,value,updated_at) VALUES (?,?,?,?)",
                (self._ns(namespace), key, json.dumps(value), time.time()),
            )
        logger.info("longterm.put ns=%s key=%s", namespace, key)
        return key

    def get(self, namespace: tuple[str, ...], key: str) -> StoredItem | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM LongTerm WHERE namespace=? AND key=?",
                (self._ns(namespace), key),
            ).fetchone()
        if row is None:
            return None
        return StoredItem(
            namespace=tuple(json.loads(row["namespace"])),
            key=row["key"],
            value=json.loads(row["value"]),
            updated_at=row["updated_at"],
        )

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        with self._conn() as c:
            c.execute(
                "DELETE FROM LongTerm WHERE namespace=? AND key=?",
                (self._ns(namespace), key),
            )

    def search(
        self,
        namespace: tuple[str, ...],
        *,
        query: str | None = None,
        limit: int = 20,
    ) -> list[StoredItem]:
        """Return all items in a namespace, optionally filtered by substring."""
        ns = self._ns(namespace)
        sql = "SELECT * FROM LongTerm WHERE namespace=?"
        params: list[Any] = [ns]
        if query:
            sql += " AND value LIKE ?"
            params.append(f"%{query}%")
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        return [
            StoredItem(
                namespace=tuple(json.loads(r["namespace"])),
                key=r["key"],
                value=json.loads(r["value"]),
                updated_at=r["updated_at"],
            )
            for r in rows
        ]


_long_term_store: SqliteLongTermStore | None = None


def get_long_term_store() -> SqliteLongTermStore:
    global _long_term_store
    if _long_term_store is None:
        _long_term_store = SqliteLongTermStore()
    return _long_term_store


# --------------------------------------------------------------------------- #
# Convenience API used by agents
# --------------------------------------------------------------------------- #

def remember_resolution(user_id: str, ticket_id: str, summary: dict) -> None:
    """Persist a resolved-ticket summary in long-term memory."""
    get_long_term_store().put(("customer", user_id), f"ticket:{ticket_id}", summary)


def recall_customer_history(user_id: str, limit: int = 5) -> list[dict]:
    items = get_long_term_store().search(("customer", user_id), limit=limit)
    return [it.value for it in items]


def remember_preference(user_id: str, name: str, value: Any) -> None:
    get_long_term_store().put(
        ("customer", user_id), f"pref:{name}", {"name": name, "value": value}
    )


def recall_preferences(user_id: str) -> dict[str, Any]:
    items = get_long_term_store().search(("customer", user_id), query="\"pref:")
    out: dict[str, Any] = {}
    for it in items:
        if it.key.startswith("pref:"):
            out[it.value["name"]] = it.value["value"]
    return out
