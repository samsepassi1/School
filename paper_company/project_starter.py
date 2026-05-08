"""Beaver's Choice Paper Company — starter database and helper functions.

This module owns the SQLite schema, initial seed data, and the seven helper
functions used by the multi-agent system's tools. Running it as a script
initializes (or re-initializes) the database and prints the opening financial
report — useful as a smoke test before exercising the agent workflow.

Helper functions exposed to agents:

* ``create_transaction``         – record a sale or stock_orders row
* ``get_all_inventory``          – stock levels for every SKU as of a date
* ``get_stock_level``            – stock for a single SKU as of a date
* ``get_supplier_delivery_date`` – heuristic supplier ETA given order size
* ``get_cash_balance``           – running cash balance up to a date
* ``generate_financial_report``  – cash + inventory snapshot
* ``search_quote_history``       – keyword search across past customer quotes
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

DB_PATH = Path(os.environ.get("PAPER_DB_PATH", Path(__file__).parent / "beavers.db"))

OPENING_CASH = 50_000.0
OPENING_DATE = "2024-12-31"


# ---------------------------------------------------------------------------
# Catalog and seed data
# ---------------------------------------------------------------------------

#: Catalog of paper SKUs. ``unit_price`` is the customer-facing list price;
#: ``supplier_price`` is what the company pays its supplier (internal only).
CATALOG: list[dict[str, Any]] = [
    {"item_name": "A4_paper_500",          "category": "paper",     "unit_price": 6.50,  "supplier_price": 4.00, "min_stock": 100},
    {"item_name": "letter_paper_500",      "category": "paper",     "unit_price": 6.00,  "supplier_price": 3.75, "min_stock": 100},
    {"item_name": "cardstock_white_250",   "category": "cardstock", "unit_price": 12.00, "supplier_price": 7.50, "min_stock": 60},
    {"item_name": "cardstock_colored_250", "category": "cardstock", "unit_price": 14.00, "supplier_price": 8.75, "min_stock": 60},
    {"item_name": "glossy_photo_50",       "category": "specialty", "unit_price": 18.00, "supplier_price": 11.00, "min_stock": 40},
    {"item_name": "poster_24x36",          "category": "specialty", "unit_price": 22.00, "supplier_price": 13.50, "min_stock": 30},
    {"item_name": "envelope_10_box",       "category": "envelope",  "unit_price": 9.00,  "supplier_price": 5.50, "min_stock": 80},
    {"item_name": "envelope_invitation",   "category": "envelope",  "unit_price": 11.00, "supplier_price": 6.75, "min_stock": 80},
    {"item_name": "notebook_college",      "category": "notebook",  "unit_price": 4.50,  "supplier_price": 2.75, "min_stock": 120},
    {"item_name": "sticky_notes_pack",     "category": "specialty", "unit_price": 3.50,  "supplier_price": 2.00, "min_stock": 150},
]

#: Initial stock by SKU. Some SKUs are intentionally lean so a few requests
#: in the eval set will exceed available stock and exercise the decline path.
INITIAL_STOCK: dict[str, int] = {
    "A4_paper_500":          400,
    "letter_paper_500":      350,
    "cardstock_white_250":   200,
    "cardstock_colored_250": 150,
    "glossy_photo_50":       80,
    "poster_24x36":          40,
    "envelope_10_box":       300,
    "envelope_invitation":   220,
    "notebook_college":      500,
    "sticky_notes_pack":     600,
}

#: Historical quotes used by ``search_quote_history``. They give the quoting
#: agent realistic anchor prices and event-type context.
SEED_QUOTE_HISTORY: list[dict[str, Any]] = [
    {"job_description": "Wedding invitations and matching envelopes for 200 guests",
     "event_type": "wedding", "items": [("cardstock_white_250", 4), ("envelope_invitation", 8)],
     "total_price": 136.00, "date": "2024-09-15"},
    {"job_description": "Conference handout packets for 500 attendees",
     "event_type": "conference", "items": [("A4_paper_500", 20), ("cardstock_white_250", 4)],
     "total_price": 178.00, "date": "2024-10-02"},
    {"job_description": "Back-to-school notebook bundle for elementary school",
     "event_type": "school", "items": [("notebook_college", 240), ("sticky_notes_pack", 100)],
     "total_price": 1430.00, "date": "2024-08-21"},
    {"job_description": "Marketing launch posters and glossy photo prints",
     "event_type": "marketing", "items": [("poster_24x36", 25), ("glossy_photo_50", 20)],
     "total_price": 910.00, "date": "2024-11-10"},
    {"job_description": "Office supply restock — letter paper for quarterly use",
     "event_type": "office", "items": [("letter_paper_500", 80)],
     "total_price": 432.00, "date": "2024-07-04"},
    {"job_description": "Birthday party invitations and envelopes",
     "event_type": "birthday", "items": [("cardstock_colored_250", 2), ("envelope_invitation", 3)],
     "total_price": 61.00, "date": "2024-06-12"},
    {"job_description": "Fundraising gala — premium invitations for 300 guests",
     "event_type": "wedding", "items": [("cardstock_colored_250", 6), ("envelope_invitation", 12)],
     "total_price": 216.00, "date": "2024-05-18"},
    {"job_description": "Trade show booth posters and photo handouts",
     "event_type": "marketing", "items": [("poster_24x36", 15), ("glossy_photo_50", 30)],
     "total_price": 870.00, "date": "2024-04-22"},
    {"job_description": "Standard office paper restock for legal firm",
     "event_type": "office", "items": [("letter_paper_500", 50), ("envelope_10_box", 30)],
     "total_price": 570.00, "date": "2024-03-08"},
    {"job_description": "School graduation programs",
     "event_type": "school", "items": [("cardstock_white_250", 8)],
     "total_price": 91.20, "date": "2024-06-01"},
    {"job_description": "Real estate marketing flyers and posters",
     "event_type": "marketing", "items": [("glossy_photo_50", 60), ("poster_24x36", 10)],
     "total_price": 1300.00, "date": "2024-02-14"},
    {"job_description": "Nonprofit annual report packets",
     "event_type": "office", "items": [("A4_paper_500", 15), ("envelope_10_box", 25)],
     "total_price": 322.50, "date": "2024-01-19"},
    {"job_description": "Wedding ceremony programs and reception cards",
     "event_type": "wedding", "items": [("cardstock_white_250", 5), ("cardstock_colored_250", 3)],
     "total_price": 96.00, "date": "2024-09-30"},
    {"job_description": "Annual conference badge holders and photo prints",
     "event_type": "conference", "items": [("glossy_photo_50", 25), ("envelope_10_box", 40)],
     "total_price": 810.00, "date": "2024-10-25"},
    {"job_description": "Elementary school art supply bundle",
     "event_type": "school", "items": [("sticky_notes_pack", 200), ("notebook_college", 100)],
     "total_price": 1138.00, "date": "2024-08-05"},
    {"job_description": "Holiday card mailing for accounting firm",
     "event_type": "office", "items": [("cardstock_colored_250", 4), ("envelope_invitation", 6)],
     "total_price": 117.20, "date": "2024-12-01"},
    {"job_description": "Tech startup product launch posters",
     "event_type": "marketing", "items": [("poster_24x36", 8), ("glossy_photo_50", 12)],
     "total_price": 379.20, "date": "2024-11-20"},
    {"job_description": "University commencement programs",
     "event_type": "school", "items": [("cardstock_white_250", 10), ("envelope_invitation", 5)],
     "total_price": 168.00, "date": "2024-05-30"},
    {"job_description": "Corporate quarterly meeting packets",
     "event_type": "conference", "items": [("A4_paper_500", 30), ("notebook_college", 80)],
     "total_price": 540.00, "date": "2024-09-08"},
    {"job_description": "Bridal shower invitations and thank-you notes",
     "event_type": "wedding", "items": [("cardstock_colored_250", 3), ("envelope_invitation", 4)],
     "total_price": 79.20, "date": "2024-07-20"},
]


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database(reset: bool = False) -> None:
    """Create tables and seed initial data.

    If ``reset`` is True (or if the DB has no rows yet) the catalog, opening
    cash entry, initial stock, and historical quotes are written. Existing
    transactions are left in place unless ``reset`` is True.
    """
    if reset and DB_PATH.exists():
        DB_PATH.unlink()

    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS inventory (
                item_name      TEXT PRIMARY KEY,
                category       TEXT NOT NULL,
                unit_price     REAL NOT NULL,
                supplier_price REAL NOT NULL,
                min_stock      INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                item_name        TEXT NOT NULL,
                transaction_type TEXT NOT NULL CHECK (transaction_type IN ('sale','stock_orders','opening_balance')),
                units            INTEGER NOT NULL,
                price            REAL NOT NULL,
                date             TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS quote_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                job_description TEXT NOT NULL,
                event_type      TEXT NOT NULL,
                items_json      TEXT NOT NULL,
                total_price     REAL NOT NULL,
                date            TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(date);
            CREATE INDEX IF NOT EXISTS idx_tx_item ON transactions(item_name);
            CREATE INDEX IF NOT EXISTS idx_qh_event ON quote_history(event_type);
            """
        )

        already_seeded = conn.execute("SELECT COUNT(*) FROM inventory").fetchone()[0] > 0
        if already_seeded and not reset:
            return

        conn.executemany(
            "INSERT OR REPLACE INTO inventory(item_name, category, unit_price, supplier_price, min_stock) "
            "VALUES (:item_name, :category, :unit_price, :supplier_price, :min_stock)",
            CATALOG,
        )

        # Opening cash recorded as a single sentinel row. ``get_cash_balance``
        # treats ``opening_balance`` rows as positive cash inflow.
        conn.execute(
            "INSERT INTO transactions(item_name, transaction_type, units, price, date) "
            "VALUES (?, 'opening_balance', 0, ?, ?)",
            ("__opening__", OPENING_CASH, OPENING_DATE),
        )

        # Initial stock recorded as ``stock_orders`` at supplier price. We
        # use the opening date so they're considered already on hand.
        for item, units in INITIAL_STOCK.items():
            supplier_price = next(c["supplier_price"] for c in CATALOG if c["item_name"] == item)
            conn.execute(
                "INSERT INTO transactions(item_name, transaction_type, units, price, date) "
                "VALUES (?, 'stock_orders', ?, ?, ?)",
                (item, units, supplier_price * units, OPENING_DATE),
            )
            # Cancel out the cost of the initial stock so opening cash stays at OPENING_CASH.
            conn.execute(
                "INSERT INTO transactions(item_name, transaction_type, units, price, date) "
                "VALUES (?, 'opening_balance', 0, ?, ?)",
                ("__opening_offset__", supplier_price * units, OPENING_DATE),
            )

        for q in SEED_QUOTE_HISTORY:
            conn.execute(
                "INSERT INTO quote_history(job_description, event_type, items_json, total_price, date) "
                "VALUES (?, ?, ?, ?, ?)",
                (q["job_description"], q["event_type"], json.dumps(q["items"]), q["total_price"], q["date"]),
            )


# ---------------------------------------------------------------------------
# Helper functions used by agent tools
# ---------------------------------------------------------------------------

def create_transaction(
    item_name: str,
    transaction_type: str,
    units: int,
    price: float,
    date: str,
) -> int:
    """Insert a transaction row and return its id.

    ``transaction_type`` must be ``'sale'`` (cash in) or ``'stock_orders'``
    (cash out). ``price`` is the *total* line value, not the unit price.
    """
    if transaction_type not in {"sale", "stock_orders"}:
        raise ValueError(f"transaction_type must be 'sale' or 'stock_orders', got {transaction_type!r}")
    if units <= 0:
        raise ValueError("units must be positive")
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO transactions(item_name, transaction_type, units, price, date) "
            "VALUES (?, ?, ?, ?, ?)",
            (item_name, transaction_type, units, float(price), date),
        )
        return cur.lastrowid


def get_all_inventory(as_of_date: str) -> dict[str, int]:
    """Return current stock for every SKU as of ``as_of_date`` (YYYY-MM-DD)."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT i.item_name AS item_name,
                   COALESCE(SUM(CASE
                       WHEN t.transaction_type = 'stock_orders' THEN t.units
                       WHEN t.transaction_type = 'sale'         THEN -t.units
                       ELSE 0
                   END), 0) AS on_hand
            FROM inventory i
            LEFT JOIN transactions t
                   ON t.item_name = i.item_name AND t.date <= ?
            GROUP BY i.item_name
            """,
            (as_of_date,),
        ).fetchall()
    return {r["item_name"]: int(r["on_hand"]) for r in rows}


def get_stock_level(item_name: str, as_of_date: str) -> int:
    """Return stock for one SKU as of ``as_of_date``."""
    return get_all_inventory(as_of_date).get(item_name, 0)


def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """Heuristic supplier ETA given an order size.

    Bigger orders take longer:
    * <= 50 units  → 2 days
    * <= 200 units → 5 days
    * <= 500 units → 8 days
    * > 500 units  → 12 days
    """
    base = datetime.strptime(input_date_str, "%Y-%m-%d")
    if quantity <= 50:
        offset = 2
    elif quantity <= 200:
        offset = 5
    elif quantity <= 500:
        offset = 8
    else:
        offset = 12
    return (base + timedelta(days=offset)).strftime("%Y-%m-%d")


def get_cash_balance(as_of_date: str) -> float:
    """Running cash balance up to and including ``as_of_date``.

    Sales add cash, stock_orders subtract cash, and opening_balance entries
    seed the starting capital.
    """
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(CASE
                WHEN transaction_type = 'sale'             THEN price
                WHEN transaction_type = 'stock_orders'     THEN -price
                WHEN transaction_type = 'opening_balance' THEN price
                ELSE 0
            END), 0.0) AS cash
            FROM transactions
            WHERE date <= ?
            """,
            (as_of_date,),
        ).fetchone()
    return float(row["cash"])


def generate_financial_report(as_of_date: str) -> dict[str, Any]:
    """Snapshot of cash, inventory value (at unit_price), and total assets."""
    cash = get_cash_balance(as_of_date)
    inventory = get_all_inventory(as_of_date)
    with _connect() as conn:
        prices = {
            r["item_name"]: r["unit_price"]
            for r in conn.execute("SELECT item_name, unit_price FROM inventory").fetchall()
        }
    inventory_value = sum(units * prices.get(item, 0.0) for item, units in inventory.items())
    return {
        "as_of_date": as_of_date,
        "cash_balance": round(cash, 2),
        "inventory_value": round(inventory_value, 2),
        "total_assets": round(cash + inventory_value, 2),
        "inventory_units": inventory,
    }


def search_quote_history(search_terms: list[str], limit: int = 5) -> list[dict[str, Any]]:
    """Keyword search across past job descriptions and event types.

    Returns rows whose ``job_description`` or ``event_type`` contains any
    of the (case-insensitive) ``search_terms``, ordered by most recent.
    """
    if not search_terms:
        return []
    where_parts = []
    params: list[str] = []
    for term in search_terms:
        like = f"%{term.lower()}%"
        where_parts.append("(LOWER(job_description) LIKE ? OR LOWER(event_type) LIKE ?)")
        params.extend([like, like])
    sql = (
        "SELECT job_description, event_type, items_json, total_price, date "
        "FROM quote_history WHERE " + " OR ".join(where_parts) +
        " ORDER BY date DESC LIMIT ?"
    )
    params.append(limit)
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        {
            "job_description": r["job_description"],
            "event_type": r["event_type"],
            "items": json.loads(r["items_json"]),
            "total_price": r["total_price"],
            "date": r["date"],
        }
        for r in rows
    ]


def get_catalog() -> list[dict[str, Any]]:
    """Return catalog rows (item_name, category, unit_price, min_stock).

    Supplier price is stripped — agents must not surface internal cost.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT item_name, category, unit_price, min_stock FROM inventory ORDER BY item_name"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Smoke test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_database(reset=True)
    report = generate_financial_report(OPENING_DATE)
    print("Beaver's Choice — opening financial report")
    print(f"  as of:           {report['as_of_date']}")
    print(f"  cash balance:    ${report['cash_balance']:,.2f}")
    print(f"  inventory value: ${report['inventory_value']:,.2f}")
    print(f"  total assets:    ${report['total_assets']:,.2f}")
    print("  inventory units:")
    for item, units in sorted(report["inventory_units"].items()):
        print(f"    {item:24s} {units:>5d}")
