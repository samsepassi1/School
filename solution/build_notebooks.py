"""Generate the three project notebooks (01/02/03) from inline cell definitions.

Usage: ``python build_notebooks.py`` (run from inside ``solution/``).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": "\n".join(lines),
    }


def write(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1))
    print(f"wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# 01 — External CultPass database setup
# --------------------------------------------------------------------------- #

NB_01: list[dict] = [
    md(
        "# 01 — External Database Setup (CultPass)",
        "",
        "Builds the **external** SQLite database that simulates the data UDA-Hub",
        "receives from its first customer, **CultPass**. This is the data the",
        "agentic system queries via the `cultpass_*` tools to enrich tickets.",
        "",
        "Schema:",
        "",
        "- `CultPassMember` — members and tier (classic / plus / elite)",
        "- `CultPassEvent` — events the member can book",
        "- `CultPassBooking` — bookings + plus-ones",
        "- `CultPassPayment` — subscription / event / refund payments",
    ),
    code(
        "import sys, pathlib",
        "ROOT = pathlib.Path.cwd()",
        "if str(ROOT) not in sys.path:",
        "    sys.path.insert(0, str(ROOT))",
        "",
        "from data.external import db, seed",
        "from agentic.config import settings",
        "settings.external_db_path",
    ),
    md("## 1. (Re)create the database and load seed rows"),
    code("counts = seed.seed_all(reset=True)", "counts"),
    md("## 2. Inspect schema"),
    code(
        "from tabulate import tabulate",
        "tables = db.fetch_all(",
        "    \"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name\"",
        ")",
        "print(tabulate(tables, headers='keys'))",
    ),
    md("## 3. Sample rows"),
    code(
        "print(tabulate(db.fetch_all('SELECT member_id, full_name, email, tier, status, city FROM CultPassMember'), headers='keys'))",
    ),
    code(
        "print(tabulate(db.fetch_all('SELECT event_id, title, city, starts_at, capacity, category FROM CultPassEvent'), headers='keys'))",
    ),
    code(
        "print(tabulate(db.fetch_all('''",
        "    SELECT b.booking_id, b.member_id, e.title AS event, b.status, b.plus_one",
        "      FROM CultPassBooking b JOIN CultPassEvent e USING(event_id)",
        "      ORDER BY b.booked_at DESC",
        "'''), headers='keys'))",
    ),
    code(
        "print(tabulate(db.fetch_all('SELECT payment_id, member_id, amount_cents, kind, status FROM CultPassPayment'), headers='keys'))",
    ),
    md(
        "## 4. Spot-check: Bob's duplicate charge",
        "",
        "Bob (`cp_m_002`) has two captured subscription payments in May — the seed",
        "scenario the Resolver/refund tool will fix in notebook 03.",
    ),
    code(
        "print(tabulate(db.fetch_all('''",
        "    SELECT * FROM CultPassPayment WHERE member_id=\"cp_m_002\" ORDER BY created_at DESC",
        "'''), headers='keys'))",
    ),
    md("---", "", "External database is ready. Continue to `02_core_db_setup.ipynb`."),
]


# --------------------------------------------------------------------------- #
# 02 — Core UDA-Hub database setup + Knowledge ingest
# --------------------------------------------------------------------------- #

NB_02: list[dict] = [
    md(
        "# 02 — Core Database Setup (UDA-Hub)",
        "",
        "Initialises the **core** SQLite database that backs UDA-Hub itself, and",
        "loads the Knowledge base from `data/external/cultpass_articles.jsonl`.",
        "",
        "Required tables (per rubric): `Account`, `User`, `Ticket`, `TicketMetadata`,",
        "`TicketMessage`, `Knowledge`.",
    ),
    code(
        "import sys, pathlib",
        "ROOT = pathlib.Path.cwd()",
        "if str(ROOT) not in sys.path:",
        "    sys.path.insert(0, str(ROOT))",
        "",
        "from data.core import db, seed",
        "from agentic.config import settings",
        "settings.core_db_path",
    ),
    md("## 1. (Re)create the database and load seed rows"),
    code("counts = seed.seed_all(reset=True)", "counts"),
    md(
        "## 2. Verify required tables",
        "",
        "Each table must exist and have at least one row.",
    ),
    code(
        "from tabulate import tabulate",
        "for tbl in ['Account','User','Ticket','TicketMetadata','TicketMessage','Knowledge']:",
        "    n = db.fetch_one(f'SELECT COUNT(*) AS n FROM {tbl}')['n']",
        "    print(f'{tbl:<16} {n}')",
    ),
    md("## 3. Knowledge base — categories and counts"),
    code(
        "kb = db.fetch_all('SELECT category, COUNT(*) AS n FROM Knowledge GROUP BY category ORDER BY category')",
        "print(tabulate(kb, headers='keys'))",
        "total = db.fetch_one('SELECT COUNT(*) AS n FROM Knowledge')['n']",
        "print(f'\\nTotal articles: {total}  (rubric requires \\u226514)')",
    ),
    code(
        "for row in db.fetch_all('SELECT article_id, title, category FROM Knowledge ORDER BY article_id'):",
        "    print(f\"{row['article_id']}  [{row['category']:<10}] {row['title']}\")",
    ),
    md("## 4. Spot-check a single article"),
    code(
        "art = db.fetch_one('SELECT * FROM Knowledge WHERE article_id=?', ('cp_kb_013',))",
        "print(art['title']); print('-'*60); print(art['body'])",
    ),
    md(
        "## 5. Cross-DB linkage — Account.external_member_id → CultPassMember",
        "",
        "Every Account in the core DB is linked to a CultPass member id, which is",
        "what the `cultpass_member_lookup` tool uses to join.",
    ),
    code(
        "print(tabulate(db.fetch_all('''",
        "    SELECT account_id, name, plan, status, external_member_id",
        "      FROM Account ORDER BY account_id",
        "'''), headers='keys'))",
    ),
    md("## 6. Build the FAISS vector index over Knowledge"),
    code(
        "import os",
        "if os.getenv('OPENAI_API_KEY'):",
        "    from agentic.retrieval import build_or_load_vectorstore",
        "    vs = build_or_load_vectorstore(rebuild=True)",
        "    print('FAISS index built with', vs.index.ntotal, 'vectors')",
        "else:",
        "    print('OPENAI_API_KEY not set — skipping FAISS index build (keyword fallback will be used at runtime).')",
    ),
    md("---", "", "Core database is ready. Continue to `03_agentic_app.ipynb` to run the workflow."),
]


# --------------------------------------------------------------------------- #
# 03 — End-to-end agentic app
# --------------------------------------------------------------------------- #

NB_03: list[dict] = [
    md(
        "# 03 — End-to-End Agentic App",
        "",
        "Runs the full LangGraph workflow on several sample tickets and shows:",
        "",
        "1. Classification + routing decisions",
        "2. RAG knowledge retrieval with confidence scoring",
        "3. Tool invocation against the core and external databases",
        "4. Resolution vs. escalation outcomes",
        "5. Short-term (per-thread) and long-term (per-user) memory",
        "6. The interactive `chat_interface()` REPL",
        "",
        "> Requires `OPENAI_API_KEY` in `.env` (or environment).",
    ),
    code(
        "import sys, pathlib",
        "ROOT = pathlib.Path.cwd()",
        "if str(ROOT) not in sys.path:",
        "    sys.path.insert(0, str(ROOT))",
        "",
        "from agentic.logging_utils import configure_logging",
        "from agentic.workflow import build_app",
        "from agentic.runner import run_ticket",
        "from utils import print_result, chat_interface",
        "",
        "configure_logging()",
        "app = build_app()",
        "print(app.get_graph().draw_mermaid())",
    ),
    md("## 1. Scenario A — KB resolution (how-to)"),
    code(
        "result = run_ticket(app, {",
        "    'ticket_id': 'tkt_demo_kb',",
        "    'user_id':   'usr_001',",
        "    'subject':   'How do I turn on 2FA?',",
        "    'body':      'I want to enable two-factor authentication on my Cult Pass account.',",
        "    'channel':   'web',",
        "    'urgency':   'normal',",
        "})",
        "print_result(result)",
    ),
    md("## 2. Scenario B — Tool-driven refund (uses cultpass_member_lookup + process_refund)"),
    code(
        "result = run_ticket(app, {",
        "    'ticket_id': 'tkt_demo_refund',",
        "    'user_id':   'usr_002',",
        "    'subject':   'Charged twice this month',",
        "    'body':      'You billed me $9.99 twice for May membership. Please refund the duplicate.',",
        "    'channel':   'email',",
        "    'urgency':   'high',",
        "})",
        "print_result(result)",
    ),
    md("## 3. Scenario C — Cancel a CultPass booking via the external tool"),
    code(
        "result = run_ticket(app, {",
        "    'ticket_id': 'tkt_demo_cancel',",
        "    'user_id':   'usr_006',",
        "    'subject':   'Need to cancel my Hamlet booking',",
        "    'body':      \"Please cancel my booking cp_b_007 (Hamlet at Donmar Warehouse). Something came up.\",",
        "    'channel':   'chat',",
        "    'urgency':   'normal',",
        "})",
        "print_result(result)",
    ),
    md("## 4. Scenario D — Escalation (no relevant article + critical urgency)"),
    code(
        "result = run_ticket(app, {",
        "    'ticket_id': 'tkt_demo_esc',",
        "    'user_id':   'usr_003',",
        "    'subject':   'Wallet QR not loading at venue right now',",
        "    'body':      'I am at the venue and the QR will not load. Doors closed in 10 minutes!',",
        "    'channel':   'chat',",
        "    'urgency':   'critical',",
        "})",
        "print_result(result)",
    ),
    md(
        "## 5. Memory in action — follow-up in the same thread",
        "",
        "Reusing the thread_id from Scenario B continues that conversation; the",
        "checkpointer makes the prior state available.",
    ),
    code(
        "result = run_ticket(app, {",
        "    'ticket_id': 'tkt_demo_followup',",
        "    'user_id':   'usr_002',",
        "    'subject':   'Quick follow-up to my refund',",
        "    'body':      'When should I see the refund on my statement?',",
        "    'channel':   'email',",
        "    'urgency':   'normal',",
        "    'thread_id': 'thread-tkt_demo_refund',",
        "})",
        "print_result(result)",
    ),
    md("## 6. Inspect long-term memory and persisted ticket state"),
    code(
        "from agentic.memory import get_long_term_store",
        "for it in get_long_term_store().search(('customer','usr_002')):",
        "    print(it.key, '->', it.value)",
    ),
    code(
        "from data.core import db",
        "from tabulate import tabulate",
        "print(tabulate(db.fetch_all('''",
        "    SELECT t.ticket_id, t.status, m.category, m.urgency, m.sentiment, m.confidence, m.routed_to",
        "      FROM Ticket t JOIN TicketMetadata m USING(ticket_id)",
        "     WHERE t.ticket_id LIKE 'tkt_demo_%'",
        "     ORDER BY t.ticket_id",
        "'''), headers='keys'))",
    ),
    md(
        "## 7. Optional — interactive chat shell",
        "",
        "Uncomment to launch `chat_interface()`. It reuses the compiled `app` and",
        "pins one `thread_id` per session, so short-term memory works.",
    ),
    code(
        "# chat_interface(app, default_user='usr_002')",
    ),
]


def main() -> None:
    write(ROOT / "01_external_db_setup.ipynb", NB_01)
    write(ROOT / "02_core_db_setup.ipynb", NB_02)
    write(ROOT / "03_agentic_app.ipynb", NB_03)


if __name__ == "__main__":
    main()
