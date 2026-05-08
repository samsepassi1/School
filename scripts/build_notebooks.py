"""Build the .ipynb files for UDA-Hub from plain-text cell definitions.

Run with: python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": "\n".join(lines),
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": "\n".join(lines),
    }


def write_nb(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1))
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# 01 — Database setup & knowledge base management
# --------------------------------------------------------------------------- #

NB_01: list[dict] = [
    md(
        "# 01 — Database Setup & Knowledge Base",
        "",
        "This notebook initialises the UDA-Hub SQLite database, seeds sample tenants/users/tickets,",
        "and loads the knowledge base used by the retrieval agent.",
        "",
        "**Schema**",
        "",
        "- `Account` — tenant/billing customer",
        "- `User` — end customer",
        "- `Ticket` — a support case",
        "- `TicketMetadata` — channel, urgency, classification, sentiment, ...",
        "- `TicketMessage` — append-only conversation log",
        "- `Knowledge` — support articles",
    ),
    code(
        "import sys, pathlib",
        "sys.path.insert(0, str(pathlib.Path.cwd().parent))  # so `uda_hub` is importable from notebooks/",
        "",
        "from uda_hub import db, seed",
        "from uda_hub.config import settings",
        "settings",
    ),
    md("## 1. Initialise (or reset) the database"),
    code(
        "counts = seed.seed_all(reset=True)",
        "counts",
    ),
    md(
        "## 2. Inspect the schema",
        "",
        "Confirm every required table exists and is populated.",
    ),
    code(
        "from tabulate import tabulate",
        "rows = db.fetch_all(",
        "    \"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name\"",
        ")",
        "print(tabulate(rows, headers='keys'))",
    ),
    code(
        "for tbl in ['Account','User','Ticket','TicketMetadata','TicketMessage','Knowledge']:",
        "    n = db.fetch_one(f'SELECT COUNT(*) AS n FROM {tbl}')['n']",
        "    print(f'{tbl:<16} {n}')",
    ),
    md("## 3. Sample rows"),
    code(
        "print(tabulate(db.fetch_all('SELECT account_id, name, plan, status FROM Account'), headers='keys'))",
    ),
    code(
        "print(tabulate(db.fetch_all('SELECT user_id, account_id, email, full_name FROM User'), headers='keys'))",
    ),
    code(
        "print(tabulate(",
        "    db.fetch_all('SELECT ticket_id, user_id, subject, status FROM Ticket'),",
        "    headers='keys', maxcolwidths=[None,None,40,None]",
        "))",
    ),
    md("## 4. Knowledge base"),
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
    md("## 5. Spot-check a single article"),
    code(
        "art = db.fetch_one('SELECT * FROM Knowledge WHERE article_id=?', ('kb_010',))",
        "print(art['title']); print('-'*60); print(art['body'])",
    ),
    md(
        "## 6. Conversation history is preserved",
        "",
        "Returning customers should see prior interactions retrieved later by the agents.",
    ),
    code(
        "print(tabulate(",
        "    db.fetch_all(\"\"\"",
        "        SELECT t.ticket_id, t.subject, m.role, m.content",
        "          FROM Ticket t JOIN TicketMessage m USING(ticket_id)",
        "         WHERE t.user_id='usr_002'",
        "         ORDER BY m.message_id",
        "    \"\"\"),",
        "    headers='keys', maxcolwidths=[None,30,None,60]",
        "))",
    ),
    md(
        "---",
        "",
        "Database is ready. Continue to `02_end_to_end_demo.ipynb` to run the multi-agent workflow.",
    ),
]


# --------------------------------------------------------------------------- #
# 02 — End-to-end demo
# --------------------------------------------------------------------------- #

NB_02: list[dict] = [
    md(
        "# 02 — End-to-End UDA-Hub Demo",
        "",
        "Runs the full LangGraph workflow on several sample tickets and shows:",
        "",
        "1. Classification + routing decisions",
        "2. Knowledge retrieval with confidence scoring",
        "3. Tool invocation against the SQLite-backed support database",
        "4. Resolution vs. escalation outcomes",
        "5. Short-term (per-thread) and long-term (per-user) memory",
        "",
        "> Requires `OPENAI_API_KEY` to be set in your environment (or `.env`).",
    ),
    code(
        "import sys, pathlib",
        "sys.path.insert(0, str(pathlib.Path.cwd().parent))",
        "",
        "from uda_hub import db, seed",
        "from uda_hub.config import settings",
        "from uda_hub.retrieval import build_or_load_vectorstore",
        "from uda_hub.graph import build_app",
        "from uda_hub.logging_utils import configure_logging, get_run_log",
        "",
        "configure_logging()",
        "seed.seed_all(reset=True)              # fresh DB",
        "build_or_load_vectorstore(rebuild=True)  # build FAISS index from KB",
    ),
    md(
        "## 1. Compile the graph",
        "",
        "`build_app` returns a compiled LangGraph app wired with a SQLite checkpointer (short-term memory)",
        "and a SQLite long-term store. The Mermaid diagram below shows the agent topology.",
    ),
    code(
        "app = build_app()",
        "print(app.get_graph().draw_mermaid())",
    ),
    md("## 2. Helper to run a single ticket"),
    code(
        "import json, uuid",
        "from uda_hub.runner import run_ticket",
        "",
        "def demo(ticket: dict):",
        "    print(f\"\\n{'='*72}\\nTICKET {ticket['ticket_id']}: {ticket['subject']}\\n{'='*72}\")",
        "    result = run_ticket(app, ticket)",
        "    print('-- final answer --')",
        "    print(result['answer'])",
        "    print('-- routing trail --')",
        "    for step in result['log']:",
        "        print(' ', step)",
        "    return result",
    ),
    md(
        "## 3. Scenario A — Knowledge-base resolution",
        "",
        "A simple how-to question that should be answered straight from the FAQ.",
    ),
    code(
        "demo({",
        "    'ticket_id': 'tkt_demo_kb',",
        "    'user_id':   'usr_001',",
        "    'subject':   'How do I turn on 2FA?',",
        "    'body':      \"I want to enable two-factor authentication on my account.\",",
        "    'channel':   'web',",
        "    'urgency':   'normal',",
        "})",
    ),
    md(
        "## 4. Scenario B — Tool-driven resolution (refund)",
        "",
        "Customer with a billing issue. The Resolver should call the `process_refund` tool",
        "and confirm the action.",
    ),
    code(
        "demo({",
        "    'ticket_id': 'tkt_demo_refund',",
        "    'user_id':   'usr_002',",
        "    'subject':   'Charged twice again this month',",
        "    'body':      'You billed me $19.99 twice on May 3rd. Please refund the duplicate.',",
        "    'channel':   'email',",
        "    'urgency':   'high',",
        "})",
    ),
    md(
        "## 5. Scenario C — Escalation (no relevant article)",
        "",
        "Out-of-scope question with no matching article and a tool not safe to auto-execute.",
        "Confidence should fall below threshold and the Escalation agent should take over.",
    ),
    code(
        "demo({",
        "    'ticket_id': 'tkt_demo_esc',",
        "    'user_id':   'usr_003',",
        "    'subject':   'Bulk export still failing — clients impacted',",
        "    'body':      \"Our nightly bulk export to S3 returns 500. Three of our enterprise clients are blocked.\",",
        "    'channel':   'email',",
        "    'urgency':   'critical',",
        "})",
    ),
    md(
        "## 6. Scenario D — Memory in action",
        "",
        "Same user, follow-up question in the same thread. The graph should pick up the",
        "checkpointed state (short-term memory) and remember the prior context.",
    ),
    code(
        "demo({",
        "    'ticket_id': 'tkt_demo_followup',",
        "    'user_id':   'usr_002',",
        "    'subject':   'Follow-up to my refund',",
        "    'body':      \"Quick check — when should I see the refund on my statement?\",",
        "    'channel':   'email',",
        "    'urgency':   'normal',",
        "    'thread_id': 'usr_002-billing',  # same thread as scenario B",
        "})",
    ),
    md("## 7. Inspect long-term memory for a returning customer"),
    code(
        "from uda_hub.memory import get_long_term_store",
        "store = get_long_term_store()",
        "for item in store.search(('customer', 'usr_002')):",
        "    print(item.key, '->', item.value)",
    ),
    md("## 8. Inspect persisted ticket state in SQLite"),
    code(
        "from tabulate import tabulate",
        "print(tabulate(db.fetch_all(\"\"\"",
        "    SELECT t.ticket_id, t.status, m.category, m.urgency, m.sentiment, m.confidence, m.routed_to",
        "      FROM Ticket t JOIN TicketMetadata m USING(ticket_id)",
        "     WHERE t.ticket_id LIKE 'tkt_demo_%'",
        "     ORDER BY t.ticket_id",
        "\"\"\"), headers='keys'))",
    ),
    code(
        "print(tabulate(db.fetch_all(\"\"\"",
        "    SELECT ticket_id, role, author, substr(content,1,80) AS preview",
        "      FROM TicketMessage",
        "     WHERE ticket_id LIKE 'tkt_demo_%'",
        "     ORDER BY message_id",
        "\"\"\"), headers='keys'))",
    ),
    md(
        "---",
        "",
        "Every decision (classification, routing, retrieval, tool call, escalation) is logged via",
        "`uda_hub.logging_utils` and written to `data/uda_hub.db` via `TicketMessage`/`TicketMetadata`",
        "for full audit replay.",
    ),
]


def main() -> None:
    write_nb(NB_DIR / "01_database_setup.ipynb", NB_01)
    write_nb(NB_DIR / "02_end_to_end_demo.ipynb", NB_02)


if __name__ == "__main__":
    main()
