"""End-to-end runnable script equivalent of `03_agentic_app.ipynb`.

Usage from the solution/ folder::

    python 03_agentic_app.py            # runs all 4 demo scenarios
    python 03_agentic_app.py --chat     # also drops into the interactive REPL

Make sure you have run notebook 01 and 02 first (or call ``setup_databases``
which does it for you).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def setup_databases(rebuild_index: bool = False) -> None:
    from data.external import seed as ext_seed
    from data.core import seed as core_seed

    ext_seed.seed_all(reset=True)
    core_seed.seed_all(reset=True)

    import os

    if os.getenv("OPENAI_API_KEY"):
        from agentic.retrieval import build_or_load_vectorstore

        build_or_load_vectorstore(rebuild=rebuild_index)


def run_demo() -> None:
    from agentic.logging_utils import configure_logging
    from agentic.runner import run_ticket
    from agentic.workflow import build_app
    from utils import print_result

    configure_logging()
    app = build_app()

    scenarios = [
        dict(ticket_id="tkt_demo_kb",     user_id="usr_001",
             subject="How do I turn on 2FA?",
             body="I want to enable two-factor authentication on my Cult Pass account.",
             channel="web", urgency="normal"),
        dict(ticket_id="tkt_demo_refund", user_id="usr_002",
             subject="Charged twice this month",
             body="You billed me $9.99 twice for May membership. Please refund the duplicate.",
             channel="email", urgency="high"),
        dict(ticket_id="tkt_demo_cancel", user_id="usr_006",
             subject="Need to cancel my Hamlet booking",
             body="Please cancel my booking cp_b_007 (Hamlet at Donmar Warehouse). "
                  "Something came up.",
             channel="chat", urgency="normal"),
        dict(ticket_id="tkt_demo_esc",    user_id="usr_003",
             subject="Wallet QR not loading at venue right now",
             body="I'm at the venue and the QR will not load. Doors closed in 10 minutes!",
             channel="chat", urgency="critical"),
    ]
    for s in scenarios:
        print(f"\n{'='*72}\nTICKET {s['ticket_id']}: {s['subject']}\n{'='*72}")
        result = run_ticket(app, s)
        print_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="UDA-Hub end-to-end demo")
    parser.add_argument("--no-setup", action="store_true",
                        help="Skip seeding the databases (assume already set up).")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Rebuild the FAISS index from scratch.")
    parser.add_argument("--chat", action="store_true",
                        help="Drop into the interactive chat_interface() after the demo.")
    args = parser.parse_args()

    if not args.no_setup:
        setup_databases(rebuild_index=args.rebuild_index)
    run_demo()
    if args.chat:
        from utils import chat_interface
        chat_interface()


if __name__ == "__main__":
    main()
