"""Shared helpers and the interactive chat shell used by 03_agentic_app."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

# Make ``solution/`` importable regardless of where the entrypoint is run from.
SOLUTION_ROOT = Path(__file__).resolve().parent
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #

def print_log(log: list[str], *, prefix: str = "  ") -> None:
    for line in log:
        print(f"{prefix}{line}")


def print_result(result: dict[str, Any]) -> None:
    print("\n=== answer ===")
    print(result.get("answer", "(no answer)"))
    print("\n=== routing trail ===")
    print_log(result.get("log", []))
    cls = result.get("classification") or {}
    if cls:
        print(
            f"\n=== classification ===\n  {cls.get('category')}/"
            f"{cls.get('urgency')}/{cls.get('sentiment')} "
            f"(conf={cls.get('confidence', 0):.2f})"
        )
    if result.get("retrieved"):
        print("\n=== retrieved articles ===")
        for d in result["retrieved"]:
            print(f"  [{d['article_id']}] {d['title']}  score={d['score']:.2f}")


# --------------------------------------------------------------------------- #
# Interactive shell
# --------------------------------------------------------------------------- #

_HELP = """\
Commands:
  /help                       show this message
  /user <user_id>             switch to a different customer (default usr_001)
  /thread <id>                pin a thread_id (so memory persists across messages)
  /history                    print recent thread state from the checkpointer
  /reset                      start a brand-new thread
  /quit | /exit               leave
Anything else is treated as a new ticket from the current user.
"""


def chat_interface(app=None, *, default_user: str = "usr_001",
                   default_channel: str = "chat") -> None:
    """Simple REPL that pushes free-text messages through the workflow.

    The same ``thread_id`` is reused for an entire session so short-term
    memory works out of the box. ``/reset`` starts a fresh one.
    """
    from agentic.runner import run_ticket
    from agentic.workflow import build_app

    if app is None:
        app = build_app()

    user_id = default_user
    thread_id = f"chat-{user_id}-{uuid.uuid4().hex[:6]}"

    print("UDA-Hub chat — type /help for commands. Ctrl-C to exit.")
    print(f"  user={user_id}  thread={thread_id}\n")

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not text:
            continue
        if text in {"/quit", "/exit"}:
            print("bye")
            return
        if text == "/help":
            print(_HELP)
            continue
        if text.startswith("/user "):
            user_id = text.split(maxsplit=1)[1].strip()
            print(f"  switched to user={user_id}")
            continue
        if text.startswith("/thread "):
            thread_id = text.split(maxsplit=1)[1].strip()
            print(f"  pinned thread={thread_id}")
            continue
        if text == "/reset":
            thread_id = f"chat-{user_id}-{uuid.uuid4().hex[:6]}"
            print(f"  new thread={thread_id}")
            continue
        if text == "/history":
            try:
                snap = app.get_state({"configurable": {"thread_id": thread_id}})
                vals = snap.values if hasattr(snap, "values") else {}
                print(json.dumps({k: vals.get(k) for k in
                                  ("ticket_id", "answer", "needs_escalation", "log")},
                                 default=str, indent=2))
            except Exception as exc:
                print(f"  (no history: {exc})")
            continue

        ticket_id = f"chat_{uuid.uuid4().hex[:8]}"
        result = run_ticket(app, {
            "ticket_id": ticket_id,
            "user_id":   user_id,
            "subject":   text[:80],
            "body":      text,
            "channel":   default_channel,
            "urgency":   "normal",
            "thread_id": thread_id,
        })
        print()
        print_result(result)
        print()
