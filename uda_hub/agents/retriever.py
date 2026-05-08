"""Retrieval agent.

Calls the FAISS-backed knowledge base, attaches retrieved articles to state,
and computes the best-match confidence used by the supervisor for routing.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from uda_hub.agents.state import AgentState
from uda_hub.retrieval import keyword_retrieve, retrieve


logger = logging.getLogger(__name__)


def retriever_node(state: AgentState) -> dict[str, Any]:
    query = f"{state.get('subject','')} {state.get('body','')}".strip()

    use_keyword_fallback = not os.getenv("OPENAI_API_KEY")
    try:
        if use_keyword_fallback:
            docs, best = keyword_retrieve(query, k=4)
        else:
            docs, best = retrieve(query, k=4)
    except Exception as exc:
        logger.warning("retriever falling back to keyword search (%s)", exc)
        docs, best = keyword_retrieve(query, k=4)

    log = list(state.get("log", []))
    if docs:
        ids = ", ".join(d["article_id"] for d in docs)
        log.append(f"retriever -> {len(docs)} docs [{ids}] best_score={best:.2f}")
    else:
        log.append("retriever -> no matching documents")

    logger.info(
        "retriever ticket=%s docs=%d best=%.2f",
        state.get("ticket_id"),
        len(docs),
        best,
    )
    return {"retrieved": docs, "retrieval_confidence": best, "log": log}
