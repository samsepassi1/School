"""RAG over the Knowledge table.

Pipeline:

  1. Read articles from the core DB ``Knowledge`` table (loaded from
     ``cultpass_articles.jsonl`` by notebook 02).
  2. Embed each ``title + body`` with OpenAI text-embedding-3-small.
  3. Index with FAISS (in-memory + on-disk persistence under
     ``data/models/uda_hub_kb``).
  4. Retrieval: ``similarity_search_with_score`` returns L2 distance over
     normalised vectors, which we convert to a cosine-style 0..1 score via
     ``1 - dist^2 / 2``.
  5. Confidence = best score returned. The supervisor compares it to
     ``settings.confidence_threshold`` to decide between Resolver and
     Escalation.

A keyword-overlap fallback is provided for offline tests / no-API-key runs
so the rest of the system stays exercisable without network access.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document

from agentic.config import settings
from data.core import db


logger = logging.getLogger(__name__)


class RetrievedArticle(TypedDict):
    article_id: str
    title: str
    category: str
    body: str
    score: float


def _docs_from_db() -> list[Document]:
    rows = db.fetch_all(
        "SELECT article_id, title, category, tags, body FROM Knowledge ORDER BY article_id"
    )
    return [
        Document(
            page_content=f"{r['title']}\n\n{r['body']}",
            metadata={
                "article_id": r["article_id"],
                "title":      r["title"],
                "category":   r["category"],
                "tags":       r["tags"],
            },
        )
        for r in rows
    ]


def _score_to_confidence(distance: float) -> float:
    sim = 1.0 - (distance * distance) / 2.0
    return max(0.0, min(1.0, sim))


def build_or_load_vectorstore(*, rebuild: bool = False):
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    path = Path(settings.vectorstore_path)
    embeddings = OpenAIEmbeddings(model=settings.embed_model)

    if path.exists() and any(path.iterdir()) and not rebuild:
        logger.info("loading FAISS index from %s", path)
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)

    docs = _docs_from_db()
    if not docs:
        raise RuntimeError("Knowledge table is empty — run notebook 02 first.")
    logger.info("building FAISS index over %d articles", len(docs))
    vs = FAISS.from_documents(docs, embeddings)
    path.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(path))
    return vs


def retrieve(query: str, k: int = 4, *, vectorstore=None) -> tuple[list[RetrievedArticle], float]:
    vs = vectorstore or build_or_load_vectorstore()
    pairs = vs.similarity_search_with_score(query, k=k)
    out: list[RetrievedArticle] = []
    best = 0.0
    for doc, dist in pairs:
        score = _score_to_confidence(float(dist))
        best = max(best, score)
        out.append(
            RetrievedArticle(
                article_id=doc.metadata["article_id"],
                title=doc.metadata["title"],
                category=doc.metadata["category"],
                body=doc.page_content,
                score=score,
            )
        )
    return out, best


# --- offline keyword fallback ------------------------------------------------

def keyword_retrieve(query: str, k: int = 4) -> tuple[list[RetrievedArticle], float]:
    tokens = {t for t in _tokenise(query) if len(t) > 2}
    if not tokens:
        return [], 0.0
    rows = db.fetch_all("SELECT article_id, title, category, tags, body FROM Knowledge")
    scored: list[tuple[float, dict]] = []
    for r in rows:
        text = f"{r['title']} {r['tags']} {r['body']}".lower()
        words = set(_tokenise(text))
        overlap = len(tokens & words)
        if overlap == 0:
            continue
        score = overlap / math.sqrt(len(words) + 1)
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return [], 0.0
    top = scored[:k]
    max_score = top[0][0]
    out: list[RetrievedArticle] = []
    for raw_score, r in top:
        out.append(
            RetrievedArticle(
                article_id=r["article_id"],
                title=r["title"],
                category=r["category"],
                body=f"{r['title']}\n\n{r['body']}",
                score=min(1.0, raw_score / (max_score + 1e-6)),
            )
        )
    best = out[0]["score"] if out else 0.0
    return out, best


def _tokenise(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())
