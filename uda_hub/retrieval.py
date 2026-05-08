"""FAISS-backed knowledge-base retrieval with confidence scoring.

The vector index is persisted to disk so subsequent runs are fast. Confidence
is derived from the cosine similarity returned by FAISS — the supervisor uses
it to decide whether to escalate.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document

from uda_hub import db
from uda_hub.config import settings


logger = logging.getLogger(__name__)


class RetrievedArticle(TypedDict):
    article_id: str
    title: str
    category: str
    body: str
    score: float  # 0..1, higher = more relevant


def _docs_from_db() -> list[Document]:
    rows = db.fetch_all(
        "SELECT article_id, title, category, tags, body FROM Knowledge ORDER BY article_id"
    )
    return [
        Document(
            page_content=f"{r['title']}\n\n{r['body']}",
            metadata={
                "article_id": r["article_id"],
                "title": r["title"],
                "category": r["category"],
                "tags": r["tags"],
            },
        )
        for r in rows
    ]


def _score_to_confidence(distance: float) -> float:
    """FAISS returns L2 distance for normalised vectors; map to 0..1 similarity.

    For unit-norm vectors, ``cos = 1 - dist^2 / 2``. We clamp to [0,1].
    """
    sim = 1.0 - (distance * distance) / 2.0
    return max(0.0, min(1.0, sim))


def build_or_load_vectorstore(*, rebuild: bool = False):
    """Build (or load) the FAISS index over the Knowledge table.

    Imports are local so the module can be inspected without optional deps installed.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    path = Path(settings.vectorstore_path)
    embeddings = OpenAIEmbeddings(model=settings.embed_model)

    if path.exists() and not rebuild:
        logger.info("loading FAISS index from %s", path)
        return FAISS.load_local(
            str(path), embeddings, allow_dangerous_deserialization=True
        )

    docs = _docs_from_db()
    if not docs:
        raise RuntimeError("Knowledge table is empty — run seed.seed_all() first.")
    logger.info("building FAISS index over %d articles", len(docs))
    vs = FAISS.from_documents(docs, embeddings)
    path.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(path))
    return vs


def retrieve(
    query: str, k: int = 4, *, vectorstore=None
) -> tuple[list[RetrievedArticle], float]:
    """Return top-k articles plus the best confidence in [0,1]."""
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


# ----- keyword fallback (used when no API key is available) -----------------

def keyword_retrieve(query: str, k: int = 4) -> tuple[list[RetrievedArticle], float]:
    """Tokenised TF-overlap fallback. Cheap and deterministic — handy for tests."""
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
        # Crude IDF-like normalisation
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
