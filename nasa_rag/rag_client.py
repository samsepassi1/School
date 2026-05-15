"""RAG retrieval client backed by ChromaDB.

Embeds the user question with the same model used during ingestion, runs a
similarity query against the configured collection, and returns the top-k
chunks formatted into a clean context block for the LLM.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI


KNOWN_MISSIONS = ("Apollo 11", "Apollo 13", "Challenger")


@dataclass
class RetrievedChunk:
    text: str
    source: str
    mission: str
    score: float  # similarity in [0, 1] (1 = identical)
    chunk_index: int = 0


@dataclass
class RetrievalResult:
    question: str
    chunks: list[RetrievedChunk] = field(default_factory=list)

    @property
    def context(self) -> str:
        return format_context(self.chunks)

    @property
    def contexts(self) -> list[str]:
        """Plain list of chunk texts, suitable for RAGAS."""
        return [c.text for c in self.chunks]


class RAGClient:
    """Thin wrapper around ChromaDB + OpenAI embeddings.

    Parameters
    ----------
    chroma_dir : Path | str, optional
        Where the Chroma persistent client should look for data. Defaults to
        the ``CHROMA_DIR`` env var, then ``./chroma_db``.
    collection_name : str, optional
        Name of the collection. Defaults to ``CHROMA_COLLECTION`` env var,
        then ``nasa_missions``.
    embed_model : str, optional
        OpenAI embedding model used for query embedding. Must match the model
        used at ingestion time. Defaults to ``OPENAI_EMBED_MODEL`` env var,
        then ``text-embedding-3-small``.
    """

    def __init__(
        self,
        chroma_dir: Path | str | None = None,
        collection_name: str | None = None,
        embed_model: str | None = None,
        openai_client: Optional[OpenAI] = None,
    ) -> None:
        load_dotenv()
        self.chroma_dir = Path(chroma_dir or os.getenv("CHROMA_DIR", "./chroma_db"))
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION", "nasa_missions")
        self.embed_model = embed_model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

        self._chroma = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self._chroma.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._openai = openai_client or OpenAI()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collection_size(self) -> int:
        return self.collection.count()

    def embed_query(self, question: str) -> list[float]:
        resp = self._openai.embeddings.create(model=self.embed_model, input=[question])
        return resp.data[0].embedding

    def search(
        self,
        question: str,
        *,
        k: int = 4,
        mission: str | None = None,
    ) -> RetrievalResult:
        """Run a similarity search.

        Parameters
        ----------
        question : str
            The user question.
        k : int
            Number of top results to return.
        mission : str | None
            If provided, restricts results to chunks tagged with this mission.
        """
        if not question or not question.strip():
            return RetrievalResult(question=question, chunks=[])

        where: dict | None = None
        if mission and mission != "All":
            where = {"mission": mission}

        embedding = self.embed_query(question)
        raw = self.collection.query(
            query_embeddings=[embedding],
            n_results=max(1, k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        seen_texts: set[str] = set()
        for doc, md, dist in zip(docs, metas, dists):
            if not doc:
                continue
            key = doc.strip()
            if key in seen_texts:
                continue
            seen_texts.add(key)
            md = md or {}
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source=md.get("source", "unknown"),
                    mission=md.get("mission", "Unknown"),
                    score=float(1.0 - dist) if dist is not None else 0.0,
                    chunk_index=int(md.get("chunk_index", 0)),
                )
            )

        chunks.sort(key=lambda c: c.score, reverse=True)
        return RetrievalResult(question=question, chunks=chunks)


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(chunks: list[RetrievedChunk]) -> str:
    """Join retrieved chunks into a single labeled context block."""
    if not chunks:
        return "(no relevant context retrieved)"
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[{i}] Mission: {chunk.mission} | Source: {chunk.source} "
            f"| Score: {chunk.score:.3f}"
        )
        parts.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n---\n\n".join(parts)
