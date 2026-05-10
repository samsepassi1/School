"""ChromaDB-backed vector store for the UdaPlay game knowledge base.

The store is persistent: documents are written to a directory on disk so
notebook 01 (ingestion) and notebook 02 (agent) can share the same index.
Embeddings come from OpenAI's text-embedding-3-small by default; if no API
key is configured the store falls back to Chroma's built-in default
(SentenceTransformer all-MiniLM-L6-v2) so the project can be evaluated
offline.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


DEFAULT_COLLECTION = "udaplay_games"
DEFAULT_PERSIST_DIR = "chromadb"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


@dataclass
class GameDocument:
    """A single game record converted into a retrievable document."""

    id: str
    text: str
    metadata: dict

    @classmethod
    def from_record(cls, record: dict, source: str) -> "GameDocument":
        name = record.get("Name", "Unknown")
        platform = record.get("Platform", "Unknown")
        year = record.get("YearOfRelease", "Unknown")
        publisher = record.get("Publisher", "Unknown")
        developer = record.get("Developer", "Unknown")
        genre = record.get("Genre", "Unknown")
        description = record.get("Description", "")

        # The document text is what the embedder sees. We keep it dense and
        # natural-language so semantic search works for "who developed X" or
        # "when was Y released" without needing a structured query.
        text = (
            f"{name} ({year}) — {genre} on {platform}. "
            f"Developed by {developer} and published by {publisher}. "
            f"{description}"
        )

        metadata = {
            "name": name,
            "platform": platform,
            "year": int(year) if str(year).isdigit() else year,
            "publisher": publisher,
            "developer": developer,
            "genre": genre,
            "source": source,
        }
        return cls(id=Path(source).stem, text=text, metadata=metadata)


def load_games_from_directory(games_dir: str | os.PathLike) -> list[GameDocument]:
    """Read every ``*.json`` file in ``games_dir`` and return GameDocuments."""
    path = Path(games_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Games directory not found: {path}")

    documents: list[GameDocument] = []
    for json_path in sorted(path.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as fp:
            record = json.load(fp)
        documents.append(GameDocument.from_record(record, source=str(json_path)))
    return documents


def _build_embedding_function(model: str | None):
    """Pick OpenAI embeddings if a key is available, otherwise the default."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and model:
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name=model
        )
    return embedding_functions.DefaultEmbeddingFunction()


class VectorStoreManager:
    """Thin wrapper around ChromaDB's PersistentClient.

    The class hides the embedding-function and collection bookkeeping so
    notebooks and the agent share one ergonomic API: ``add_games`` /
    ``query`` / ``count`` / ``reset``.
    """

    def __init__(
        self,
        persist_directory: str | os.PathLike = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str | None = DEFAULT_EMBED_MODEL,
    ) -> None:
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self.embedding_function = _build_embedding_function(embedding_model)

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    # --- ingestion -----------------------------------------------------
    def add_games(self, documents: Sequence[GameDocument]) -> int:
        """Upsert game documents into the collection. Returns rows written."""
        if not documents:
            return 0
        self.collection.upsert(
            ids=[d.id for d in documents],
            documents=[d.text for d in documents],
            metadatas=[d.metadata for d in documents],
        )
        return len(documents)

    def add_text(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        """Add a free-form note (used by long-term memory / web search caching)."""
        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

    # --- query ---------------------------------------------------------
    def query(self, question: str, k: int = 4) -> list[dict]:
        """Return the top-k matches for a question with text + distance."""
        result = self.collection.query(query_texts=[question], n_results=k)
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        hits: list[dict] = []
        for i, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append(
                {
                    "id": i,
                    "text": doc,
                    "metadata": meta or {},
                    "distance": float(dist) if dist is not None else None,
                }
            )
        return hits

    # --- maintenance ---------------------------------------------------
    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection (use sparingly — destructive)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def peek(self, n: int = 5) -> Iterable[dict]:
        """Return up to ``n`` documents for inspection / debugging."""
        out = self.collection.peek(limit=n)
        for i, doc, meta in zip(out.get("ids", []), out.get("documents", []), out.get("metadatas", [])):
            yield {"id": i, "text": doc, "metadata": meta or {}}
