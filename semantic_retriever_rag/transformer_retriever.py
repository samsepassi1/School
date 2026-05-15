"""Transformer-based semantic retriever.

Encodes documents and queries with a pretrained sentence-transformer model
and retrieves the top-k documents per query using cosine similarity.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np


class TransformerRetriever:
    """Dense retriever backed by a Hugging Face sentence-transformer model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # TODO: load a pretrained transformer model from sentence-transformers
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.documents: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None

    def build_index(self, documents: Iterable[str]) -> np.ndarray:
        """Encode the document corpus into dense embeddings.

        Returns the embedding matrix with shape ``(n_documents, embedding_dim)``.
        """
        # TODO: encode the document corpus into embeddings
        self.documents = list(documents)
        self.doc_embeddings = self.model.encode(
            self.documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self.doc_embeddings

    def encode_queries(self, queries: Iterable[str]) -> np.ndarray:
        """Encode incoming queries with the same transformer model."""
        # TODO: encode the queries
        return self.model.encode(
            list(queries),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> Dict[int, List[int]]:
        """Return the top-k document indices for each query.

        Cosine similarity is used to rank documents. Because embeddings are
        L2-normalized during encoding, the cosine similarity reduces to a
        dot product between query and document vectors.
        """
        if self.doc_embeddings is None:
            raise RuntimeError("build_index() must be called before retrieve().")

        query_embeddings = self.encode_queries(queries)

        # Cosine similarity == dot product on L2-normalized vectors.
        similarity = query_embeddings @ self.doc_embeddings.T  # (n_queries, n_docs)

        results: Dict[int, List[int]] = {}
        for q_idx, scores in enumerate(similarity):
            # argsort descending; take top_k indices.
            top_indices = np.argsort(-scores)[:top_k]
            results[q_idx] = [int(i) for i in top_indices]
        return results

    def retrieve_with_scores(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> Dict[int, List[tuple]]:
        """Like :meth:`retrieve` but also returns the similarity scores."""
        if self.doc_embeddings is None:
            raise RuntimeError("build_index() must be called before retrieve().")

        query_embeddings = self.encode_queries(queries)
        similarity = query_embeddings @ self.doc_embeddings.T

        out: Dict[int, List[tuple]] = {}
        for q_idx, scores in enumerate(similarity):
            top_indices = np.argsort(-scores)[:top_k]
            out[q_idx] = [(int(i), float(scores[i])) for i in top_indices]
        return out


__all__ = ["TransformerRetriever"]
