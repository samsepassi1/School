"""Word2Vec retriever.

Trains a small Word2Vec model on the corpus, then represents each document
and query as the mean of its in-vocabulary word vectors. Retrieval is done
with cosine similarity. Includes a simple grid search over the main
hyperparameters (``vector_size``, ``window``, ``min_count``) that picks the
configuration with the best Recall@k on a held-out query set.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from bm25_retriever import tokenize


class Word2VecRetriever:
    """Word2Vec-backed retriever with optional hyperparameter search."""

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        epochs: int = 50,
        seed: int = 42,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self.model = None
        self.documents: List[str] = []
        self.doc_vectors: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def build_index(self, documents: Iterable[str]) -> np.ndarray:
        """Train the Word2Vec model and embed each document."""
        from gensim.models import Word2Vec

        self.documents = list(documents)
        tokenized = [tokenize(doc) for doc in self.documents]

        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=1,
            seed=self.seed,
            epochs=self.epochs,
        )

        self.doc_vectors = np.vstack(
            [self._embed_tokens(tokens) for tokens in tokenized]
        )
        return self.doc_vectors

    def _embed_tokens(self, tokens: Sequence[str]) -> np.ndarray:
        """Mean-pool the vectors for the in-vocabulary tokens."""
        assert self.model is not None, "build_index() must be called first."
        vectors = [
            self.model.wv[t] for t in tokens if t in self.model.wv  # type: ignore[index]
        ]
        if not vectors:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(vectors, axis=0)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> Dict[int, List[int]]:
        if self.doc_vectors is None:
            raise RuntimeError("build_index() must be called before retrieve().")

        query_vectors = np.vstack(
            [self._embed_tokens(tokenize(q)) for q in queries]
        )

        # Cosine similarity = normalized dot product.
        def _normalize(matrix: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return matrix / norms

        q_norm = _normalize(query_vectors)
        d_norm = _normalize(self.doc_vectors)
        similarity = q_norm @ d_norm.T

        results: Dict[int, List[int]] = {}
        for q_idx, scores in enumerate(similarity):
            top_indices = np.argsort(-scores)[:top_k]
            results[q_idx] = [int(i) for i in top_indices]
        return results

    # ------------------------------------------------------------------
    # Hyperparameter search
    # ------------------------------------------------------------------
    @classmethod
    def grid_search(
        cls,
        documents: Sequence[str],
        queries: Sequence[str],
        relevant: Dict[int, List[int]],
        k: int = 3,
        vector_sizes: Sequence[int] = (50, 100, 200),
        windows: Sequence[int] = (3, 5),
        min_counts: Sequence[int] = (1, 2),
    ) -> Tuple["Word2VecRetriever", Dict[str, float]]:
        """Brute-force grid search that maximises Recall@k.

        Returns the trained retriever for the best configuration along with a
        dictionary of the chosen hyperparameters and the achieved Recall@k.
        """
        from evaluator import recall_at_k

        best_score = -1.0
        best_params: Dict[str, float] = {}
        best_retriever: Optional[Word2VecRetriever] = None

        for vs in vector_sizes:
            for w in windows:
                for mc in min_counts:
                    retriever = cls(vector_size=vs, window=w, min_count=mc)
                    retriever.build_index(documents)
                    retrieved = retriever.retrieve(list(queries), top_k=k)
                    score = recall_at_k(retrieved, relevant, k=k)
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "vector_size": vs,
                            "window": w,
                            "min_count": mc,
                            f"recall_at_{k}": score,
                        }
                        best_retriever = retriever

        assert best_retriever is not None
        return best_retriever, best_params


__all__ = ["Word2VecRetriever"]
