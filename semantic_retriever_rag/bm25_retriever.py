"""Classical BM25 keyword retriever.

Implements Okapi BM25 from scratch so we do not need ``rank_bm25`` as a
dependency. The tokenizer is intentionally simple — lower-case, drop
punctuation, split on whitespace, remove a small stop-word list.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "i", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "was", "were", "with", "do", "how", "what", "when", "where", "which",
    "who", "why", "my", "me", "you", "your", "we", "our", "this", "these",
    "those", "can", "could", "would", "should", "will",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lower-case, split on non-alphanumerics, strip stop-words."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Retriever:
    """Okapi BM25 retriever with the standard k1=1.5, b=0.75 parameters."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.tokenized: List[List[str]] = []
        self.doc_freqs: List[Counter] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.idf: Dict[str, float] = {}

    def build_index(self, documents: Iterable[str]) -> None:
        """Tokenize the corpus and pre-compute term frequencies and IDF."""
        self.documents = list(documents)
        self.tokenized = [tokenize(doc) for doc in self.documents]
        self.doc_freqs = [Counter(tokens) for tokens in self.tokenized]
        self.doc_lens = [len(tokens) for tokens in self.tokenized]
        self.avg_doc_len = (
            sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        )

        n_docs = len(self.documents)
        df: Counter = Counter()
        for tokens in self.tokenized:
            for term in set(tokens):
                df[term] += 1

        # Okapi BM25 IDF with the +1 smoothing inside the log to keep values
        # non-negative for terms that appear in every document.
        self.idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def _score(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * len(self.documents)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_idx, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avg_doc_len
                )
                scores[doc_idx] += idf * (tf * (self.k1 + 1)) / denom
        return scores

    def retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> Dict[int, List[int]]:
        """Return the top-k document indices for each query."""
        results: Dict[int, List[int]] = {}
        for q_idx, query in enumerate(queries):
            scores = self._score(tokenize(query))
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
            results[q_idx] = ranked[:top_k]
        return results


__all__ = ["BM25Retriever", "tokenize"]
