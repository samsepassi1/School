"""Information-retrieval evaluation metrics.

All three metrics expect the same data shapes:

* ``retrieved``: ``{query_index: [doc_id, ...]}`` from a retriever's
  ``retrieve`` method, ordered by descending relevance.
* ``relevant``: ``{query_index: [doc_id, ...]}`` ground-truth relevant
  document ids per query.
"""

from __future__ import annotations

from typing import Dict, List


def recall_at_k(
    retrieved: Dict[int, List[int]],
    relevant: Dict[int, List[int]],
    k: int = 5,
) -> float:
    """Mean Recall@k across queries.

    Recall@k = (# relevant docs in top-k) / (# total relevant docs).
    """
    # TODO: implement recall@k
    if not retrieved:
        return 0.0

    scores: List[float] = []
    for q_idx, retrieved_ids in retrieved.items():
        gold = set(relevant.get(q_idx, []))
        if not gold:
            continue
        top_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in top_k if doc_id in gold)
        scores.append(hits / len(gold))

    return sum(scores) / len(scores) if scores else 0.0


def precision_at_k(
    retrieved: Dict[int, List[int]],
    relevant: Dict[int, List[int]],
    k: int = 5,
) -> float:
    """Mean Precision@k across queries.

    Precision@k = (# relevant docs in top-k) / k.
    """
    # TODO: implement precision@k
    if not retrieved or k <= 0:
        return 0.0

    scores: List[float] = []
    for q_idx, retrieved_ids in retrieved.items():
        gold = set(relevant.get(q_idx, []))
        if not gold:
            continue
        top_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in top_k if doc_id in gold)
        scores.append(hits / k)

    return sum(scores) / len(scores) if scores else 0.0


def mrr(
    retrieved: Dict[int, List[int]],
    relevant: Dict[int, List[int]],
) -> float:
    """Mean Reciprocal Rank across queries.

    For each query we find the rank (1-indexed) of the first relevant
    document in the retrieved list and average ``1/rank``. Queries with no
    relevant document in the list contribute zero.
    """
    # TODO: implement MRR
    if not retrieved:
        return 0.0

    reciprocal_ranks: List[float] = []
    for q_idx, retrieved_ids in retrieved.items():
        gold = set(relevant.get(q_idx, []))
        if not gold:
            continue
        rank_score = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in gold:
                rank_score = 1.0 / rank
                break
        reciprocal_ranks.append(rank_score)

    return (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks
        else 0.0
    )


def evaluate_all(
    retrieved: Dict[int, List[int]],
    relevant: Dict[int, List[int]],
    k: int = 3,
) -> Dict[str, float]:
    """Convenience wrapper that returns all three metrics at once."""
    return {
        f"recall@{k}": recall_at_k(retrieved, relevant, k=k),
        f"precision@{k}": precision_at_k(retrieved, relevant, k=k),
        "mrr": mrr(retrieved, relevant),
    }


__all__ = ["recall_at_k", "precision_at_k", "mrr", "evaluate_all"]
