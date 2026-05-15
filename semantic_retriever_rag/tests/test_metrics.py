"""Unit tests for evaluator metrics — runnable without any model downloads."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator import mrr, precision_at_k, recall_at_k  # noqa: E402


def test_recall_at_k_perfect():
    retrieved = {0: [1, 2, 3], 1: [4, 5]}
    relevant = {0: [1], 1: [4]}
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_at_k_miss():
    retrieved = {0: [2, 3, 4]}
    relevant = {0: [1]}
    assert recall_at_k(retrieved, relevant, k=3) == 0.0


def test_precision_at_k():
    retrieved = {0: [1, 2, 3]}
    relevant = {0: [1, 3]}
    # 2 hits out of 3 retrieved = 0.6666...
    assert abs(precision_at_k(retrieved, relevant, k=3) - 2 / 3) < 1e-9


def test_mrr_first_position():
    retrieved = {0: [1, 2, 3]}
    relevant = {0: [1]}
    assert mrr(retrieved, relevant) == 1.0


def test_mrr_third_position():
    retrieved = {0: [9, 8, 7]}
    relevant = {0: [7]}
    assert abs(mrr(retrieved, relevant) - 1 / 3) < 1e-9


def test_mrr_no_hit():
    retrieved = {0: [9, 8, 7]}
    relevant = {0: [1]}
    assert mrr(retrieved, relevant) == 0.0
