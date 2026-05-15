"""Real-time and batch evaluation for the NASA RAG system.

Computes RAGAS faithfulness + response relevancy for every (question,
contexts, answer) triple. BLEU and ROUGE-L are also computed when a reference
answer is provided. The module is designed to fail soft: if RAGAS or its
optional dependencies aren't installed it falls back to the lexical metrics
and reports the missing pieces in the result, instead of crashing.
"""
from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv


# Lazy imports kept inside helpers so the rest of the app can run even if
# RAGAS isn't installed in the local environment.


@dataclass
class EvalScores:
    question: str
    answer: str
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "metrics": self.metrics,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Lexical fallbacks (no RAGAS / no OpenAI required)
# ---------------------------------------------------------------------------

def _safe_bleu(reference: str, hypothesis: str) -> float | None:
    if not reference or not hypothesis:
        return None
    try:
        from sacrebleu import sentence_bleu
        return float(sentence_bleu(hypothesis, [reference]).score) / 100.0
    except Exception:  # pragma: no cover - optional dep
        return None


def _safe_rouge_l(reference: str, hypothesis: str) -> float | None:
    if not reference or not hypothesis:
        return None
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(reference, hypothesis)["rougeL"].fmeasure)
    except Exception:  # pragma: no cover - optional dep
        return None


def _context_precision_lexical(contexts: Iterable[str], answer: str) -> float | None:
    """Cheap lexical precision: fraction of contexts that share a >=4-word
    n-gram with the answer. Useful as a sanity-check signal when RAGAS isn't
    available.
    """
    contexts = list(contexts)
    if not contexts or not answer:
        return None
    answer_tokens = answer.lower().split()
    if len(answer_tokens) < 4:
        return None
    answer_ngrams = {
        " ".join(answer_tokens[i : i + 4])
        for i in range(len(answer_tokens) - 3)
    }
    hits = 0
    for ctx in contexts:
        ctx_tokens = ctx.lower().split()
        ctx_ngrams = {
            " ".join(ctx_tokens[i : i + 4])
            for i in range(len(ctx_tokens) - 3)
        }
        if answer_ngrams & ctx_ngrams:
            hits += 1
    return hits / len(contexts)


# ---------------------------------------------------------------------------
# RAGAS integration
# ---------------------------------------------------------------------------

def _try_import_ragas():
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
        return evaluate, faithfulness, answer_relevancy, Dataset
    except Exception as exc:  # pragma: no cover - depends on env
        return None, None, None, None


def _ragas_scores(
    question: str,
    contexts: list[str],
    answer: str,
    reference: str | None,
) -> tuple[dict[str, float], list[str]]:
    """Compute Faithfulness + Response Relevancy via RAGAS.

    Returns (metrics, errors). On failure, returns ({}, [error_msg]) so the
    caller can fall back gracefully.
    """
    errors: list[str] = []
    evaluate, faithfulness, answer_relevancy, Dataset = _try_import_ragas()
    if evaluate is None:
        return {}, ["ragas / datasets not installed - skipping RAGAS metrics"]

    if not contexts or not answer or not question:
        return {}, ["empty question/answer/contexts - skipping RAGAS metrics"]

    sample: dict[str, list] = {
        "question": [question],
        "contexts": [contexts],
        "answer": [answer],
    }
    if reference:
        sample["ground_truth"] = [reference]
        sample["reference"] = [reference]

    try:
        dataset = Dataset.from_dict(sample)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        scores: dict[str, float] = {}
        # RAGAS >=0.1 exposes a Result object that behaves like a dict of
        # metric_name -> list[float] (per-sample) or a single mean.
        try:
            df = result.to_pandas()
            for col in df.columns:
                if col in {"question", "contexts", "answer", "ground_truth", "reference"}:
                    continue
                value = df[col].iloc[0]
                try:
                    scores[col] = float(value)
                except (TypeError, ValueError):
                    continue
        except Exception:
            # Fallback: try dict access
            for key in ("faithfulness", "answer_relevancy"):
                if key in result:
                    try:
                        scores[key] = float(result[key])
                    except Exception:  # noqa: BLE001
                        pass
        return scores, errors
    except Exception as exc:  # noqa: BLE001
        errors.append(f"ragas evaluation failed: {exc}")
        return {}, errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RAGASEvaluator:
    """Evaluate (question, contexts, answer) triples.

    Real-time use (single triple) is via :meth:`score`. Batch evaluation over
    a test set is via :meth:`evaluate_batch`.
    """

    def __init__(self) -> None:
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            # RAGAS's faithfulness and answer_relevancy default to using
            # OpenAI judges; we leave the key check to runtime but record it.
            self._missing_key = True
        else:
            self._missing_key = False

    def score(
        self,
        question: str,
        contexts: list[str] | str,
        answer: str,
        *,
        reference: str | None = None,
    ) -> EvalScores:
        if isinstance(contexts, str):
            contexts_list = [contexts] if contexts.strip() else []
        else:
            contexts_list = [c for c in contexts if c and c.strip()]

        result = EvalScores(question=question or "", answer=answer or "")

        if not question or not question.strip():
            result.errors.append("question is empty")
            return result
        if not answer or not answer.strip():
            result.errors.append("answer is empty")
            return result
        if not contexts_list:
            result.errors.append("no non-empty contexts provided")

        ragas_metrics, ragas_errors = _ragas_scores(
            question, contexts_list, answer, reference
        )
        result.metrics.update(ragas_metrics)
        result.errors.extend(ragas_errors)

        # Lexical / reference-based metrics (always available)
        if reference:
            bleu = _safe_bleu(reference, answer)
            rouge_l = _safe_rouge_l(reference, answer)
            if bleu is not None:
                result.metrics["bleu"] = bleu
            if rouge_l is not None:
                result.metrics["rouge_l"] = rouge_l

        lex_precision = _context_precision_lexical(contexts_list, answer)
        if lex_precision is not None:
            result.metrics["context_precision_lexical"] = lex_precision

        return result

    def evaluate_batch(
        self,
        test_cases: list[dict],
    ) -> dict[str, Any]:
        """Evaluate a list of test cases.

        Each ``test_cases`` entry should look like::

            {
                "question": str,
                "contexts": list[str],
                "answer": str,
                "reference": str | None,  # optional
            }
        """
        per_question: list[dict[str, Any]] = []
        metric_values: dict[str, list[float]] = {}

        for case in test_cases:
            scored = self.score(
                question=case.get("question", ""),
                contexts=case.get("contexts", []),
                answer=case.get("answer", ""),
                reference=case.get("reference"),
            )
            per_question.append(scored.to_dict())
            for name, value in scored.metrics.items():
                metric_values.setdefault(name, []).append(value)

        aggregate: dict[str, dict[str, float]] = {}
        for name, values in metric_values.items():
            if not values:
                continue
            aggregate[name] = {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
            if len(values) > 1:
                aggregate[name]["stdev"] = statistics.pstdev(values)

        return {"per_question": per_question, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Eval-set loader
# ---------------------------------------------------------------------------

def load_evaluation_dataset(path: Path) -> list[dict[str, Any]]:
    """Load an evaluation set.

    Supports two formats:

    1. ``test_questions.json``: a JSON list of ``{question, mission?, reference?,
       category?}`` objects.
    2. ``evaluation_dataset.txt``: blocks separated by blank lines, each block
       starts with ``Q:`` followed by optional ``Mission:``, ``Category:``,
       and ``Expected:`` lines.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation set not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open() as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("JSON eval set must be a list of objects")
        return data

    cases: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if current.get("question"):
                cases.append(current)
                current = {}
            continue
        lower = line.lower()
        if lower.startswith("q:"):
            if current.get("question"):
                cases.append(current)
            current = {"question": line[2:].strip()}
        elif lower.startswith("mission:"):
            current["mission"] = line.split(":", 1)[1].strip()
        elif lower.startswith("category:"):
            current["category"] = line.split(":", 1)[1].strip()
        elif lower.startswith("expected:"):
            current["reference"] = line.split(":", 1)[1].strip()
        else:
            # Continuation of the most recent field
            if "reference" in current:
                current["reference"] = (current["reference"] + " " + line.strip()).strip()
    if current.get("question"):
        cases.append(current)
    return cases
