"""Batch evaluation runner.

Loads ``evaluation_dataset.txt`` (or a custom path / a JSON file), runs each
question through the live RAG + LLM stack, and scores every triple with
:class:`RAGASEvaluator`. Prints a per-question summary plus aggregate metrics.

Example:
    python run_evaluation.py --top-k 4
    python run_evaluation.py --eval-set test_questions.json --out report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from llm_client import LLMClient
from rag_client import RAGClient
from ragas_evaluator import RAGASEvaluator, load_evaluation_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch evaluation over the NASA RAG system")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path(__file__).parent / "evaluation_dataset.txt",
        help="Path to evaluation_dataset.txt or a JSON list",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional path to write the JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_arg_parser().parse_args(argv)

    rag = RAGClient()
    if rag.collection_size() == 0:
        print(
            "ERROR: Chroma collection is empty. Run `python embedding_pipeline.py` "
            "before evaluating.",
            file=sys.stderr,
        )
        return 2

    llm = LLMClient(model=args.model) if args.model else LLMClient()
    evaluator = RAGASEvaluator()

    cases = load_evaluation_dataset(args.eval_set)
    print(f"Loaded {len(cases)} evaluation cases from {args.eval_set}\n")

    triples: list[dict] = []
    for i, case in enumerate(cases, start=1):
        question = case["question"]
        mission = case.get("mission") or None
        reference = case.get("reference")
        category = case.get("category", "uncategorized")

        # Fresh LLM history per question keeps the eval comparable across runs.
        llm.reset()

        retrieval = rag.search(question, k=args.top_k, mission=mission)
        answer = llm.generate(question, retrieval.context, record_history=False)

        print(f"[{i}/{len(cases)}] ({category}) {question}")
        print(f"  retrieved {len(retrieval.chunks)} chunks; "
              f"top score={retrieval.chunks[0].score:.3f}" if retrieval.chunks else "  (no chunks)")
        print(f"  answer: {answer[:200]}{'…' if len(answer) > 200 else ''}\n")

        triples.append(
            {
                "question": question,
                "category": category,
                "mission": mission,
                "contexts": retrieval.contexts,
                "answer": answer,
                "reference": reference,
            }
        )

    print("Scoring with RAGAS…")
    report = evaluator.evaluate_batch(triples)

    print("\n=== Per-question metrics ===")
    for case, per in zip(triples, report["per_question"]):
        print(f"- ({case.get('category', '')}) {case['question']}")
        if per["metrics"]:
            for name, value in per["metrics"].items():
                print(f"    {name}: {value:.3f}")
        if per["errors"]:
            print(f"    errors: {per['errors']}")

    print("\n=== Aggregate metrics ===")
    for name, stats in report["aggregate"].items():
        line = f"- {name}: mean={stats['mean']:.3f} min={stats['min']:.3f} max={stats['max']:.3f} n={stats['n']}"
        if "stdev" in stats:
            line += f" stdev={stats['stdev']:.3f}"
        print(line)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
