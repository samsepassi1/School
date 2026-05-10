"""Agent tools: retrieval, evaluation, and Tavily-backed web search.

Each tool is a small class with a callable interface so it slots into either
a hand-written state machine or a tool-using LLM loop. Pydantic models give
us structured outputs that the agent can reason over without parsing strings.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from .vector_store import VectorStoreManager


# ---------------------------------------------------------------------------
# Structured payloads
# ---------------------------------------------------------------------------


class GameRecord(BaseModel):
    """A single retrieved game, normalised for downstream prompting."""

    id: str
    name: str
    platform: Optional[str] = None
    year: Optional[int | str] = None
    publisher: Optional[str] = None
    developer: Optional[str] = None
    genre: Optional[str] = None
    text: str
    distance: Optional[float] = None
    source: Optional[str] = None


class RetrievalEvaluation(BaseModel):
    """LLM judge verdict on whether retrieved hits answer the question."""

    sufficient: bool = Field(description="True if internal knowledge answers the question.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the verdict, 0-1.")
    reasoning: str = Field(description="One short paragraph justifying the verdict.")
    missing_info: Optional[str] = Field(default=None, description="What's missing if insufficient.")


class WebSearchResult(BaseModel):
    """A single web hit returned by Tavily (or the offline stub)."""

    title: str
    url: str
    content: str
    score: Optional[float] = None


# ---------------------------------------------------------------------------
# Tool 1: retrieve_game
# ---------------------------------------------------------------------------


class GameRetrievalTool:
    """Semantic search over the local ChromaDB collection."""

    name = "retrieve_game"
    description = (
        "Look up games in the UdaPlay internal knowledge base. Input is a natural "
        "language question or game-related phrase. Returns the top matches with "
        "their metadata."
    )

    def __init__(self, store: VectorStoreManager, top_k: int = 4) -> None:
        self.store = store
        self.top_k = top_k

    def __call__(self, query: str, k: Optional[int] = None) -> list[GameRecord]:
        hits = self.store.query(query, k=k or self.top_k)
        records: list[GameRecord] = []
        for hit in hits:
            meta = hit.get("metadata") or {}
            records.append(
                GameRecord(
                    id=hit["id"],
                    name=meta.get("name") or hit["id"],
                    platform=meta.get("platform"),
                    year=meta.get("year"),
                    publisher=meta.get("publisher"),
                    developer=meta.get("developer"),
                    genre=meta.get("genre"),
                    text=hit["text"],
                    distance=hit.get("distance"),
                    source=meta.get("source"),
                )
            )
        return records


# ---------------------------------------------------------------------------
# Tool 2: evaluate_retrieval
# ---------------------------------------------------------------------------


# Heuristic distance threshold for the "no LLM" fallback. Lower distance ==
# closer match in Chroma's L2 space, so values below this are treated as
# good-quality hits when an OpenAI key is unavailable.
HEURISTIC_DISTANCE_THRESHOLD = 1.0


class EvaluationTool:
    """LLM judge that decides whether internal retrieval is good enough.

    Uses an OpenAI chat model when ``OPENAI_API_KEY`` is set; otherwise falls
    back to a deterministic distance-based heuristic so the project still
    runs in offline / classroom-grading environments.
    """

    name = "evaluate_retrieval"
    description = (
        "Judge whether the retrieved games actually answer the user's question. "
        "Returns a structured verdict (sufficient, confidence, reasoning) so the "
        "agent can decide whether to fall back to web search."
    )

    SYSTEM_PROMPT = (
        "You are a strict evaluator for a video-game research agent. Given a user "
        "question and the top retrieved documents from an internal knowledge base, "
        "decide whether the retrieved information is sufficient and on-topic to "
        "answer the question. Be conservative: if a key fact is missing or the "
        "match is only tangentially related, mark sufficient=false. Always reply "
        "with a JSON object matching the requested schema."
    )

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def _heuristic(
        self, question: str, retrieved: list[GameRecord]
    ) -> RetrievalEvaluation:
        if not retrieved:
            return RetrievalEvaluation(
                sufficient=False,
                confidence=0.9,
                reasoning="No documents retrieved from the internal knowledge base.",
                missing_info="any candidate game record",
            )
        best = retrieved[0]
        dist = best.distance if best.distance is not None else 99.0
        if dist <= HEURISTIC_DISTANCE_THRESHOLD:
            return RetrievalEvaluation(
                sufficient=True,
                confidence=max(0.5, 1.0 - dist / 2),
                reasoning=(
                    f"Top hit '{best.name}' has distance {dist:.2f}, below the "
                    f"heuristic threshold of {HEURISTIC_DISTANCE_THRESHOLD}."
                ),
                missing_info=None,
            )
        return RetrievalEvaluation(
            sufficient=False,
            confidence=min(0.9, dist / 2),
            reasoning=(
                f"Best distance ({dist:.2f}) exceeds heuristic threshold "
                f"{HEURISTIC_DISTANCE_THRESHOLD}; retrieval looks weak."
            ),
            missing_info="a closer match for the question",
        )

    def __call__(
        self, question: str, retrieved: list[GameRecord]
    ) -> RetrievalEvaluation:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._heuristic(question, retrieved)

        # Lazy import so the package is only required when actually used.
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        snippets = "\n\n".join(
            f"[{i+1}] {r.text} (distance={r.distance})" for i, r in enumerate(retrieved)
        ) or "<no documents retrieved>"

        try:
            resp = client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Retrieved documents:\n{snippets}\n\n"
                            "Reply with a JSON object with keys: sufficient (bool), "
                            "confidence (float 0-1), reasoning (string), "
                            "missing_info (string or null)."
                        ),
                    },
                ],
            )
            payload = json.loads(resp.choices[0].message.content)
            return RetrievalEvaluation(**payload)
        except Exception as exc:  # noqa: BLE001 — keep agent alive on judge failure
            heur = self._heuristic(question, retrieved)
            heur.reasoning += f" (LLM judge failed: {exc})"
            return heur


# ---------------------------------------------------------------------------
# Tool 3: game_web_search
# ---------------------------------------------------------------------------


@dataclass
class WebSearchTool:
    """Tavily-backed web search with a small offline stub for grading.

    When ``TAVILY_API_KEY`` is set we hit the real API; otherwise we return an
    empty result list so the agent can still produce a graceful "I don't know,
    here is what I have" report.
    """

    name: str = "game_web_search"
    description: str = (
        "Search the public web for video game information. Use this when the "
        "internal knowledge base does not have the answer."
    )
    max_results: int = 5

    def __call__(self, query: str) -> list[WebSearchResult]:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return []
        try:
            # Tavily ships its own Python client; import lazily to keep the
            # base requirements thin.
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=query,
                search_depth="advanced",
                include_answer=False,
                max_results=self.max_results,
            )
            results = response.get("results", []) if isinstance(response, dict) else []
            return [
                WebSearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    score=r.get("score"),
                )
                for r in results
            ]
        except Exception:
            return []
