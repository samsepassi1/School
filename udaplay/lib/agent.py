"""UdaPlay stateful research agent.

The agent runs a small, explicit state machine:

    START -> RETRIEVE -> EVALUATE -> (REPORT | WEB_SEARCH -> REPORT) -> END

It maintains conversation history across calls (so the user can ask follow-up
questions) and persists web findings into the vector store as long-term
memory, so the second time the same fact is asked the answer comes from the
fast internal path.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from .tools import (
    EvaluationTool,
    GameRecord,
    GameRetrievalTool,
    RetrievalEvaluation,
    WebSearchResult,
    WebSearchTool,
)
from .vector_store import VectorStoreManager


class AgentState(str, Enum):
    START = "start"
    RETRIEVE = "retrieve"
    EVALUATE = "evaluate"
    WEB_SEARCH = "web_search"
    REPORT = "report"
    DONE = "done"


@dataclass
class TraceEvent:
    """One step in the agent's reasoning trace, surfaced in the report."""

    state: AgentState
    detail: str
    payload: dict | None = None

    def as_dict(self) -> dict:
        return {
            "state": self.state.value,
            "detail": self.detail,
            "payload": self.payload or {},
        }


class Citation(BaseModel):
    label: str
    source: str
    snippet: str


class AgentReport(BaseModel):
    """Structured response surfaced to the caller for every query."""

    question: str
    answer: str
    confidence: float
    used_web_search: bool
    citations: list[Citation]
    trace: list[dict]


@dataclass
class AgentMemory:
    """Conversation history kept across calls for follow-up questions."""

    turns: list[dict] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.turns.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def transcript(self, limit: int = 6) -> str:
        if not self.turns:
            return "(no prior turns)"
        recent = self.turns[-limit:]
        return "\n".join(f"{t['role']}: {t['content']}" for t in recent)


class UdaPlayAgent:
    """Stateful agent driving retrieval -> evaluation -> (web) -> report."""

    SYSTEM_PROMPT = (
        "You are UdaPlay, a video-game research assistant. Answer the user's "
        "question using ONLY the provided context (internal knowledge base hits "
        "and optional web search results). Be concise, factual, and cite the "
        "sources you used by their bracket label, e.g. [KB-1] or [WEB-2]. If the "
        "context truly does not answer the question, say so plainly."
    )

    def __init__(
        self,
        store: VectorStoreManager,
        model: str = "gpt-4o-mini",
        top_k: int = 4,
        confidence_floor: float = 0.55,
        persist_web_findings: bool = True,
    ) -> None:
        self.store = store
        self.model = model
        self.confidence_floor = confidence_floor
        self.persist_web_findings = persist_web_findings

        self.retrieval_tool = GameRetrievalTool(store, top_k=top_k)
        self.evaluation_tool = EvaluationTool(model=model)
        self.web_search_tool = WebSearchTool()

        self.memory = AgentMemory()

    # ---- public API ---------------------------------------------------
    def ask(self, question: str) -> AgentReport:
        trace: list[TraceEvent] = []
        state = AgentState.START
        retrieved: list[GameRecord] = []
        evaluation: Optional[RetrievalEvaluation] = None
        web_hits: list[WebSearchResult] = []

        trace.append(TraceEvent(state, f"Received question: {question}"))

        # --- RETRIEVE
        state = AgentState.RETRIEVE
        retrieved = self.retrieval_tool(question)
        trace.append(
            TraceEvent(
                state,
                f"Retrieved {len(retrieved)} hit(s) from internal knowledge base.",
                payload={"hits": [r.model_dump() for r in retrieved[:3]]},
            )
        )

        # --- EVALUATE
        state = AgentState.EVALUATE
        evaluation = self.evaluation_tool(question, retrieved)
        trace.append(
            TraceEvent(
                state,
                f"Judge: sufficient={evaluation.sufficient} "
                f"confidence={evaluation.confidence:.2f}",
                payload=evaluation.model_dump(),
            )
        )

        used_web_search = False
        if not evaluation.sufficient or evaluation.confidence < self.confidence_floor:
            # --- WEB_SEARCH (fallback)
            state = AgentState.WEB_SEARCH
            web_hits = self.web_search_tool(question)
            used_web_search = True
            trace.append(
                TraceEvent(
                    state,
                    f"Internal retrieval insufficient; web search returned "
                    f"{len(web_hits)} result(s).",
                    payload={
                        "results": [w.model_dump() for w in web_hits[:3]],
                    },
                )
            )
            if web_hits and self.persist_web_findings:
                self._remember_web_findings(question, web_hits)

        # --- REPORT
        state = AgentState.REPORT
        answer, citations = self._compose_answer(
            question, retrieved, evaluation, web_hits
        )
        trace.append(
            TraceEvent(state, "Composed final answer with citations.")
        )

        # bookkeeping
        self.memory.add("user", question)
        self.memory.add("assistant", answer)

        report = AgentReport(
            question=question,
            answer=answer,
            confidence=evaluation.confidence if not used_web_search else min(
                0.85, evaluation.confidence + 0.2 if web_hits else evaluation.confidence
            ),
            used_web_search=used_web_search,
            citations=citations,
            trace=[t.as_dict() for t in trace],
        )
        return report

    def reset_memory(self) -> None:
        self.memory = AgentMemory()

    # ---- internals ----------------------------------------------------
    def _remember_web_findings(
        self, question: str, hits: list[WebSearchResult]
    ) -> None:
        """Persist web search results into the vector store as long-term memory."""
        for i, hit in enumerate(hits):
            doc_id = f"web::{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}::{i}"
            text = f"Q: {question}\nA: {hit.title}\n{hit.content}"
            self.store.add_text(
                doc_id,
                text,
                metadata={
                    "kind": "web_memory",
                    "url": hit.url,
                    "title": hit.title,
                    "question": question,
                },
            )

    def _compose_answer(
        self,
        question: str,
        retrieved: list[GameRecord],
        evaluation: RetrievalEvaluation,
        web_hits: list[WebSearchResult],
    ) -> tuple[str, list[Citation]]:
        citations: list[Citation] = []
        kb_block_lines: list[str] = []
        for i, r in enumerate(retrieved, start=1):
            label = f"KB-{i}"
            kb_block_lines.append(f"[{label}] {r.text}")
            citations.append(
                Citation(
                    label=label,
                    source=r.source or f"internal::{r.id}",
                    snippet=r.text[:240],
                )
            )

        web_block_lines: list[str] = []
        for i, w in enumerate(web_hits, start=1):
            label = f"WEB-{i}"
            web_block_lines.append(f"[{label}] {w.title} — {w.content}")
            citations.append(
                Citation(label=label, source=w.url, snippet=w.content[:240])
            )

        # If we have an OpenAI key, ask the LLM to write the answer.
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            answer = self._llm_compose(
                question, kb_block_lines, web_block_lines
            )
        else:
            answer = self._template_compose(
                question, retrieved, evaluation, web_hits
            )

        return answer, citations

    def _llm_compose(
        self,
        question: str,
        kb_lines: list[str],
        web_lines: list[str],
    ) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        kb_block = "\n".join(kb_lines) or "(no internal hits)"
        web_block = "\n".join(web_lines) or "(no web results)"
        history = self.memory.transcript(limit=6)
        try:
            resp = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Conversation so far:\n{history}\n\n"
                            f"Internal knowledge base hits:\n{kb_block}\n\n"
                            f"Web search results:\n{web_block}\n\n"
                            f"Question: {question}\n\n"
                            "Write a concise answer (2-4 sentences). Cite sources "
                            "inline using their [KB-i] or [WEB-i] labels."
                        ),
                    },
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001 — degrade to template
            return self._template_compose(
                question, [], None, []
            ) + f"\n\n(LLM compose failed: {exc})"

    @staticmethod
    def _template_compose(
        question: str,
        retrieved: list[GameRecord],
        evaluation: Optional[RetrievalEvaluation],
        web_hits: list[WebSearchResult],
    ) -> str:
        """Deterministic answer used when no LLM is available."""
        if retrieved and (evaluation is None or evaluation.sufficient):
            top = retrieved[0]
            return (
                f"Based on the internal knowledge base, the closest match for "
                f"\"{question}\" is {top.name} ({top.year}) on {top.platform}, "
                f"developed by {top.developer} and published by {top.publisher}. "
                f"[KB-1]"
            )
        if web_hits:
            top = web_hits[0]
            return (
                f"Internal knowledge was insufficient. Web search suggests: "
                f"{top.title} — {top.content[:200]} [WEB-1]"
            )
        return (
            "I don't have a confident answer for that question. The internal "
            "knowledge base did not contain a strong match and no web results "
            "are available."
        )

    def to_json(self, report: AgentReport) -> str:
        """Serialise a report for logs / dashboards."""
        return json.dumps(report.model_dump(), indent=2, default=str)
