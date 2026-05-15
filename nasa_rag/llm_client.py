"""OpenAI chat client for the NASA RAG assistant.

Wraps a chat-completions call with:
  * a NASA-mission-expert system prompt that requires citing context,
  * conversation history threaded through each call, and
  * a strict "answer from context only" instruction.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


SYSTEM_PROMPT = """You are NASA Mission Intelligence, an expert assistant on
historic NASA crewed spaceflight. You specialize in Apollo 11, Apollo 13, and
the Space Shuttle Challenger (STS-51-L). Your audience is astronauts,
researchers, engineers, and historians who need accurate, source-grounded
answers.

Rules you MUST follow:
1. Ground every factual claim in the retrieved CONTEXT block provided with
   each user question. Do not invent dates, names, callouts, or technical
   details that are not supported by that context.
2. When you use a fact from the context, cite the source bracket number, e.g.
   "(see [1])". If multiple sources support a claim, cite all of them.
3. If the context is empty, contradictory, or insufficient to answer, say so
   plainly. Do not fall back to general knowledge to fill gaps - instead,
   acknowledge the limitation and suggest what additional source material
   would help.
4. Be precise and concise. Prefer structured answers (short paragraphs or
   bullets) for technical or timeline questions.
5. Maintain a professional, mission-operations tone. No speculation or
   dramatization.
"""


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class LLMClient:
    """Stateful chat client. Persists conversation history across turns."""

    model: str | None = None
    temperature: float = 0.2
    max_history_turns: int = 8  # user+assistant pairs retained per call
    system_prompt: str = SYSTEM_PROMPT
    client: Optional[OpenAI] = None
    history: list[ChatMessage] = field(default_factory=list)

    def __post_init__(self) -> None:
        load_dotenv()
        if self.model is None:
            self.model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        if self.client is None:
            self.client = OpenAI()

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.history.clear()

    def _trimmed_history(self) -> list[ChatMessage]:
        # Keep the most recent N turns. A "turn" is one user + one assistant
        # message, so we keep up to 2 * max_history_turns history messages.
        limit = self.max_history_turns * 2
        return self.history[-limit:] if limit > 0 else []

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _build_user_message(self, question: str, context: str) -> str:
        return (
            "Use the CONTEXT below to answer the question. If the context does"
            " not contain the answer, say so explicitly.\n\n"
            f"=== CONTEXT ===\n{context}\n=== END CONTEXT ===\n\n"
            f"Question: {question}"
        )

    def generate(
        self,
        question: str,
        context: str,
        *,
        record_history: bool = True,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send the question + retrieved context to the LLM.

        ``record_history=True`` appends the user question (plain, no context)
        and the assistant answer to ``self.history`` so the next call can use
        them as conversational memory.
        """
        if self.client is None:  # pragma: no cover - set in __post_init__
            self.client = OpenAI()

        used_model = model or self.model
        used_temperature = self.temperature if temperature is None else temperature

        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        for m in self._trimmed_history():
            messages.append({"role": m.role, "content": m.content})
        messages.append(
            {"role": "user", "content": self._build_user_message(question, context)}
        )

        completion = self.client.chat.completions.create(
            model=used_model,
            messages=messages,
            temperature=used_temperature,
        )
        answer = (completion.choices[0].message.content or "").strip()

        if record_history:
            # Store the *clean* user question (without the verbose context
            # block) so history stays small and reusable across turns.
            self.history.append(ChatMessage(role="user", content=question))
            self.history.append(ChatMessage(role="assistant", content=answer))

        return answer
