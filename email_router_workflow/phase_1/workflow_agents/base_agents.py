"""Reusable agent classes for building agentic workflows."""

from __future__ import annotations

import os
import re
from typing import Callable, Dict, List, Optional

import numpy as np
from openai import OpenAI


# ---------------------------------------------------------------------------
# 1. Direct Prompt Agent
# ---------------------------------------------------------------------------
class DirectPromptAgent:
    """Sends a prompt directly to the LLM with no system message or context."""

    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key

    def respond(self, prompt: str) -> str:
        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 2. Augmented Prompt Agent
# ---------------------------------------------------------------------------
class AugmentedPromptAgent:
    """Adopts a defined persona via a system prompt before answering."""

    def __init__(self, openai_api_key: str, persona: str):
        self.openai_api_key = openai_api_key
        self.persona = persona

    def respond(self, prompt: str) -> str:
        client = OpenAI(api_key=self.openai_api_key)
        system_message = (
            f"You are {self.persona}. Forget all previous context."
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 3. Knowledge Augmented Prompt Agent
# ---------------------------------------------------------------------------
class KnowledgeAugmentedPromptAgent:
    """Answers using a provided persona and a fixed body of knowledge."""

    def __init__(self, openai_api_key: str, persona: str, knowledge: str):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge

    def respond(self, prompt: str) -> str:
        client = OpenAI(api_key=self.openai_api_key)
        system_message = (
            f"You are {self.persona} knowledge-based assistant. "
            f"Forget all previous context.\n"
            f"Use only the following knowledge to answer, do not use your own "
            f"knowledge: {self.knowledge}\n"
            f"Answer the prompt based on this knowledge, not your own."
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 4. RAG Knowledge Prompt Agent (provided)
# ---------------------------------------------------------------------------
class RAGKnowledgePromptAgent:
    """Retrieval-augmented agent: chunks a knowledge corpus, embeds the chunks,
    and at query time fetches the most similar chunks to supply as context."""

    def __init__(
        self,
        openai_api_key: str,
        persona: str,
        knowledge: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
    ):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.client = OpenAI(api_key=openai_api_key)
        self.chunks = self._chunk_text(knowledge)
        self.chunk_embeddings = [self._embed(c) for c in self.chunks] if self.chunks else []

    def _chunk_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        chunks: List[str] = []
        start = 0
        step = max(1, self.chunk_size - self.chunk_overlap)
        while start < len(text):
            chunks.append(text[start : start + self.chunk_size])
            start += step
        return chunks

    def _embed(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
        )
        return np.array(response.data[0].embedding)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _retrieve(self, query: str) -> str:
        if not self.chunk_embeddings:
            return ""
        q_emb = self._embed(query)
        scored = [
            (self._cosine(q_emb, emb), chunk)
            for emb, chunk in zip(self.chunk_embeddings, self.chunks)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [chunk for _, chunk in scored[: self.top_k]]
        return "\n---\n".join(top)

    def respond(self, prompt: str) -> str:
        context = self._retrieve(prompt)
        system_message = (
            f"You are {self.persona}. Forget all previous context.\n"
            f"Use only the following retrieved knowledge to answer:\n{context}\n"
            f"If the answer is not in the retrieved knowledge, say so."
        )
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 5. Evaluation Agent
# ---------------------------------------------------------------------------
class EvaluationAgent:
    """Evaluates a worker agent's response against criteria, iterating with
    correction instructions until the response passes or max_interactions hit."""

    def __init__(
        self,
        openai_api_key: str,
        persona: str,
        evaluation_criteria: str,
        agent_to_evaluate,
        max_interactions: int = 10,
    ):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.agent_to_evaluate = agent_to_evaluate
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt: str) -> Dict:
        client = OpenAI(api_key=self.openai_api_key)
        prompt_for_worker = initial_prompt
        worker_response = ""
        evaluation = ""
        iterations = 0

        for i in range(self.max_interactions):
            iterations = i + 1

            # Step 1: get a response from the worker agent.
            worker_response = self.agent_to_evaluate.respond(prompt_for_worker)

            # Step 2: ask the LLM to judge the response against criteria.
            eval_messages = [
                {"role": "system", "content": f"You are {self.persona}."},
                {
                    "role": "user",
                    "content": (
                        "Evaluate the following answer against these criteria.\n"
                        f"Criteria: {self.evaluation_criteria}\n\n"
                        f"Answer: {worker_response}\n\n"
                        "Reply with 'Yes' on the first line if the answer fully "
                        "meets the criteria, otherwise reply with 'No' on the "
                        "first line followed by an explanation."
                    ),
                },
            ]
            evaluation = (
                client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=eval_messages,
                    temperature=0,
                )
                .choices[0]
                .message.content
            )

            if evaluation.strip().lower().startswith("yes"):
                break

            # Step 3: ask the LLM for instructions to correct the response.
            instruction_messages = [
                {"role": "system", "content": f"You are {self.persona}."},
                {
                    "role": "user",
                    "content": (
                        "An answer failed the evaluation. Provide concise, "
                        "actionable instructions the worker agent should "
                        "follow to fix it.\n"
                        f"Criteria: {self.evaluation_criteria}\n"
                        f"Answer: {worker_response}\n"
                        f"Evaluation feedback: {evaluation}"
                    ),
                },
            ]
            instructions = (
                client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=instruction_messages,
                    temperature=0,
                )
                .choices[0]
                .message.content
            )

            # Step 4: re-prompt the worker with the original prompt + feedback.
            prompt_for_worker = (
                f"Original prompt: {initial_prompt}\n"
                f"Your previous answer: {worker_response}\n"
                f"Feedback for correction: {instructions}\n"
                "Please produce an improved answer that satisfies the criteria."
            )

        return {
            "final_response": worker_response,
            "evaluation": evaluation,
            "iterations": iterations,
        }


# ---------------------------------------------------------------------------
# 6. Routing Agent
# ---------------------------------------------------------------------------
class RoutingAgent:
    """Selects the most semantically relevant downstream agent based on the
    cosine similarity of the prompt embedding to each agent's description."""

    def __init__(self, openai_api_key: str, agents: Optional[List[Dict]] = None):
        self.openai_api_key = openai_api_key
        self.agents: List[Dict] = agents or []

    def get_embedding(self, text: str) -> np.ndarray:
        client = OpenAI(api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
        )
        return np.array(response.data[0].embedding)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def route(self, user_input: str):
        if not self.agents:
            raise ValueError("RoutingAgent has no agents configured.")
        prompt_embedding = self.get_embedding(user_input)
        best_agent: Optional[Dict] = None
        best_score = -1.0
        for agent in self.agents:
            description = agent.get("description", agent.get("name", ""))
            score = self._cosine(prompt_embedding, self.get_embedding(description))
            if score > best_score:
                best_score = score
                best_agent = agent
        assert best_agent is not None
        print(
            f"[RoutingAgent] selected '{best_agent.get('name', '?')}' "
            f"(similarity={best_score:.3f})"
        )
        return best_agent["func"](user_input)


# ---------------------------------------------------------------------------
# 7. Action Planning Agent
# ---------------------------------------------------------------------------
class ActionPlanningAgent:
    """Extracts an ordered list of actionable steps for a task, using the
    domain knowledge provided at construction time."""

    def __init__(self, openai_api_key: str, knowledge: str):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge
        self.client = OpenAI(api_key=openai_api_key)

    def extract_steps_from_prompt(self, prompt: str) -> List[str]:
        system_message = (
            "You are an Action Planning Agent. Using the knowledge below, "
            "extract the ordered, actionable steps required to fulfill the "
            "user's request. Output one step per line and nothing else.\n"
            f"Knowledge:\n{self.knowledge}"
        )
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content or ""
        steps: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip leading list markers like "1.", "-", "*", "Step 1:".
            line = re.sub(r"^(?:step\s*\d+[:.\)]?|\d+[.\)]|[-*•])\s*", "", line, flags=re.IGNORECASE)
            if line:
                steps.append(line)
        return steps


__all__ = [
    "DirectPromptAgent",
    "AugmentedPromptAgent",
    "KnowledgeAugmentedPromptAgent",
    "RAGKnowledgePromptAgent",
    "EvaluationAgent",
    "RoutingAgent",
    "ActionPlanningAgent",
]
