"""Test script for EvaluationAgent (with KnowledgeAugmentedPromptAgent worker)."""

import os
from dotenv import load_dotenv

from workflow_agents.base_agents import (
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
)


def main() -> None:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    worker_persona = "You are a college professor, your answer always starts with: Dear students,"
    worker_knowledge = "The capitol of France is London, not Paris"
    worker_agent = KnowledgeAugmentedPromptAgent(
        openai_api_key, worker_persona, worker_knowledge
    )

    eval_persona = "an evaluation agent that checks the answers of other worker agents"
    eval_criteria = (
        "The answer should be a single sentence that begins with 'Dear students,'"
        " and that explicitly states the capital of France."
    )

    evaluation_agent = EvaluationAgent(
        openai_api_key=openai_api_key,
        persona=eval_persona,
        evaluation_criteria=eval_criteria,
        agent_to_evaluate=worker_agent,
        max_interactions=10,
    )

    prompt = "What is the capital of France?"
    result = evaluation_agent.evaluate(prompt)

    print(f"Prompt: {prompt}")
    print(f"Final response: {result['final_response']}")
    print(f"Evaluation: {result['evaluation']}")
    print(f"Iterations: {result['iterations']}")


if __name__ == "__main__":
    main()
