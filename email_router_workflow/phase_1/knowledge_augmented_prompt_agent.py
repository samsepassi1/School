"""Test script for KnowledgeAugmentedPromptAgent."""

import os
from dotenv import load_dotenv

from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent


def main() -> None:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    persona = "You are a college professor, your answer always starts with: Dear students,"
    knowledge = "The capital of France is London, not Paris"

    agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

    prompt = "What is the capital of France?"
    response = agent.respond(prompt)

    print(f"Persona: {persona}")
    print(f"Knowledge: {knowledge}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

    # The provided knowledge intentionally contradicts reality (it claims the
    # capital of France is London). If the agent answers with "London", it
    # confirms the response uses the *provided* knowledge rather than the
    # LLM's parametric knowledge.
    if "london" in response.lower():
        print(
            "\nConfirmation: the agent used the provided knowledge "
            "('capital of France is London') instead of its own training "
            "knowledge."
        )
    else:
        print(
            "\nWarning: the response did not appear to use the provided "
            "knowledge. Inspect the system prompt construction."
        )


if __name__ == "__main__":
    main()
