"""Test script for DirectPromptAgent."""

import os
from dotenv import load_dotenv

from workflow_agents.base_agents import DirectPromptAgent


def main() -> None:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    direct_agent = DirectPromptAgent(openai_api_key)

    prompt = "What is the Capital of France?"
    response = direct_agent.respond(prompt)

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

    # The DirectPromptAgent does not pass any system prompt, persona, or
    # external knowledge. The answer is therefore drawn from the parametric
    # knowledge of the underlying gpt-3.5-turbo model -- i.e. facts that the
    # LLM internalised during pre-training.
    print(
        "\nKnowledge source: this answer comes from the pre-training data of "
        "the gpt-3.5-turbo model. No persona, system prompt, or retrieved "
        "knowledge is supplied by the agent."
    )


if __name__ == "__main__":
    main()
