"""Test script for AugmentedPromptAgent."""

import os
from dotenv import load_dotenv

from workflow_agents.base_agents import AugmentedPromptAgent


def main() -> None:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    persona = "a helpful college professor who always begins with 'Dear students,'"
    augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

    prompt = "What is the Capital of France?"
    augmented_agent_response = augmented_agent.respond(prompt)

    print(f"Persona: {persona}")
    print(f"Prompt: {prompt}")
    print(f"Response: {augmented_agent_response}")

    # Knowledge source: the answer still comes from the gpt-3.5-turbo
    # pre-training data -- no external knowledge is provided. The persona only
    # changes *how* the answer is presented.
    #
    # Persona effect: by setting a system prompt that asks the model to act as
    # a college professor and to start with "Dear students,", the response
    # adopts that voice. Without the persona, the answer would be a plain,
    # unstyled fact; with the persona, the same fact is wrapped in an
    # instructional, professorial tone.


if __name__ == "__main__":
    main()
