"""Test script for ActionPlanningAgent."""

import os
from dotenv import load_dotenv

from workflow_agents.base_agents import ActionPlanningAgent


KNOWLEDGE = """
How to make scrambled eggs:
1. Crack the eggs into a bowl.
2. Whisk the eggs with a pinch of salt and a splash of milk.
3. Heat butter in a non-stick pan over medium-low heat.
4. Pour the eggs into the pan.
5. Stir gently and continuously with a spatula until the eggs are softly set.
6. Remove from heat just before they look fully done.
7. Serve immediately.
"""


def main() -> None:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    agent = ActionPlanningAgent(openai_api_key, knowledge=KNOWLEDGE)

    prompt = "One morning I wanted to have scrambled eggs"
    steps = agent.extract_steps_from_prompt(prompt)

    print(f"Prompt: {prompt}")
    print("Extracted steps:")
    for i, step in enumerate(steps, start=1):
        print(f"  {i}. {step}")


if __name__ == "__main__":
    main()
