"""Test script for RoutingAgent."""

import os
from dotenv import load_dotenv

from workflow_agents.base_agents import (
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)


def main() -> None:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    texas_agent = KnowledgeAugmentedPromptAgent(
        openai_api_key,
        persona="a Texas history expert",
        knowledge=(
            "Rome, Texas is an unincorporated community in Fannin County in "
            "northeastern Texas. It was settled in the mid-1800s and was named "
            "after Rome, Italy by early settlers."
        ),
    )

    europe_agent = KnowledgeAugmentedPromptAgent(
        openai_api_key,
        persona="a European history expert",
        knowledge=(
            "Rome, Italy is the capital city of Italy. Founded traditionally "
            "in 753 BC, it became the heart of the Roman Republic and Roman "
            "Empire and is one of the oldest continuously inhabited cities in "
            "Europe."
        ),
    )

    math_agent = KnowledgeAugmentedPromptAgent(
        openai_api_key,
        persona="a software-engineering math tutor",
        knowledge=(
            "When estimating Agile work, total effort in days equals the "
            "number of stories multiplied by the days per story."
        ),
    )

    routing_agent = RoutingAgent(openai_api_key)
    routing_agent.agents = [
        {
            "name": "Texas history agent",
            "description": "Answers questions about places, events, and history in the U.S. state of Texas, including small towns named after European cities.",
            "func": lambda x: texas_agent.respond(x),
        },
        {
            "name": "European history agent",
            "description": "Answers questions about the history of European countries and cities, including Rome, Italy and the Roman Empire.",
            "func": lambda x: europe_agent.respond(x),
        },
        {
            "name": "Math agent",
            "description": "Answers basic math word problems, including Agile estimation and arithmetic with stories, days, and effort.",
            "func": lambda x: math_agent.respond(x),
        },
    ]

    prompts = [
        "Tell me about the history of Rome, Texas",
        "Tell me about the history of Rome, Italy",
        "One story takes 2 days, and there are 20 stories",
    ]

    for prompt in prompts:
        print(f"\n=== Prompt: {prompt} ===")
        response = routing_agent.route(prompt)
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
