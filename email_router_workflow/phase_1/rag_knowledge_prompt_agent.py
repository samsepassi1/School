"""Test script for RAGKnowledgePromptAgent."""

import os
from dotenv import load_dotenv

from workflow_agents.base_agents import RAGKnowledgePromptAgent


KNOWLEDGE = """
InnovateNext Solutions is a startup that builds productivity software for
small and medium businesses. Their flagship product is the Email Router, a
service that classifies and routes inbound email to the appropriate team
mailbox. The Email Router uses a rules engine plus an LLM-based classifier.
Pricing for the Email Router starts at $29 per seat per month for the
Standard tier and $79 per seat per month for the Enterprise tier, which
includes SSO, audit logs, and a 99.95% uptime SLA. Support is provided by
the customer success team and is available 24/7 for Enterprise customers.
"""


def main() -> None:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    persona = "a customer support specialist for InnovateNext Solutions"
    agent = RAGKnowledgePromptAgent(
        openai_api_key=openai_api_key,
        persona=persona,
        knowledge=KNOWLEDGE,
        chunk_size=200,
        chunk_overlap=40,
        top_k=2,
    )

    prompt = "How much does the Email Router Enterprise tier cost and what does it include?"
    response = agent.respond(prompt)

    print(f"Persona: {persona}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
