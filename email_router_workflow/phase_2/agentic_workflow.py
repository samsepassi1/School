"""End-to-end agentic workflow for project management.

Pilot: turn the Product-Spec-Email-Router.txt specification into a
comprehensive project plan -- user stories, product features, and
engineering tasks -- by orchestrating four agent classes from
``workflow_agents.base_agents``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List

from dotenv import load_dotenv

# TODO 1: Import the agents from the workflow_agents library.
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)


# ---------------------------------------------------------------------------
# TODO 2: Load the OpenAI API key from the environment.
# ---------------------------------------------------------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it to your environment or a .env file."
    )


# ---------------------------------------------------------------------------
# TODO 3: Load the Product-Spec-Email-Router.txt document.
# ---------------------------------------------------------------------------
SPEC_PATH = Path(__file__).parent / "Product-Spec-Email-Router.txt"
product_spec = SPEC_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# TODO 4: Instantiate the Action Planning Agent.
# ---------------------------------------------------------------------------
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a persona, an "
    "action, and a desired outcome for each story. Each story represents a "
    "specific functionality of the product described in the specification.\n"
    "Features are defined by grouping related user stories.\n"
    "Tasks are defined for each story and represent the engineering work "
    "required to develop the product.\n"
    "A development plan for a product contains all stories, features, and "
    "tasks required to build the product.\n"
    "When the user asks for a full project plan, the steps are: "
    "1) define the user stories, "
    "2) group the user stories into product features, "
    "3) break each story down into engineering tasks."
)

action_planning_agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge_action_planning,
)


# ---------------------------------------------------------------------------
# TODO 5 + 6: Product Manager knowledge agent.
# ---------------------------------------------------------------------------
persona_product_manager = (
    "You are a Product Manager. You define user stories from a product "
    "specification."
)
knowledge_product_manager = (
    "User stories are written as: 'As a [type of user], I want [an action or "
    "feature] so that [benefit/value].'\n"
    "Cover every distinct persona that appears in the product spec. Each "
    "persona should have multiple stories that map to the functional "
    "requirements relevant to them. Do not invent functionality that is not "
    "in the spec.\n\n"
    "Product specification:\n"
    f"{product_spec}"
)

product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager,
)


# ---------------------------------------------------------------------------
# TODO 7: Product Manager evaluation agent.
# ---------------------------------------------------------------------------
persona_product_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents"
)
evaluation_criteria_product_manager = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that "
    "[benefit/value]."
)

product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_product_manager,
    agent_to_evaluate=product_manager_knowledge_agent,
    max_interactions=5,
)


# ---------------------------------------------------------------------------
# Program Manager knowledge agent (before TODO 8).
# ---------------------------------------------------------------------------
persona_program_manager = (
    "You are a Program Manager. You group user stories into product features."
)
knowledge_program_manager = (
    "Features of a product are defined by organising similar user stories "
    "into cohesive groups. For each feature provide:\n"
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)

program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager,
)


# ---------------------------------------------------------------------------
# TODO 8: Program Manager evaluation agent.
# ---------------------------------------------------------------------------
persona_program_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents"
)
evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)

program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=evaluation_criteria_program_manager,
    agent_to_evaluate=program_manager_knowledge_agent,
    max_interactions=5,
)


# ---------------------------------------------------------------------------
# Development Engineer knowledge agent (before TODO 9).
# ---------------------------------------------------------------------------
persona_dev_engineer = (
    "You are a Development Engineer. You break user stories into concrete "
    "engineering tasks."
)
knowledge_dev_engineer = (
    "Each engineering task must be expressed using exactly these fields:\n"
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)

development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer,
)


# ---------------------------------------------------------------------------
# TODO 9: Development Engineer evaluation agent.
# ---------------------------------------------------------------------------
persona_dev_engineer_eval = (
    "You are an evaluation agent that checks the answers of other worker agents"
)
evaluation_criteria_dev_engineer = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier  for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)

development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=evaluation_criteria_dev_engineer,
    agent_to_evaluate=development_engineer_knowledge_agent,
    max_interactions=5,
)


# ---------------------------------------------------------------------------
# TODO 11: Support functions invoked by the routing agent.
# ---------------------------------------------------------------------------
def product_manager_support_function(query: str) -> str:
    """Generate user stories and pass them through the PM evaluation loop."""
    initial_response = product_manager_knowledge_agent.respond(query)
    evaluation = product_manager_evaluation_agent.evaluate(query)
    return evaluation.get("final_response", initial_response)


def program_manager_support_function(query: str) -> str:
    """Generate product features and pass them through the PgM evaluation loop."""
    initial_response = program_manager_knowledge_agent.respond(query)
    evaluation = program_manager_evaluation_agent.evaluate(query)
    return evaluation.get("final_response", initial_response)


def development_engineer_support_function(query: str) -> str:
    """Generate engineering tasks and pass them through the Dev evaluation loop."""
    initial_response = development_engineer_knowledge_agent.respond(query)
    evaluation = development_engineer_evaluation_agent.evaluate(query)
    return evaluation.get("final_response", initial_response)


# ---------------------------------------------------------------------------
# TODO 10: Routing Agent with one route per role.
# ---------------------------------------------------------------------------
routing_agent = RoutingAgent(openai_api_key=openai_api_key)
routing_agent.agents = [
    {
        "name": "Product Manager",
        "description": (
            "Responsible for defining product personas and user stories only. "
            "Does not define features or tasks. Does not group stories. "
            "Choose this route when the step asks for user stories in the "
            "form 'As a [user], I want [action] so that [benefit]'."
        ),
        "func": lambda x: product_manager_support_function(x),
    },
    {
        "name": "Program Manager",
        "description": (
            "Responsible for grouping user stories into product features. "
            "Does not write user stories or engineering tasks. "
            "Choose this route when the step asks for features described "
            "with Feature Name, Description, Key Functionality, and User "
            "Benefit."
        ),
        "func": lambda x: program_manager_support_function(x),
    },
    {
        "name": "Development Engineer",
        "description": (
            "Responsible for breaking user stories into detailed engineering "
            "tasks. Does not write user stories or features. "
            "Choose this route when the step asks for tasks with Task ID, "
            "Task Title, Related User Story, Description, Acceptance "
            "Criteria, Estimated Effort, and Dependencies."
        ),
        "func": lambda x: development_engineer_support_function(x),
    },
]


# ---------------------------------------------------------------------------
# TODO 12: Run the workflow.
# ---------------------------------------------------------------------------
workflow_prompt = (
    "I am the Technical Project Manager for the Email Router project. "
    "Using the product specification, produce a complete project plan that "
    "contains: (1) the user stories, (2) the product features, and (3) the "
    "engineering tasks needed to deliver the Email Router."
)


def run_workflow() -> List[str]:
    print("=" * 72)
    print("Agentic Project Management Workflow -- Email Router pilot")
    print("=" * 72)
    print(f"\nWorkflow prompt:\n{workflow_prompt}\n")

    workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
    print(f"Action plan ({len(workflow_steps)} steps):")
    for i, step in enumerate(workflow_steps, start=1):
        print(f"  {i}. {step}")

    completed_steps: List[str] = []
    for i, step in enumerate(workflow_steps, start=1):
        print("\n" + "-" * 72)
        print(f"Step {i}/{len(workflow_steps)}: {step}")
        print("-" * 72)
        try:
            result = routing_agent.route(step)
        except Exception as exc:  # surface routing/LLM errors but keep going
            result = f"[ERROR while processing step: {exc}]"
            print(result)
        completed_steps.append(result)
        print(f"\nResult of step {i}:\n{result}")

    print("\n" + "=" * 72)
    print("Final workflow output (last completed step):")
    print("=" * 72)
    if completed_steps:
        print(completed_steps[-1])
    else:
        print("(no steps were produced by the action planning agent)")

    return completed_steps


if __name__ == "__main__":
    run_workflow()
