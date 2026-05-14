"""Text moderation agent.

Analyzes a single agent-to-customer message and returns a structured
``ModerationResult`` describing whether it contains PII, sounds unfriendly,
or sounds unprofessional.
"""

from __future__ import annotations

from pydantic_ai import Agent

from schemas.moderation_result import ModerationResult

from ._common import moderation_model_name

TEXT_MODERATION_SYSTEM_PROMPT = """\
You are a content moderation agent for ACME Enterprise customer service.

A trainee customer-service agent is about to send the following message to a
customer. Read it carefully and decide whether it is appropriate.

Flag the message on every dimension that applies:

- contains_pii: set to true if the message contains personally identifiable
  information (full names, addresses, phone numbers, email addresses, SSNs,
  credit card numbers, dates of birth, etc.). Generic greetings and product
  names are NOT PII.
- is_unfriendly: set to true if the tone is rude, sarcastic, dismissive,
  hostile, or otherwise unfriendly toward the customer.
- is_unprofessional: set to true if the message uses slang, profanity,
  personal opinions, careless phrasing, or anything else inappropriate in a
  professional customer-service exchange.

Always populate ``rationale`` with one or two short sentences explaining your
decision (even when nothing was flagged). Be concise and concrete.

If everything looks fine, return all flags as false and explain briefly why
the message is acceptable.
"""


text_agent: Agent[None, ModerationResult] = Agent(
    moderation_model_name(),
    output_type=ModerationResult,
    system_prompt=TEXT_MODERATION_SYSTEM_PROMPT,
    name="text_moderation_agent",
)


async def moderate_text(message: str) -> ModerationResult:
    """Run the text moderation agent on ``message`` and return the result."""
    result = await text_agent.run(message)
    return result.output
