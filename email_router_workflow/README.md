# AI-Powered Agentic Workflow for Project Management

Pilot client: **InnovateNext Solutions**. Pilot product: the **Email Router**.

The project is delivered in two phases:

```
email_router_workflow/
├── phase_1/                     reusable agent library + 7 test scripts
│   ├── workflow_agents/
│   │   ├── __init__.py
│   │   └── base_agents.py
│   ├── direct_prompt_agent.py
│   ├── augmented_prompt_agent.py
│   ├── knowledge_augmented_prompt_agent.py
│   ├── rag_knowledge_prompt_agent.py
│   ├── evaluation_agent.py
│   ├── routing_agent.py
│   └── action_planning_agent.py
└── phase_2/                     end-to-end workflow on the Email Router spec
    ├── workflow_agents/         (copy of the Phase 1 library)
    ├── Product-Spec-Email-Router.txt
    └── agentic_workflow.py
```

## Setup

```bash
cd email_router_workflow
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # then edit .env and paste your key
```

The `.env` file must contain:

```
OPENAI_API_KEY=sk-...
```

## Capture all execution evidence in one command

Rubric items 5 and 12 require captured outputs from every Phase 1 test and the
Phase 2 workflow run. The bundled runner does this in one shot:

```bash
python capture_outputs.py
```

It runs each Phase 1 test script and `phase_2/agentic_workflow.py` and writes
combined stdout+stderr to:

- `phase_1/outputs/<agent_script>.txt` (one per agent)
- `phase_2/outputs/agentic_workflow.txt`

Commit those files alongside the code so the reviewer has the evidence inline.

## Phase 1 – run the agent tests manually (optional)

If you'd rather run tests one at a time, each test script can be run from
inside `phase_1/`:

```bash
cd phase_1
python direct_prompt_agent.py
python augmented_prompt_agent.py
python knowledge_augmented_prompt_agent.py
python rag_knowledge_prompt_agent.py
python evaluation_agent.py
python routing_agent.py
python action_planning_agent.py
```

Capture the terminal output (text or screenshot) for each script. The seven
agents are:

| # | Agent | Purpose |
|---|-------|---------|
| 1 | `DirectPromptAgent` | Pass the user prompt straight to the LLM. |
| 2 | `AugmentedPromptAgent` | Add a system-prompt persona. |
| 3 | `KnowledgeAugmentedPromptAgent` | Persona + fixed knowledge. |
| 4 | `RAGKnowledgePromptAgent` | Persona + retrieved knowledge (chunk → embed → top-k). |
| 5 | `EvaluationAgent` | Iterative critique of a worker agent's response. |
| 6 | `RoutingAgent` | Cosine-similarity routing across specialist agents. |
| 7 | `ActionPlanningAgent` | Extract ordered steps from a prompt + knowledge. |

## Phase 2 – run the agentic workflow

```bash
cd phase_2
python agentic_workflow.py
```

The script:

1. Reads `Product-Spec-Email-Router.txt`.
2. Uses the **Action Planning Agent** to break the high-level TPM prompt
   into ordered steps.
3. Uses the **Routing Agent** to dispatch each step to the matching team:
   - **Product Manager** team — `KnowledgeAugmentedPromptAgent` +
     `EvaluationAgent` producing user stories in the form
     *"As a [user], I want [action] so that [benefit]."*
   - **Program Manager** team — produces features with
     `Feature Name / Description / Key Functionality / User Benefit`.
   - **Development Engineer** team — produces tasks with
     `Task ID / Task Title / Related User Story / Description /
     Acceptance Criteria / Estimated Effort / Dependencies`.
4. Prints each step's evaluated, validated output and the final plan.

## Submission checklist

- [x] `phase_1/workflow_agents/base_agents.py`
- [x] Seven Phase 1 test scripts under `phase_1/`
- [ ] Captured outputs (screenshots or text) for each Phase 1 test
      → `python capture_outputs.py` writes them to `phase_1/outputs/`
- [x] `phase_2/agentic_workflow.py`
- [ ] Captured output of the Phase 2 workflow run
      → `python capture_outputs.py` writes it to `phase_2/outputs/`

## Notes

- The library is duplicated under `phase_2/workflow_agents/` so each phase can
  be run independently. If you change one copy, mirror the change to the
  other.
- The evaluation loops in Phase 2 are capped at 5 iterations per step to keep
  API usage bounded; raise this in `agentic_workflow.py` if you want more
  aggressive correction.
