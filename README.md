# UDA-Hub — Universal Decision Agent

A LangGraph-powered multi-agent system that ingests customer support tickets,
classifies them, retrieves relevant knowledge, takes action via support tools
when appropriate, and either resolves the ticket autonomously or escalates it
to a human — with persistent short- and long-term memory.

> Course project for the Udacity "AI Agents" / agentic systems track.

## Architecture at a glance

```
START -> hydrate -> classifier -> retriever -> [supervisor router]
                                                   |        |
                                              resolver   escalation
                                                   \        /
                                                memory_writer -> END
```

Six specialised nodes share a typed `AgentState` and follow a **Supervisor**
pattern. See [`docs/architecture.md`](docs/architecture.md) for the full
design, diagram, and rubric mapping.

## Project layout

```
uda_hub/                 core package (db, retrieval, tools, memory, agents, graph)
notebooks/
  01_database_setup.ipynb       initialise the DB + load the knowledge base
  02_end_to_end_demo.ipynb      run the full workflow on 4 sample scenarios
docs/architecture.md     architecture design document
data/                    SQLite databases + FAISS index (gitignored)
scripts/build_notebooks.py
```

## Prerequisites

- Python 3.11+
- An OpenAI API key (for embeddings + the chat model)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # then edit .env to set OPENAI_API_KEY
```

## Run

1. **Initialise the database and knowledge base**:
   ```bash
   jupyter notebook notebooks/01_database_setup.ipynb
   ```
2. **Run the end-to-end demo**:
   ```bash
   jupyter notebook notebooks/02_end_to_end_demo.ipynb
   ```

Or programmatically:

```python
from uda_hub import seed
from uda_hub.retrieval import build_or_load_vectorstore
from uda_hub.graph import build_app
from uda_hub.runner import run_ticket

seed.seed_all(reset=True)
build_or_load_vectorstore(rebuild=True)
app = build_app()

result = run_ticket(app, {
    "ticket_id": "tkt_demo_001",
    "user_id":   "usr_002",
    "subject":   "Charged twice this month",
    "body":      "Please refund the duplicate $19.99 charge.",
    "channel":   "email",
    "urgency":   "high",
})
print(result["answer"])
print("\n".join(result["log"]))
```

## Rubric mapping

| Rubric area                                  | Where it lives                                                  |
|---------------------------------------------|-----------------------------------------------------------------|
| DB schema (Account/User/Ticket/.../Knowledge) | `uda_hub/db.py`                                                 |
| Database management notebook                 | `notebooks/01_database_setup.ipynb`                             |
| ≥14 KB articles across categories             | `uda_hub/seed.py` (16 articles)                                  |
| Architecture design doc + diagram             | `docs/architecture.md`                                          |
| ≥4 specialised agents                         | `uda_hub/agents/{classifier,retriever,resolver,escalation,supervisor}.py` |
| LangGraph orchestration                       | `uda_hub/graph.py`                                              |
| Routing based on classification + metadata    | `uda_hub/agents/supervisor.py::supervisor_router`               |
| KB retrieval + confidence                     | `uda_hub/retrieval.py`                                          |
| Escalation when no relevant article           | `uda_hub/agents/escalation.py`                                  |
| ≥2 support-operation tools                    | `uda_hub/tools.py` (4 tools)                                    |
| Persistent customer history                   | `uda_hub/memory.py::SqliteLongTermStore` + `db.TicketMessage`   |
| Short-term (per-thread) memory                | `uda_hub/memory.py::get_checkpointer` (SqliteSaver)             |
| Long-term memory                              | `uda_hub/memory.py::SqliteLongTermStore`                        |
| End-to-end demo + logging                     | `notebooks/02_end_to_end_demo.ipynb` + `uda_hub/logging_utils.py` |

## Tests

```bash
python -m pytest tests/ -q
```

The smoke test verifies imports, schema creation, seeding, and routing logic
without making LLM calls.
