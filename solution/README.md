# UDA-Hub — Universal Decision Agent (solution)

A LangGraph multi-agent system that ingests Cult Pass support tickets,
classifies them, retrieves grounded knowledge, takes action through tools
against the customer (Cult Pass) and core (UDA-Hub) databases, and either
resolves the ticket autonomously or escalates it — with persistent
short- and long-term memory.

## Layout

```
solution/
├── agentic/
│   ├── agents/             classifier, retriever, resolver, escalation, supervisor, state
│   ├── tools/              uda_account, cultpass, mcp_server (optional)
│   ├── design/             architecture.md, data_model.md, rag.md
│   ├── workflow.py         LangGraph wiring (built from scratch, no prebuilt agent)
│   ├── runner.py           run_ticket helper
│   ├── retrieval.py        FAISS + keyword fallback
│   ├── memory.py           SqliteSaver checkpointer + long-term store
│   ├── llm.py              ChatOpenAI factory
│   └── logging_utils.py
├── data/
│   ├── core/               db.py, seed.py — Account, User, Ticket*, Knowledge
│   ├── external/           db.py, seed.py, cultpass_articles.jsonl (16 articles)
│   └── models/             FAISS index (gitignored)
├── tests/                  pytest smoke suite
├── 01_external_db_setup.ipynb
├── 02_core_db_setup.ipynb
├── 03_agentic_app.ipynb
├── 03_agentic_app.py       runnable equivalent
├── utils.py                chat_interface() and display helpers
├── build_notebooks.py      regenerate the .ipynb files
├── requirements.txt
└── .env.example
```

## Setup

Python 3.11+ is required.

```bash
cd solution
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env to set OPENAI_API_KEY
```

### Run order (notebooks)

1. `01_external_db_setup.ipynb` — builds `data/external/cultpass.db` (CultPass members, events, bookings, payments).
2. `02_core_db_setup.ipynb`     — builds `data/core/uda_hub.db` and loads `data/external/cultpass_articles.jsonl` into the `Knowledge` table; builds the FAISS index when an OpenAI key is present.
3. `03_agentic_app.ipynb`       — runs the full LangGraph workflow on four sample tickets and inspects memory + persisted state. The optional final cell launches `chat_interface()` from `utils.py`.

### Run order (script)

```bash
python 03_agentic_app.py            # seeds DBs + runs all 4 demo scenarios
python 03_agentic_app.py --chat     # also drops into the interactive REPL
```

### Tests

```bash
python -m pytest tests/ -q
```

The 11-case smoke suite covers:

- External and core DB schemas + seeding
- ≥14 KB articles loaded from `cultpass_articles.jsonl`
- Keyword retrieval (FAISS-free)
- All seven tools — happy and error paths
- Long-term store roundtrip
- Supervisor routing rules
- Graph compilation

## What's implemented (rubric mapping)

| Rubric area                                                          | Where it lives                                                                 |
|---------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Database/KB infrastructure (Account/User/Ticket*/Knowledge)         | `data/core/db.py`, `data/core/seed.py`                                         |
| External CultPass database                                           | `data/external/db.py`, `data/external/seed.py`, `data/external/cultpass.db`    |
| `cultpass_articles.jsonl` expanded to 16 articles                   | `data/external/cultpass_articles.jsonl`                                        |
| Database management notebooks                                        | `01_external_db_setup.ipynb`, `02_core_db_setup.ipynb`                        |
| Architecture design + diagram                                        | `agentic/design/architecture.md` (Mermaid + ASCII)                             |
| Data model design                                                    | `agentic/design/data_model.md`                                                 |
| RAG documentation                                                    | `agentic/design/rag.md`                                                        |
| ≥4 specialised agents                                                | `agentic/agents/{classifier,retriever,resolver,escalation,supervisor}.py`     |
| LangGraph workflow built from scratch                                | `agentic/workflow.py`                                                          |
| Routing on classification + retrieval confidence                     | `agentic/agents/supervisor.py::supervisor_router`                              |
| Knowledge retrieval + confidence (RAG)                               | `agentic/retrieval.py`                                                         |
| Escalation when no relevant article                                  | `agentic/agents/escalation.py`                                                 |
| ≥2 support-operation tools (4 core + 3 CultPass = 7 total)           | `agentic/tools/uda_account.py`, `agentic/tools/cultpass.py`                    |
| Persistent customer history                                          | `agentic/memory.py::SqliteLongTermStore` + `TicketMessage`                     |
| Short-term memory (per `thread_id`)                                  | `agentic/memory.py::get_checkpointer` (SqliteSaver)                            |
| Long-term memory                                                     | `agentic/memory.py::SqliteLongTermStore`                                       |
| End-to-end demo + logging                                            | `03_agentic_app.ipynb`, `03_agentic_app.py`, `agentic/logging_utils.py`        |
| Chat interface                                                       | `utils.py::chat_interface`                                                     |
| Tests                                                                | `tests/test_smoke.py`                                                          |
| MCP-ready tools (recommended)                                        | `agentic/tools/mcp_server.py` (optional FastMCP wrapper)                       |

## Configuration

All paths and model identifiers come from `agentic/config.py` and can be
overridden via environment variables (or `.env`):

| Variable                    | Default                                  |
|-----------------------------|------------------------------------------|
| `OPENAI_API_KEY`            | (required for FAISS + LLM calls)        |
| `UDA_LLM_MODEL`             | `gpt-4o-mini`                           |
| `UDA_EMBED_MODEL`           | `text-embedding-3-small`                |
| `UDA_CONFIDENCE_THRESHOLD`  | `0.55`                                  |
| `UDA_CORE_DB`               | `data/core/uda_hub.db`                  |
| `UDA_EXTERNAL_DB`           | `data/external/cultpass.db`             |
| `UDA_VECTORSTORE`           | `data/models/uda_hub_kb`                |
| `UDA_CHECKPOINT_DB`         | `data/core/checkpoints.db`              |
| `UDA_LONGTERM_DB`           | `data/core/longterm.db`                 |
| `UDA_CULTPASS_ARTICLES`     | `data/external/cultpass_articles.jsonl` |

## Pinned dependencies

See `requirements.txt`. Headline:

- `langgraph==0.2.62`, `langgraph-checkpoint-sqlite==2.0.1`
- `langchain==0.3.7`, `langchain-openai==0.2.10`, `langchain-community==0.3.7`
- `faiss-cpu==1.8.0.post1`, `openai==1.50.0`
- `pydantic==2.8.2`, `python-dotenv==1.0.1`
- `pytest==8.3.3`, `notebook==7.2.2`
