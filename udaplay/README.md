# UdaPlay — AI Research Agent for Video Games

**Author:** Sam Sepassi

UdaPlay is a question-answering agent for the gaming-analytics scenario. It
answers natural-language questions like *"Who developed FIFA 21?"* or
*"When was God of War Ragnarok released?"* by:

1. Retrieving from an internal ChromaDB vector store of curated game records.
2. Letting an LLM judge evaluate whether the retrieval is sufficient.
3. Falling back to Tavily web search when internal knowledge is weak,
   persisting new findings into the store as long-term memory.
4. Returning a structured report with answer, confidence, citations, and the
   full state-machine trace.

```
udaplay/
├── games/                           20 game records (one JSON per game)
├── lib/
│   ├── vector_store.py              ChromaDB-backed VectorStoreManager
│   ├── tools.py                     retrieve_game, evaluate_retrieval,
│   │                                game_web_search
│   └── agent.py                     UdaPlayAgent state machine + memory
├── Udaplay_01_solution_project.ipynb  Build & test the RAG pipeline
├── Udaplay_02_solution_project.ipynb  Run the agent on example queries
├── build_notebooks.py               Regenerates the two notebooks
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
cd udaplay
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # then fill in OPENAI_API_KEY / TAVILY_API_KEY
```

Both keys are optional. Without `OPENAI_API_KEY` the store falls back to
Chroma's default `all-MiniLM-L6-v2` embeddings and the evaluator switches to
a deterministic distance-based heuristic. Without `TAVILY_API_KEY` the web
search tool returns an empty list (the agent reports honestly that it could
not find an answer rather than fabricating one).

## Run

1. **Build the index** — open `Udaplay_01_solution_project.ipynb` and run all
   cells. This populates `chromadb/` from the `games/` directory.
2. **Run the agent** — open `Udaplay_02_solution_project.ipynb`. It loads the
   persistent store and runs the agent against four sample queries.

## State machine

```
START -> RETRIEVE -> EVALUATE -> ┬──► REPORT  (high-confidence internal hit)
                                 └─► WEB_SEARCH -> REPORT
```

`UdaPlayAgent.ask()` returns an `AgentReport` with:

| field             | description                                             |
|-------------------|---------------------------------------------------------|
| `answer`          | natural-language answer with `[KB-i]` / `[WEB-i]` cites |
| `confidence`      | 0–1 score from the evaluator                            |
| `used_web_search` | whether the web fallback was triggered                  |
| `citations`       | list of `Citation(label, source, snippet)`              |
| `trace`           | step-by-step record of every state transition           |

`agent.to_json(report)` serialises the same payload for dashboards.

## Rubric mapping

| Rubric criterion                                    | Where it lives                                                     |
|-----------------------------------------------------|---------------------------------------------------------------------|
| Process & embed local dataset                       | `lib/vector_store.py`, notebook 01 cells 2–4                        |
| Persistent vector DB queryable for semantic search  | `chromadb/` directory + notebook 01 cell 5                          |
| Tool: retrieve from vector DB                       | `lib/tools.py::GameRetrievalTool`                                   |
| Tool: evaluate retrieval quality                    | `lib/tools.py::EvaluationTool` (LLM judge + heuristic fallback)     |
| Tool: web search fallback                           | `lib/tools.py::WebSearchTool` (Tavily)                              |
| Agent: try internal first → evaluate → fallback     | `lib/agent.py::UdaPlayAgent.ask`                                    |
| Stateful conversation                               | `lib/agent.py::AgentMemory`                                         |
| State-machine workflow                              | `lib/agent.py::AgentState` enum + ordered transitions in `ask`      |
| Structured, well-cited answers                      | `AgentReport` + `Citation` pydantic models                          |
| Demonstrate ≥3 example queries                      | Notebook 02 sections 4 & 6                                          |
| Personalised dataset                                | `games/` (20 games across 7 platforms, 9 publishers)                |
| Long-term memory from web search                    | `_remember_web_findings` in `lib/agent.py`; notebook 02 section 7   |
| Structured JSON output                              | `agent.to_json(report)` in notebook 02 section 5                    |
