"""Build the two graded notebooks for UdaPlay.

Notebooks are constructed in code (not hand-edited JSON) to keep diffs small
and reviewable. Run ``python build_notebooks.py`` from the ``udaplay/``
directory to regenerate ``Udaplay_01_solution_project.ipynb`` and
``Udaplay_02_solution_project.ipynb``.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).parent


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in lines],
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# Notebook 01: RAG pipeline
# ---------------------------------------------------------------------------

NB1 = notebook(
    [
        md(
            "# UdaPlay 01 — RAG Pipeline",
            "",
            "**Author: Sam Sepassi**",
            "",
            "This notebook prepares a persistent ChromaDB vector store from the local",
            "`games/` JSON dataset so the UdaPlay agent (notebook 02) can answer",
            "questions via Retrieval-Augmented Generation.",
            "",
            "Pipeline:",
            "",
            "1. Load every `games/NNN.json` record",
            "2. Convert each record into a single embeddable document",
            "3. Embed and persist into ChromaDB (`chromadb/` directory)",
            "4. Sanity-check with three semantic queries",
        ),
        md("## 1. Setup"),
        code(
            "import json",
            "from pathlib import Path",
            "",
            "from dotenv import load_dotenv",
            "load_dotenv()",
            "",
            "from lib.vector_store import VectorStoreManager, load_games_from_directory",
        ),
        md("## 2. Inspect the raw dataset"),
        code(
            "GAMES_DIR = Path('games')",
            "files = sorted(GAMES_DIR.glob('*.json'))",
            "print(f'Found {len(files)} game JSON files')",
            "with files[0].open() as fp:",
            "    print(json.dumps(json.load(fp), indent=2))",
        ),
        md(
            "## 3. Convert JSON records into embeddable documents",
            "",
            "`load_games_from_directory` reads each file and produces a `GameDocument`",
            "whose `text` is a single natural-language summary — that's what the",
            "embedder sees, so the document text deliberately includes name, year,",
            "platform, developer, publisher, genre, and description.",
        ),
        code(
            "documents = load_games_from_directory(GAMES_DIR)",
            "print(f'Built {len(documents)} documents')",
            "print('---')",
            "print(documents[0].text)",
            "print('---')",
            "print(documents[0].metadata)",
        ),
        md(
            "## 4. Build the persistent vector store",
            "",
            "ChromaDB writes everything to the `chromadb/` directory so the agent",
            "notebook can re-use the same index without re-embedding. We `reset()`",
            "first so re-runs are idempotent.",
        ),
        code(
            "store = VectorStoreManager(persist_directory='chromadb')",
            "store.reset()",
            "written = store.add_games(documents)",
            "print(f'Wrote {written} documents; collection now contains {store.count()} rows')",
        ),
        md("Peek at the first few persisted docs to confirm the round-trip:"),
        code(
            "for row in list(store.peek(3)):",
            "    print(row['id'], '->', row['metadata'].get('name'))",
        ),
        md(
            "## 5. Semantic search smoke tests",
            "",
            "Three queries that map onto the project specification's example",
            "questions. We print the top hit and its distance.",
        ),
        code(
            "QUERIES = [",
            "    'Who developed FIFA 21?',",
            "    'When was God of War Ragnarok released?',",
            "    'What platform was Pokemon Red launched on?',",
            "]",
            "",
            "for q in QUERIES:",
            "    hits = store.query(q, k=3)",
            "    print(f'Q: {q}')",
            "    for h in hits:",
            "        meta = h['metadata']",
            "        print(f\"  - {meta.get('name')} ({meta.get('year')}) [dist={h['distance']:.3f}]\")",
            "    print()",
        ),
        md(
            "## 6. Done",
            "",
            "The vector store is now persisted to `chromadb/`. Move on to",
            "`Udaplay_02_solution_project.ipynb` to run the agent.",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Notebook 02: Agent
# ---------------------------------------------------------------------------

NB2 = notebook(
    [
        md(
            "# UdaPlay 02 — Agent",
            "",
            "**Author: Sam Sepassi**",
            "",
            "This notebook wires the three tools (`retrieve_game`,",
            "`evaluate_retrieval`, `game_web_search`) into a stateful agent and runs",
            "it against several example queries.",
            "",
            "State machine:",
            "",
            "```",
            "START -> RETRIEVE -> EVALUATE -> (REPORT | WEB_SEARCH -> REPORT) -> DONE",
            "```",
            "",
            "When evaluation says retrieval is insufficient, the agent falls back to",
            "Tavily web search **and** writes the new findings back into the vector",
            "store as long-term memory.",
        ),
        md("## 1. Setup"),
        code(
            "from dotenv import load_dotenv",
            "load_dotenv()",
            "",
            "from lib.vector_store import VectorStoreManager",
            "from lib.agent import UdaPlayAgent",
        ),
        md(
            "## 2. Load the persistent vector store",
            "",
            "We don't re-ingest here — that was notebook 01's job. We just open the",
            "existing collection.",
        ),
        code(
            "store = VectorStoreManager(persist_directory='chromadb')",
            "print(f'Vector store contains {store.count()} documents')",
            "assert store.count() > 0, 'Run Udaplay_01_solution_project.ipynb first to build the index'",
        ),
        md(
            "## 3. Construct the agent",
            "",
            "`UdaPlayAgent` owns three tools, the conversation memory, and the state",
            "machine. `confidence_floor=0.55` means anything below 0.55 confidence",
            "from the evaluator triggers a web-search fallback.",
        ),
        code(
            "agent = UdaPlayAgent(store=store, top_k=4, confidence_floor=0.55)",
            "print('Tools registered:')",
            "for tool in (agent.retrieval_tool, agent.evaluation_tool, agent.web_search_tool):",
            "    print(f'  - {tool.name}: {tool.description}')",
        ),
        md(
            "## 4. Example queries",
            "",
            "Run several questions through the agent in sequence. Because the agent",
            "remembers prior turns, the third query references context from the",
            "first.",
        ),
        code(
            "QUERIES = [",
            "    'Who developed FIFA 21?',",
            "    'When was God of War Ragnarok released?',",
            "    'What platform was Pokemon Red launched on?',",
            "    'What is Rockstar Games working on right now?',",
            "]",
        ),
        code(
            "def show_report(report):",
            "    print('=' * 80)",
            "    print(f'Q: {report.question}')",
            "    print('-' * 80)",
            "    print(f'Answer:        {report.answer}')",
            "    print(f'Confidence:    {report.confidence:.2f}')",
            "    print(f'Web fallback?: {report.used_web_search}')",
            "    print('Citations:')",
            "    for c in report.citations:",
            "        print(f'  [{c.label}] {c.source}')",
            "    print('Trace:')",
            "    for step in report.trace:",
            "        print(f\"  -> {step['state']}: {step['detail']}\")",
            "    print()",
            "",
            "for q in QUERIES:",
            "    show_report(agent.ask(q))",
        ),
        md(
            "## 5. Structured output",
            "",
            "The same report is also available as JSON for downstream integrations",
            "(dashboards, evaluation pipelines, etc.).",
        ),
        code(
            "report = agent.ask('Who developed The Witcher 3?')",
            "print(agent.to_json(report))",
        ),
        md(
            "## 6. Conversation memory",
            "",
            "Across calls the agent keeps a short transcript so follow-up questions",
            "have context.",
        ),
        code(
            "_ = agent.ask('Who published Elden Ring?')",
            "_ = agent.ask('And which studio developed it?')",
            "print(agent.memory.transcript())",
        ),
        md(
            "## 7. Long-term memory from web search",
            "",
            "When the agent falls back to the web (e.g. for the Rockstar question),",
            "the results are persisted into the vector store under `kind=web_memory`",
            "so a follow-up question can answer from the local index.",
        ),
        code(
            "web_memories = [row for row in store.peek(20) if row['metadata'].get('kind') == 'web_memory']",
            "print(f'{len(web_memories)} web-memory rows persisted')",
            "for m in web_memories[:3]:",
            "    print('-', m['id'], '->', m['metadata'].get('title'))",
        ),
        md(
            "## 8. Report",
            "",
            "Each agent run above prints, for every query:",
            "",
            "- the final answer with inline `[KB-i]` / `[WEB-i]` citations,",
            "- a confidence score from the evaluator,",
            "- a boolean indicating whether the web fallback was triggered,",
            "- the citation list (source paths or URLs), and",
            "- the full state-machine trace.",
            "",
            "That trace is the agent's reasoning record, satisfying the rubric",
            "requirement to surface tool usage and reasoning alongside the answer.",
        ),
    ]
)


def main() -> None:
    (NB_DIR / "Udaplay_01_solution_project.ipynb").write_text(
        json.dumps(NB1, indent=1) + "\n", encoding="utf-8"
    )
    (NB_DIR / "Udaplay_02_solution_project.ipynb").write_text(
        json.dumps(NB2, indent=1) + "\n", encoding="utf-8"
    )
    print("Notebooks written.")


if __name__ == "__main__":
    main()
