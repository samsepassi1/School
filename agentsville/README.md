# AgentsVille Trip Planner

A two-agent system that drafts, evaluates, and revises a personalized
day-by-day vacation itinerary for the (fictional) city of **AgentsVille**
using the OpenAI API.

## Layout

| File | Purpose |
| --- | --- |
| `project_lib.py` | Pydantic models (`Traveler`, `Activity`, `Weather`, `TravelPlan`, …) and a mock "external API" for AgentsVille weather and activities (June 2025). |
| `project_starter.ipynb` | The full notebook: vacation setup, `ItineraryAgent`, evaluations, tools, and the ReAct `ItineraryRevisionAgent`. |
| `_build_notebook.py` | Source-of-truth script that regenerates `project_starter.ipynb` from inline cell definitions. |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then put your real OPENAI_API_KEY in .env
jupyter notebook project_starter.ipynb
```

Run the cells top-to-bottom. The notebook will:

1. Build a `VacationInfo` for two travelers visiting AgentsVille.
2. Pull the matching weather and activity data.
3. Run the `ItineraryAgent` to draft an initial `TravelPlan`.
4. Run the evaluation suite (including an LLM-powered weather-compatibility
   check).
5. Run the `ItineraryRevisionAgent` (ReAct loop) until every evaluation
   passes and the travelers' "≥ 2 activities per day" feedback is satisfied.
6. Generate a fun narrative summary of the trip.

## Editing the notebook

The notebook is generated from `_build_notebook.py`. Prefer editing the cell
definitions there and re-running:

```bash
python _build_notebook.py
```

That keeps diffs reviewable and avoids hand-editing JSON.
