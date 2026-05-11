# AgentsVille Trip Planner

**Author:** Sam Sepassi  
**Program:** Agentic AI Nanodegree (nd900)  
**Project:** AgentsVille Trip Planner — A Multi-Agent Travel Assistant System

A two-agent system that drafts, evaluates, and revises a personalized
day-by-day vacation itinerary for the (fictional) city of **AgentsVille**
using the OpenAI API.

---

## Rubric Mapping

| Rubric Requirement | File / Cell | Implementation |
| --- | --- | --- |
| Define `VacationInfo` with typed fields and validation | `project_starter.ipynb` — Cell 5 | Pydantic model with `field_validator` for date ordering, min budget, min travelers |
| Use role-based prompting for the ItineraryAgent | Cell 9 | System prompt assigns role of "expert travel planner" with chain-of-thought instructions |
| Generate a structured `TravelPlan` JSON output | Cell 10–11 | `client.beta.chat.completions.parse` with `response_format=TravelPlan` |
| Implement programmatic evaluations | Cell 13 | 6 evaluators: city, dates, activities exist, cost correctness, budget, min activities/day |
| Implement LLM-powered weather compatibility check | Cell 14 | `llm_check_weather_compatibility()` using `_CompatibilityDecision` Pydantic model |
| Register at least one tool for the revision agent | Cell 18 | 4 tools: `calculator`, `get_activities_for_date`, `add_activity_to_plan`, `run_evaluations` |
| Implement ReAct (THOUGHT → ACTION → OBSERVATION) loop | Cell 20–21 | `run_revision_agent()` with regex-based THOUGHT/ACTION parsing and tool dispatch |
| Agent revises plan until all evaluations pass | Cell 22 | ReAct loop with `max_steps=14`, exits on `FINAL_ANSWER` tool call after all evals pass |
| Generate a narrative summary | Cell 24 | LLM travel-writer prompt producing a 2–3 paragraph human-readable summary |

---

## System Architecture

```
VacationInfo (Pydantic)
      │
      ▼
[Mock APIs: get_weather_data, get_activities_data]
      │
      ▼
ItineraryAgent ──────────────────────────────────────────────────────────────
│  Role-based prompt + Chain-of-Thought + structured output (TravelPlan)   │
│  OpenAI gpt-4o-mini via client.beta.chat.completions.parse               │
└──────────────────────────────────────────────────────────────────────────┘
      │
      ▼  initial TravelPlan
Evaluation Suite
  ├── eval_matches_city
  ├── eval_matches_dates
  ├── eval_activities_exist
  ├── eval_cost_correct
  ├── eval_within_budget
  ├── eval_weather_compatible (LLM-powered)
  └── eval_min_activities_per_day
      │
      ▼  EvalResult list (some may FAIL)
ItineraryRevisionAgent (ReAct)
│  THOUGHT → ACTION → OBSERVATION loop                                      │
│  Tools: calculator | get_activities_for_date | add_activity_to_plan |    │
│          run_evaluations | FINAL_ANSWER                                   │
└──────────────────────────────────────────────────────────────────────────┘
      │
      ▼  revised TravelPlan (all evals PASS)
Narrative Summary (travel-writer LLM prompt)
```

**Design decisions:**
- **Typed data contracts everywhere.** Every agent input and output is a Pydantic model. This prevents hallucinated fields, makes validation trivial, and documents the API surface explicitly.
- **ReAct over one-shot revision.** A single revision prompt would risk over-correcting. The ReAct loop lets the agent inspect the actual evaluation output and make targeted, incremental fixes.
- **Tool registry pattern.** Tools are registered as `ToolSpec` objects and auto-injected into the system prompt so any new tool is immediately available without prompt rewriting.
- **Graceful degradation.** Every component handles missing API keys (raises a clean `RuntimeError`), missing weather data (skips LLM check with a clear message), and unknown activity IDs (flags them in the eval result rather than crashing).

---

## File Layout

| File | Purpose |
| --- | --- |
| `project_lib.py` | Pydantic models (`Traveler`, `Activity`, `Weather`, `ItineraryActivity`, `ItineraryDay`, `TravelPlan`) and mock APIs for AgentsVille weather and activities (June 2025). |
| `project_starter.ipynb` | The full notebook: vacation setup, `ItineraryAgent`, evaluations, tools, `ItineraryRevisionAgent`, and narrative summary. |
| `_build_notebook.py` | Source-of-truth script that regenerates `project_starter.ipynb` from inline cell definitions. |
| `requirements.txt` | Python dependencies: `openai>=1.40.0`, `pydantic>=2.6`, `python-dotenv>=1.0`, `jupyter>=1.0`. |

---

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your-key-here   # or add to a .env file
jupyter notebook project_starter.ipynb
```

Run cells top-to-bottom. The notebook will:

1. Build a `VacationInfo` for two travelers (Ada and Grace) visiting AgentsVille, June 10–14, 2025.
2. Fetch matching weather and activity data from the mock APIs.
3. Run the `ItineraryAgent` to draft an initial `TravelPlan`.
4. Run the evaluation suite (including an LLM-powered weather-compatibility check).
5. Run the `ItineraryRevisionAgent` (ReAct loop) until every evaluation passes and the travelers' "≥ 2 activities per day" feedback is satisfied.
6. Generate a fun narrative summary of the trip.

---

## Editing the Notebook

The notebook is generated from `_build_notebook.py`. Prefer editing the cell
definitions there and re-running:

```bash
python _build_notebook.py
```

That keeps diffs reviewable and avoids hand-editing JSON.
