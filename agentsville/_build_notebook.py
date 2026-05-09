"""Generate ``project_starter.ipynb`` from inline Python cell definitions.

This script is the single source of truth for the notebook. Edit the cells
below and re-run with ``python _build_notebook.py`` to regenerate.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def md(text: str) -> dict:
    """Build a markdown cell."""

    text = dedent(text).strip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    """Build a code cell."""

    text = dedent(text).strip("\n") + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS: list[dict] = []
CELLS += [

# ---------------------------------------------------------------------------
md("""
    # AgentsVille Trip Planner

    A multi-agent system that builds, evaluates, and revises a personalized
    vacation itinerary for the city of **AgentsVille** using an LLM.

    The notebook walks through five steps:

    1. **Vacation details** — capture trip dates, travelers, and constraints in a
       `VacationInfo` Pydantic model.
    2. **Data gathering** — pull mock weather and activity data for AgentsVille.
    3. **`ItineraryAgent`** — a one-shot LLM call that drafts a structured
       day-by-day itinerary (`TravelPlan`).
    4. **Evaluations** — programmatic and LLM-powered checks for budget, dates,
       hallucinations, and weather compatibility.
    5. **`ItineraryRevisionAgent`** — a ReAct (THOUGHT → ACTION → OBSERVATION)
       agent that uses tools to refine the itinerary until every evaluation
       passes and the travelers' feedback is satisfied.

    All helper data (Pydantic models for `Activity`, `Weather`, `TravelPlan`,
    plus the mock APIs) lives in `project_lib.py`.
"""),

# ---------------------------------------------------------------------------
md("""
    ## 0. Setup

    Install dependencies (only needed once) and import the pieces we'll use.
    Provide your OpenAI API key either via a `.env` file at the project root
    or by setting the `OPENAI_API_KEY` environment variable before launching
    Jupyter.
"""),

code("""
    # !pip install openai pydantic python-dotenv
"""),

code("""
    from __future__ import annotations

    import json
    import os
    import re
    from datetime import date, datetime, timedelta
    from typing import Any, Callable, Dict, List, Optional

    from pydantic import BaseModel, Field, field_validator

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from openai import OpenAI

    from project_lib import (
        CITY,
        Activity,
        ItineraryActivity,
        ItineraryDay,
        Traveler,
        TravelPlan,
        Weather,
        WeatherCondition,
        get_activities_by_date,
        get_activities_data,
        get_weather_data,
        safe_calculator,
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file or export it "
            "before launching Jupyter."
        )

    client = OpenAI(api_key=api_key)
    PLANNER_MODEL = "gpt-4o-mini"
    REASONING_MODEL = "gpt-4o-mini"
    print("OpenAI client ready.")
"""),

# ---------------------------------------------------------------------------
md("""
    ## 1. Define the vacation details

    `VacationInfo` is the input contract for the planner. The model is built so
    that:

    - the destination is always a non-empty string,
    - `end_date` cannot precede `start_date`,
    - at least one traveler is provided,
    - the budget is a positive number expressed in USD.
"""),

code("""
    class VacationInfo(BaseModel):
        \"\"\"Everything the planner needs to know about the upcoming trip.\"\"\"

        destination: str = Field(..., min_length=1, description="City the travelers are visiting.")
        start_date: date = Field(..., description="First day of the trip (inclusive).")
        end_date: date = Field(..., description="Last day of the trip (inclusive).")
        travelers: List[Traveler] = Field(..., min_length=1)
        budget_usd: float = Field(..., gt=0, description="Total trip budget in USD.")
        traveler_feedback: Optional[str] = Field(
            default=None,
            description=(
                "Free-form notes from the travelers (e.g. 'we want at least "
                "2 activities per day'). Consumed by the revision agent."
            ),
        )

        @field_validator("end_date")
        @classmethod
        def _end_after_start(cls, v: date, info):  # type: ignore[override]
            start = info.data.get("start_date")
            if start is not None and v < start:
                raise ValueError("end_date must be on or after start_date")
            return v

        @property
        def num_days(self) -> int:
            return (self.end_date - self.start_date).days + 1

        @property
        def all_interests(self) -> List[str]:
            seen: List[str] = []
            for t in self.travelers:
                for i in t.interests:
                    if i not in seen:
                        seen.append(i)
            return seen


    vacation_info = VacationInfo(
        destination=CITY,
        start_date=date(2025, 6, 10),
        end_date=date(2025, 6, 14),
        travelers=[
            Traveler(name="Ada", age=34, interests=["food", "art", "music"]),
            Traveler(name="Grace", age=36, interests=["history", "outdoor", "wellness"]),
        ],
        budget_usd=1500.0,
        traveler_feedback="Please make sure each day has at least 2 activities.",
    )

    print(vacation_info.model_dump_json(indent=2))
"""),

# ---------------------------------------------------------------------------
md("""
    ## 2. Gather weather and activities

    The mock APIs in `project_lib.py` simulate two external services. We use
    the start and end dates from `VacationInfo` to fetch only the relevant
    rows.
"""),

code("""
    weather_forecast = get_weather_data(vacation_info.start_date, vacation_info.end_date)
    available_activities = get_activities_data(vacation_info.start_date, vacation_info.end_date)

    print(f"Weather days fetched: {len(weather_forecast)}")
    for w in weather_forecast:
        print(f"  {w.date}  {w.condition.value:<14} {w.description}")

    print(f"\\nActivities fetched: {len(available_activities)}")
    print(f"First 3 examples:")
    for a in available_activities[:3]:
        print(f"  {a.activity_id}  {a.date}  {a.name}  ${a.price_usd:.0f}")
"""),

# ---------------------------------------------------------------------------
md("""
    ## 3. The `ItineraryAgent`

    A single LLM call that produces a fully-structured `TravelPlan`. We:

    1. Render the `TravelPlan` JSON Schema and inject it into the prompt so
       the model knows the exact output contract.
    2. Provide the entire `VacationInfo`, weather forecast and activities list
       as JSON context so the model has no need to invent anything.
    3. Use OpenAI's `beta.chat.completions.parse` helper to validate the
       response against the Pydantic model in one shot.
"""),

code("""
    TRAVEL_PLAN_SCHEMA = json.dumps(TravelPlan.model_json_schema(), indent=2)


    ITINERARY_AGENT_SYSTEM_PROMPT = f\"\"\"
    You are an **expert travel planner** specializing in {{city}} vacations. Your
    superpower is turning a traveler's preferences into a thoughtful, day-by-day
    itinerary that fits the weather, the budget, and what they actually enjoy.

    # YOUR TASK
    Given a `VacationInfo` object, the daily weather forecast, and the catalog
    of available activities, design a **complete day-by-day itinerary** for the
    trip and return it as a single JSON object that conforms to the
    `TravelPlan` schema below.

    # HARD CONSTRAINTS
    1. The itinerary must take place in `VacationInfo.destination` and nowhere
       else. The `city` field of the `TravelPlan` must equal that destination.
    2. The itinerary's `start_date` and `end_date` must exactly equal the
       dates in `VacationInfo`.
    3. **Never invent activities.** Every `activity_id` in the plan MUST appear
       verbatim in the provided activities list. Times, prices and locations
       must also be copied from that list.
    4. Each `ItineraryDay` must contain at least one activity. Aim for 2–4
       activities per day for good pacing.
    5. Avoid primarily-outdoor activities on days whose weather is `rainy`,
       `stormy`, or otherwise unsafe (high precipitation, thunderstorms,
       extreme winds). Prefer indoor or covered options on those days.
    6. Prefer activities whose `related_interests` overlap with the travelers'
       interests in `VacationInfo.travelers[*].interests`.
    7. `total_cost_usd` must equal the sum of `price_usd` of every activity in
       the plan, and must be **less than or equal to** `VacationInfo.budget_usd`.

    # REASONING (Chain-of-Thought — internal)
    Before producing the JSON, walk through the following plan **silently**:
      1. List each trip day and write down its weather summary.
      2. For each day, list candidate activities (filter by date, by weather
         compatibility, and by traveler interests).
      3. Pick 2–3 activities per day, varying themes (food / culture / outdoor
         / etc.) and pacing (active mornings, relaxed evenings).
      4. Sum the prices and verify the running total is within budget. If it
         exceeds budget, swap costlier activities for cheaper alternatives.
      5. Write a one-sentence `weather_summary` per day and a short overall
         `notes` field.

    Do **not** include this reasoning in the final answer. Return only the JSON.

    # OUTPUT FORMAT
    Return ONLY a single JSON object that conforms exactly to this schema
    (the names and types must match):

    ```
    {{travel_plan_schema}}
    ```

    No markdown, no commentary, no code fences — just the JSON object.

    # EXAMPLE OUTPUT (truncated, for shape only)
    ```
    {{example}}
    ```
    \"\"\".strip()


    EXAMPLE_PLAN = {
      "city": "AgentsVille",
      "start_date": "2025-06-10",
      "end_date": "2025-06-11",
      "travelers": ["Ada", "Grace"],
      "days": [
        {
          "date": "2025-06-10",
          "weather_summary": "Bright and sunny, perfect for outdoor strolls.",
          "activities": [
            {
              "activity_id": "ACT-001",
              "name": "Riverside Park Walking Tour",
              "start_time": "09:00",
              "end_time": "11:00",
              "location": "Riverside Park",
              "description": "Guided walking tour of the riverside park.",
              "price_usd": 25.0
            }
          ]
        }
      ],
      "total_cost_usd": 25.0,
      "notes": "A balanced two-day taster of AgentsVille."
    }


    rendered_itinerary_system_prompt = (
        ITINERARY_AGENT_SYSTEM_PROMPT
        .replace("{city}", vacation_info.destination)
        .replace("{travel_plan_schema}", TRAVEL_PLAN_SCHEMA)
        .replace("{example}", json.dumps(EXAMPLE_PLAN, indent=2))
    )

    print(rendered_itinerary_system_prompt[:1200])
    print("...")
"""),

code("""
    def run_itinerary_agent(
        vacation_info: VacationInfo,
        weather_forecast: List[Weather],
        available_activities: List[Activity],
        model: str = PLANNER_MODEL,
    ) -> TravelPlan:
        \"\"\"Call the ItineraryAgent and parse its response into a `TravelPlan`.\"\"\"

        user_payload = {
            "vacation_info": json.loads(vacation_info.model_dump_json()),
            "weather_forecast": [json.loads(w.model_dump_json()) for w in weather_forecast],
            "available_activities": [
                json.loads(a.model_dump_json()) for a in available_activities
            ],
        }

        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": rendered_itinerary_system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Build the itinerary now. Here is the full context:\\n\\n"
                        + json.dumps(user_payload, indent=2)
                    ),
                },
            ],
            response_format=TravelPlan,
            temperature=0.2,
        )
        return response.choices[0].message.parsed
"""),

code("""
    initial_plan = run_itinerary_agent(
        vacation_info, weather_forecast, available_activities
    )
    print(initial_plan.model_dump_json(indent=2))
"""),

# ---------------------------------------------------------------------------
md("""
    ## 4. Evaluating the itinerary

    Each `EvalResult` is a small record returned by an evaluation function. We
    define one evaluator per concern:

    | Evaluator | Concern |
    | --- | --- |
    | `eval_matches_city` | Plan visits the right city |
    | `eval_matches_dates` | Plan covers the requested date range |
    | `eval_activities_exist` | No hallucinated activities |
    | `eval_cost_correct` | `total_cost_usd` matches the sum of activities |
    | `eval_within_budget` | Total cost ≤ traveler budget |
    | `eval_weather_compatible` | LLM-powered weather safety check |
    | `eval_min_activities_per_day` | Traveler feedback: ≥ 2 activities / day |
"""),

code("""
    class EvalResult(BaseModel):
        name: str
        passed: bool
        message: str

        def __str__(self) -> str:
            mark = "PASS" if self.passed else "FAIL"
            return f"[{mark}] {self.name}: {self.message}"


    def eval_matches_city(plan: TravelPlan, vacation_info: VacationInfo) -> EvalResult:
        ok = plan.city.strip().lower() == vacation_info.destination.strip().lower()
        return EvalResult(
            name="matches_city",
            passed=ok,
            message=(
                "City matches the requested destination."
                if ok
                else f"Plan city {plan.city!r} != requested {vacation_info.destination!r}"
            ),
        )


    def eval_matches_dates(plan: TravelPlan, vacation_info: VacationInfo) -> EvalResult:
        start_ok = plan.start_date == vacation_info.start_date
        end_ok = plan.end_date == vacation_info.end_date
        plan_dates = {d.date for d in plan.days}
        expected = {
            vacation_info.start_date + timedelta(days=i)
            for i in range(vacation_info.num_days)
        }
        days_ok = plan_dates == expected
        ok = start_ok and end_ok and days_ok
        return EvalResult(
            name="matches_dates",
            passed=ok,
            message=(
                "Plan dates and per-day entries cover the requested range."
                if ok
                else (
                    f"start={plan.start_date} vs {vacation_info.start_date}, "
                    f"end={plan.end_date} vs {vacation_info.end_date}, "
                    f"missing={sorted(expected - plan_dates)}, "
                    f"extra={sorted(plan_dates - expected)}"
                )
            ),
        )


    def eval_activities_exist(
        plan: TravelPlan, available_activities: List[Activity]
    ) -> EvalResult:
        catalog = {a.activity_id: a for a in available_activities}
        bad: List[str] = []
        for day in plan.days:
            for act in day.activities:
                src = catalog.get(act.activity_id)
                if src is None:
                    bad.append(f"unknown id {act.activity_id!r}")
                    continue
                if src.date != day.date:
                    bad.append(
                        f"{act.activity_id} scheduled on {day.date} but is offered on {src.date}"
                    )
                if abs(src.price_usd - act.price_usd) > 0.01:
                    bad.append(
                        f"{act.activity_id} price {act.price_usd} != catalog {src.price_usd}"
                    )
        return EvalResult(
            name="activities_exist",
            passed=not bad,
            message=(
                "All activities exist in the catalog and match dates/prices."
                if not bad
                else "; ".join(bad)
            ),
        )


    def eval_cost_correct(plan: TravelPlan) -> EvalResult:
        summed = sum(a.price_usd for d in plan.days for a in d.activities)
        ok = abs(summed - plan.total_cost_usd) < 0.01
        return EvalResult(
            name="cost_correct",
            passed=ok,
            message=(
                f"total_cost_usd ({plan.total_cost_usd:.2f}) matches activity sum."
                if ok
                else f"total_cost_usd={plan.total_cost_usd:.2f} but sum of activities={summed:.2f}"
            ),
        )


    def eval_within_budget(plan: TravelPlan, vacation_info: VacationInfo) -> EvalResult:
        ok = plan.total_cost_usd <= vacation_info.budget_usd + 1e-6
        return EvalResult(
            name="within_budget",
            passed=ok,
            message=(
                f"Total cost {plan.total_cost_usd:.2f} <= budget {vacation_info.budget_usd:.2f}."
                if ok
                else f"Plan exceeds budget by {plan.total_cost_usd - vacation_info.budget_usd:.2f}"
            ),
        )


    def eval_min_activities_per_day(
        plan: TravelPlan, minimum: int = 2
    ) -> EvalResult:
        bad = [d.date.isoformat() for d in plan.days if len(d.activities) < minimum]
        return EvalResult(
            name="min_activities_per_day",
            passed=not bad,
            message=(
                f"Every day has at least {minimum} activities."
                if not bad
                else f"Days with < {minimum} activities: {', '.join(bad)}"
            ),
        )
"""),

code("""
    ACTIVITY_AND_WEATHER_ARE_COMPATIBLE_SYSTEM_PROMPT = \"\"\"
    You are a careful **travel safety reviewer**. You decide whether a proposed
    activity is COMPATIBLE with the day's weather forecast.

    # TASK
    You will be given:
      - an activity description (name, location, indoor/outdoor flag, summary)
      - a daily weather forecast (condition, temperatures, precipitation chance,
        free-text description)

    Decide whether attending the activity in that weather is reasonable.

    # OUTPUT FORMAT
    Return ONLY a single JSON object — no markdown, no prose around it:

    {"compatible": <true|false>, "reason": "<one short sentence>"}

    # RULES
    - Indoor activities are essentially always compatible with weather.
    - Primarily-outdoor activities are INCOMPATIBLE if the forecast shows:
        * thunderstorms or stormy conditions, OR
        * heavy rain (precipitation_chance > 0.5), OR
        * dangerous winds, OR
        * extreme/unsafe temperatures.
    - Mild rain or wind is fine for indoor or covered activities.
    - When in doubt, err toward safety: if the weather is stormy, mark any
      outdoor activity as incompatible.

    # EXAMPLES

    Activity: "Mountain Bike Adventure — outdoor mountain trails (is_outdoor=true)"
    Weather: "stormy, precipitation_chance=0.95, 'Thunderstorms expected — outdoor activities should be avoided.'"
    => {"compatible": false, "reason": "Mountain biking in a thunderstorm is unsafe."}

    Activity: "Pottery Workshop — indoor studio (is_outdoor=false)"
    Weather: "rainy, precipitation_chance=0.85, 'Steady rain throughout the day.'"
    => {"compatible": true, "reason": "The workshop is indoors so the rain doesn't matter."}

    Activity: "Riverside Park Walking Tour — outdoor (is_outdoor=true)"
    Weather: "sunny, precipitation_chance=0.05, 'Bright and sunny all day.'"
    => {"compatible": true, "reason": "Conditions are ideal for an outdoor walk."}

    Activity: "Sunset Jazz Cruise — outdoor boat (is_outdoor=true)"
    Weather: "windy, precipitation_chance=0.10, 'Strong winds across the harbour.'"
    => {"compatible": false, "reason": "Strong winds make a boat cruise uncomfortable and unsafe."}
    \"\"\".strip()


    class _CompatibilityDecision(BaseModel):
        compatible: bool
        reason: str


    def llm_check_weather_compatibility(
        activity: Activity, weather: Weather, model: str = REASONING_MODEL
    ) -> _CompatibilityDecision:
        \"\"\"Ask the LLM whether `activity` is compatible with `weather`.\"\"\"

        user_msg = json.dumps(
            {
                "activity": {
                    "name": activity.name,
                    "location": activity.location,
                    "description": activity.description,
                    "is_outdoor": activity.is_outdoor,
                },
                "weather": {
                    "condition": weather.condition.value,
                    "temperature_high_c": weather.temperature_high_c,
                    "temperature_low_c": weather.temperature_low_c,
                    "precipitation_chance": weather.precipitation_chance,
                    "description": weather.description,
                },
            },
            indent=2,
        )

        resp = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": ACTIVITY_AND_WEATHER_ARE_COMPATIBLE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format=_CompatibilityDecision,
            temperature=0.0,
        )
        return resp.choices[0].message.parsed


    def eval_weather_compatible(
        plan: TravelPlan,
        weather_forecast: List[Weather],
        available_activities: List[Activity],
    ) -> EvalResult:
        catalog = {a.activity_id: a for a in available_activities}
        weather_by_date = {w.date: w for w in weather_forecast}
        bad: List[str] = []
        for day in plan.days:
            weather = weather_by_date.get(day.date)
            if weather is None:
                bad.append(f"no weather data for {day.date}")
                continue
            for act in day.activities:
                catalog_act = catalog.get(act.activity_id)
                if catalog_act is None:
                    continue  # caught by eval_activities_exist
                decision = llm_check_weather_compatibility(catalog_act, weather)
                if not decision.compatible:
                    bad.append(
                        f"{day.date} {act.activity_id} ({act.name}): {decision.reason}"
                    )
        return EvalResult(
            name="weather_compatible",
            passed=not bad,
            message=(
                "All activities are weather-compatible."
                if not bad
                else "; ".join(bad)
            ),
        )
"""),

code("""
    def run_all_evals(
        plan: TravelPlan,
        vacation_info: VacationInfo,
        weather_forecast: List[Weather],
        available_activities: List[Activity],
        min_activities_per_day: int = 2,
        include_weather_llm_check: bool = True,
    ) -> List[EvalResult]:
        results = [
            eval_matches_city(plan, vacation_info),
            eval_matches_dates(plan, vacation_info),
            eval_activities_exist(plan, available_activities),
            eval_cost_correct(plan),
            eval_within_budget(plan, vacation_info),
            eval_min_activities_per_day(plan, min_activities_per_day),
        ]
        if include_weather_llm_check:
            results.append(
                eval_weather_compatible(plan, weather_forecast, available_activities)
            )
        return results
"""),

code("""
    initial_eval_results = run_all_evals(
        initial_plan,
        vacation_info,
        weather_forecast,
        available_activities,
    )
    for r in initial_eval_results:
        print(r)
"""),

# ---------------------------------------------------------------------------
md("""
    ## 5. Tools for the revision agent

    The revision agent can call four tools. Each tool is a thin Python wrapper
    that returns a string (or a stop signal). The tool registry collects
    metadata used to build the agent's system prompt dynamically.
"""),

code("""
    class ToolSpec(BaseModel):
        name: str
        description: str
        parameters: Dict[str, str]
        # `func` is stored separately because Pydantic does not love callables.

        model_config = {"arbitrary_types_allowed": True}


    TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}


    def register_tool(spec: ToolSpec, func: Callable[..., Any]) -> None:
        TOOL_REGISTRY[spec.name] = {"spec": spec, "func": func}


    # -- calculator_tool ----------------------------------------------------------
    def calculator_tool(expression: str) -> str:
        \"\"\"Evaluate a basic arithmetic expression and return the numeric result as a string.

        Args:
            expression: Arithmetic expression using digits, '+', '-', '*', '/',
                '%', '**', and parentheses. Example: '(85+30+25)*2 + 95'.

        Returns:
            The result as a string, e.g. '375.0'. Raises ValueError on bad input.
        \"\"\"

        return str(safe_calculator(expression))


    register_tool(
        ToolSpec(
            name="calculator_tool",
            description=(
                "Evaluate a basic arithmetic expression and return the numeric "
                "result as a string. Use this whenever you need to recompute "
                "totals (e.g. summing activity prices)."
            ),
            parameters={
                "expression": (
                    "str — arithmetic expression using digits, '+', '-', '*', "
                    "'/', '%', '**', and parentheses, e.g. '(85+30+25)*2 + 95'."
                ),
            },
        ),
        calculator_tool,
    )


    # -- get_activities_by_date_tool ---------------------------------------------
    def get_activities_by_date_tool(date_str: str) -> str:
        \"\"\"Return every activity offered in AgentsVille on a specific calendar date.

        Args:
            date_str (str): The target date in ISO format 'YYYY-MM-DD',
                e.g. '2025-06-13'. Must correspond to a real day during the
                trip's date window — out-of-range or malformed dates raise
                ValueError.

        Returns:
            str: A JSON-encoded list of activity objects. Each object has the
                following fields:
                    - activity_id (str): Stable catalog identifier, e.g. 'ACT-018'.
                    - name (str): Human-readable activity name.
                    - date (str): The activity's date in 'YYYY-MM-DD' format.
                    - start_time (str): 24-hour 'HH:MM' start time.
                    - end_time (str): 24-hour 'HH:MM' end time.
                    - location (str): Where the activity takes place.
                    - description (str): Short summary of the activity.
                    - price_usd (float): Price per person in USD.
                    - related_interests (list[str]): Tags this activity matches.
                    - is_outdoor (bool): True if primarily outdoors.

            If no activities are offered on the given date, returns '[]'.

        Use this tool to look up real activities before adding them to the
        itinerary. NEVER invent an activity_id — only use ids returned by this
        tool or already present in the current plan.
        \"\"\"

        try:
            target = date.fromisoformat(date_str)
        except ValueError as exc:
            raise ValueError(f"Invalid date {date_str!r}; expected 'YYYY-MM-DD'.") from exc
        acts = get_activities_by_date(target)
        return json.dumps([json.loads(a.model_dump_json()) for a in acts], indent=2)


    register_tool(
        ToolSpec(
            name="get_activities_by_date_tool",
            description=(
                "Return every activity offered in AgentsVille on a specific "
                "date as a JSON list of activity records (activity_id, name, "
                "date, start_time, end_time, location, description, price_usd, "
                "related_interests, is_outdoor)."
            ),
            parameters={
                "date_str": "str — ISO date 'YYYY-MM-DD', e.g. '2025-06-13'.",
            },
        ),
        get_activities_by_date_tool,
    )


    # -- run_evals_tool -----------------------------------------------------------
    def run_evals_tool(plan_json: str) -> str:
        \"\"\"Run the full evaluation suite on a candidate TravelPlan.

        Args:
            plan_json: JSON string conforming to the TravelPlan schema.

        Returns:
            A multi-line string listing PASS/FAIL for each evaluator.
        \"\"\"

        try:
            plan = TravelPlan.model_validate_json(plan_json)
        except Exception as exc:
            return f"INVALID_PLAN: TravelPlan failed to parse — {exc}"

        results = run_all_evals(
            plan,
            vacation_info,
            weather_forecast,
            available_activities,
        )
        lines = [str(r) for r in results]
        all_pass = all(r.passed for r in results)
        lines.append(f"\\nALL_PASS: {all_pass}")
        return "\\n".join(lines)


    register_tool(
        ToolSpec(
            name="run_evals_tool",
            description=(
                "Run the full evaluation suite on a candidate TravelPlan and "
                "return a PASS/FAIL summary for every check. Always run this "
                "tool BEFORE proposing changes and AGAIN before calling "
                "final_answer_tool."
            ),
            parameters={
                "plan_json": (
                    "str — a JSON string that validates against the TravelPlan "
                    "Pydantic schema."
                ),
            },
        ),
        run_evals_tool,
    )


    # -- final_answer_tool --------------------------------------------------------
    class _FinalAnswerSentinel(Exception):
        \"\"\"Raised internally to terminate the ReAct loop.\"\"\"

        def __init__(self, plan: TravelPlan):
            self.plan = plan


    def final_answer_tool(plan_json: str) -> str:
        \"\"\"Submit the final, revised TravelPlan and exit the ReAct loop.

        Args:
            plan_json: JSON string conforming to the TravelPlan schema.

        Returns:
            Never returns to the agent — the surrounding loop catches the
            sentinel and ends the conversation with this plan as the answer.
        \"\"\"

        plan = TravelPlan.model_validate_json(plan_json)
        raise _FinalAnswerSentinel(plan)


    register_tool(
        ToolSpec(
            name="final_answer_tool",
            description=(
                "Submit the final TravelPlan and end the ReAct loop. ONLY call "
                "this tool after run_evals_tool has reported ALL_PASS: True on "
                "the latest version of the plan."
            ),
            parameters={
                "plan_json": (
                    "str — a JSON string that validates against the TravelPlan "
                    "Pydantic schema. This is the final, revised plan."
                ),
            },
        ),
        final_answer_tool,
    )


    print("Registered tools:", list(TOOL_REGISTRY))
"""),

# ---------------------------------------------------------------------------
md("""
    ## 6. The `ItineraryRevisionAgent`

    A ReAct agent that loops THOUGHT → ACTION → OBSERVATION until it submits
    a final answer. The system prompt is built dynamically from the tool
    registry so that any new tool added above is automatically described.
"""),

code("""
    def _format_tools_block(registry: Dict[str, Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for name, entry in registry.items():
            spec: ToolSpec = entry["spec"]
            param_lines = "\\n".join(
                f"    - {pname}: {pdesc}" for pname, pdesc in spec.parameters.items()
            ) or "    (none)"
            blocks.append(
                f"- **{name}**\\n  Purpose: {spec.description}\\n  Parameters:\\n{param_lines}"
            )
        return "\\n".join(blocks)


    ITINERARY_REVISION_AGENT_SYSTEM_PROMPT = \"\"\"
    You are an **expert travel-planning revision agent** for trips to {city}.
    You receive an existing draft `TravelPlan`, a `VacationInfo`, and free-form
    feedback from the travelers. Your job is to revise the plan until **every
    evaluation passes** and the travelers' feedback is satisfied — in
    particular, every day must contain at least 2 activities.

    # OPERATING PROTOCOL — THINK / ACT / OBSERVE
    You operate inside a loop. On every step you MUST emit ONE message that
    contains BOTH of the following sections, in this order, and nothing else:

    THOUGHT: <your private reasoning — what you know, what you still need,
              which tool you will call next and why>
    ACTION: <a single JSON object exactly of the form
              {"tool_name": "<tool_name>", "arguments": {"arg1": "value1", ...}}>

    The runtime will execute the tool described in your ACTION and reply with
    a single OBSERVATION message containing the tool's output as a string. You
    then produce another THOUGHT/ACTION pair.

    Strict rules:
      - THOUGHT and ACTION must BOTH appear on every step. Never omit either.
      - The ACTION MUST be a single, valid JSON object on its own line(s).
        Do not wrap it in code fences. Do not include trailing commentary.
      - Use only the tools listed below; never invent tool names.

    # AVAILABLE TOOLS
    {tools_block}

    # WORKFLOW YOU MUST FOLLOW
    1. **Diagnose first.** Begin by calling `run_evals_tool` on the CURRENT
       draft plan to see exactly which checks are failing. Read the output
       carefully — it tells you what to fix.
    2. **Look up real activities.** When you decide to add or replace an
       activity on a given day, call `get_activities_by_date_tool` for that
       date and pick a real `activity_id` from the result. NEVER invent
       activity_ids.
    3. **Recompute totals.** Use `calculator_tool` to recompute `total_cost_usd`
       whenever you change the activity list, so the new value is exact.
    4. **Re-evaluate.** Once you have a revised plan that you believe is
       correct, call `run_evals_tool` AGAIN on the revised plan. You MUST run
       `run_evals_tool` a second time before finalizing — this is non-negotiable.
    5. **Finalize.** Only after `run_evals_tool` reports `ALL_PASS: True` may
       you call `final_answer_tool` with the final plan JSON. Calling
       `final_answer_tool` is the ONLY way to exit the loop — never stop
       on your own.

    # FOCUS
    - Honor the travelers' feedback. The travelers said: "{traveler_feedback}".
    - Keep the city, dates and travelers identical to the original VacationInfo.
    - Stay within the budget. Prefer cheaper alternatives if needed.
    - Keep activities weather-appropriate (avoid outdoor activities on stormy
      or heavily rainy days).

    # TRAVELPLAN SCHEMA
    Every plan you produce — including the one passed to `final_answer_tool` —
    must conform to this JSON Schema:

    ```
    {travel_plan_schema}
    ```
    \"\"\".strip()


    rendered_revision_system_prompt = (
        ITINERARY_REVISION_AGENT_SYSTEM_PROMPT
        .replace("{city}", vacation_info.destination)
        .replace("{tools_block}", _format_tools_block(TOOL_REGISTRY))
        .replace("{traveler_feedback}", vacation_info.traveler_feedback or "(none)")
        .replace("{travel_plan_schema}", TRAVEL_PLAN_SCHEMA)
    )

    print(rendered_revision_system_prompt[:1800])
    print("...")
"""),

code("""
    THOUGHT_RE = re.compile(r"THOUGHT\\s*:\\s*(.+?)(?=ACTION\\s*:)", re.DOTALL | re.IGNORECASE)
    ACTION_RE = re.compile(r"ACTION\\s*:\\s*(\\{.*\\})\\s*$", re.DOTALL | re.IGNORECASE)


    class _ParseError(ValueError):
        pass


    def _parse_react_message(msg: str) -> tuple[str, dict]:
        \"\"\"Pull the THOUGHT prose and ACTION JSON from a single LLM message.\"\"\"

        thought_match = THOUGHT_RE.search(msg)
        action_match = ACTION_RE.search(msg)
        if not thought_match or not action_match:
            raise _ParseError(
                "Message must contain both THOUGHT: and ACTION: sections, "
                "with ACTION: followed by a single JSON object."
            )
        thought = thought_match.group(1).strip()
        try:
            action = json.loads(action_match.group(1))
        except json.JSONDecodeError as exc:
            raise _ParseError(f"ACTION JSON failed to parse: {exc}") from exc
        if not isinstance(action, dict) or "tool_name" not in action or "arguments" not in action:
            raise _ParseError(
                'ACTION must be {"tool_name": "...", "arguments": {...}}.'
            )
        return thought, action


    def run_revision_agent(
        initial_plan: TravelPlan,
        vacation_info: VacationInfo,
        max_steps: int = 12,
        model: str = REASONING_MODEL,
        verbose: bool = True,
    ) -> TravelPlan:
        \"\"\"Run the ReAct loop until `final_answer_tool` is invoked.\"\"\"

        user_intro = (
            "Here is the current state. Begin the THINK/ACT/OBSERVE loop now.\\n\\n"
            + json.dumps(
                {
                    "vacation_info": json.loads(vacation_info.model_dump_json()),
                    "current_plan": json.loads(initial_plan.model_dump_json()),
                    "traveler_feedback": vacation_info.traveler_feedback,
                },
                indent=2,
            )
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": rendered_revision_system_prompt},
            {"role": "user", "content": user_intro},
        ]

        for step in range(1, max_steps + 1):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            assistant_msg = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": assistant_msg})

            if verbose:
                print(f"\\n=== Step {step} — assistant ===")
                print(assistant_msg)

            try:
                thought, action = _parse_react_message(assistant_msg)
            except _ParseError as exc:
                obs = f"PARSE_ERROR: {exc}\\nReply with both THOUGHT: and ACTION: sections."
                messages.append({"role": "user", "content": f"OBSERVATION: {obs}"})
                if verbose:
                    print(f"OBSERVATION: {obs}")
                continue

            tool_name = action["tool_name"]
            args = action.get("arguments", {}) or {}
            entry = TOOL_REGISTRY.get(tool_name)
            if entry is None:
                obs = f"UNKNOWN_TOOL: {tool_name!r}. Choose from {list(TOOL_REGISTRY)}."
                messages.append({"role": "user", "content": f"OBSERVATION: {obs}"})
                if verbose:
                    print(f"OBSERVATION: {obs}")
                continue

            try:
                result = entry["func"](**args)
            except _FinalAnswerSentinel as final:
                if verbose:
                    print("\\n=== final_answer_tool invoked — exiting loop ===")
                return final.plan
            except TypeError as exc:
                obs = f"BAD_ARGUMENTS for {tool_name}: {exc}"
            except Exception as exc:  # noqa: BLE001
                obs = f"TOOL_ERROR ({tool_name}): {exc}"
            else:
                obs = str(result)

            if verbose:
                preview = obs if len(obs) < 800 else obs[:800] + "... [truncated]"
                print(f"OBSERVATION: {preview}")
            messages.append({"role": "user", "content": f"OBSERVATION: {obs}"})

        raise RuntimeError(
            f"ItineraryRevisionAgent exceeded max_steps={max_steps} without "
            "calling final_answer_tool."
        )
"""),

code("""
    revised_plan = run_revision_agent(
        initial_plan=initial_plan,
        vacation_info=vacation_info,
        max_steps=14,
        verbose=True,
    )

    print("\\n=== Final revised plan ===")
    print(revised_plan.model_dump_json(indent=2))

    print("\\n=== Final eval results ===")
    for r in run_all_evals(
        revised_plan, vacation_info, weather_forecast, available_activities
    ):
        print(r)
"""),

# ---------------------------------------------------------------------------
md("""
    ## 7. A fun narrative summary

    To wrap things up we ask the LLM to turn the final structured plan into a
    short, narrative travel-magazine-style write-up the travelers can read
    before the trip.
"""),

code("""
    NARRATIVE_SYSTEM_PROMPT = (
        "You are a travel writer. Given a structured TravelPlan and the names "
        "of the travelers, write a fun, vivid 2–3 paragraph narrative summary "
        "of the trip in second person ('you and ...'). Mention specific "
        "activities, weather notes, and the overall arc of the trip. Keep it "
        "under 250 words. Do not output JSON."
    )


    narrative_response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[
            {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "travelers": [t.name for t in vacation_info.travelers],
                        "plan": json.loads(revised_plan.model_dump_json()),
                    },
                    indent=2,
                ),
            },
        ],
        temperature=0.7,
    )

    print(narrative_response.choices[0].message.content)
"""),

# ---------------------------------------------------------------------------
md("""
    ## Done!

    You now have:

    - A typed `VacationInfo` describing the trip.
    - A draft `TravelPlan` from the one-shot `ItineraryAgent`.
    - A revised `TravelPlan` from the ReAct `ItineraryRevisionAgent` that
      passes every programmatic and LLM-powered evaluation.
    - A fun narrative summary to share with the travelers.

    To plan a different trip, edit `vacation_info` at the top and re-run the
    notebook from top to bottom.
"""),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "codemirror_mode": {"name": "ipython", "version": 3},
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).parent / "project_starter.ipynb"
    out.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
