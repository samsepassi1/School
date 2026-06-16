# Employee Events Dashboard

A FastHTML data dashboard for monitoring employee and team productivity and predicting recruitment risk.

## Overview

Built for a manufacturing company's data science team, this dashboard allows managers to:
- Monitor an individual employee's or a team's cumulative positive/negative performance events
- View the predicted likelihood of an employee (or team average) being recruited by a competitor
- Browse manager notes for the selected employee or team

## Tech Stack

- **FastHTML** — Python-native web framework for the dashboard UI
- **SQLite** (`employee_events.db`) — stores performance events, notes, employees, teams
- **scikit-learn** — pre-trained model (`assets/model.pkl`) for recruitment risk prediction
- **Matplotlib** — embedded charts (line chart + bar chart) rendered as base64 PNGs
- **pytest** — database existence and schema tests with GitHub Actions CI

## Repository Structure

```
├── assets/
│   ├── model.pkl          # Pre-trained sklearn recruitment risk model
│   └── report.css         # Dashboard stylesheet
├── python-package/
│   ├── employee_events/
│   │   ├── __init__.py
│   │   ├── sql_execution.py   # QueryMixin — SQLite connection handling
│   │   ├── query_base.py      # QueryBase — shared SQL queries
│   │   ├── employee.py        # Employee subclass
│   │   ├── team.py            # Team subclass
│   │   └── employee_events.db # SQLite database
│   └── setup.py
├── report/
│   ├── base_components/       # BaseComponent, Dropdown, Radio, MatplotlibViz, DataTable
│   ├── combined_components/   # CombinedComponent, FormGroup
│   ├── dashboard.py           # FastHTML app + all dashboard subclasses + routes
│   └── utils.py               # load_model() helper
├── tests/
│   └── test_employee_events.py # pytest: DB exists + 3 table checks
├── requirements.txt
└── .github/workflows/test.yml  # CI: runs pytest on every push to main
```

## Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/samsepassi1/School.git
cd School/employee-dashboard

# 2. Install all dependencies (includes the python package)
pip install -r requirements.txt

# 3. Run the dashboard
cd report
python dashboard.py
# Open http://localhost:5001
```

## Running Tests

```bash
pytest tests/ -v
```

## Libraries Used

| Library | Version | Purpose |
|---|---|---|
| python-fasthtml | 0.8.0 | Web framework / HTML generation |
| scikit-learn | 1.5.2 | ML model for recruitment risk |
| matplotlib | 3.9.2 | Embedded chart visualizations |
| pandas | 2.2.3 | SQL query results / data frames |
| numpy | 2.1.2 | Numerical operations |
| scipy | 1.14.1 | ML dependency |
| pytest | latest | Database and schema tests |

## Dashboard Features

- **Employee view** — cumulative events line chart + recruitment risk bar chart + notes table
- **Team view** — same charts aggregated across all team members
- **Radio toggle** — switch between Employee / Team filter in one click
- **Dropdown** — select any employee or team from the database
- **Predicted recruitment risk** — color-coded horizontal bar (0–1 probability)

## Acknowledgements

- Udacity Data Scientist Nanodegree — Dashboard Project
- SQLite `employee_events.db` schema designed by the Udacity data team
