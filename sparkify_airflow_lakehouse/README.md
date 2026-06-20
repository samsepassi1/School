# AWS Data Lakehouse Pipeline for Sparkify

**Author:** Sam Sepassi  
**Course:** Udacity Data Engineering with AWS Nanodegree

---

## ⚠️ Reviewer Note

> **Please review this project located in `sparkify_airflow_lakehouse/`.**
>
> - **Setup DAG:** `sparkify_airflow_lakehouse/setup/run_pipeline.py` (original starter — not modified)
> - **Raw DAG:** `sparkify_airflow_lakehouse/raw/dag.py`
> - **Transactions DAG:** `sparkify_airflow_lakehouse/transactions/dag.py`
> - **Analytics DAG:** `sparkify_airflow_lakehouse/analytics/dag.py`
> - **Validation SQL:** `sparkify_airflow_lakehouse/validation/athena_checks.sql`

---

## Architecture

```
setup/run_pipeline.py
    │ emits Dataset("s3://sparkify/pipeline_requested")
    ▼
raw/dag.py
    │ ingests S3 landing → Iceberg raw layer (Glue)
    │ emits Dataset("s3://sparkify/raw_complete")
    ▼
transactions/dag.py
    │ transforms raw → transactions layer (Iceberg, idempotent)
    │ emits Dataset("s3://sparkify/transactions_complete")
    ▼
analytics/dag.py
    │ builds analytics marts (Iceberg, partition overwrite)
    │ SQL validation via athena_checks.sql
```

---

## DAGs

| DAG | Schedule | Purpose |
|-----|----------|---------|
| `run_pipeline` | Manual (schedule=None) | Emits pipeline_requested asset |
| `raw` | Asset-triggered | Discovers & ingests landing tables into Iceberg raw |
| `transactions` | Asset-triggered | Cleans & conforms raw → transactions layer |
| `analytics` | Asset-triggered | Builds analytics marts from transactions |

---

## Key Design Decisions

- **Event-driven:** All DAGs trigger via Airflow Dataset (Asset) events, not cron
- **Dynamic table discovery:** `raw/dag.py` inspects S3 at runtime — no hardcoded table names
- **Idempotent overwrites:** Iceberg partition overwrite per data_interval
- **SQL validation:** Athena checks run before emitting downstream assets
- **Original setup DAG preserved:** `setup/run_pipeline.py` is the unmodified starter file

---

## Project Structure

```
sparkify_airflow_lakehouse/
├── setup/
│   └── run_pipeline.py          ← original starter (not modified)
├── raw/
│   ├── dag.py
│   └── glue_script.py
├── transactions/
│   ├── dag.py
│   ├── glue_script.py
│   └── sql/
│       ├── artists.sql
│       ├── events.sql
│       ├── song_versions.sql
│       ├── songs.sql
│       ├── user_levels.sql
│       └── users.sql
├── analytics/
│   ├── dag.py
│   ├── glue_script.py
│   └── sql/
│       ├── artist_popularity.sql
│       ├── songplay_facts.sql
│       └── user_activity_daily.sql
├── validation/
│   └── athena_checks.sql
└── README.md                    ← this file
```
