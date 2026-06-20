# AWS Data Lakehouse Pipeline for Sparkify

**Author:** Sam Sepassi  
**Course:** Udacity Data Engineering with AWS Nanodegree

---

## ⚠️ Reviewer Note

> **Please review this project located in `sparkify_airflow_lakehouse/`.**
>
> - **Setup DAG:** `setup/run_pipeline.py`
> - **Raw DAG:** `raw/dag.py` (includes SQLCheckOperator)
> - **Transactions DAG:** `transactions/dag.py` (includes SQLCheckOperator)
> - **Analytics DAG:** `analytics/dag.py`
> - **Validation SQL:** `validation/athena_checks.sql`

---

## Architecture

```
setup/run_pipeline.py
    │ emits Dataset("s3://sparkify/pipeline_requested") with metadata
    ▼
raw/dag.py
    │ discovers S3 landing tables → Glue job ingests to Iceberg "raw" database
    │ SQLCheckOperator verifies raw.logs is non-empty
    │ emits Dataset("s3://sparkify/raw_complete") with metadata
    ▼
transactions/dag.py
    │ promotes raw → transactions in dependency order:
    │   artists → users → song_versions → songs → user_levels → events
    │ Each table deduplicated on its explicit primary key
    │ SQLCheckOperator verifies events have no duplicate event_id
    │ SQLCheckOperator verifies users table is non-empty
    │ emits Dataset("s3://sparkify/transactions_complete") with metadata
    ▼
analytics/dag.py
    │ builds analytics snapshots using PySpark DataFrame API (NO SQL)
    │   songplay_facts, user_activity_daily, artist_popularity
    │ full overwrite (DROP + create) — no append/insert
    │ emits Dataset("s3://sparkify/analytics_complete")
```

---

## DAGs

| DAG | Schedule | Purpose |
|-----|----------|---------|
| `run_pipeline` | Manual (schedule=None) | Emits pipeline_requested asset with metadata |
| `raw` | Asset-triggered (pipeline_requested) | Discovers & ingests landing tables into Iceberg raw layer |
| `transactions` | Asset-triggered (raw_complete) | Normalizes raw → transactions layer with deduplication |
| `analytics` | Asset-triggered (transactions_complete) | Builds analytics marts as full snapshots (PySpark DataFrames) |

---

## Athena Database Names

Per rubric requirements, the Athena/Glue databases are named:

| Database | Tables |
|----------|--------|
| `raw` | `logs`, `songs` |
| `transactions` | `events`, `users`, `artists`, `songs`, `song_versions`, `user_levels` |
| `analytics` | `songplay_facts`, `user_activity_daily`, `artist_popularity` |

---

## Key Design Decisions

- **Event-driven:** All DAGs trigger via Airflow Dataset (Asset) events, not cron
- **Dynamic table discovery:** `raw/dag.py` inspects S3 at runtime
- **Glue job arguments:** All GlueJobOperator calls pass `--BUCKET`, `--DATA_INTERVAL`, `--SQL_FILE`, and `--TABLE_NAME`
- **Table-specific primary keys:** Each transactions table deduplicates on its explicit PK (event_id, user_id, song_id, artist_id) — not generic _id matching
- **event_id generation:** `concat_ws('-', ts, userId, sessionId, page)` creates a stable unique key per event
- **Analytics without SQL:** `analytics/glue_script.py` uses pure PySpark DataFrame API (filter, join, groupBy, agg) — no `spark.sql()` calls
- **Full snapshot overwrites:** Analytics tables are dropped and recreated each run (no append/insert)
- **SQL Check operators:** `SQLCheckOperator` in raw and transactions DAGs validates data quality at runtime
- **Original setup DAG preserved:** `setup/run_pipeline.py` is the starter file (not modified beyond returning metadata)

---

## Project Structure

```
sparkify_airflow_lakehouse/
├── setup/
│   └── run_pipeline.py              ← pipeline trigger (emits pipeline_requested asset)
├── raw/
│   ├── dag.py                       ← raw ingestion DAG + SQLCheckOperator
│   └── glue_script.py               ← Glue job: S3 JSON → Iceberg raw tables
├── transactions/
│   ├── dag.py                       ← transactions promotion DAG + SQLCheckOperator
│   ├── glue_script.py               ← Glue job: SQL-based transformation + deduplication
│   └── sql/
│       ├── artists.sql
│       ├── events.sql               ← includes event_id (stable primary key)
│       ├── song_versions.sql
│       ├── songs.sql
│       ├── user_levels.sql
│       └── users.sql
├── analytics/
│   ├── dag.py                       ← analytics snapshot DAG
│   ├── glue_script.py               ← Glue job: PySpark DataFrame API (NO SQL)
│   └── sql/
│       ├── artist_popularity.sql    ← reference SQL (not executed by Glue)
│       ├── songplay_facts.sql       ← reference SQL (not executed by Glue)
│       └── user_activity_daily.sql  ← reference SQL (not executed by Glue)
├── validation/
│   └── athena_checks.sql            ← Athena validation queries
└── README.md                        ← this file
```
