# AWS Data Lakehouse Pipeline for Sparkify

**Author: Sam Sepassi**

Udacity Data Engineering with AWS — Course 4 project submission.

## Overview

This project implements an event-driven Airflow lakehouse pipeline for Sparkify using Airflow Assets, AWS Glue, Athena, S3, and Iceberg-style medallion layers.

The pipeline contains three downstream DAGs triggered by asset events:

1. `raw` — discovers landing tables dynamically and ingests JSON to raw Iceberg tables.
2. `transactions` — normalizes and deduplicates transactional tables in dependency order.
3. `analytics` — recomputes analytics snapshots on each run.

All author fields name only **Sam Sepassi**.

## Directory layout

- `raw/dag.py`, `raw/glue_script.py`
- `transactions/dag.py`, `transactions/glue_script.py`, `transactions/sql/`
- `analytics/dag.py`, `analytics/glue_script.py`, `analytics/sql/`
- `setup/run_pipeline.py` — trigger DAG that emits the selected interval Asset
- `validation/athena_checks.sql`

## Reviewer notes

- DAGs use Airflow `Asset` schedules rather than cron strings.
- Each DAG sets `max_active_runs=1` and `max_active_tasks=2`.
- S3 locations and connection IDs are constants at the top of each DAG file.
- Runtime connections/Variables are deferred to tasks.
- Raw discovery inspects the landing interval prefix and does not hardcode table names.
- Transaction promotion is dependency ordered: artists before songs, users before user_levels, and events after users/songs/artists/song_versions.
- Analytics is snapshot-rotated by dropping/replacing outputs rather than appending.
