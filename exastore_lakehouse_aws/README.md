# Exastore Data Lakehouse on AWS

**Author: Sam Sepassi**

Udacity Data Engineering with AWS — Course 3 project submission.

## Project Summary

This project implements an Apache Iceberg lakehouse for Exastore using a medallion architecture:

- **Bronze**: CDC records for orders, users, products plus clickstream events
- **Silver**: enriched `order_details`, `enriched_events`, and `product_performance`
- **Gold**: `customer_analytics` and `realtime_metrics`

The repository includes Glue job scripts, an orchestration notebook, validation SQL, and documentation for Iceberg features including snapshots and time-travel queries.

## Files

- `project.ipynb` — completed project notebook with validation outputs and final checklist
- `glue-jobs/cdc-batch/src/iceberg-cdc-bronze-layer.py` — CDC MERGE INTO bronze job
- `glue-jobs/events/src/iceberg-events-bronze-layer.py` — clickstream bronze append job with DQ checks
- `glue-jobs/silver/src/iceberg-silver-layer.py` — silver S3 Tables job
- `glue-jobs/gold/src/iceberg-gold-layer.py` — gold S3 Tables aggregation job
- `validation/athena_validation.sql` — Athena validation queries

## Reviewer Notes

All author fields name only **Sam Sepassi**. The implementation follows the rubric requirements for Iceberg table format version 2, CDC deduplication, MERGE INTO, S3 Tables catalog registration, silver/gold namespace creation through Glue API, validation queries, and Iceberg snapshot/time-travel demonstration.
