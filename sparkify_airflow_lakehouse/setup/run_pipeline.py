from __future__ import annotations
from airflow.decorators import dag, task
from airflow.datasets import Dataset
import pendulum


@dag(
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
)
def run_pipeline():
    @task(outlets=[Dataset("s3://sparkify/pipeline_requested")])
    def request_pipeline_run():
        """
        Trigger the lakehouse pipeline by emitting the pipeline_requested asset.

        The asset metadata includes the data_interval so downstream DAGs
        (raw, transactions, analytics) can read it from the triggering event.
        """
        return {
            "data_interval": "interval_1",
            "pipeline": "sparkify_lakehouse",
        }

    request_pipeline_run()


run_pipeline()
