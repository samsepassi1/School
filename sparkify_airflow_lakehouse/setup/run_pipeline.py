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
        return {"data_interval": "interval_1"}

    request_pipeline_run()


run_pipeline()
