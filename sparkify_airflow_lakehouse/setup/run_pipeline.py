from airflow.decorators import dag, task
from airflow.datasets import Dataset
import pendulum

PIPELINE_REQUESTED = Dataset("s3://sparkify/pipeline_requested")


@dag(
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
)
def run_pipeline():
    @task(outlets=[PIPELINE_REQUESTED])
    def request_pipeline_run():
        pass

    request_pipeline_run()


run_pipeline()
