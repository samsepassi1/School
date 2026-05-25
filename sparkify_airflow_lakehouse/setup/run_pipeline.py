from airflow import DAG
from airflow.decorators import task
from airflow.datasets import Dataset as Asset
import pendulum
PIPELINE_REQUESTED = Asset('s3://sparkify/pipeline_requested')
with DAG('run_pipeline', start_date=pendulum.datetime(2025,1,1,tz='UTC'), schedule=None, catchup=False, params={'data_interval':'interval_1'}, max_active_runs=1, max_active_tasks=2) as dag:
    @task(outlets=[PIPELINE_REQUESTED])
    def emit_request(**context):
        interval=context['params']['data_interval']
        return {'data_interval': interval, 'tables': ['logs','songs']}
    emit_request()
