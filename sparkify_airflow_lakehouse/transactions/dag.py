from __future__ import annotations
import pendulum
from airflow import DAG
from airflow.decorators import task
from airflow.datasets import Dataset as Asset
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

AWS_CONN_ID = 'aws_default'
S3_BUCKET_VAR = 'sparkify_bucket'
LANDING_ROOT = 'landing'
SCRIPTS_ROOT = 'glue-scripts'
RAW_ASSET = Asset('s3://sparkify/raw_complete')
TRANSACTIONS_ASSET = Asset('s3://sparkify/transactions_complete')
ANALYTICS_ASSET = Asset('s3://sparkify/analytics_complete')

def interval_from_context(**context):
    events = context.get('triggering_dataset_events') or context.get('triggering_asset_events') or {}
    for evs in events.values():
        if evs:
            return evs[-1].extra.get('data_interval', context['params'].get('data_interval','interval_1'))
    return context['params'].get('data_interval','interval_1')

with DAG('transactions', start_date=pendulum.datetime(2025,1,1,tz='UTC'), schedule=[RAW_ASSET], catchup=False, max_active_runs=1, max_active_tasks=2, doc_md='Transactions layer normalizes raw Sparkify data.') as dag:
    SQL_ORDER=['artists','users','song_versions','songs','user_levels','events']
    @task
    def selected_interval(**context): return interval_from_context(**context)
    interval=selected_interval()
    prev=None
    for name in SQL_ORDER:
        op=GlueJobOperator(task_id=f'promote_{name}', job_name='sparkify-transactions', script_location='s3://{{ var.value.sparkify_bucket }}/glue-scripts/transactions/glue_script.py', iam_role_name='GlueServiceRole', aws_conn_id=AWS_CONN_ID, script_args={'--DATA_INTERVAL':'{{ ti.xcom_pull(task_ids="selected_interval") }}','--SQL_FILE':f'sql/{name}.sql','--TABLE_NAME':name}, wait_for_completion=True)
        if prev: prev >> op
        else: interval >> op
        prev=op
    @task(outlets=[TRANSACTIONS_ASSET])
    def emit_transactions(data_interval): return {'data_interval': data_interval, 'tables': SQL_ORDER}
    prev >> emit_transactions(interval)
