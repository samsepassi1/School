import sys,boto3
from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession
args=getResolvedOptions(sys.argv,['JOB_NAME','ACCOUNT_ID','TABLE_BUCKET_NAME'])
ACCOUNT_ID=args['ACCOUNT_ID']; TABLE_BUCKET_NAME=args['TABLE_BUCKET_NAME']; NS='ecommerce_gold'
spark=(SparkSession.builder.appName(args['JOB_NAME'])
 .config('spark.sql.catalog.s3tablescatalog','org.apache.iceberg.spark.SparkCatalog')
 .config('spark.sql.catalog.s3tablescatalog.catalog-impl','org.apache.iceberg.aws.glue.GlueCatalog')
 .config('spark.sql.catalog.s3tablescatalog.glue.id',f'{ACCOUNT_ID}:s3tablescatalog/{TABLE_BUCKET_NAME}')
 .config('spark.sql.catalog.s3tablescatalog.io-impl','org.apache.iceberg.aws.s3.S3FileIO')
 .config('spark.sql.extensions','org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions').getOrCreate())
try: boto3.client('glue').create_database(CatalogId=f'{ACCOUNT_ID}:s3tablescatalog/{TABLE_BUCKET_NAME}',DatabaseInput={'Name':NS})
except Exception as e: print('namespace exists or could not be created:',e)
def create_gold_table(name, query):
    df=spark.sql(query); fqn=f's3tablescatalog.{NS}.{name}'
    spark.sql(f'DROP TABLE IF EXISTS {fqn} PURGE')
    df.writeTo(fqn).using('iceberg').tableProperty('format-version','2').create()
create_gold_table('customer_analytics',"""SELECT date_trunc('month',created_at) month,country,customer_segment,count(distinct user_id) customers,count(distinct order_id) orders,sum(order_total) revenue,avg(order_total) avg_order_value FROM s3tablescatalog.ecommerce_silver.order_details GROUP BY 1,2,3""")
create_gold_table('realtime_metrics',"""SELECT date_trunc('hour',event_ts) event_hour,traffic_source,event_type,count(*) event_count,count(distinct user_id) users,sum(revenue) revenue FROM s3tablescatalog.ecommerce_silver.enriched_events GROUP BY 1,2,3""")
