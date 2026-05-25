import sys,boto3
from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession
args=getResolvedOptions(sys.argv,['JOB_NAME','ACCOUNT_ID','TABLE_BUCKET_NAME','BRONZE_DB'])
ACCOUNT_ID=args['ACCOUNT_ID']; TABLE_BUCKET_NAME=args['TABLE_BUCKET_NAME']; BRONZE_DB=args['BRONZE_DB']; NS='ecommerce_silver'
spark=(SparkSession.builder.appName(args['JOB_NAME'])
 .config('spark.sql.catalog.s3tablescatalog','org.apache.iceberg.spark.SparkCatalog')
 .config('spark.sql.catalog.s3tablescatalog.catalog-impl','org.apache.iceberg.aws.glue.GlueCatalog')
 .config('spark.sql.catalog.s3tablescatalog.glue.id',f'{ACCOUNT_ID}:s3tablescatalog/{TABLE_BUCKET_NAME}')
 .config('spark.sql.catalog.s3tablescatalog.io-impl','org.apache.iceberg.aws.s3.S3FileIO')
 .config('spark.sql.catalog.glue_catalog','org.apache.iceberg.spark.SparkCatalog')
 .config('spark.sql.catalog.glue_catalog.catalog-impl','org.apache.iceberg.aws.glue.GlueCatalog')
 .config('spark.sql.extensions','org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions').getOrCreate())
try: boto3.client('glue').create_database(CatalogId=f'{ACCOUNT_ID}:s3tablescatalog/{TABLE_BUCKET_NAME}',DatabaseInput={'Name':NS})
except Exception as e: print('namespace exists or could not be created:',e)
def create_silver_table(name, query):
    df=spark.sql(query); fqn=f's3tablescatalog.{NS}.{name}'
    spark.sql(f'DROP TABLE IF EXISTS {fqn} PURGE')
    df.writeTo(fqn).using('iceberg').tableProperty('format-version','2').create()
create_silver_table('order_details',f"""SELECT o.order_id,o.user_id,u.email,u.country,o.status,o.order_total,o.created_at, row_number() over(partition by o.user_id order by o.created_at) order_sequence, sum(o.order_total) over(partition by o.user_id order by o.created_at rows between unbounded preceding and current row) lifetime_value, case when sum(o.order_total) over(partition by o.user_id) >= 1000 then 'VIP' when count(*) over(partition by o.user_id)>=5 then 'Loyal' when count(*) over(partition by o.user_id)>=2 then 'Repeat' else 'New' end customer_segment FROM glue_catalog.{BRONZE_DB}.orders o JOIN glue_catalog.{BRONZE_DB}.users u ON o.user_id=u.id WHERE o.op<>'d'""")
create_silver_table('enriched_events',f"""SELECT e.*,u.country,p.name product_name,p.category,case when e.event_type='purchase' then coalesce(e.price,p.price,0) else 0 end revenue,row_number() over(partition by e.session_id order by e.event_ts) session_sequence,hour(e.event_ts) event_hour,date(e.event_ts) event_date FROM glue_catalog.{BRONZE_DB}.events e LEFT JOIN glue_catalog.{BRONZE_DB}.users u ON e.user_id=u.id LEFT JOIN glue_catalog.{BRONZE_DB}.products p ON e.product_id=p.id""")
create_silver_table('product_performance',f"""SELECT p.id product_id,p.name,p.category,count(case when e.event_type='page_view' then 1 end) page_views,count(case when e.event_type='purchase' then 1 end) purchases,cast(count(case when e.event_type='purchase' then 1 end) as double)/nullif(count(case when e.event_type='page_view' then 1 end),0) conversion_rate FROM glue_catalog.{BRONZE_DB}.products p LEFT JOIN glue_catalog.{BRONZE_DB}.events e ON p.id=e.product_id GROUP BY p.id,p.name,p.category""")
