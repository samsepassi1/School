import sys
from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession
args=getResolvedOptions(sys.argv,['JOB_NAME','BUCKET','BRONZE_DB'])
BUCKET=args['BUCKET']; BRONZE_DB=args['BRONZE_DB']
spark=(SparkSession.builder.appName(args['JOB_NAME'])
 .config('spark.sql.catalog.glue_catalog','org.apache.iceberg.spark.SparkCatalog')
 .config('spark.sql.catalog.glue_catalog.catalog-impl','org.apache.iceberg.aws.glue.GlueCatalog')
 .config('spark.sql.catalog.glue_catalog.warehouse',f's3://{BUCKET}/bronze/warehouse')
 .config('spark.sql.catalog.glue_catalog.io-impl','org.apache.iceberg.aws.s3.S3FileIO')
 .config('spark.sql.extensions','org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions').getOrCreate())
spark.sql(f'CREATE DATABASE IF NOT EXISTS glue_catalog.{BRONZE_DB}')
spark.sql(f"""CREATE TABLE IF NOT EXISTS glue_catalog.{BRONZE_DB}.events (
 event_id STRING,user_id STRING,product_id STRING,session_id STRING,event_type STRING,event_ts TIMESTAMP,
 traffic_source STRING,device_type STRING,price DECIMAL(10,2),year INT,month INT,day INT,_processed_at TIMESTAMP)
USING iceberg PARTITIONED BY (year, month) LOCATION 's3://{BUCKET}/bronze/events'
TBLPROPERTIES ('format-version'='2','write.format.default'='parquet','write.parquet.compression-codec'='snappy')""")
spark.read.json(f's3://{BUCKET}/datasets/event_data/small/events_batch.json').createOrReplaceTempView('raw_events')
spark.sql("""CREATE OR REPLACE TEMP VIEW parsed_events AS SELECT cast(event_id as string) event_id, cast(user_id as string) user_id, cast(product_id as string) product_id, cast(session_id as string) session_id, cast(event_type as string) event_type, to_timestamp(event_timestamp) event_ts, cast(traffic_source as string) traffic_source, cast(device_type as string) device_type, cast(price as decimal(10,2)) price, year(to_timestamp(event_timestamp)) year, month(to_timestamp(event_timestamp)) month, day(to_timestamp(event_timestamp)) day, current_timestamp() _processed_at FROM raw_events""")
spark.sql("""SELECT count(*) total, sum(case when event_type not in ('page_view','add_to_cart','purchase','click') then 1 else 0 end) invalid_types, sum(case when event_ts > current_timestamp() then 1 else 0 end) future_events, sum(case when user_id is null then 1 else 0 end) null_users FROM parsed_events""").show(truncate=False)
spark.sql("""CREATE OR REPLACE TEMP VIEW valid_events AS SELECT * FROM parsed_events WHERE event_type in ('page_view','add_to_cart','purchase','click') AND event_ts <= current_timestamp() AND user_id IS NOT NULL""")
spark.sql(f"INSERT INTO glue_catalog.{BRONZE_DB}.events SELECT * FROM valid_events")
spark.sql(f"SELECT event_type,count(*) FROM glue_catalog.{BRONZE_DB}.events GROUP BY event_type").show()
