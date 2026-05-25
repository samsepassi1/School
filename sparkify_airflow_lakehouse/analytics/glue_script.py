import sys,boto3
from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession
args=getResolvedOptions(sys.argv,['BUCKET','SQL_FILE','TABLE_NAME'])
spark=SparkSession.builder.appName('sparkify-analytics').config('spark.sql.extensions','org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions').getOrCreate()
spark.sql('CREATE DATABASE IF NOT EXISTS sparkify_analytics')
s3=boto3.client('s3'); sql=s3.get_object(Bucket=args['BUCKET'],Key=f"analytics/{args['SQL_FILE']}")['Body'].read().decode()
df=spark.sql(sql)
target=f"sparkify_analytics.{args['TABLE_NAME']}"
spark.sql(f'DROP TABLE IF EXISTS {target} PURGE')
df.writeTo(target).using('iceberg').tableProperty('format-version','2').create()
