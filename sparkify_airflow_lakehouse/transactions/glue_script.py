import sys,boto3
from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession
args=getResolvedOptions(sys.argv,['BUCKET','DATA_INTERVAL','SQL_FILE','TABLE_NAME'])
spark=SparkSession.builder.appName('sparkify-transactions').config('spark.sql.extensions','org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions').getOrCreate()
spark.sql('CREATE DATABASE IF NOT EXISTS sparkify_transactions')
s3=boto3.client('s3'); sql=s3.get_object(Bucket=args['BUCKET'],Key=f"transactions/{args['SQL_FILE']}")['Body'].read().decode()
sql=sql.replace('{{ data_interval }}',args['DATA_INTERVAL'])
df=spark.sql(sql)
target=f"sparkify_transactions.{args['TABLE_NAME']}"
# Deduplicate on id-like primary keys before full replacement/upsert simulation.
keys=[c for c in df.columns if c.endswith('_id') or c=='id']
if keys: df=df.dropDuplicates([keys[0]])
df.writeTo(target).using('iceberg').tableProperty('format-version','2').createOrReplace()
