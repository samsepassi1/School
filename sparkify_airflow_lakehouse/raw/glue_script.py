import sys
from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession
args=getResolvedOptions(sys.argv,['BUCKET','DATA_INTERVAL','TABLES'])
spark=SparkSession.builder.appName('sparkify-raw').config('spark.sql.extensions','org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions').getOrCreate()
bucket=args['BUCKET']; interval=args['DATA_INTERVAL']; tables=[t for t in args['TABLES'].split(',') if t]
spark.sql('CREATE DATABASE IF NOT EXISTS sparkify_raw')
for table in tables:
    path=f's3://{bucket}/landing/{interval}/{table}/'
    df=spark.read.json(path).withColumn('data_interval', spark.sql(f"SELECT '{interval}'").first()[0])
    target=f'sparkify_raw.{table}'
    df.writeTo(target).using('iceberg').tableProperty('format-version','2').partitionedBy('data_interval').createOrReplace()
