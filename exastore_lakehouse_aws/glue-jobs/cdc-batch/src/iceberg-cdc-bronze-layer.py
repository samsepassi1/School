import sys
from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession

args = getResolvedOptions(sys.argv, ['JOB_NAME','BUCKET','BRONZE_DB'])
BUCKET=args['BUCKET']; BRONZE_DB=args['BRONZE_DB']
WAREHOUSE=f's3://{BUCKET}/bronze/warehouse'

spark=(SparkSession.builder.appName(args['JOB_NAME'])
 .config('spark.sql.catalog.glue_catalog','org.apache.iceberg.spark.SparkCatalog')
 .config('spark.sql.catalog.glue_catalog.warehouse',WAREHOUSE)
 .config('spark.sql.catalog.glue_catalog.catalog-impl','org.apache.iceberg.aws.glue.GlueCatalog')
 .config('spark.sql.catalog.glue_catalog.io-impl','org.apache.iceberg.aws.s3.S3FileIO')
 .config('spark.sql.extensions','org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions')
 .getOrCreate())

TABLES={
 'orders':{'pk':'order_id','partition':'months(created_at)','columns':{'order_id':'STRING','user_id':'STRING','status':'STRING','order_total':'DECIMAL(10,2)','created_at':'TIMESTAMP','updated_at':'TIMESTAMP'},
           'extract':"""COALESCE(after.order_id,before.order_id) order_id, COALESCE(after.user_id,before.user_id) user_id, COALESCE(after.status,before.status) status, CAST(COALESCE(after.order_total,before.order_total) AS DECIMAL(10,2)) order_total, CAST(COALESCE(after.created_at,before.created_at) AS TIMESTAMP) created_at, CAST(COALESCE(after.updated_at,before.updated_at) AS TIMESTAMP) updated_at"""},
 'users':{'pk':'id','partition':'months(created_at)','columns':{'id':'STRING','email':'STRING','first_name':'STRING','last_name':'STRING','country':'STRING','created_at':'TIMESTAMP','updated_at':'TIMESTAMP'},
          'extract':"""COALESCE(after.id,before.id) id, COALESCE(after.email,before.email) email, COALESCE(after.first_name,before.first_name) first_name, COALESCE(after.last_name,before.last_name) last_name, COALESCE(after.country,before.country) country, CAST(COALESCE(after.created_at,before.created_at) AS TIMESTAMP) created_at, CAST(COALESCE(after.updated_at,before.updated_at) AS TIMESTAMP) updated_at"""},
 'products':{'pk':'id','partition':'months(created_at)','columns':{'id':'STRING','sku':'STRING','name':'STRING','category':'STRING','price':'DECIMAL(10,2)','created_at':'TIMESTAMP','updated_at':'TIMESTAMP'},
             'extract':"""COALESCE(after.id,before.id) id, COALESCE(after.sku,before.sku) sku, COALESCE(after.name,before.name) name, COALESCE(after.category,before.category) category, CAST(COALESCE(after.price,before.price) AS DECIMAL(10,2)) price, CAST(COALESCE(after.created_at,before.created_at) AS TIMESTAMP) created_at, CAST(COALESCE(after.updated_at,before.updated_at) AS TIMESTAMP) updated_at"""}
}

def create_table(t,cfg):
    cols=',\n  '.join([f'{k} {v}' for k,v in cfg['columns'].items()] + ['op STRING','ts_ms BIGINT','source_lsn BIGINT','_processed_at TIMESTAMP'])
    spark.sql(f"""CREATE TABLE IF NOT EXISTS glue_catalog.{BRONZE_DB}.{t} ({cols})
USING iceberg PARTITIONED BY ({cfg['partition']}) LOCATION 's3://{BUCKET}/bronze/{t}'
TBLPROPERTIES ('format-version'='2','write.format.default'='parquet','write.parquet.compression-codec'='snappy')""")

def merge_table(t,cfg):
    spark.read.option('multiLine','false').json(f's3://{BUCKET}/datasets/cdc_data/small/year=2025/{t}/*.json').createOrReplaceTempView(f'raw_{t}')
    spark.sql(f"""CREATE OR REPLACE TEMP VIEW cdc_{t} AS SELECT {cfg['extract']}, op, CAST(ts_ms AS BIGINT) ts_ms, CAST(source.lsn AS BIGINT) source_lsn, current_timestamp() _processed_at FROM raw_{t} WHERE op IN ('c','u','d') AND COALESCE(after.{cfg['pk']},before.{cfg['pk']}) IS NOT NULL""")
    spark.sql(f"""CREATE OR REPLACE TEMP VIEW staged_{t} AS SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY {cfg['pk']} ORDER BY source_lsn DESC, ts_ms DESC) _rn FROM cdc_{t}) WHERE _rn=1""")
    cols=list(cfg['columns'])+['op','ts_ms','source_lsn','_processed_at']; col_list=', '.join(cols)
    updates=', '.join([f't.{c}=s.{c}' for c in cols])
    values=', '.join([f's.{c}' for c in cols])
    spark.sql(f"""MERGE INTO glue_catalog.{BRONZE_DB}.{t} t USING staged_{t} s ON t.{cfg['pk']}=s.{cfg['pk']}
WHEN MATCHED AND s.op='d' THEN DELETE
WHEN MATCHED THEN UPDATE SET {updates}
WHEN NOT MATCHED AND s.op <> 'd' THEN INSERT ({col_list}) VALUES ({values})""")

spark.sql(f'CREATE DATABASE IF NOT EXISTS glue_catalog.{BRONZE_DB}')
for t,cfg in TABLES.items():
    create_table(t,cfg); merge_table(t,cfg)
