# tests/test_main.py
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from datetime import datetime
from pyspark.sql.functions import input_file_name, regexp_extract, lit
from unittest.mock import MagicMock, patch

from src.utils.tools import processing_reviews, save_dataframe
from src.metrics.metrics import validate_ingest

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.master("local[1]").appName("TestReadData").getOrCreate()

def apple_store_schema_bronze():
    return StructType([
        StructField('author_name', StringType(), True),
        StructField('author_uri', StringType(), True),
        StructField('content', StringType(), True),
        StructField('content_attributes_label', StringType(), True),
        StructField('content_attributes_term', StringType(), True),
        StructField('id', StringType(), True),
        StructField('im_rating', StringType(), True),
        StructField('im_version', StringType(), True),
        StructField('im_votecount', StringType(), True),
        StructField('im_votesum', StringType(), True),
        StructField('link_attributes_href', StringType(), True),
        StructField('link_attributes_related', StringType(), True),
        StructField('title', StringType(), True),
        StructField('updated', StringType(), True)
    ])

def data_apple():
    return [
        ("keneddy santos", "https://itunes.apple.com/br/reviews/id1291186914", "Está na hora de atualizar...", "Aplicativo", "Application", "11670010101", "4", "24.7.7", "0", "0", "https://itunes.apple.com/...","related", "Face ID", "2024-08-30T08:10:10-07:00"),
        # ... (outros dados iguais aos anteriores)
    ]

def test_read_data(spark):
    df_test = spark.createDataFrame(data_apple(), apple_store_schema_bronze())
    datePath = datetime.now().strftime("%Y%m%d")
    test_parquet_path = f"/tmp/test_apple_data/appleStore/banco-santander-br/odate={datePath}/"
    df_test.write.mode("overwrite").parquet(test_parquet_path)
    df = spark.read.parquet(test_parquet_path).withColumn("app", regexp_extract(input_file_name(), "/appleStore/(.*?)/odate=", 1))
    df_processado = processing_reviews(df)
    assert df_processado.count() == 7

def test_processamento_reviews(spark):
    df_test = spark.createDataFrame(data_apple(), apple_store_schema_bronze())
    datePath = datetime.now().strftime("%Y%m%d")
    test_parquet_path = f"/tmp/test_apple_data/odate={datePath}/"
    df_test.write.mode("overwrite").parquet(test_parquet_path)
    df = spark.read.parquet(test_parquet_path).withColumn("app", regexp_extract(input_file_name(), "/appleStore/(.*?)/odate=", 1))
    df_processado = processing_reviews(df)
    assert df_processado.count() > 0

def test_validate_ingest(spark):
    df = spark.createDataFrame(data_apple(), apple_store_schema_bronze()).withColumn("app", lit("banco-santander-br"))
    df_processado = processing_reviews(df)
    valid_df, invalid_df, validation_results = validate_ingest(spark, df_processado)
    assert valid_df.count() > 0
    assert len(validation_results) > 0

def test_save_data(spark):
    df = spark.createDataFrame(data_apple(), apple_store_schema_bronze()).withColumn("app", lit("banco-santander-br"))
    df_processado = processing_reviews(df)
    valid_df, _, _ = validate_ingest(spark, df_processado)
    datePath = datetime.now().strftime("%Y%m%d")
    path_target = f"/tmp/fake/path/valid/odate={datePath}/"
    with patch("pyspark.sql.DataFrameWriter.parquet", MagicMock()) as mock_parquet:
        save_dataframe(valid_df, path_target, "bronze")
        mock_parquet.assert_any_call(path_target)
