# tests/test_main.py
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
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
        StructField('im_rating', IntegerType(), True),
        StructField('im_version', StringType(), True),
        StructField('im_votecount', IntegerType(), True),
        StructField('im_votesum', IntegerType(), True),
        StructField('link_attributes_href', StringType(), True),
        StructField('link_attributes_related', StringType(), True),
        StructField('title', StringType(), True),
        StructField('updated', StringType(), True),
        StructField('segmento', StringType(), True)  # Adicionado como 15º campo
    ])

def data_apple():
    return [
        (
            "keneddy santos",  # author_name
            "https://itunes.apple.com/br/reviews/id1291186914",  # author_uri
            "Está na hora de atualizar...",  # content
            "Aplicativo",  # content_attributes_label
            "Application",  # content_attributes_term
            "11670010101",  # id
            4,  # im_rating
            "24.7.7",  # im_version
            0,  # im_votecount
            0,  # im_votesum
            "https://itunes.apple.com/...",  # link_attributes_href
            "related",  # link_attributes_related
            "Face ID",  # title
            "2024-08-30T08:10:10-07:00",  # updated
            "pf"  # segmento
        )
    ]

def test_read_data(spark):
    df_test = spark.createDataFrame(data_apple(), apple_store_schema_bronze())
    datePath = datetime.now().strftime("%Y%m%d")
    test_parquet_path = f"/tmp/test_apple_data/appleStore/banco-santander-br/odate={datePath}/"
    df_test.write.mode("overwrite").parquet(test_parquet_path)
    df = spark.read.parquet(test_parquet_path).withColumn("app", regexp_extract(input_file_name(), "/appleStore/(.*?)/odate=", 1))
    df_processado = processing_reviews(df)
    assert df_processado.count() == 1

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
