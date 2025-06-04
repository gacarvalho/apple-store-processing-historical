# tests/test_main.py
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from datetime import datetime
from pyspark.sql.functions import input_file_name, regexp_extract, lit
from unittest.mock import MagicMock, patch, PropertyMock
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
        StructField('segmento', StringType(), True)
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

    print("functions: test_processamento_reviews")
    df.take(10)
    assert df_processado.count() > 0

def test_validate_ingest(spark):
    df = spark.createDataFrame(data_apple(), apple_store_schema_bronze()).withColumn("app", lit("banco-santander-br"))
    df_processado = processing_reviews(df)
    valid_df, invalid_df, validation_results = validate_ingest(spark, df_processado)
    assert valid_df.count() > 0
    assert len(validation_results) > 0

def test_save_data(spark):
    from src.utils import tools  # importa dentro do teste para patch correto

    df = spark.createDataFrame(data_apple(), apple_store_schema_bronze()).withColumn("app", lit("banco-santander-br"))
    df_processado = processing_reviews(df)
    valid_df, _, _ = validate_ingest(spark, df_processado)
    datePath = datetime.now().strftime("%Y%m%d")
    path_target = f"/tmp/fake/path/valid/odate={datePath}"

    mock_parquet = MagicMock()
    mock_partitionBy = MagicMock()
    mock_partitionBy.parquet = mock_parquet

    mock_mode = MagicMock()
    mock_mode.partitionBy.return_value = mock_partitionBy

    mock_option = MagicMock()
    mock_option.mode.return_value = mock_mode

    mock_write = MagicMock()
    mock_write.option.return_value = mock_option

    with patch.object(type(valid_df), "write", new_callable=PropertyMock) as mock_write_property:
        mock_write_property.return_value = mock_write

        tools.save_dataframe(
            df=valid_df,
            path=path_target,
            label="valido",
            partition_column="odate",
            compression="snappy"
        )

        mock_write.option.assert_called_once_with("compression", "snappy")
        mock_option.mode.assert_called_once_with("overwrite")
        mock_mode.partitionBy.assert_called_once_with("odate")
        mock_parquet.assert_called_once_with(path_target)

