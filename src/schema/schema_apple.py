from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType

def apple_store_schema_silver():
    return StructType([
        StructField("id", StringType(), True),
        StructField("name_client", StringType(), True),
        StructField("app", StringType(), True),
        StructField("im_version", StringType(), True),
        StructField("im_rating", StringType(), True),
        StructField("title", StringType(), True),
        StructField("content", StringType(), True),
        StructField("updated", StringType(), True),
        StructField("segmento", StringType(), True),
        StructField("historical_data", ArrayType(
            ArrayType(StructType([
                StructField("title", StringType(), True),
                StructField("content", StringType(), True),
                StructField("app", StringType(), True),
                StructField("segmento", StringType(), True),
                StructField("im_version", StringType(), True),
                StructField("im_rating", StringType(), True)
            ]), True), True))
    ])


