import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract
from datetime import datetime
from metrics import MetricsCollector, validate_ingest
from tools import *

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Criação da sessão Spark
        with spark_session() as spark:
            # Coleta de métricas
            metrics_collector = MetricsCollector(spark)
            metrics_collector.start_collection()

            # Definindo caminhos
            datePath = datetime.now().strftime("%Y%m%d")
            format = "parquet"
            pathSource = f"/santander/bronze/compass/reviews/appleStore/*/odate={datePath}/*.{format}"
            path_target = f"/santander/silver/compass/reviews/appleStore/odate={datePath}/"
            path_target_fail = f"/santander/silver/compass/reviews_fail/appleStore/odate={datePath}/"

            # Leitura do arquivo Parquet
            df = spark.read.parquet(pathSource).withColumn("app", regexp_extract(input_file_name(), "/appleStore/(.*?)/odate=", 1))

            # Processamento dos dados
            df_processado = processing_reviews(df)

            # Valida o DataFrame e coleta resultados
            valid_df, invalid_df, validation_results = validate_ingest(spark, df_processado)

            valid_df.show(50, truncate=False)
            invalid_df.show(50, truncate=False)

            # Salvar dados válidos
            save_dataframe(valid_df, path_target, "valido")

            # Salvar dados inválidos
            save_dataframe(invalid_df, path_target_fail, "invalido")

            metrics_collector.end_collection()

            # Coleta métricas após o processamento
            metrics_json = metrics_collector.collect_metrics(valid_df, invalid_df, validation_results, "silver_apple_store")
            
            # Salvar métricas no MongoDB
            save_metrics(spark, metrics_json)

            logging.info(f"Métricas da aplicação: {metrics_json}")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        spark.stop()

def spark_session():
    try:
        spark = SparkSession.builder \
            .appName("App Reviews [apple store]") \
            .config("spark.jars.packages", "org.apache.spark:spark-measure_2.12:0.16") \
            .getOrCreate()
        return spark

    except Exception as e:
        logging.error(f"Failed to create SparkSession: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
