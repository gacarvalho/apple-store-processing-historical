import logging
import sys
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract
from datetime import datetime
from metrics import MetricsCollector, validate_ingest
try:
    from tools import *
except ModuleNotFoundError:
    from src.utils.tools import *


# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():

    # Capturar argumentos da linha de comando
    args = sys.argv

    # Verificar se o número correto de argumentos foi passado
    if len(args) != 2:
        print("[*] Usage: spark-submit app.py <env> ")
        sys.exit(1)

    # Criação da sessão Spark
    spark = spark_session()

    try:

        # Entrada e captura de variaveis e parametros
        env = args[1]

        # Coleta de métricas
        metrics_collector = MetricsCollector(spark)
        metrics_collector.start_collection()

        # Definindo caminhos
        datePath = datetime.now().strftime("%Y%m%d")
        format = "parquet"
        pathSource_pf = f"/santander/bronze/compass/reviews/appleStore/*_pf/odate={datePath}/*.{format}"
        pathSource_pj = f"/santander/bronze/compass/reviews/appleStore/*_pj/odate={datePath}/*.{format}"
        path_target = f"/santander/silver/compass/reviews/appleStore/odate={datePath}/"
        path_target_fail = f"/santander/silver/compass/reviews_fail/appleStore/odate={datePath}/"

        # Leitura do arquivo Parquet
        logging.info(f"[*] Iniciando leitura dos path origens.", exc_info=True)
        df_pf = read_source_parquet(spark, pathSource_pf)
        df_pj = read_source_parquet(spark, pathSource_pj)

        # Mantém apenas os DataFrames que possuem dados
        dfs = [df for df in [df_pf, df_pj] if df is not None]

        # Se houver pelo menos um DataFrame com dados, une os resultados
        if dfs:
            df = dfs[0] if len(dfs) == 1 else dfs[0].unionByName(dfs[1])
        else:
            print("[*] Nenhum dado encontrado! Criando DataFrame vazio...")
            empty_schema = spark.read.parquet(pathSource_pf).schema
            df = spark.createDataFrame([], schema=empty_schema)

        # Processamento dos dados
        logging.info(f"[*] Iniciando o processamento da funcao processing_reviews", exc_info=True)
        df_processado = processing_reviews(df)

        if env == "pre":
            df_processado.show()


        # Valida o DataFrame e coleta resultados
        logging.info(f"[*] Iniciando o processamento da funcao validate_ingest", exc_info=True)
        valid_df, invalid_df, validation_results = validate_ingest(spark, df_processado)

        if env == "pre":
            valid_df.show()
            invalid_df.show()

        # Salvar dados válidos
        logging.info(f"[*] Iniciando a gravacao do dataframe do {valid_df} no path {path_target}", exc_info=True)
        save_dataframe(valid_df, path_target, "valido")

        # Salvar dados inválidos
        logging.info(f"[*] Iniciando a gravacao do dataframe do {invalid_df} no path {path_target_fail}", exc_info=True)
        save_dataframe(invalid_df, path_target_fail, "invalido")

        logging.info(f"[*] Finalizacao da coleta de metricas", exc_info=True)
        metrics_collector.end_collection()

        # Coleta métricas após o processamento
        logging.info(f"[*] Estruturando retorno das metricas em json", exc_info=True)
        metrics_json = metrics_collector.collect_metrics(valid_df, invalid_df, validation_results, "silver_apple_store")

        logging.info(f"[*] Salvando metricas", exc_info=True)
        # Salvar métricas no MongoDB
        save_metrics(metrics_json,valid_df)

        logging.info(f"[*] Métricas da aplicação: {metrics_json}")

    except Exception as e:
        logging.error(f"[*] An error occurred: {e}", exc_info=True)
        log_error(e, df)

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
        logging.error(f"[*] Failed to create SparkSession: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
