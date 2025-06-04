import logging
import sys
from datetime import datetime
from pyspark.sql import SparkSession
from metrics import MetricsCollector, validate_ingest
try:
    from tools import read_source_parquet, save_dataframe, processing_reviews, save_metrics
    from schema_apple import apple_store_schema_silver
except ModuleNotFoundError:
    from src.utils.tools import read_source_parquet, save_dataframe, processing_reviews, save_metrics
    from src.schema.schema_apple import apple_store_schema_silver


# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes da Aplicação ---
FORMAT = "parquet"
PATH_BRONZE_BASE = "/santander/bronze/compass/reviews/appleStore/"
PATH_SILVER_BASE = "/santander/silver/compass/reviews/appleStore/"
PATH_SILVER_FAIL_BASE = "/santander/silver/compass/reviews_fail/appleStore/"
ENV_PRE_VALUE = "pre"
ELASTIC_INDEX_SUCCESS = "compass_dt_datametrics"
ELASTIC_INDEX_FAIL = "compass_dt_datametrics_fail"

def spark_session():
    """
    Cria e retorna uma SparkSession configurada para a aplicação.
    """
    try:
        spark = SparkSession.builder \
            .appName("App Reviews Silver [apple store]") \
            .config("spark.jars.packages", "org.apache.spark:spark-measure_2.12:0.16") \
            .getOrCreate()
        return spark
    except Exception as e:
        logging.error(f"[*] Falha ao criar SparkSession: {e}", exc_info=True)
        raise

def main():
    """
    Processa dados da camada bronze para a silver, validando e salvando os dados e métricas.
    """
    args_list = sys.argv

    # Verificar se o número correto de argumentos foi passado
    if len(args_list) != 2:
        logging.error("[*] Uso: spark-submit app.py <env>")
        sys.exit(1)

    env = args_list[1]
    spark = spark_session()

    try:
        # Inicialização da coleta de métricas
        metrics_collector = MetricsCollector(spark)
        metrics_collector.start_collection()

        # Data atual e caminhos
        date_str = datetime.now().strftime("%Y%m%d")
        path_source_pf = f"{PATH_BRONZE_BASE}*_pf/odate={date_str}/"
        path_source_pj = f"{PATH_BRONZE_BASE}*_pj/odate={date_str}/"
        path_target = PATH_SILVER_BASE
        path_target_fail = PATH_SILVER_FAIL_BASE

        # Leitura das fontes
        logging.info("[*] Iniciando leitura dos paths de origem.")
        df_pf = read_source_parquet(spark, path_source_pf)
        df_pj = read_source_parquet(spark, path_source_pj)

        dfs = [df for df in [df_pf, df_pj] if df is not None]

        if dfs:
            df = dfs[0] if len(dfs) == 1 else dfs[0].unionByName(dfs[1])
        else:
            logging.warning("[*] Nenhum dado encontrado. Criando DataFrame vazio.")
            empty_schema = spark.read.parquet(path_source_pf).schema
            df = spark.createDataFrame([], schema=empty_schema)

        # Processamento
        logging.info("[*] Iniciando o processamento das avaliações.")
        df_processado = processing_reviews(df)

        if env == ENV_PRE_VALUE:
            df_processado.show()

        # Validação
        logging.info("[*] Validando os dados processados.")
        valid_df, invalid_df, validation_results = validate_ingest(spark, df_processado)

        if env == ENV_PRE_VALUE:
            valid_df.take(10)
            invalid_df.take(10)

        # Salvando dados válidos
        logging.info(f"[*] Salvando dados válidos em {path_target}")
        save_dataframe(
            df=valid_df,
            path=path_target,
            label="valido",
            schema=apple_store_schema_silver(),
            partition_column="odate", # data de carga referencia
            compression="snappy"
        )

        # Salvando dados inválidos
        logging.info(f"[*] Salvando dados inválidos em {path_target_fail}")
        # Salvar dados inválidos
        save_dataframe(
            df=invalid_df,
            path=path_target_fail,
            label="invalido",
            partition_column="odate", # data de carga referencia
            compression="snappy"
        )

        # Finaliza coleta de métricas
        metrics_collector.end_collection()

        # Estrutura e salva métricas
        metrics_json = metrics_collector.collect_metrics(valid_df, invalid_df, validation_results, "silver_apple_store")
        save_metrics(
            metrics_type='success',
            index=ELASTIC_INDEX_SUCCESS,
            df=valid_df,
            metrics_data=metrics_json
        )

        logging.info(f"[*] Métricas da aplicação: {metrics_json}")

    except Exception as e:
        logging.error(f"[*] Ocorreu um erro: {e}", exc_info=True)
        save_metrics(
            metrics_type="fail",
            index=ELASTIC_INDEX_FAIL,
            error=e
        )

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
