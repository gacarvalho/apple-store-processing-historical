import logging
import sys
from datetime import datetime
from typing import Tuple
from pyspark.sql import SparkSession, DataFrame

"""    Necessidade para rodar os testes unitarios como processo de integracao!  """
try:
    from tools import read_source_parquet, save_dataframe, processing_reviews
    from metrics import MetricsCollector, validate_ingest, save_metrics
    from schema_apple import apple_store_schema_silver
except ModuleNotFoundError:
    from src.utils.tools import read_source_parquet, save_dataframe, processing_reviews
    from src.utils.metrics import MetricsCollector, validate_ingest, save_metrics
    from src.schema.schema_apple import apple_store_schema_silver

# Configuração centralizada
FORMAT = "parquet"
PATH_BRONZE_BASE = "/santander/bronze/compass/reviews/appleStore/"
PATH_SILVER_BASE = "/santander/silver/compass/reviews/appleStore/"
PATH_SILVER_FAIL_BASE = "/santander/silver/compass/reviews_fail/appleStore/"
ENV_PRE_VALUE = "pre"
ELASTIC_INDEX_SUCCESS = "compass_dt_datametrics"
ELASTIC_INDEX_FAIL = "compass_dt_datametrics_fail"
COMPRESSION_TYPE = "snappy"
PARTITION_COLUMN = "odate"

# Configuração de logging estruturado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineConfig:
    """Configuracao do pipeline"""
    def __init__(self, env: str):
        self.env = env
        self.date_str = datetime.now().strftime("%Y%m%d")
        self.path_source_pf = f"{PATH_BRONZE_BASE}*_pf/odate={self.date_str}/"
        self.path_source_pj = f"{PATH_BRONZE_BASE}*_pj/odate={self.date_str}/"
        self.path_target = PATH_SILVER_BASE
        self.path_target_fail = PATH_SILVER_FAIL_BASE

def create_spark_session() -> SparkSession:
    """Criacao da sessao do Spark"""
    try:
        spark = SparkSession.builder \
            .appName("App Reviews Silver [apple store]") \
            .config("spark.jars.packages", "org.apache.spark:spark-measure_2.12:0.16") \
            .getOrCreate()
        logger.info("[*] Spark Session criado com sucesso.")
        return spark
    except Exception as e:
        logger.error(f"[*] Falha ao criar o Spark Session: {e}", exc_info=True)
        raise

def read_and_union_data(spark: SparkSession, config: PipelineConfig) -> DataFrame:
    """Ler e unir todos os dados de origem"""
    logger.info("[*] Lendo dados de origem dos caminhos PF e PJ.")

    try:
        df_pf = read_source_parquet(spark, config.path_source_pf)
        df_pj = read_source_parquet(spark, config.path_source_pj)

        dfs = [df for df in [df_pf, df_pj] if df is not None]

        if not dfs:
            logger.warning("[*] Nenhum dado encontrado - criando DataFrame vazio.")
            empty_schema = spark.read.parquet(config.path_source_pf).schema
            return spark.createDataFrame([], schema=empty_schema)

        return dfs[0] if len(dfs) == 1 else dfs[0].unionByName(dfs[1])
    except Exception as e:
        logger.error("[*] Falha ao ler e unir dados.", exc_info=True)
        raise

def process_and_validate_data(spark: SparkSession, df: DataFrame, config: PipelineConfig) -> Tuple[DataFrame, DataFrame, dict]:
    """[*] Processar e validar dados com coleta de métricas"""
    metrics_collector = MetricsCollector(spark)
    metrics_collector.start_collection()

    try:
        logger.info("[*] Processando dados de avaliações")
        df_processed = processing_reviews(df)

        logger.info("[*] Validando a qualidade dos dados")
        valid_df, invalid_df, validation_results = validate_ingest(spark, df_processed)

        if config.env == ENV_PRE_VALUE:
            logger.info("[*] Dados de amostra: valid_df")
            valid_df.take(10)
            logger.info("[*] Dados de amostra: invalid_df")
            invalid_df.take(10)

        return valid_df, invalid_df, validation_results
    finally:
        metrics_collector.end_collection()

def save_output_data(valid_df: DataFrame, invalid_df: DataFrame, config: PipelineConfig) -> None:
    """Salvar dados de saida com tratamento de erros adequado"""
    try:
        logger.info(f"[*] Salvando dados validos em {config.path_target}")
        save_dataframe(
            df=valid_df,
            path=config.path_target,
            label="valido",
            schema=apple_store_schema_silver(),
            partition_column=PARTITION_COLUMN,
            compression=COMPRESSION_TYPE
        )

        logger.info(f"[*] Salvando dados invalidos em {config.path_target_fail}")
        save_dataframe(
            df=invalid_df,
            path=config.path_target_fail,
            label="invalido",
            partition_column=PARTITION_COLUMN,
            compression=COMPRESSION_TYPE
        )
    except Exception as e:
        logger.error("[*] Falha ao salvar os dados de saida", exc_info=True)
        raise

def execute_pipeline(spark: SparkSession, config: PipelineConfig) -> None:
    """Executa o pipeline completo"""
    try:
        # Pipeline steps
        source_df = read_and_union_data(spark, config)
        valid_df, invalid_df, validation_results = process_and_validate_data(spark, source_df, config)
        save_output_data(valid_df, invalid_df, config)

        # Metrics collection and saving
        metrics_collector = MetricsCollector(spark)
        metrics_json = metrics_collector.collect_metrics(
            valid_df, invalid_df, validation_results, "silver_apple_store"
        )

        save_metrics(
            metrics_type='success',
            index=ELASTIC_INDEX_SUCCESS,
            df=valid_df,
            metrics_data=metrics_json
        )

        logger.info(f"[*] Pipeline metrics: {metrics_json}")
        logger.info("[*] Pipeline executado com sucesso.")
    except Exception as e:
        logger.error("[*] Pipeline executado com falha!", exc_info=True)
        save_metrics(
            metrics_type="fail",
            index=ELASTIC_INDEX_FAIL,
            error=e
        )
        raise

def main():

    if len(sys.argv) != 2:
        logger.error("Usage: spark-submit app.py <env>")
        sys.exit(1)

    env = sys.argv[1]
    config = PipelineConfig(env)
    spark = None

    try:
        spark = create_spark_session()
        execute_pipeline(spark, config)
    except Exception as e:
        logger.error(f"Erro no pipeline: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    main()