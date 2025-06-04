import importlib
import logging
import json
import re
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import  col, count, when, current_timestamp, date_format, collect_list, explode
from pyspark.sql.types import ArrayType, StructType, StringType, StructField
from pyspark.sql import functions as F
from sparkmeasure import StageMetrics
try:
    from tools import processing_old_new
except ImportError:
    from src.utils.tools import processing_old_new


# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MetricsCollector:
    """
    Coleta métricas de execução de um estágio Spark.

    Attributes:
        spark (SparkSession): Referência para a SparkSession ativa.
        stage_metrics (StageMetrics): Instância da classe StageMetrics para coletar métricas de estágio.
        start_time (datetime): Data e hora de início da coleta de métricas.
    """

    def __init__(self, spark):
        self.spark = spark
        self.stage_metrics = StageMetrics(spark)
        self.start_time = None
        self.end_time = None

    def start_collection(self):
        self.stage_metrics.begin()
        self.start_time = datetime.now()

    def end_collection(self):
        self.end_time = datetime.now()
        self.stage_metrics.end()

    def parse_stage_metrics(self, stage_metrics_str: str) -> dict:
        stage_metrics = {}
        lines = stage_metrics_str.strip().split("\n")
        for line in lines:
            if line.strip() and not line.startswith("Scheduling mode"):
                key_value = re.split(r" => | = ", line.strip())
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    stage_metrics[key] = value
        return stage_metrics


    def collect_metrics(self, valid_df: DataFrame, invalid_df: DataFrame, validation_results: dict, id_app) -> str:
        """
        Coleta métricas de processamento, validação e recursos utilizados.

        Args:
            valid_df (DataFrame): DataFrame contendo os registros válidos.
            invalid_df (DataFrame): DataFrame contendo os registros inválidos.
            validation_results (dict): Dicionário com os resultados da validação.

        Returns:
            str: Um JSON com as métricas coletadas, incluindo:
                - Informações gerais da aplicação (ID, sigla do projeto).
                - Contagem de registros válidos e inválidos.
                - Tempo total de processamento.
                - Uso de memória.
                - Número de nós de dados.
                - Métricas de estágio do Spark.
                - Resultados da validação.
                - Contadores de sucesso e erro da validação.

        Esta função coleta diversas métricas relacionadas ao processo de ingestão e validação de dados.
        As métricas são calculadas com base nos DataFrames de dados válidos e inválidos, nos resultados da validação
        e nas métricas de execução do Spark. O resultado final é um JSON que pode ser utilizado para monitoramento e análise.
        """
        if self.start_time is None or self.end_time is None:
            raise ValueError("[*] O tempo de início ou término não foi definido corretamente. Verifique a execução dos métodos start_collection e end_collection.")

        total_time = self.end_time - self.start_time
        formatted_time = f"{total_time.total_seconds() / 60:.2f} min"
        start_ts = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_ts = self.end_time.strftime("%Y-%m-%d %H:%M:%S")
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        memory_used = self.spark.sparkContext._jvm.org.apache.spark.util.SizeEstimator.estimate(valid_df._jdf) / (1024 * 1024)
        count_valid = valid_df.count()
        count_invalid = invalid_df.count()
        total_records = count_valid + count_invalid
        percentage_valid = (count_valid / total_records * 100) if total_records > 0 else 0.0
        stage_metrics_raw = self.stage_metrics.report()
        stage_metrics_dict = self.parse_stage_metrics(stage_metrics_raw)

        validation_metrics = {
            "duplicate_check": validation_results["duplicate_check"],
            "null_check": validation_results["null_check"],
            "type_consistency_check": validation_results["type_consistency_check"],
        }

        # Contadores de sucesso e erro
        success_count = sum(1 for result in validation_metrics.values() if result["status"])
        error_count = len(validation_metrics) - success_count

        # Convertendo "segmento" para uma lista de strings
        segmentos_unicos = [str(row["segmento"]) if row["segmento"] is not None else "UNKNOWN"
                            for row in valid_df.select("segmento").distinct().collect()]


        metrics = {
            "application_id": self.spark.sparkContext.applicationId,
            "owner": {
                "sigla": "DT",
                "projeto": "compass",
                "layer_lake": "silver"
            },
            "valid_data": {
                "count": count_valid,
                "percentage": percentage_valid
            },
            "invalid_data": {
                "count": count_invalid,
                "percentage": (count_invalid / total_records * 100) if total_records > 0 else 0.0
            },
            "total_records": total_records,
            "total_processing_time": formatted_time,
            "memory_used": memory_used,
            "stages": stage_metrics_dict,
            "validation_results": validation_metrics,
            "success_count": success_count,
            "error_count": error_count,
            "type_client": ",".join(segmentos_unicos).upper() if segmentos_unicos else "NA",
            "source": {
                "app": id_app,
                "search": "apple_store"
            },
            "_ts": {
                "compass_start_ts": start_ts,
                "compass_end_ts": end_ts
            },
            "timestamp": timestamp

        }

        return json.dumps(metrics)

def print_validation_results(results: dict):
    print(f"\n[*] Validação da ingestão concluída para {results['total_records']} registros.\n")

    for check, result in results.items():
        if check != "total_records":
            status = "PASSOU" if result["status"] else "FALHOU"
            print(f"{check.replace('_', ' ').title()}: {status}")
            if result["message"]:
                print(f"  -> {result['message']}\n")



def validate_ingest(spark: SparkSession, df: DataFrame) -> tuple:
    """
    Valida um DataFrame de dados de ingestão e compara com dados históricos.

    Args:
        df (DataFrame): O DataFrame a ser validado.

    Returns:
        tuple: Uma tupla contendo:
            - DataFrame de registros válidos.
            - DataFrame de registros inválidos.
            - Dicionário com os resultados da validação.

    Realiza as seguintes validações:
        - Verifica a existência de duplicatas.
        - Verifica a existência de valores nulos em colunas críticas.
        - Verifica a consistência dos tipos de dados.

    Codigos de retorno:
        200: Sucesso (Nenhum problema encontrado)
        400: Erro nos dados (Valores nulos ou tipos inválidos)
        409: Conflito de dados (Registros duplicados encontrados)
    """
    logging.info(f"[*] def validate_ingest: Iniciando o processamento da funcao processing_old_new", exc_info=True)
    df = processing_old_new(spark, df)

    # Validações (duplicidade, valores nulos, consistência de tipos)
    validation_results = {
        "duplicate_check": {"message": "", "status": True, "code": 200},
        "null_check": {"message": "", "status": True, "code": 200},
        "type_consistency_check": {"message": "", "status": True, "code": 200},
        "total_records": df.count(),
    }

    logging.info(f"[*] def validate_ingest: Iniciando o processo de data quality", exc_info=True)
    # Verificação de duplicidade
    duplicates = df.groupBy("id").count().filter(col("count") > 1)
    duplicate_count = duplicates.count()
    if duplicate_count > 0:
        validation_results["duplicate_check"]["status"] = False
        validation_results["duplicate_check"]["code"] = 409
        validation_results["duplicate_check"]["message"] = f"Registros duplicados encontrados: {duplicate_count} registros com base no 'id'."
    else:
        validation_results["duplicate_check"]["status"] = True
        validation_results["duplicate_check"]["code"] = 200
        validation_results["duplicate_check"]["message"] = "Nenhum registro duplicado encontrado."

    print(f"Número de duplicatas: {duplicate_count}")

    # Verificação de valores nulos
    critical_columns = ["id", "content", "im_rating", "im_version"]
    nulls_df = df.select([count(when(col(c).isNull(), c)).alias(c) for c in critical_columns])
    null_counts = nulls_df.collect()[0].asDict()

    null_issues = {col: count for col, count in null_counts.items() if count > 0}
    if null_issues:
        validation_results["null_check"]["status"] = False
        validation_results["null_check"]["code"] = 400
        validation_results["null_check"]["message"] = f"Valores nulos encontrados nas colunas: {null_issues}."
    else:
        validation_results["null_check"]["status"] = True
        validation_results["null_check"]["code"] = 200
        validation_results["null_check"]["message"] = "Nenhum valor nulo encontrado nas colunas criticas."

    # Consistência dos tipos de dados
    invalid_rating = df.filter(~df.im_rating.cast("int").isNotNull())
    if invalid_rating.count() > 0:
        validation_results["type_consistency_check"]["status"] = False
        validation_results["type_consistency_check"]["code"] = 400
        validation_results["type_consistency_check"]["message"] = "Valores não numericos encontrados na coluna 'im_rating'."
    else:
        validation_results["type_consistency_check"]["status"] = True
        validation_results["type_consistency_check"]["code"] = 200
        validation_results["type_consistency_check"]["message"] = "Consistencia dos tipos de dados verificada com sucesso."

    logging.info(f"[*] def validate_ingest: Finalizando a visao final de invalids_records e valids_records", exc_info=True)
    valid_records = df
    invalid_records = df.select("id","name_client","app","im_version","im_rating","title","content","updated","segmento","historical_data").filter(col("id").isNull() | col("name_client").isNull() | col("content").isNull())

    # Se houver duplicatas, adicionamos ao DataFrame de inválidos
    # Seleciona apenas a coluna "id" do duplicates
    duplicate_ids = duplicates.select("id")

    # Filtra os registros completos de df com base nos IDs duplicados
    duplicates_records = df.join(duplicate_ids, on="id", how="inner").select(
        df["id"],
        df["name_client"],
        df["app"],
        df["im_version"],
        df["im_rating"],
        df["title"],
        df["content"],
        df["segmento"],
        df["updated"]
    )

    if duplicate_count > 0:
        invalid_records = invalid_records.union(duplicates_records)


    valid_records = valid_records.subtract(invalid_records)

    # Função de impressão de resultados
    print_validation_results(validation_results)

    return valid_records, invalid_records, validation_results

