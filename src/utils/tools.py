"""
Módulo com funções utilitárias para processamento de dados no pipeline de reviews da Apple Store.

Funções principais:
- read_source_parquet: Leitura de arquivos Parquet com tratamento de erros
- processing_reviews: Processamento básico dos dados de reviews
- save_dataframe: Salvamento robusto de DataFrames
- save_metrics: Integração com Elasticsearch para métricas
- processing_old_new: Comparação entre dados novos e históricos
"""

import json
import os
import logging
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from urllib.parse import quote_plus
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    coalesce, collect_list, struct, array, col,
    count, when, to_date, regexp_extract,
    input_file_name, lit, broadcast, rand, floor
)
from pyspark.sql.utils import AnalysisException
from pyspark.sql.types import (
    StructType, StructField, StringType,
    ArrayType, IntegerType
)
from unidecode import unidecode
from elasticsearch import Elasticsearch

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Função para remoção de acentos
def remove_accents(s):
    """Remove acentos e caracteres especiais de uma string"""
    return unidecode(s)

remove_accents_udf = F.udf(remove_accents, StringType())

def read_source_parquet(spark: SparkSession, path: str) -> Optional[DataFrame]:
    """
    Tenta ler um arquivo Parquet e retorna None se não houver dados.
    
    Args:
        spark: Sessão do Spark
        path: Caminho do arquivo Parquet
        
    Returns:
        DataFrame com os dados ou None se falhar
    """
    try:
        df = spark.read.parquet(path)
        if df.isEmpty():
            logger.error(f"[*] Nenhum dado encontrado em: {path}")
            return None

        return df.withColumn("app", regexp_extract(input_file_name(), "/appleStore/(.*?)/odate=", 1)) \
            .withColumn("segmento", regexp_extract(input_file_name(), r"/appleStore/[^/_]+_([pfj]+)/odate=", 1))

    except AnalysisException:
        logger.error(f"[*] Falha ao ler: {path}. O arquivo pode não existir.")
        return None

def processing_reviews(df: DataFrame) -> DataFrame:
    """
    Processa o DataFrame de reviews aplicando transformações básicas.
    
    Args:
        df: DataFrame com os dados brutos de reviews
        
    Returns:
        DataFrame processado
    """
    logger.info(f"{datetime.now().strftime('%Y%m%d %H:%M:%S.%f')} [*] Processando o tratamento da camada historica")

    return df.select(
        "id",
        F.upper(remove_accents_udf(F.col("author_name"))).alias("name_client"),
        "updated",
        "author_uri",
        F.upper(remove_accents_udf(F.col("title"))).alias("title"),
        "im_version",
        "im_rating",
        "im_votecount",
        "im_votesum",
        "app",
        "content_attributes_label",
        "content_attributes_term",
        "segmento",
        F.upper(remove_accents_udf(F.col("content"))).alias("content")
    )

def get_schema(df: DataFrame, schema: StructType) -> DataFrame:
    """
    Ajusta o DataFrame para seguir o schema especificado.
    
    Args:
        df: DataFrame a ser ajustado
        schema: Schema desejado
        
    Returns:
        DataFrame com schema aplicado
    """
    for field in schema.fields:
        if field.dataType == IntegerType():
            df = df.withColumn(field.name, df[field.name].cast(IntegerType()))
        elif field.dataType == StringType():
            df = df.withColumn(field.name, df[field.name].cast(StringType()))
    return df.select([field.name for field in schema.fields])

def save_dataframe(
            df: DataFrame,
            path: str,
            label: str,
            schema: Optional[StructType] = None,
            partition_column: str = "odate",
            compression: str = "snappy"
    ) -> bool:
    """
    Salva um DataFrame Spark no formato Parquet de forma robusta.

    Args:
        df: DataFrame a ser salvo
        path: Caminho de destino
        label: Identificação para logs (ex: 'valido', 'invalido')
        schema: Schema opcional para validação
        partition_column: Coluna de partição
        compression: Tipo de compressão

    Returns:
        bool: True se salvou com sucesso, False caso contrário

    Raises:
        ValueError: Se os parâmetros forem inválidos
        IOError: Se houver problemas ao escrever no filesystem
    """
    if not isinstance(df, DataFrame):
        logger.error(f"[*] Objeto passado não é um DataFrame Spark: {type(df)}")
        return False

    if not path:
        logger.error("Caminho de destino não pode ser vazio")
        return False

    current_date = datetime.now().strftime('%Y%m%d')
    full_path = Path(path)

    try:
        if schema:
            logger.info(f"[*] Aplicando schema para dados {label}")
            df = get_schema(df, schema)

        df_partition = df.withColumn(partition_column, lit(current_date))

        if not df_partition.head(1):
            logger.warning(f"[*] Nenhum dado {label} encontrado para salvar")
            return False

        try:
            full_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"[*] Diretório {full_path} verificado/criado")
        except Exception as dir_error:
            logger.error(f"[*] Falha ao preparar diretório {full_path}: {dir_error}")
            raise IOError(f"[*] Erro de diretório: {dir_error}") from dir_error

        logger.info(f"[*] Salvando {df_partition.count()} registros ({label}) em {full_path}")

        (df_partition.write
         .option("compression", compression)
         .mode("overwrite")
         .partitionBy(partition_column)
         .parquet(str(full_path)))

        logger.info(f"[*] Dados {label} salvos com sucesso em {full_path}")
        return True

    except Exception as e:
        error_msg = f"[*] Falha ao salvar dados {label} em {full_path}"
        logger.error(error_msg, exc_info=True)
        logger.error(f"[*] Detalhes do erro: {str(e)}\n{traceback.format_exc()}")
        return False

def save_metrics(
            metrics_type: str,
            index: str,
            error: Optional[Exception] = None,
            df: Optional[DataFrame] = None,
            metrics_data: Optional[Union[dict, str]] = None
    ) -> None:
    """
    Salva métricas no Elasticsearch com estruturas específicas.
    
    Args:
        metrics_type: 'success' ou 'fail'
        index: Nome do índice no Elasticsearch
        error: Objeto de exceção (para tipo 'fail')
        df: DataFrame (para extrair segmentos)
        metrics_data: Dados das métricas (para tipo 'success')
        
    Raises:
        ValueError: Se os parâmetros forem inválidos
    """
    metrics_type = metrics_type.lower()

    if metrics_type not in ('success', 'fail'):
        raise ValueError("[*] O tipo deve ser 'success' ou 'fail'")

    if metrics_type == 'fail' and not error:
        raise ValueError("[*] Para tipo 'fail', o parâmetro 'error' é obrigatório")

    if metrics_type == 'success' and not metrics_data:
        raise ValueError("[*] Para tipo 'success', 'metrics_data' é obrigatório")

    ES_HOST = os.getenv("ES_HOST", "http://elasticsearch:9200")
    ES_USER = os.getenv("ES_USER")
    ES_PASS = os.getenv("ES_PASS")

    if not all([ES_USER, ES_PASS]):
        raise ValueError("[*] Credenciais do Elasticsearch não configuradas")

    if metrics_type == 'fail':
        try:
            segmentos_unicos = [row["segmento"] for row in df.select("segmento").distinct().collect()] if df else ["UNKNOWN_CLIENT"]
        except Exception:
            logger.warning("[*] Não foi possível extrair segmentos. Usando 'UNKNOWN_CLIENT'.")
            segmentos_unicos = ["UNKNOWN_CLIENT"]

        document = {
            "timestamp": datetime.now().isoformat(),
            "layer": "silver",
            "project": "compass",
            "job": "apple_store_reviews",
            "priority": "0",
            "tower": "SBBR_COMPASS",
            "client": segmentos_unicos,
            "error": str(error) if error else "Erro desconhecido"
        }
    else:
        if isinstance(metrics_data, str):
            try:
                document = json.loads(metrics_data)
            except json.JSONDecodeError as e:
                raise ValueError("[*] metrics_data não é um JSON válido") from e
        else:
            document = metrics_data

    try:
        es = Elasticsearch(
            hosts=[ES_HOST],
            basic_auth=(ES_USER, ES_PASS),
            request_timeout=30
        )

        response = es.index(
            index=index,
            document=document
        )

        logger.info(f"[*] Métricas salvas com sucesso no índice {index}. ID: {response['_id']}")
        return response

    except Exception as es_error:
        logger.error(f"[*] Falha ao salvar no Elasticsearch: {str(es_error)}")
        raise
    except Exception as e:
        logger.error(f"[*] Erro inesperado: {str(e)}")
        raise

def path_exists() -> bool:
    """
    Verifica se o caminho de dados históricos existe no HDFS.
    
    Returns:
        bool: True se existir, False caso contrário
    """
    historical_data_path = "/santander/silver/compass/reviews/appleStore/"
    hdfs_path_exists = os.system(f"hadoop fs -test -e {historical_data_path} ") == 0

    if not hdfs_path_exists:
        logger.warning(f"[*] O caminho {historical_data_path} não existe no HDFS.")
        return False

    try:
        cmd = f"hdfs dfs -ls {historical_data_path}"
        result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)

        if "odate=" in result.stdout:
            logger.info("[*] Partições 'odate=*' encontradas no HDFS.")
            return True
        else:
            logger.info("[*] Nenhuma partição com 'odate=*' foi encontrada no HDFS.")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"[*] Erro ao acessar o HDFS: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"[*] Ocorreu um erro inesperado: {str(e)}")
        return False

def processing_old_new(spark: SparkSession, df: DataFrame) -> DataFrame:
    """
    Compara dados novos com históricos, gerando registro de alterações.
    
    Args:
        spark: Sessão Spark
        df: DataFrame com dados novos
        
    Returns:
        DataFrame com dados combinados e histórico de mudanças
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    historical_data_path = f"/santander/silver/compass/reviews/appleStore/"

    if path_exists():
        df_historical = (spark.read.option("basePath", historical_data_path)
                         .parquet(f"{historical_data_path}odate=*")
                         .withColumn("odate", to_date(col("odate"), "yyyyMMdd"))
                         .filter(f"odate < '{current_date}'")
                         .drop("odate"))
    else:
        schema = StructType([
            StructField("id", StringType(), True),
            StructField("name_client", StringType(), True),
            StructField("app", StringType(), True),
            StructField("im_version", StringType(), True),
            StructField("im_rating", StringType(), True),
            StructField("title", StringType(), True),
            StructField("content", StringType(), True),
            StructField("updated", StringType(), True),
            StructField("segmento", StringType(), True),
            StructField("historical_data", ArrayType(StringType()), True)
        ])
        df_historical = spark.createDataFrame([], schema)

    if 'segmento' not in df_historical.columns:
        df_historical = df_historical.withColumn("segmento", lit("NA"))


    """
    Aplicamos Salting que é técnica para evitar skew (desequilíbrio de diff de dados (files=particoes) no join.
    
        -> Adicionamos uma coluna extra com valores aleatórios (salt) para dividir a chave principal, distribuindo melhor os dados entre as tasks do Spark.
        -> Por exemplo: se a chave do join for 'id' e adicionarmos um 'salt' de 0 a 9, o join passa a ser feito por ('id', 'salt') em vez de apenas 'id'.
    
        Isso distribui melhor os dados entre as tasks do Spark, evitando sobrecarga em uma única partição.
    
    Após o join, a coluna 'salt' é removida para manter a integridade dos dados.
    """

    max_broadcast_size_mb = 50

    # Estimativa simples em bytes do histórico
    size_in_bytes = df_historical.rdd.map(lambda r: len(str(r))).sum()
    size_in_mb = size_in_bytes / (1024 * 1024)

    if size_in_mb <= max_broadcast_size_mb:
        # Se o DataFrame histórico for pequeno (até 64MB), usamos broadcast join.
        # O Spark envia o histórico para todos os executores, evitando shuffle.
        # Isso melhora a performance e evita problemas com partições desbalanceadas,
        # onde os dados antigos são leves e os dados novos são muito pesados.
        joined_reviews_df = df.alias("new").join(broadcast(df_historical).alias("old"), "id", "outer")
    else:
        # Se o histórico for grande, usamos salting para evitar skew no join.
        # Adicionamos uma coluna 'salt' com valores aleatórios de 0 a 9 em ambos os DataFrames.
        # O join é feito por (id, salt), espalhando a carga entre várias partições.
        # Isso distribui melhor os dados, evitando que uma única chave sobrecarregue uma task.
        salt_count = 10
        df_salted = df.withColumn("salt", (floor(rand() * salt_count)).cast("int"))
        df_hist_salted = df_historical.withColumn("salt", (floor(rand() * salt_count)).cast("int"))

        joined_reviews_df = df_salted.alias("new").join(df_hist_salted.alias("old"),(col("new.id") == col("old.id")) & (col("new.salt") == col("old.salt")),"outer").drop("new.salt", "old.salt")


    result_df = joined_reviews_df \
        .withColumn(
        "historical_data_temp",
        F.when(
            (F.col("new.title").isNotNull()) &
            (F.col("old.title").isNotNull()) &
            (F.col("new.title") != F.col("old.title")),
            F.array(F.struct(
                F.col("old.title").alias("title"),
                F.col("old.content").alias("content"),
                F.col("old.app").alias("app"),
                F.col("old.segmento").alias("segmento"),
                F.col("old.im_version").cast("string").alias("im_version"),
                F.col("old.im_rating").cast("string").alias("im_rating")
            ))
        ).when(
            (F.col("new.content").isNotNull()) &
            (F.col("old.content").isNotNull()) &
            (F.col("new.content") != F.col("old.content")),
            F.array(F.struct(
                F.col("old.title").alias("title"),
                F.col("old.content").alias("content"),
                F.col("old.app").alias("app"),
                F.col("old.segmento").alias("segmento"),
                F.col("old.im_version").cast("string").alias("im_version"),
                F.col("old.im_rating").cast("string").alias("im_rating")
            ))
        ).when(
            (F.col("new.im_version").isNotNull()) &
            (F.col("old.im_version").isNotNull()) &
            (F.col("new.im_version") != F.col("old.im_version")),
            F.array(F.struct(
                F.col("old.title").alias("title"),
                F.col("old.content").alias("content"),
                F.col("old.app").alias("app"),
                F.col("old.segmento").alias("segmento"),
                F.col("old.im_version").cast("string").alias("im_version"),
                F.col("old.im_rating").cast("string").alias("im_rating")
            ))
        ).when(
            (F.col("new.im_rating").isNotNull()) &
            (F.col("old.im_rating").isNotNull()) &
            (F.col("new.im_rating") != F.col("old.im_rating")),
            F.array(F.struct(
                F.col("old.title").alias("title"),
                F.col("old.content").alias("content"),
                F.col("old.app").alias("app"),
                F.col("old.segmento").alias("segmento"),
                F.col("old.im_version").cast("string").alias("im_version"),
                F.col("old.im_rating").cast("string").alias("im_rating")
            ))
        ).otherwise(None)
    ).distinct()

    df_final = result_df.groupBy("id").agg(
        F.coalesce(F.first("new.name_client"), F.first("old.name_client")).alias("name_client"),
        F.coalesce(F.first("new.app"), F.first("old.app")).alias("app"),
        F.coalesce(F.first("new.im_version"), F.first("old.im_version")).alias("im_version"),
        F.coalesce(F.first("new.im_rating"), F.first("old.im_rating")).alias("im_rating"),
        F.coalesce(F.first("new.title"), F.first("old.title")).alias("title"),
        F.coalesce(F.first("new.content"), F.first("old.content")).alias("content"),
        F.first("new.updated").alias("updated"),
        F.first("new.segmento").alias("segmento"),
        F.flatten(F.collect_list("historical_data_temp")).alias("historical_data")
    )

    logger.info("[*] Visao final do processing_reviews")
    df_final.show()

    return df_final