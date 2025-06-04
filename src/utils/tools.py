import json
import os
import logging
import subprocess
import pyspark.sql.functions as F
from urllib.parse import quote_plus
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import coalesce, collect_list, struct, array, col, count, when, to_date, regexp_extract, input_file_name, lit
from pyspark.sql.utils import AnalysisException
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType
from datetime import datetime
from pathlib import Path
from unidecode import unidecode
from elasticsearch import Elasticsearch
from typing import Optional, Union
import traceback

# Função para remover acentos
def remove_accents(s):
    return unidecode(s)

remove_accents_udf = F.udf(remove_accents, StringType())

def read_source_parquet(spark, path):
    """Tenta ler um Parquet e retorna None se não houver dados"""
    try:
        df = spark.read.parquet(path)
        if df.isEmpty():
            logging.error(f"[*] Nenhum dado encontrado em: {path}")
            return None
        return df.withColumn("app", regexp_extract(input_file_name(), "/appleStore/(.*?)/odate=", 1)) \
                 .withColumn("segmento", regexp_extract(input_file_name(), r"/appleStore/[^/_]+_([pfj]+)/odate=", 1))
    except AnalysisException:
        logging.error(f"[*] Falha ao ler: {path}. O arquivo pode não existir.")
        return None

# Função de processamento
def processing_reviews(df: DataFrame):
    logging.info(f"{datetime.now().strftime('%Y%m%d %H:%M:%S.%f')} [*] Processando o tratamento da camada historica")

    # Aplicando as transformações no DataFrame
    df_select = df.select(
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

    return df_select

def get_schema(df, schema):
    """
    Obtém o DataFrame a seguir o schema especificado.
    """
    for field in schema.fields:
        if field.dataType == IntegerType():
            df = df.withColumn(field.name, df[field.name].cast(IntegerType()))
        elif field.dataType == StringType():
            df = df.withColumn(field.name, df[field.name].cast(StringType()))
    return df.select([field.name for field in schema.fields])

def save_dataframe(df: DataFrame,
                   path: str,
                   label: str,
                   schema: Optional[dict] = None,
                   partition_column: str = "odate",
                   compression: str = "snappy") -> bool:
    """
    Salva um DataFrame Spark no formato Parquet de forma robusta e profissional.

    Parâmetros:
        df: DataFrame Spark a ser salvo
        path: Caminho de destino para salvar os dados
        label: Identificação do tipo de dados para logs (ex: 'valido', 'invalido')
        schema: Schema opcional para validação dos dados
        partition_column: Nome da coluna de partição (default: 'odate')
        compression: Tipo de compressão (default: 'snappy')

    Retorno:
        bool: True se salvou com sucesso, False caso contrário

    Exceções:
        ValueError: Se os parâmetros obrigatórios forem inválidos
        IOError: Se houver problemas ao escrever no filesystem
    """
    # Validação dos parâmetros
    if not isinstance(df, DataFrame):
        logging.error(f"Objeto passado não é um DataFrame Spark: {type(df)}")
        return False

    if not path:
        logging.error("Caminho de destino não pode ser vazio")
        return False

    # Configuração
    current_date = datetime.now().strftime('%Y%m%d')
    full_path = Path(path)

    try:
        # Aplicar schema se fornecido
        if schema:
            logging.info(f"[*] Aplicando schema para dados {label}")
            df = get_schema(df, schema)

        # Adicionar coluna de partição
        df_partition = df.withColumn(partition_column, lit(current_date))

        # Verificar se há dados
        if not df_partition.head(1):
            logging.warning(f"[*] Nenhum dado {label} encontrado para salvar")
            return False

        # Preparar diretório
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            logging.debug(f"[*] Diretório {full_path} verificado/criado")
        except Exception as dir_error:
            logging.error(f"[*] Falha ao preparar diretório {full_path}: {dir_error}")
            raise IOError(f"[*] Erro de diretório: {dir_error}") from dir_error

        # Escrever dados
        logging.info(f"[*] Salvando {df_partition.count()} registros ({label}) em {full_path}")

        (df_partition.write
         .option("compression", compression)
         .mode("overwrite")
         .partitionBy(partition_column)
         .parquet(str(full_path)))

        logging.info(f"[*] Dados {label} salvos com sucesso em {full_path}")
        return True

    except Exception as e:
        error_msg = f"[*] Falha ao salvar dados {label} em {full_path}"
        logging.error(error_msg, exc_info=True)
        logging.error(f"[*] Detalhes do erro: {str(e)}\n{traceback.format_exc()}")
        return False

def save_metrics(metrics_type: str,
                 index: str,
                 error: Optional[Exception] = None,
                 df: Optional[DataFrame] = None,
                 metrics_data: Optional[Union[dict, str]] = None) -> None:
    """
    Salva métricas no Elasticsearch mantendo estruturas específicas para cada tipo.

    Parâmetros:
        metrics_type: 'success' ou 'fail' (case insensitive)
        index: Nome do índice no Elasticsearch
        error: Objeto de exceção (obrigatório para tipo 'fail')
        client: Nome do cliente (opcional)
        metrics_data: Dados das métricas (obrigatório para 'success', ignorado para 'fail')

    Estruturas:
        - FAIL: Mantém estrutura fixa com informações de erro
        - SUCCESS: Mantém exatamente o que foi passado em metrics_data
    """
    # Converter para minúsculas para padronização
    metrics_type = metrics_type.lower()

    # Validações iniciais
    if metrics_type not in ('success', 'fail'):
        raise ValueError("[*] O tipo deve ser 'success' ou 'fail'")

    if metrics_type == 'fail' and not error:
        raise ValueError("[*] Para tipo 'fail', o parâmetro 'error' é obrigatório")

    if metrics_type == 'success' and not metrics_data:
        raise ValueError("[*] Para tipo 'success', 'metrics_data' é obrigatório")

    # Configuração do Elasticsearch
    ES_HOST = os.getenv("ES_HOST", "http://elasticsearch:9200")
    ES_USER = os.getenv("ES_USER")
    ES_PASS = os.getenv("ES_PASS")

    if not all([ES_USER, ES_PASS]):
        raise ValueError("[*] Credenciais do Elasticsearch não configuradas")

    # Construção do documento
    if metrics_type == 'fail':
        try:
            segmentos_unicos = [row["segmento"] for row in df.select("segmento").distinct().collect()] if df else ["UNKNOWN_CLIENT"]
        except Exception:
            logging.warning("[*] Não foi possível extrair segmentos. Usando 'UNKNOWN_CLIENT'.")
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

    # Conexão e envio para Elasticsearch
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

        logging.info(f"[*] Métricas salvas com sucesso no índice {index}. ID: {response['_id']}")
        return response

    except Exception as es_error:
        logging.error(f"[*] Falha ao salvar no Elasticsearch: {str(es_error)}")
        raise
    except Exception as e:
        logging.error(f"[*] Erro inesperado: {str(e)}")
        raise

def path_exists() -> bool:

    # Caminho para os dados históricos
    historical_data_path = "/santander/silver/compass/reviews/appleStore/"

    # Verificando se o caminho existe no HDFS
    hdfs_path_exists = os.system(f"hadoop fs -test -e {historical_data_path} ") == 0

    if not hdfs_path_exists:
        logging.warning(f"[*] O caminho {historical_data_path} não existe no HDFS.")
        return False  # Retorna False se o caminho não existir no HDFS

    try:
        # Comando para listar os diretórios no HDFS
        cmd = f"hdfs dfs -ls {historical_data_path}"

        # Executar o comando HDFS
        result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)

        # Verificar se há partições "odate="
        if "odate=" in result.stdout:
            logging.info("[*] Partições 'odate=*' encontradas no HDFS.")
            return True  # Retorna True se as partições forem encontradas
        else:
            logging.info("[*] Nenhuma partição com 'odate=*' foi encontrada no HDFS.")
            return False  # Retorna False se não houver partições

    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao acessar o HDFS: {e.stderr}")
        return False  # Retorna False se ocorrer erro ao acessar o HDFS
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado: {str(e)}")
        return False  # Retorna False para outros erros


def processing_old_new(spark: SparkSession, df: DataFrame):
    """
    Compara os dados novos de avaliações de um aplicativo com os dados antigos já armazenados. 
    Quando há diferenças entre os novos e os antigos, essas mudanças são salvas em uma nova coluna chamada 'historical_data'.
    O objetivo é manter um registro das alterações nas avaliações ao longo do tempo, permitindo ver como os dados evoluíram.

    
    Args:
        spark (SparkSession): A sessão ativa do Spark para ler e processar os dados.
        df (DataFrame): DataFrame contendo os novos dados de reviews.

    Returns:
        DataFrame: DataFrame resultante contendo os dados combinados e o histórico de mudanças.
    """


    # Obtenha a data atual
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Caminho para os dados históricos
    historical_data_path = f"/santander/silver/compass/reviews/appleStore/"

   # Verificando se o caminho existe no HDFS
    hdfs_path_exists = path_exists()

    if hdfs_path_exists:
        # Se o caminho existir, lê os dados do Parquet e ajusta os tipos
        df_historical = (spark.read.option("basePath", historical_data_path)
                           .parquet(f"{historical_data_path}odate=*")
                           .withColumn("odate", to_date(col("odate"), "yyyyMMdd"))
                           .filter(f"odate < '{current_date}'") # Nao considerar o odate atual em caso de reprocessamento
                           .drop("odate")) # Para nao trazer duplicados no output
    else:
        # Se o caminho não existir, cria um DataFrame vazio com o mesmo esquema esperado
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

    # Definindo aliases para os DataFrames
    new_reviews_df_alias = df.alias("new")  # DataFrame de novos reviews
    historical_reviews_df_alias = df_historical.alias("old")  # DataFrame de reviews históricos



    # Junção dos DataFrames
    joined_reviews_df = new_reviews_df_alias.join(historical_reviews_df_alias, "id", "outer")

    # Criação da coluna historical_data
    result_df = joined_reviews_df \
        .withColumn(
            "historical_data_temp",
            # Cria uma nova coluna 'historical_data_temp' com informações de dados históricos
            # se houver diferenças nas colunas relevantes entre os DataFrames novo e antigo.
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
            ).otherwise(
                None  # Quando não houver diferenças
            )
        ).distinct()

    # Agrupando e coletando históricos
    df_final = result_df.groupBy("id").agg(
        # Coleta o primeiro valor não nulo das colunas relevantes dos DataFrames novo e antigo
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


    logging.info(f"[*] Visao final do processing_reviews", exc_info=True)
    df_final.show()
    
    return df_final

