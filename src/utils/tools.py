import json
import os
import logging
import pymongo
import subprocess
import pyspark.sql.functions as F
from urllib.parse import quote_plus
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import coalesce, collect_list, struct, array, col, count, when, to_date
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType
from datetime import datetime
from pathlib import Path
from unidecode import unidecode
from src.schema.schema_apple import apple_store_schema_silver

# Função para remover acentos
def remove_accents(s):
    return unidecode(s)

remove_accents_udf = F.udf(remove_accents, StringType())


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


def save_reviews(reviews_df: DataFrame, directory: str):
    """
    Salva os dados do DataFrame no formato Delta no diretório especificado.

    Args:
        reviews_df (DataFrame): DataFrame PySpark contendo as avaliações.
        directory (str): Caminho do diretório onde os dados serão salvos.
    """
    try:
        # Verifica se o diretório existe e cria-o se não existir
        Path(directory).mkdir(parents=True, exist_ok=True)

        reviews_df.write.option("compression", "snappy").mode("overwrite").parquet(directory)
        logging.info(f"[*] Dados salvos em {directory} no formato Delta")
    except Exception as e:
        logging.error(f"[*] Erro ao salvar os dados: {e}")
        exit(1)


def save_dataframe(df, path, label):
    """
    Salva o DataFrame em formato parquet e loga a operação.
    """
    try:
        schema = apple_store_schema_silver()

        # Alinhar o DataFrame ao schema definido
        df = get_schema(df, schema)

        if df.limit(1).count() > 0:  # Verificar existência de dados
            logging.info(f"[*] Salvando dados {label} para: {path}")
            df.printSchema()
            save_reviews(df, path)
        else:
            logging.warning(f"[*] Nenhum dado {label} foi encontrado!")
    except Exception as e:
        logging.error(f"[*] Erro ao salvar dados {label}: {e}", exc_info=True)

def save_metrics(spark, metrics_json):
    """
    Salva as métricas no MongoDB.
    """
    try:
        metrics_data = json.loads(metrics_json)
        write_to_mongo(metrics_data, "dt_datametrics_compass")
        logging.info(f"[*] Métricas da aplicação salvas: {metrics_json}")
    except json.JSONDecodeError as e:
        logging.error(f"[*] Erro ao processar métricas: {e}", exc_info=True)

def write_to_mongo(dados_feedback: dict, table_id: str):
    """
    Salva os dados no MongoDB, obtendo as credenciais pelas variaveis de ambiente.
    Args:
        dados_feedback (dict): DataFrame PySpark contendo as avaliações.
        table_id (str): Caminho do diretório onde os dados serão salvos.
    """

    mongo_user = os.environ["MONGO_USER"]
    mongo_pass = os.environ["MONGO_PASS"]
    mongo_host = os.environ["MONGO_HOST"]
    mongo_port = os.environ["MONGO_PORT"]
    mongo_db = os.environ["MONGO_DB"]

    # ---------------------------------------------- Escapar nome de usuário e senha ----------------------------------------------
    # A função quote_plus transforma caracteres especiais em seu equivalente escapado, de modo que o 
    # URI seja aceito pelo MongoDB. Por exemplo, m@ngo será convertido para m%40ngo.
    escaped_user = quote_plus(mongo_user)
    escaped_pass = quote_plus(mongo_pass)


    # ---------------------------------------------- Conexão com MongoDB ----------------------------------------------------------
    # Quando definimos maxPoolSize=1, estamos dizendo ao MongoDB para manter apenas uma conexão aberta no pool. 
    # Isso implica que cada vez que uma nova operação precisa de uma conexão, a conexão existente será 
    # reutilizada em vez de criar uma nova.
    mongo_uri = f"mongodb://{escaped_user}:{escaped_pass}@{mongo_host}:{mongo_port}/{mongo_db}?authSource={mongo_db}&maxPoolSize=1"

    client = pymongo.MongoClient(mongo_uri)

    try:
        db = client[mongo_db]
        collection = db[table_id]

        # Inserir dados no MongoDB
        if isinstance(dados_feedback, dict):  # Verifica se os dados são um dicionário
            collection.insert_one(dados_feedback)
        elif isinstance(dados_feedback, list):  # Verifica se os dados são uma lista
            collection.insert_many(dados_feedback)
        else:
            print("[*] Os dados devem ser um dicionário ou uma lista de dicionários.")
    finally:
        # Garante que a conexão será fechada
        client.close()

 

def path_exists() -> bool:

    # Caminho para os dados históricos
    historical_data_path = "/santander/silver/compass/reviews/appleStore/"

    # Verificando se o caminho existe no HDFS
    hdfs_path_exists = os.system(f"hadoop fs -test -e {historical_data_path} ") == 0

    if not hdfs_path_exists:
        print(f"[*] O caminho {historical_data_path} não existe no HDFS.")
        return False  # Retorna False se o caminho não existir no HDFS

    try:
        # Comando para listar os diretórios no HDFS
        cmd = f"hdfs dfs -ls {historical_data_path}"

        # Executar o comando HDFS
        result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)

        # Verificar se há partições "odate="
        if "odate=" in result.stdout:
            print("[*] Partições 'odate=*' encontradas no HDFS.")
            return True  # Retorna True se as partições forem encontradas
        else:
            print("[*] Nenhuma partição com 'odate=*' foi encontrada no HDFS.")
            return False  # Retorna False se não houver partições

    except subprocess.CalledProcessError as e:
        print(f"Erro ao acessar o HDFS: {e.stderr}")
        return False  # Retorna False se ocorrer erro ao acessar o HDFS
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {str(e)}")
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
            StructField("historical_data", ArrayType(StringType()), True)
        ])
        df_historical = spark.createDataFrame([], schema)

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
                    F.col("old.im_version").cast("string").alias("im_version"),
                    F.col("old.im_rating").cast("string").alias("im_rating")
                ))
            ).otherwise(
                None  # Quando não houver diferenças
            )
        )

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
        # Coleta uma lista de todos os dados históricos
        F.collect_list("historical_data_temp").alias("historical_data")  
    )

    df_final.show()
    
    return df_final

