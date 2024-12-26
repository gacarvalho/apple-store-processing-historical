
import pytest
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from datetime import datetime
from pyspark.sql.functions import input_file_name, regexp_extract, lit
from unittest.mock import MagicMock, patch
from src.utils.tools import processing_reviews, save_dataframe
from src.metrics.metrics import validate_ingest



@pytest.fixture(scope="session")
def spark():
    """
    Fixture que inicializa o SparkSession para os testes.
    """
    spark = SparkSession.builder.master("local").appName("TestApp").getOrCreate()
    yield spark
    spark.stop()

def apple_store_schema_bronze():
    """
    Define o esquema para os dados da Apple Store na camada bronze.

    Retorna:
        StructType: Estrutura do DataFrame com os campos e tipos correspondentes.
    """
    return StructType([
        StructField('author_name', StringType(), True),
        StructField('author_uri', StringType(), True),
        StructField('content', StringType(), True),
        StructField('content_attributes_label', StringType(), True),
        StructField('content_attributes_term', StringType(), True),
        StructField('id', StringType(), True),
        StructField('im_rating', StringType(), True),
        StructField('im_version', StringType(), True),
        StructField('im_votecount', StringType(), True),
        StructField('im_votesum', StringType(), True),
        StructField('link_attributes_href', StringType(), True),
        StructField('link_attributes_related', StringType(), True),
        StructField('title', StringType(), True),
        StructField('updated', StringType(), True)
    ])



# Configuração do pytest para o Spark
@pytest.fixture(scope="session")
def spark():
    # Inicializar uma SparkSession para os testes
    return SparkSession.builder.master("local[1]").appName("TestReadData").getOrCreate()

def data_apple():
    return [
        ("keneddy santos", "https://itunes.apple.com/br/reviews/id1291186914", "Está na hora de atualizar para acessar com o Face ID , seria bom se fosse possível comprar euro pelo app iria facilitar mais para nos clientes", "Aplicativo", "Application", "11670010101", "4", "24.7.7", "0", "0", "https://itunes.apple.com/br/review?id=6462515499&type=Purple%20Software", "related", "Face ID", "2024-08-30T08:10:10-07:00"),
        ("keneddy santos", "https://itunes.apple.com/br/reviews/id1291186914", "Está na hora de atualizar para acessar com o Face ID , seria bom se fosse possível comprar euro pelo app iria facilitar mais para nos clientes", "Aplicativo", "Application", "11670010101", "4", "24.7.7", "0", "0", "https://itunes.apple.com/br/review?id=6462515499&type=Purple%20Software", "related", "Face ID", "2024-08-30T08:10:10-07:00"),
        ("Vanessawu866", "https://itunes.apple.com/br/reviews/id469508303", "Taxas competitivas, aplicativo super rápido e prático!", "Aplicativo", "Application", "11480999346", "5", "24.6.2", "0", "0", "https://itunes.apple.com/br/review?id=6462515499&type=Purple%20Software", "related", "Ótimas taxas!", "2024-07-11T07:54:53-07:00"),
        ("Grbdsrai", "https://itunes.apple.com/br/reviews/id1513508851", "Só pelo fato de eu ser cliente e não conseguir usar o app imagino como deve ser a usabilidade", "Aplicativo", "Application", "12018236188", "1", "24.10.1", "0", "0", "https://itunes.apple.com/br/review?id=6462515499&type=Purple%20Software", "related", "Não funciona", "2024-12-02T11:15:58-07:00"),
        ("renatahoris", "https://itunes.apple.com/br/reviews/id364090042", "Fora do mercado! Dólar 5,55 nas outras instituições e 5,67 no app", "Aplicativo", "Application", "11589888946", "2", "24.7.7", "0", "0", "https://itunes.apple.com/br/review?id=6462515499&type=Purple%20Software", "related", "Pessima conversão", "2024-08-09T04:58:18-07:00"),
        ("renatahoris", "https://itunes.apple.com/br/reviews/id364090042", "Fora do mercado! Dólar 5,55 nas outras instituições e 5,67 no app", "Aplicativo", "Application", "11589888946", "2", "24.7.7", "0", "0", "https://itunes.apple.com/br/review?id=6462515499&type=Purple%20Software", "related", "Pessima conversão", "2024-08-09T04:58:18-07:00"),
        ("Usuario123@1", "https://itunes.apple.com/br/reviews/id173146519", "Utilizei na Europa sem nenhum problema. Passou em todos os lugares. Muito prático. Agora que tem Apple Pay vai ser melhor ainda!", "Aplicativo", "Application", "11745305763", "5", "24.8.3", "0", "0", "https://itunes.apple.com/br/review?id=6462515499&type=Purple%20Software", "related", "Ótima experiência!", "2024-09-20T09:58:57-07:00")
    ]


# Teste unitário para a função read_data
def test_read_data(spark):
    # Criando um DataFrame de teste com dados fictícios
    test_data = data_apple()

    schema = apple_store_schema_bronze()
    # Criar o DataFrame com os dados de teste
    df_test = spark.createDataFrame(test_data, schema)

    # Salvar o DataFrame como um arquivo Parquet temporário
    datePath = datetime.now().strftime("%Y%m%d")
    test_parquet_path = f"/tmp/test_apple_data/appleStore/banco-santander-br/odate={datePath}/"
    df_test.write.mode("overwrite").parquet(test_parquet_path)

    df = spark.read.parquet(test_parquet_path).withColumn("app", regexp_extract(input_file_name(), "/appleStore/(.*?)/odate=", 1))

    df_processado = processing_reviews(df)

    # Verifique se o número de registros no DataFrame é o esperado
    assert df_processado.count() == 7, f"Esperado 7 registros, mas encontrou {df_processado.count()}."

# Teste unitário para a função de extração
def test_processamento_reviews(spark):
    # Criando um DataFrame de teste com dados fictícios
    test_data = data_apple()

    schema = apple_store_schema_bronze()
    # Criar o DataFrame com os dados de teste
    df_test = spark.createDataFrame(test_data, schema)

    # Salvar o DataFrame como um arquivo Parquet temporário
    datePath = datetime.now().strftime("%Y%m%d")
    test_parquet_path = f"/tmp/test_apple_data/odate={datePath}/"
    df_test.write.mode("overwrite").parquet(test_parquet_path)

    df = spark.read.parquet(test_parquet_path).withColumn("app", regexp_extract(input_file_name(), "/appleStore/(.*?)/odate=", 1))


    df_processado = processing_reviews(df)

    # Verifica se a volumetria é maior que 0 e o df está preenchido
    assert df_processado.count() > 0, "[*] Dataframe vazio"


def test_validate_ingest(spark):
    """
    Testa a função de validação de ingestão para garantir que os DataFrames têm dados e que a validação gera resultados.
    """
    # Criando um DataFrame de teste com dados fictícios
    test_data = data_apple()

    schema = apple_store_schema_bronze()
    # Criar o DataFrame com os dados de teste
    df = spark.createDataFrame(test_data, schema)

    df = df.withColumn("app", lit("banco-santander-br"))

    df_processado = processing_reviews(df)

    # Valida o DataFrame e coleta resultados
    valid_df, invalid_df, validation_results = validate_ingest(spark, df_processado)

    print(f"Total de registros válidos: {valid_df.count()}")
    print(f"Total de registros inválidos: {invalid_df.count()}")
    print(f"Resultados da validação: {validation_results}")


    assert valid_df.count() > 0, "[*] O DataFrame válido está vazio!"
    assert len(validation_results) > 0, "[*] Não foram encontrados resultados de validação!"

    # Exibir resultados para depuração
    print("Testes realizados com sucesso!")

def test_save_data(spark):
    ## Criando um DataFrame de teste com dados fictícios
    test_data = data_apple()

    schema = apple_store_schema_bronze()
    # Criar o DataFrame com os dados de teste
    df = spark.createDataFrame(test_data, schema)

    df = df.withColumn("app", lit("banco-santander-br"))

    df_processado = processing_reviews(df)

    # Valida o DataFrame e coleta resultados
    valid_df, invalid_df, validation_results = validate_ingest(spark, df_processado)

    # Definindo caminhos
    datePath = datetime.now().strftime("%Y%m%d")
    path_target = f"/tmp/fake/path/valid/odate={datePath}/"

    # Mockando o método parquet
    with patch("pyspark.sql.DataFrameWriter.parquet", MagicMock()) as mock_parquet:
        # Chamando a função a ser testada
        save_dataframe(valid_df, path_target, "bronze")

        # Verificando se o método parquet foi chamado com os caminhos corretos
        mock_parquet.assert_any_call(path_target)

    print("[*] Teste de salvar dados concluído com sucesso!")