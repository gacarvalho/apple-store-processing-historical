🧭 ♨️ COMPASS
---

<p align="left">
  <img src="https://img.shields.io/badge/projeto-Compass-blue?style=flat-square" alt="Projeto">
  <img src="https://img.shields.io/badge/versão aplicação-1.0.1-blue?style=flat-square" alt="Versão Aplicação">
  <img src="https://img.shields.io/badge/status-deployed-green?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/autor-Gabriel_Carvalho-lightgrey?style=flat-square" alt="Autor">
</p>

Essa aplicação faz parte do projeto **compass-deployment** que é uma solução desenvolvida no contexto do programa Data Master, promovido pela F1rst Tecnologia, com o objetivo de disponibilizar uma plataforma robusta e escalável para captura, processamento e análise de feedbacks de clientes do Banco Santander.


![<data-master-compass>](https://github.com/gacarvalho/repo-spark-delta-iceberg/blob/main/header.png?raw=true)



`📦 artefato` `iamgacarvalho/dmc-app-silver-reviews-apple-store`

- **Versão:** `1.0.1`
- **Repositório:** [GitHub](https://github.com/gacarvalho/apple-store-processing-historical)
- **Imagem Docker:** [Docker Hub](https://hub.docker.com/repository/docker/iamgacarvalho/dmc-app-silver-reviews-apple-store/tags/1.0.1/sha256-a35d88d3c69b78abcecfff0a53906201fab48bdd8b2e5579057e935f58b6fe41)
- **Descrição:**  Coleta avaliações de clientes nos canais via API do Itunes na Apple Store ingeridos no Data Lake, realizando a ingestão a partir da camada Bronze, processando e aplicando tratamento de dados e armazenando no HDFS em formato **Parquet**.
- **Parâmetros:**


    - `$CONFIG_ENV` (`Pre`, `Pro`) → Define o ambiente: `Pre` (Pré-Produção), `Pro` (Produção).

| Componente          | Descrição                                                                            |
|---------------------|--------------------------------------------------------------------------------------|
| **Objetivo**        | Coletar, processar e armazenar avaliações de apps da Apple Store para a camada Silver |
| **Entrada**         | Ambiente (pre/prod)                                                                  |
| **Saída**           | Dados válidos/inválidos em Parquet + métricas no Elasticsearch                       |
| **Tecnologias**     | PySpark, Elasticsearch, Parquet, SparkMeasure                                        |
| **Fluxo Principal** | 1. Coleta dos dados brutos → 2. Aplica padronização → 3. Separação → 4. Armazenamento |
| **Validações**      | Duplicatas, nulos em campos críticos, consistência de tipos                          |
| **Particionamento** | Por data referencia de carga (odate)                                                 |
| **Métricas**        | Tempo execução, memória, registros válidos/inválidos, performance Spark              |
| **Tratamento Erros**| Logs detalhados, armazenamento separado de dados inválidos                           |
| **Execução**        | `spark-submit repo_extc_apple_store.py <env>`             |
