#!/bin/bash

# Função para imprimir logs com timestamp
log() {
  local message=$1
  echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO] $message"
}

# Função para tratar erros
error_exit() {
  local message=$1
  echo "$(date +"%Y-%m-%d %H:%M:%S") [ERROR] $message" >&2
  exit 1
}

# Função para validar as variáveis necessárias
validate_params() {
  if [[ -z "$SPARK_HOME" ]]; then
    error_exit "Variável SPARK_HOME não definida!"
  fi
}

# Função para montar e executar o Spark Submit
run_spark_submit() {

  if [ -f /app/.env ]; then
      export $(grep -v '^#' /app/.env | xargs)
  else
    echo "Arquivo .env não encontrado!"
    exit 1
  fi

  local spark_cmd="$SPARK_HOME/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --name 'Data Master - Compass - Generator Feedbacks to Mongo' \
    --conf spark.executor.memory=4g \
    --conf spark.driver.memory=4g \
    --conf spark.executor.cores=2 \
    --conf spark.driver.cores=2 \
    --conf spark.network.timeout=600s \
    --conf spark.rpc.askTimeout=600s \
    --conf spark.pyspark.python=/usr/bin/python3 \
    --conf spark.pyspark.driver.python=/usr/bin/python3 \
    --conf 'spark.driver.extraJavaOptions=-Dcom.sun.management.jmxremote -Dcom.sun.management.jmxremote.port=8090 -Dcom.sun.management.jmxremote.rmi.port=8091 -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Djava.rmi.server.hostname=10.101.34.131' \
    --conf spark.metrics.conf=/usr/local/spark/conf/metrics.properties \
    --conf spark.ui.prometheus.enabled=true \
    --conf spark.executor.processTreeMetrics.enabled=true \
    --packages org.mongodb.spark:mongo-spark-connector_2.12:2.4.1,ch.cern.sparkmeasure:spark-measure_2.12:0.16 \
    --py-files /app/dependencies.zip,/app/metrics.py,/app/tools.py \
    --conf spark.executorEnv.MONGO_USER=$MONGO_USER \
    --conf spark.executorEnv.MONGO_PASS=$MONGO_PASS\
    --conf spark.executorEnv.MONGO_HOST=mongodb \
    --conf spark.executorEnv.MONGO_PORT=27017 \
    --conf spark.executorEnv.MONGO_DB=compass \
    --conf spark.driverEnv.MONGO_USER=$MONGO_USER \
    --conf spark.driverEnv.MONGO_PASS=$MONGO_PASS\
    --conf spark.driverEnv.MONGO_HOST=mongodb \
    --conf spark.driverEnv.MONGO_PORT=27017 \
    --conf spark.driverEnv.MONGO_DB=compass \
    --conf spark.yarn.appMasterEnv.MONGO_USER=$MONGO_USER \
    --conf spark.yarn.appMasterEnv.MONGO_PASS=$MONGO_PASS\
    --conf spark.yarn.appMasterEnv.MONGO_HOST=mongodb \
    --conf spark.yarn.appMasterEnv.MONGO_PORT=27017 \
    --conf spark.yarn.appMasterEnv.MONGO_DB=compass \
    --name dmc-app-silver-reviews-apple-store \
   /app/app-code-compass-processing-hist-apple-store.py"

  # Exibe o comando para depuração
  log "Comando spark-submit que será executado: $spark_cmd"

  # Executa o Spark Submit e captura o código de retorno
  eval $spark_cmd
  local exit_code=$?

  if [[ $exit_code -ne 0 ]]; then
    error_exit "Falha ao executar o Spark Submit (código de saida: $exit_code)."
  else
    log "Spark Submit executado com sucesso!"
  fi
}


# Início do Script
log "************************************************************"
log "Iniciando Execução de Spark Submit"
log "************************************************************"

# Valida as variáveis de ambiente e parâmetros
validate_params

# Executa o Spark Submit
run_spark_submit

log "************************************************************"
log "Finalizando Execução de Spark Submit"
log "************************************************************"