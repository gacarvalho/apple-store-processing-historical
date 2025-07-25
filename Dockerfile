# Usando ARG para passar a variável de pasta durante o build
ARG FOLDER_REPO="dmc-app-silver-reviews-apple-store"

FROM iamgacarvalho/spark-base-data-in-compass:0.0.3

ENV SPARK_MASTER_URL="spark://spark-master:7077"
ENV SPARK_SUBMIT_ARGS=""
ENV SPARK_APPLICATION_ARGS=""
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3

# Definindo o PYTHONPATH com os caminhos específicos
ENV PYTHONPATH="/app:$PYTHONPATH"

# Copiar os scripts e código para o diretório /app
RUN mkdir /app/

COPY conf/spark-default.conf /usr/local/spark/conf/
COPY spark-conf/* /app/
COPY /src/* /app/
COPY spark-submit.sh /app/spark-submit.sh
COPY requirements.txt /app/requirements.txt


# Corrigir os repositórios desatualizados do Debian Buster
RUN sed -i 's|http://deb.debian.org/debian|http://archive.debian.org/debian|g' /etc/apt/sources.list \
    && sed -i 's|http://security.debian.org/debian-security|http://archive.debian.org/debian-security|g' /etc/apt/sources.list \
    && apt-get update -o Acquire::Check-Valid-Until=false \
    && apt-get install -y zip \
    && python3 -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt


# Criar o arquivo .zip com as dependências Python
RUN pip install --target /app/dependencies -r /app/requirements.txt \
    && cd /app/dependencies \
    && zip -r /app/dependencies.zip .

RUN wget https://repo1.maven.org/maven2/com/sun/jersey/jersey-client/1.19.4/jersey-client-1.19.4.jar -P /usr/local/spark/jars/ && \
    wget https://repo1.maven.org/maven2/com/sun/jersey/jersey-core/1.19.4/jersey-core-1.19.4.jar -P /usr/local/spark/jars/

RUN apt-get update && apt-get install -y \
    curl \
    && curl -L https://github.com/mikefarah/yq/releases/download/v4.15.1/yq_linux_amd64 -o /usr/bin/yq \
    && chmod +x /usr/bin/yq

# Garantir que o script tenha permissões executáveis
RUN chmod +x /app/spark-submit.sh

# Definir o comando CMD para executar o script
CMD ["/bin/bash", "/app/spark-submit.sh"]
