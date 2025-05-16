FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    wget \
    curl \
    git \
    unzip \
    nginx \
    ca-certificates \
    gpg-agent \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

RUN apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python /usr/bin/python3.12 && \
    update-alternatives --set python3 /usr/bin/python3.12

RUN python --version && \
    pip3 --version

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    /usr/local/bin/aws --version && \
    rm -rf awscliv2.zip aws


RUN pip3 install --no-cache-dir --ignore-installed  \
    numpy>=1.26.0 \
    scipy>=1.11.3 \
    pandas>=2.1.0 \
    scikit-learn>=1.3.2 \
    pillow>=10.1.0 \
    cython>=3.0.0 \
    catboost==1.2.8 \
    shap>=0.43.0 \
    flask \
    gevent \
    gunicorn \
    boto3 \
    s3fs \
    joblib \
    loguru 


RUN mkdir -p /opt/ml/input/config /opt/ml/input/data/training /opt/ml/model /opt/ml/output /opt/program


ENV PYTHONPATH=/opt/program
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV SAGEMAKER_PROGRAM=serve


COPY code/* /opt/program/
WORKDIR /opt/program/
RUN chmod +x /opt/program/serve

ENTRYPOINT ["python", "serve"]