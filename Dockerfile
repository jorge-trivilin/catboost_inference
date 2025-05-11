FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/conda/bin:${PATH}"

# Install OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget bzip2 ca-certificates python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download, verify, and install Miniconda
ARG MINICONDA_VERSION="latest"
ARG MINICONDA_SHA256="53a86109463cfd70ba7acab396d416e623012914eee004729e1ecd6fe94e8c69"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo "${MINICONDA_SHA256}  /tmp/miniconda.sh" | sha256sum --check && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm -f /tmp/miniconda.sh

# Clean up conda caches
RUN conda clean --all --yes

# Install Python packages via pip (using conda's pip)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy and configure your application
COPY container /opt/program
WORKDIR /opt/program    
RUN chmod +x serve

ENTRYPOINT ["python", "serve"]
