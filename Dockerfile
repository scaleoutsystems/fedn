# Base image
ARG BASE_IMG=python:3.10-slim
FROM $BASE_IMG

ARG GRPC_HEALTH_PROBE_VERSION=""

# Requirements (use MNIST Keras as default)
ARG REQUIREMENTS=""

# Add FEDn and default configs
COPY . /app
COPY config/settings-client.yaml.template /app/config/settings-client.yaml
COPY config/settings-combiner.yaml.template /app/config/settings-combiner.yaml
COPY config/settings-hooks.yaml.template /app/config/settings-hooks.yaml
COPY config/settings-reducer.yaml.template /app/config/settings-reducer.yaml
COPY $REQUIREMENTS /app/config/requirements.txt

# Install developer tools (needed for psutil)
RUN apt-get update && apt-get install -y python3-dev gcc

# Install grpc health probe checker
RUN if [ ! -z "$GRPC_HEALTH_PROBE_VERSION" ]; then \
  apt-get install -y wget && \
  wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
  chmod +x /bin/grpc_health_probe && \
  apt-get remove -y wget && apt autoremove -y; \
  else \
  echo "No grpc_health_probe version specified, skipping installation"; \
  fi

# Setup working directory
WORKDIR /app

# Create FEDn app directory
SHELL ["/bin/bash", "-c"]
RUN mkdir -p /app \
  && mkdir -p /app/client \
  && mkdir -p /app/certs \
  && mkdir -p /app/client/package \
  && mkdir -p /app/certs \
  #
  # Install FEDn and requirements
  && python -m venv /venv \
  && /venv/bin/pip install --upgrade pip \
  && /venv/bin/pip install --no-cache-dir 'setuptools>=65' \
  && /venv/bin/pip install --no-cache-dir -e . \
  && if [[ ! -z "$REQUIREMENTS" ]]; then \
  /venv/bin/pip install --no-cache-dir -r /app/config/requirements.txt; \
  fi \
  #
  # Clean up
  && rm -r /app/config/requirements.txt

ENTRYPOINT [ "/venv/bin/fedn" ]