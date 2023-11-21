# Base image
ARG BASE_IMG=python:3.10-slim
FROM $BASE_IMG

# Requirements (use MNIST Keras as default)
ARG REQUIREMENTS=""

# Add FEDn and default configs
COPY fedn /app/fedn
COPY config/settings-client.yaml.template /app/config/settings-client.yaml
COPY config/settings-combiner.yaml.template /app/config/settings-combiner.yaml
COPY config/settings-reducer.yaml.template /app/config/settings-reducer.yaml
COPY $REQUIREMENTS /app/config/requirements.txt

# Install developer tools (needed for psutil)
RUN apt-get update && apt-get install -y python3-dev gcc

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
  && /venv/bin/pip install --no-cache-dir -e /app/fedn \
  && if [[ ! -z "$REQUIREMENTS" ]]; then \
  /venv/bin/pip install --no-cache-dir -r /app/config/requirements.txt; \
  fi \
  #
  # Clean up
  && rm -r /app/config/requirements.txt

# Setup working directory
WORKDIR /app
ENTRYPOINT [ "/venv/bin/fedn" ]