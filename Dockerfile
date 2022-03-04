# Base image
ARG BASE_IMG=python:3.8.9-slim
FROM $BASE_IMG

# Requirements (use MNIST Keras as default)
ARG REQUIREMENTS=examples/mnist-keras/requirements.txt
ARG INSTALL_REQ=False

# Add FEDn and default configs
COPY fedn /app/fedn
COPY config/settings-client.yaml.template /app/config/settings-client.yaml
COPY config/settings-combiner.yaml.template /app/config/settings-combiner.yaml
COPY config/settings-reducer.yaml.template /app/config/settings-reducer.yaml
COPY $REQUIREMENTS /app/config/requirements.txt

# Create FEDn app directory
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
  && [[ "$INSTALL_REQ" == True ]] && /venv/bin/pip install --no-cache-dir -r /app/config/requirements.txt

# Setup working directory
WORKDIR /app
ENTRYPOINT [ "/venv/bin/fedn" ]