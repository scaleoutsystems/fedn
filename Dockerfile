# Base image
ARG BASE_IMG=python:3.8.9-slim
FROM $BASE_IMG

# Non-root user with sudo access
ARG USERNAME=default
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Add FEDn and default configs
COPY fedn /app/fedn
COPY config/settings-client.yaml.template /app/config/settings-client.yaml
COPY config/settings-combiner.yaml.template /app/config/settings-combiner.yaml
COPY config/settings-reducer.yaml.template /app/config/settings-reducer.yaml

# Create non-root user
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  #
  # Create FEDn app directory
  && mkdir -p /app \
  && mkdir -p /app/client \
  && mkdir -p /app/certs \
  && mkdir -p /app/client/package \
  && mkdir -p /app/certs \
  && chown -R $USERNAME /app \
  #
  # Install FEDn
  && mkdir /venv \
  && python3 -m venv /venv/fedn \
  && /venv/fedn/bin/pip install --no-cache-dir -e /app/fedn \
  && chown -R $USERNAME /venv

# Setup username, working directory and entrypoint
USER $USERNAME
WORKDIR /app
ENTRYPOINT [ "/venv/fedn/bin/fedn" ]