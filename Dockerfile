# Base image
ARG BASE_IMG=python:3.8.9
FROM $BASE_IMG

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Non-root user with sudo access
ARG USERNAME=default
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Add FEDn
COPY fedn /app/fedn

# Install apt deps
RUN apt-get update \
  && apt-get -y install --no-install-recommends \
  apt-utils \
  dialog 2>&1 \
  #
  # More apt deps
  && apt-get install -y --no-install-recommends \
  sudo \
  #
  # Create a non-root user
  && groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  #
  # Create FEDn app directory
  && mkdir -p /app \
  && mkdir -p /app/client \
  && mkdir -p /app/certs \
  && mkdir -p /app/client/package \
  && chown -R $USERNAME /app/client/package \
  #
  # Install fed
  && pip install --no-cache-dir -e /app/fedn \
  #
  # Cleanup
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

# Setup entrypoint
USER $USERNAME
ENTRYPOINT [ "/bin/bash" ]