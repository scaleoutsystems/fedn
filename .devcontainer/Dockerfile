FROM python:3.10-slim

# Non-root user with sudo access
ARG USERNAME=default
ARG USER_UID=1000
ARG USER_GID=$USER_UID


# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install apt deps
RUN apt-get --allow-releaseinfo-change update \
  && apt-get -y install --no-install-recommends \
  apt-utils \
  dialog 2>&1 \
  #
  # More apt deps
  && apt-get install -y --no-install-recommends \
  sudo \
  ca-certificates \
  wget \
  curl \
  git \
  vim \
  ssh \
  #
  # Create a non-root user to use if preferred
  && groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  #
  # Cleanup
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /home/$USERNAME

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog