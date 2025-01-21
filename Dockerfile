# Stage 1: Builder
ARG BASE_IMG=python:3.12-slim
FROM $BASE_IMG as builder

ARG GRPC_HEALTH_PROBE_VERSION=""
ARG REQUIREMENTS=""

WORKDIR /build

# Temporarily add the testing repository to install zlib1g 1.3.1
RUN echo "deb http://deb.debian.org/debian testing main" > /etc/apt/sources.list.d/testing.list \
  && apt-get update \
  && apt-get install -y --no-install-recommends -t testing zlib1g=1:1.3.dfsg+really1.3.1-1+b1 zlib1g-dev=1:1.3.dfsg+really1.3.1-1+b1 \
  && rm -rf /etc/apt/sources.list.d/testing.list \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install build dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends python3-dev gcc wget \
  && rm -rf /var/lib/apt/lists/*

# Add FEDn and default configs
COPY . /build
COPY $REQUIREMENTS /build/requirements.txt

# Install dependencies
RUN python -m venv /venv \
  && /venv/bin/pip install --upgrade pip \
  && /venv/bin/pip install --no-cache-dir 'setuptools>=65' \
  && /venv/bin/pip install --no-cache-dir . \
  && if [[ ! -z "$REQUIREMENTS" ]]; then \
  /venv/bin/pip install --no-cache-dir -r /build/requirements.txt; \
  fi \
  && rm -rf /build/requirements.txt


# Install grpc health probe
RUN if [ ! -z "$GRPC_HEALTH_PROBE_VERSION" ]; then \
  wget -qO /build/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
  chmod +x /build/grpc_health_probe; \
  fi

# Stage 2: Runtime
FROM $BASE_IMG

WORKDIR /app

# Copy application and venv from the builder stage
COPY --from=builder /venv /venv
COPY --from=builder /build /app

# Use a non-root user
RUN set -ex \
  # Create a non-root user
  && addgroup --system --gid 1001 appgroup \
  && adduser --system --uid 1001 --gid 1001 --no-create-home appuser \
  # Creare application specific tmp directory, set ENV TMPDIR to /app/tmp
  && mkdir -p /app/tmp \
  && chown -R appuser:appgroup /venv /app \
  # Temporarily add the testing repository to install zlib 1.3.1
  && echo "deb http://deb.debian.org/debian testing main" > /etc/apt/sources.list.d/testing.list \
  && apt-get update \
  && apt-get install -y --no-install-recommends -t testing zlib1g=1:1.3.dfsg+really1.3.1-1+b1 \
  && rm -rf /etc/apt/sources.list.d/testing.list \
  # Clean up
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

RUN dpkg -l | grep zlib

USER appuser

ENTRYPOINT [ "/venv/bin/fedn" ]

