# Stage 1: Builder
ARG BASE_IMG=python:3.12-slim
FROM $BASE_IMG as builder

ARG GRPC_HEALTH_PROBE_VERSION=""
ARG REQUIREMENTS=""

WORKDIR /build

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
  wget -qO /bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
  chmod +x /bin/grpc_health_probe; \
  fi

# Stage 2: Runtime
FROM $BASE_IMG

WORKDIR /app

# Copy application and venv from the builder stage
COPY --from=builder /venv /venv
COPY --from=builder /build /app

# Use a non-root user
RUN useradd -m appuser && chown -R appuser /venv /app
USER appuser

ENTRYPOINT [ "/venv/bin/fedn" ]

