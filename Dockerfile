ARG BASE_IMG=python:3.11-slim
ARG RUNTIME_IMG=gcr.io/distroless/python3
FROM $BASE_IMG AS builder

ARG GRPC_HEALTH_PROBE_VERSION=""

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev gcc wget zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

# Install grpc health probe
RUN if [ ! -z "$GRPC_HEALTH_PROBE_VERSION" ]; then \
  wget -qO /build/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
  chmod +x /build/grpc_health_probe; \
  fi

COPY . /build

RUN mkdir /python-dist \
  && pip install --upgrade pip \
  && pip install --prefix=/python-dist --no-cache-dir 'setuptools>=65' .

# Stage 2: Distroless Runtime

FROM $RUNTIME_IMG

COPY --from=builder /python-dist /python-dist
COPY --from=builder /build /app

ENV PYTHONPATH=/python-dist/lib/python3.11/site-packages

ENTRYPOINT ["python3", "/python-dist/bin/fedn"]

