FROM python:3.11-slim AS builder


WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev gcc wget zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

COPY . /build

RUN mkdir /python-dist \
  && pip install --upgrade pip \
  && pip install --prefix=/python-dist --no-cache-dir 'setuptools>=65' .

# Stage 2: Distroless Runtime
FROM gcr.io/distroless/python3

COPY --from=builder /python-dist /python-dist

ENV PYTHONPATH=/python-dist/lib/python3.11/site-packages

ENTRYPOINT ["python3", "/python-dist/bin/fedn"]

