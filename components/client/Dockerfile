FROM scaleoutsystems/fedn-base:latest

COPY components/client/requirements.txt /app/requirements.txt
RUN mkdir -p /app && \
    mkdir -p /app/client && \
    mkdir -p /app/certs
COPY fedn /app/fedn
RUN pip install -r /app/requirements.txt && \
    pip install -e /app/fedn

WORKDIR /app

