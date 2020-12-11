FROM scaleoutsystems/fedn-base:latest
COPY components/reducer/requirements.txt /app/requirements.txt
RUN mkdir -p /app && \
    mkdir -p /app/client &&\
    mkdir -p /app/certs && \
    mkdir -p /app/client/package && \
    chmod -R 777 /app/client/package
COPY fedn /app/fedn
RUN pip install -r /app/requirements.txt && \
    pip install -e /app/fedn

WORKDIR /app
