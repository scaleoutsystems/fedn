FROM morganekmefjord/basesci:latest

RUN mkdir -p /app/client
RUN mkdir /app/certs
COPY fedn /app/fedn
RUN pip install -e /app/fedn
WORKDIR /app
