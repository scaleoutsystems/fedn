FROM morganekmefjord/basesci:latest

RUN mkdir -p /app/client
RUN mkdir /app/certs
COPY fedn /app/fedn
ADD GeoLite2-City.mmdb app/client
RUN pip install -e /app/fedn
WORKDIR /app
