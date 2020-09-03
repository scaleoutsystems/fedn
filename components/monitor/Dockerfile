FROM morganekmefjord/basesci:latest

RUN mkdir /app
COPY fedn/ /app/fedn
COPY components/monitor /app
WORKDIR /app
RUN pip install -e fedn
