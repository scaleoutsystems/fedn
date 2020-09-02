FROM morganekmefjord/basesci:latest

RUN mkdir /app
COPY fedn /app/fedn
COPY cli /app/cli
WORKDIR /app
RUN pip install -e cli
RUN pip install -e fedn