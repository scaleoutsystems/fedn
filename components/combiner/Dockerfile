FROM morganekmefjord/basesci:latest

RUN mkdir /app
COPY components/combiner/. /app
COPY fedn /app/fedn
COPY cli /app/cli
WORKDIR /app

RUN pip install -e cli
RUN pip install -e fedn
RUN pip install -r requirements-serve.txt