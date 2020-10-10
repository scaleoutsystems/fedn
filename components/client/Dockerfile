ARG baseimage="python:3.8"
FROM ${baseimage}

RUN mkdir -p /app/client
RUN mkdir /app/certs
COPY fedn /app/fedn
RUN pip install numpy
RUN pip install -e /app/fedn
WORKDIR /app
