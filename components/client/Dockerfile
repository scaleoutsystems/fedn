FROM scaleoutsystems/fedn-base:latest

RUN mkdir -p /app/client
RUN mkdir /app/certs
COPY fedn /app/fedn
COPY components/client/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN pip install -e /app/fedn
WORKDIR /app

