FROM python:3.8

RUN mkdir /app
COPY components/monitor/. /app
COPY cli /app/cli
COPY sdk /app/sdk
WORKDIR /app

RUN pip install -e sdk
RUN pip install -r requirements.txt

CMD ./run.sh