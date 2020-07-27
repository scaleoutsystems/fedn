FROM python:3.8


RUN mkdir /app
COPY components/monitor/. /app
COPY sdk /app/sdk
COPY cli /app/cli
WORKDIR /app
RUN pip install -e cli
RUN pip install -e sdk
RUN pip install -r requirements.txt

CMD ./run.sh