FROM python:3.8


RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

RUN mkdir /app
COPY components/dashboard/. /app
WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]
