FROM python:3.8


RUN mkdir /app
COPY fedn/ /app/fedn
WORKDIR /app
RUN pip install -e fedn

CMD ./run.sh