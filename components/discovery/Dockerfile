FROM python:3.6 as pybuild1
#-alpine
LABEL maintainer="morgan@scaleoutsystems.com"
#RUN apk update
#RUN apk add musl-dev postgresql-dev gcc python3-dev linux-headers
RUN mkdir /app
#COPY requirements.txt /app/
#RUN pip install -r /app/requirements.txt

FROM pybuild1

#COPY . /app/

WORKDIR /app